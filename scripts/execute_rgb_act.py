#!/usr/bin/env python3
"""Send fresh ACT chunks to a separately started, locally armed ROS trial bridge."""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

from predict_rgb_act import NoRedirect, RGBActPredictor, read_ros_observation, checkpoint_dataset_root
from rgb_act_execution import validate_config
from rgb_act_ros_observer import ACTION_NAMES, STATE_NAMES


def request(session, route, payload=None):
    url = urllib.parse.urlparse(session["url"])
    if url.scheme != "http" or url.hostname != "127.0.0.1" or url.path or url.query or url.fragment or url.username:
        raise ValueError("Executor must be a loopback HTTP endpoint")
    body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    req = urllib.request.Request(session["url"] + route, data=body,
                                 headers={"Authorization": "Bearer " + session["token"], "Content-Type": "application/json"})
    with urllib.request.build_opener(NoRedirect(), urllib.request.ProxyHandler({})).open(req, timeout=1.0) as response:
        raw = response.read(16385)
    if len(raw) > 16384:
        raise ValueError("Oversized executor response")
    result = json.loads(raw)
    if result.get("schema_version") != 1 or result.get("execute") is not session["execute"]:
        raise ValueError("Executor mode/protocol changed")
    return result


def validate_checkpoint_contract(checkpoint, config):
    dataset = checkpoint_dataset_root(checkpoint)
    if dataset is None:
        raise ValueError("Execution requires the checkpoint's dataset metadata")
    info = json.loads((dataset / "meta/info.json").read_text())
    if info["fps"] != config["rate_hz"] or info.get("action_source") != "mux" or \
            info["features"]["action"]["names"] != ACTION_NAMES or \
            info["features"]["observation.state"]["names"] != STATE_NAMES:
        raise ValueError("Dataset action source, feature order, or frame rate differs from the robot contract")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--executor-session", type=Path, required=True)
    parser.add_argument("--observer-url", default="http://127.0.0.1:8766/observation")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--allow-execution", action="store_true", help="Permit connecting to an execution-enabled bridge")
    args = parser.parse_args()
    if args.executor_session.stat().st_mode & 0o077:
        parser.error("Executor session file must be private (mode 0600)")
    session = json.loads(args.executor_session.read_text())
    if session.get("schema_version") != 1 or session.get("routing") != "mux" or type(session.get("execute")) is not bool:
        parser.error("Invalid executor session")
    if session["execute"] and not args.allow_execution:
        parser.error("Execution-enabled bridge requires --allow-execution; default is dry run only")
    if args.allow_execution and not session["execute"]:
        parser.error("Bridge is in dry run mode; remove --allow-execution")
    config = validate_config(session["config"])
    checkpoint = args.checkpoint.expanduser().resolve()
    validate_checkpoint_contract(checkpoint, config)
    import torch

    torch.set_num_threads(1)
    predictor = RGBActPredictor(checkpoint, args.device, session.get("live_act", False))
    if predictor.contract["action_shape"] != [config["chunk_size"], 14]:
        raise ValueError("Checkpoint chunk size differs from the execution contract")
    return run_client(session, predictor, args.observer_url, config)


def run_client(session, predictor, observer_url, config):
    """Run an already loaded policy; the ROS bridge alone owns local ARM/stop input."""
    if session.get("live_act"):
        return run_live_client(session, predictor, observer_url, config)
    print("Model loaded. Waiting for local ARM in the bridge terminal. Ctrl-C stops this client.", flush=True)
    started = False
    previous_readiness = None
    previous_phase = None
    try:
        while True:
            status = request(session, "/status")
            if not status["armed"]:
                if started:
                    print(status["reason"], flush=True)
                    break
                readiness = (status['ready'], status['error'])
                if readiness != previous_readiness:
                    print('Bridge ready. Type ARM + Enter to start.' if status['ready'] else
                          'Test blocked: ' + str(status['error']), flush=True)
                    previous_readiness = readiness
                time.sleep(0.1)
                continue
            started = True
            if not status["ready"]:
                raise ValueError("Bridge lost readiness: " + str(status["error"]))
            phase = status.get("phase", "policy")
            if phase != previous_phase:
                print("Moving slowly to dataset episode 0, frame 0." if phase == "reset" else
                      "First-frame positioning complete; starting live ACT." if previous_phase == "reset" else
                      "Starting live ACT.", flush=True)
                previous_phase = phase
            if phase == "reset":
                time.sleep(0.05)
                continue
            if status["remaining"]:
                time.sleep(min(0.05, status["remaining"] / config["rate_hz"]))
                continue
            observation, metadata = read_ros_observation(observer_url, timeout=0.5,
                                                         max_age=config["observation_timeout_s"])
            if any(stamp <= status.get("policy_after", 0.0) for stamp in metadata["source_stamps"].values()):
                time.sleep(0.05)
                continue
            actions = predictor.predict(observation)
            result = request(session, "/chunk", {"sequence": status["sequence"] + 1,
                                                "rate_hz": config["rate_hz"], "action_names": ACTION_NAMES,
                                                "source_stamps": metadata["source_stamps"],
                                                "observation_stamp": metadata["observation_stamp"],
                                                "actions": actions.tolist()})
            print("Accepted chunk %d (%s)" % (result["sequence"], "EXECUTION" if session["execute"] else "DRY RUN"), flush=True)
    finally:
        try:
            request(session, "/stop", {})
        except Exception as exc:
            print("Could not request stop: %s; check the bridge/hardware stop. Motor power may remain on." % exc, flush=True)
    return 0


def run_live_client(session, predictor, observer_url, config):
    """TicTac-style live observations -> temporal-ensembled single actions."""
    print('Model loaded. Local ARM starts live ACT. Space pauses; Enter resets; q/Ctrl-C returns to start, then exits.', flush=True)
    epoch, previous_stamp, previous_phase = None, 0., None
    started, previous_readiness = False, None
    count, window = 0, time.monotonic()
    interrupted = False
    try:
        while True:
            tick = time.monotonic()
            status = request(session, '/status')
            if status.get('local_exit_requested'):
                break
            if not status['armed']:
                if started:
                    raise ValueError('Live ACT stopped: ' + status['reason'])
                readiness = (status['ready'], status['error'])
                if readiness != previous_readiness:
                    print('Bridge ready. Type ARM + Enter to start.' if status['ready'] else
                          'ACT blocked: ' + str(status['error']), flush=True)
                    previous_readiness = readiness
                time.sleep(.1)
                continue
            started = True
            if not status['ready']:
                raise ValueError('Live ACT lost readiness: ' + str(status['error']))
            if status.get('returning_to_start'):
                time.sleep(.03)
                continue
            current_epoch = status['policy_epoch']
            if current_epoch != epoch:
                predictor.reset()
                epoch, previous_stamp = current_epoch, 0.
                count, window = 0, time.monotonic()
            phase = status['phase']
            if phase != previous_phase:
                print({'reset': 'Positioning to dataset first frame.',
                       'paused': 'Holding pose. Space resumes; Enter returns to first frame.',
                       'policy': 'Live ACT running with fresh observations and temporal ensemble.'}[phase], flush=True)
                previous_phase = phase
                if phase == 'policy':
                    count, window = 0, time.monotonic()
            if phase != 'policy':
                time.sleep(.03)
                continue
            observation, metadata = read_ros_observation(observer_url, timeout=.5,
                                                         max_age=config['observation_timeout_s'])
            if (metadata['observation_stamp'] <= previous_stamp or
                    any(t <= status['policy_after'] for t in metadata['source_stamps'].values())):
                time.sleep(.01)
                continue
            action = predictor.select_action(observation)
            result = request(session, '/step', {'sequence': status['sequence'] + 1,
                'policy_epoch': epoch, 'rate_hz': config['rate_hz'], 'action_names': ACTION_NAMES,
                'source_stamps': metadata['source_stamps'], 'observation_stamp': metadata['observation_stamp'],
                'action': action.tolist()})
            if not result['accepted']:
                # A local pause/reset may happen while inference is in flight.
                predictor.reset()
                epoch = None
                continue
            previous_stamp = metadata['observation_stamp']
            count += 1
            now = time.monotonic()
            if now - window >= 1.:
                print('Live ACT: %.1f accepted steps/s; latest sequence %d' %
                      (count / (now - window), result['sequence']), flush=True)
                count, window = 0, now
            time.sleep(max(0., 1. / config['rate_hz'] - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        interrupted = True
        return 0
    finally:
        try:
            request(session, '/quit' if interrupted else '/stop', {})
        except (OSError, ValueError) as error:
            print('Could not request ACT stop; check the bridge terminal: ' + str(error), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("ACT client stopped; bridge stop requested.")
    except urllib.error.HTTPError as exc:
        raise SystemExit("Execution rejected (%d): %s" % (exc.code, exc.read(4096).decode(errors="replace")))
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        raise SystemExit("ACT client stopped: " + str(exc))
