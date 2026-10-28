#!/usr/bin/env python3
"""Locally armed, bounded ACT-to-ROS bridge. Dry run unless --execute is given."""
import argparse
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import select
import signal
import sys
import termios
import threading
import time
import tty
import urllib.request
import xmlrpc.client

from rgb_act_execution import COMMAND_TOPICS, MUX_TOPICS, ExecutionGuard, command_messages
from rgb_act_ros_observer import IMAGE_TOPICS, STATE_TOPICS, ObservationBuffer
from recorded_replay import RecordedReplayGuard
from live_act import LiveACTGuard


class ReplayKeyboard:
    """Read local replay keys immediately and restore the terminal on every exit path."""
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        self.line = ''
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *_exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def read(self):
        if not select.select([self.fd], [], [], 0)[0]:
            return []
        data = os.read(self.fd, 256)
        events = []
        if not data:
            return ['quit']
        for key in data.decode('ascii', errors='ignore'):
            if key == '\x04' or (key == 'q' and not self.line):
                events.append('quit')
            elif key == ' ' and not self.line:
                events.append('pause')
            elif key in '\r\n':
                events.append({'ARM': 'arm', 'QUIT': 'quit', '': 'rewind'}.get(self.line.upper(), 'invalid'))
                self.line = ''
                print(flush=True)
            elif key in ('\x7f', '\b'):
                self.line = self.line[:-1]
                print('\b \b', end='', flush=True)
            elif key.isprintable():
                self.line += key
                print(key, end='', flush=True)
        return events


def graph_connections(node_name):
    body = xmlrpc.client.dumps((node_name,), methodname="getSystemState").encode()
    req = urllib.request.Request(os.environ.get("ROS_MASTER_URI", "http://127.0.0.1:11311"), data=body,
                                 headers={"Content-Type": "text/xml"})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=0.05) as response:
        raw = response.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("Oversized ROS graph response")
    code, message, state = xmlrpc.client.loads(raw)[0][0]
    if code != 1:
        raise ValueError("ROS graph unavailable: " + message)
    return dict(state[0]), dict(state[1])


def select_mux(ros, side, topic, message_type):
    from topic_tools.srv import MuxSelect

    service = MUX_TOPICS[side] + "_select"
    confirmed = threading.Event()
    subscriber = ros.Subscriber(MUX_TOPICS[side] + "/selected", message_type,
                                lambda msg: confirmed.set() if msg.data == topic else None, queue_size=1)
    try:
        ros.wait_for_service(service, timeout=0.5)
        ros.ServiceProxy(service, MuxSelect)(topic)
        # A latched previous selection can arrive before the new service result.
        if not confirmed.wait(0.5):
            raise ValueError("Mux selection was not confirmed for " + side)
    except (ros.ROSException, ros.ServiceException) as exc:
        raise ValueError("Mux switch failed for %s: %s" % (side, exc)) from exc
    finally:
        subscriber.unregister()
    return topic


class Trial:
    def __init__(self, config, observations, ros, execute, publishers, selected_subscribers, event_log, message_type,
                 reset_positions=None, states_only=False, recording_plan=None, live_act=False, mux_select=None):
        if live_act and (states_only or recording_plan is not None):
            raise ValueError('Live ACT requires live RGB/state observations and no recording plan')
        if recording_plan is not None and not states_only:
            raise ValueError('Recording replay requires state-only observations')
        self.guard = (RecordedReplayGuard(config, reset_positions, recording_plan) if recording_plan else
                      LiveACTGuard(config, reset_positions) if live_act else
                      ExecutionGuard(config, reset_positions, states_only=states_only))
        self.live_act = live_act
        self.command_topics = COMMAND_TOPICS
        self.mux_select = mux_select
        self.mux_owned = set()
        self.observations, self.ros, self.execute = observations, ros, execute
        self.publishers, self.selected_subscribers, self.event_log = publishers, selected_subscribers, event_log
        self.message_type = message_type
        self.selected, self.diagnostics, self.gravity = {}, {}, {}
        self.lock = threading.RLock()
        self.telemetry_lock = threading.Lock()
        self.graph_checked = 0.0
        self.graph_error = "ROS graph has not been checked"
        self.mux_outputs_seen = set()
        self.mux_pending = set()
        self.first_publish_time = None
        self.exit_requested = False
        self.local_exit_requested = False
        self.returning_to_start = False

    def event(self, event, **fields):
        self.event_log.write(json.dumps({"unix_time": time.time(), "event": event, **fields}, allow_nan=False) + "\n")
        self.event_log.flush()

    def check_graph(self, now):
        if now - self.graph_checked < 0.25:
            if self.graph_error:
                raise ValueError(self.graph_error)
            return
        self.graph_checked = now
        try:
            self.check_connections(now)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self.graph_error = str(exc)
            raise
        self.graph_error = None

    def check_connections(self, now):
        published, subscribed = graph_connections(self.ros.get_name())
        self.mux_pending = set()
        rules = getattr(self.guard, 'recording_rules', None)
        if rules:
            for side, params in rules['drivers'].items():
                for key, expected in params.items():
                    name = '/piper_ctrl_%s_node/%s' % (side, key)
                    if self.ros.get_param_cached(name, None) != expected:
                        raise ValueError('Current control parameter differs from recording: ' + name)
        for side, topic in self.command_topics.items():
            expected = {self.ros.get_name()} if self.execute else set()
            if set(published.get(topic, [])) != expected:
                raise ValueError("Unexpected command publishers on %s: %s; expected %s" %
                                 (topic, sorted(published.get(topic, [])), sorted(expected)))
            if not self.execute:
                continue
            mux_node = "/mux_arm_%s_joint" % side
            output = MUX_TOPICS[side]
            actual = set(published.get(output, []))
            if actual == {mux_node}:
                self.mux_outputs_seen.add(side)
            elif actual:
                raise ValueError("Unexpected publishers on %s: %s; expected %s" %
                                 (output, sorted(actual), mux_node))
            elif side in self.mux_outputs_seen:
                raise ValueError("Mux output disappeared: " + output)
            else:
                # topic_tools/mux advertises its ShapeShifter output only after the first input message.
                self.mux_pending.add(side)
                if self.first_publish_time is not None and now - self.first_publish_time > 1.0:
                    raise ValueError("Mux output did not initialize after ARM: " + output)
            if set(published.get(output + '/selected', [])) != {mux_node}:
                raise ValueError("Unexpected mux selected-status publisher: " + side)
            if set(subscribed.get(topic, [])) != {mux_node}:
                raise ValueError("Expected only %s to subscribe to %s" % (mux_node, topic))
            if "/piper_ctrl_%s_node" % side not in subscribed.get(output, []):
                raise ValueError("Follower driver is not subscribed to " + output)
            if self.selected_subscribers[side].get_num_connections() != 1:
                raise ValueError("Mux selected-status publisher is missing or ambiguous: " + side)
            if self.execute and self.publishers[side].get_num_connections() != 1:
                raise ValueError("Expected exactly one mux subscriber for " + topic)
            node = "/piper_ctrl_%s_node/" % side
            if self.live_act:
                for key, expected_value in {"ctrl_mode": "mit", "mit/enable_pos": True,
                                            "mit/enable_vel": False, "mit/enable_tor": False}.items():
                    if self.ros.get_param_cached(node + key, None) != expected_value:
                        raise ValueError('Live ACT requires position-only MIT driver: ' + node + key)
            params = {"gripper_val_mutiple": 1.0, "gripper_reverse": False, "enable_gripper": True,
                      "gripper_cmd_limit_enable": True, "gripper_cmd_min_raw": 0, "gripper_cmd_max_raw": 80000}
            for key, expected_value in params.items():
                defaults = {"gripper_val_mutiple": 1.0, "gripper_reverse": False, "enable_gripper": True}
                if self.ros.get_param_cached(node + key, defaults.get(key)) != expected_value:
                    raise ValueError("Check driver gripper parameter: " + node + key)

    def ready(self, now):
        self.check_graph(now)
        with self.telemetry_lock:
            now = time.monotonic()
            ros_now = self.ros.Time.now().to_sec()
            snapshot = self.observations.snapshot(ros_now, now)
            measured = self.guard.feedback(snapshot, self.diagnostics, self.selected, self.gravity, ros_now, now,
                                           require_control=self.execute,
                                           require_routing=self.guard.active or bool(self.mux_owned))
        return measured, ros_now

    def status(self):
        with self.lock:
            if self.guard.active:
                self.guard.last_client = time.monotonic()
            try:
                measured, _ = self.ready(time.monotonic())
                ready, error = True, None
            except (ValueError, KeyError, TypeError, OSError) as exc:
                ready, error = False, str(exc)
            progress = (self.guard.reset_progress(measured, time.monotonic())
                        if ready and self.guard.active and self.guard.phase == 'reset' else None)
            return {"schema_version": 1, "execute": self.execute, "armed": self.guard.active,
                    "ready": ready, "error": error, "reason": self.guard.reason,
                    "phase": self.guard.phase, "policy_after": self.guard.policy_after,
                    "policy_epoch": getattr(self.guard, "policy_epoch", 0),
                    "local_exit_requested": self.local_exit_requested,
                    "returning_to_start": self.returning_to_start,
                    "reset_progress": progress,
                    "sequence": self.guard.sequence, "remaining": len(self.guard.queue)}

    def arm(self):
        with self.lock:
            self.graph_checked = 0.0
            measured, _ = self.ready(time.monotonic())
            with self.telemetry_lock:
                self.guard.arm(measured, self.diagnostics, time.monotonic())
            try:
                if self.execute:
                    # Seed the policy input with a measured hold only after local ARM.
                    self.publish(measured)
                    for side, topic in COMMAND_TOPICS.items():
                        self.mux_owned.add(side)
                        self.selected[side] = None
                        self.selected[side] = self.mux_select(side, topic)
                    self.graph_checked = 0.0
                    measured, _ = self.ready(time.monotonic())
                    # Service/selection confirmation time is outside the control-loop budget.
                    self.guard.active = False
                    with self.telemetry_lock:
                        self.guard.arm(measured, self.diagnostics, time.monotonic())
                self.event("armed", execute=self.execute, measured=measured, reset_positions=self.guard.reset_positions)
            except Exception as exc:
                self.stop("Mux handoff failed: " + str(exc))
                raise

    def restore_teleop(self):
        for side in tuple(self.mux_owned):
            try:
                # Respect an operator's handoff to another input.
                if self.selected.get(side) in (None, COMMAND_TOPICS[side]):
                    topic = "/teleop/arm_%s/joint_states_single" % side
                    self.selected[side] = self.mux_select(side, topic)
                    self.event("mux_restored", side=side, topic=topic)
                self.mux_owned.remove(side)
            except Exception as exc:
                self.event("mux_restore_failed", side=side, error=str(exc))
                print("Could not restore teleop mux for %s: %s; use Teleop menu 7." % (side, exc), flush=True)

    def publish(self, command):
        if not self.execute:
            return
        if self.first_publish_time is None:
            self.first_publish_time = time.monotonic()
        stamp = self.ros.Time.now()
        for side, msg in zip(STATE_TOPICS, command_messages(command, self.message_type, stamp)):
            self.publishers[side].publish(msg)

    def replay_key(self, key):
        with self.lock:
            if key == 'quit':
                self.quit(exit_bridge=self.live_act)
                return
            if self.local_exit_requested or self.returning_to_start:
                return
            if key == 'arm':
                if self.guard.active:
                    raise ValueError('Already armed; use Space or Enter')
                self.arm()
            elif key in ('pause', 'rewind'):
                measured, _ = self.ready(time.monotonic())
                now = time.monotonic()
                if key == 'pause':
                    if self.live_act:
                        self.guard.toggle_pause(measured, now)
                    else:
                        self.guard.toggle_pause(now)
                else:
                    self.guard.rewind(measured, now)
                if self.live_act:
                    self.guard.policy_after = self.ros.Time.now().to_sec()
                self.event(key, phase=self.guard.phase)
            else:
                raise ValueError('Use ARM + Enter, Space, Enter, or q')
            print(self.guard.reason, flush=True)

    def quit(self, exit_bridge=True):
        with self.lock:
            if self.returning_to_start:
                return
            if self.live_act and self.guard.active and self.guard.reset_positions is not None:
                try:
                    measured, _ = self.ready(time.monotonic())
                    self.guard.rewind(measured, time.monotonic())
                    self.guard.policy_after = self.ros.Time.now().to_sec()
                    self.returning_to_start = True
                    self.event('exit_return_started', reset_positions=self.guard.reset_positions)
                    print('Returning slowly to dataset first frame before exit. Ctrl-C again stops the return.', flush=True)
                    return
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    self.stop('Exit return unavailable: ' + str(exc))
            else:
                self.stop('Local operator exit')
            self.exit_requested = exit_bridge
            self.local_exit_requested = True

    def stop(self, reason):
        with self.lock:
            if self.returning_to_start:
                self.returning_to_start = False
                self.exit_requested = self.local_exit_requested = True
            was_active = self.guard.active
            self.guard.stop(reason)
            if not was_active:
                self.restore_teleop()
                return
            held = False
            try:
                # Best effort only. Never write after a manual mux takeover or loss of trustworthy feedback.
                measured, _ = self.ready(time.monotonic())
                self.publish(measured)
                held = self.execute
            except Exception:
                pass
            self.restore_teleop()
            self.event("stopped", reason=str(reason), measured_hold_sent=held)
            print(self.guard.reason + "; motor power is unchanged", flush=True)

    def submit(self, payload):
        with self.lock:
            try:
                now = time.monotonic()
                measured, ros_now = self.ready(now)
                if self.live_act and (not self.guard.active or self.guard.phase != 'policy' or
                                        payload.get('policy_epoch') != self.guard.policy_epoch):
                    return False
                self.guard.submit(payload, measured, ros_now, now)
                self.event("step" if self.live_act else "chunk", sequence=self.guard.sequence,
                           source_stamps=payload["source_stamps"],
                           **({"raw_action": payload["action"], "guarded_target": self.guard.target}
                              if self.live_act else {}))
                return True
            except Exception as exc:
                self.stop(("Rejected step: " if self.live_act else "Rejected chunk: ") + str(exc))
                raise

    def tick(self):
        with self.lock:
            if not self.guard.active:
                return
            try:
                now = time.monotonic()
                measured, _ = self.ready(now)
                if self.returning_to_start:
                    # Local shutdown owns this reset even after the model client exits.
                    self.guard.last_client = time.monotonic()
                if self.execute and self.mux_pending:
                    now = time.monotonic()
                    if self.first_publish_time is not None and now - self.first_publish_time > 1.0:
                        raise ValueError('Mux output initialization timed out: ' + ', '.join(sorted(self.mux_pending)))
                    # After local ARM, prime a cold mux with the stationary measured pose before moving.
                    self.guard.last_plan = now
                    self.guard.check_step(measured, now)
                    self.guard.started += now - self.guard.last_tick
                    self.guard.last_tick = now
                    self.publish(self.guard.last_command)
                    self.event('mux_initializing', sides=sorted(self.mux_pending))
                    return
                previous_phase = self.guard.phase
                with self.telemetry_lock:
                    stationary = all(abs(v) <= 0.03 for item, _ in self.diagnostics.values() for v in item['velocities_rad_s'])
                command = self.guard.step(measured, time.monotonic(), stationary=stationary)
                if previous_phase == "reset" and self.guard.phase == "policy":
                    self.guard.policy_after = self.ros.Time.now().to_sec()
                    self.event("first_frame_reached", measured=measured)
                self.publish(command)
                self.event("command" if self.execute else "dry_run_command", positions=command,
                           measured=measured, remaining=len(self.guard.queue), phase=self.guard.phase)
                if self.returning_to_start and self.guard.phase == 'paused':
                    self.event('exit_return_reached', measured=measured)
                    self.stop('Returned to dataset first frame; exiting')
            except Exception as exc:
                self.stop(exc)


class CommandHandler(BaseHTTPRequestHandler):
    def reply(self, code, payload):
        data = json.dumps(payload, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def authorized(self):
        if not secrets.compare_digest(self.headers.get("Authorization", "").encode(),
                                      ("Bearer " + self.server.token).encode()):
            self.reply(403, {"error": "Invalid local session token"})
            return False
        return True

    def do_GET(self):
        if self.authorized():
            if self.path == "/status":
                self.reply(200, self.server.trial.status())
            else:
                self.reply(404, {})

    def do_POST(self):
        if not self.authorized():
            return
        if self.path not in ("/chunk", "/step", "/stop", "/quit"):
            self.reply(404, {})
            return
        try:
            self.connection.settimeout(0.2)
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 128 * 1024 or self.headers.get("Transfer-Encoding"):
                raise ValueError("Expected a JSON body of at most 128 KiB")
            raw = self.rfile.read(size)
            if len(raw) != size:
                raise ValueError("Truncated command request")
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object")
            if self.path in ("/stop", "/quit"):
                if self.path == "/quit":
                    if payload.get('immediate'):
                        self.server.trial.stop('ACT client aborted')
                    self.server.trial.quit()
                else:
                    self.server.trial.stop("ACT client requested stop")
            else:
                if (self.path == '/step') != self.server.trial.live_act:
                    raise ValueError('Use /step for live ACT and /chunk for chunk execution')
                accepted = self.server.trial.submit(payload)
                self.reply(200, dict(self.server.trial.status(), accepted=accepted))
                return
            self.reply(200, self.server.trial.status())
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self.server.trial.stop("Invalid request: " + str(exc))
            self.reply(409, {"error": str(exc)})

    def log_message(self, *_args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--session-dir", type=Path, required=True, help="New, private session directory")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--execute", action="store_true", help="Allow actuator publishers; still requires local ARM")
    parser.add_argument('--live-act', action='store_true', help='Live RGB ACT through mux; J2/J3 use the recorded MIT protocol bounds')
    parser.add_argument('--states-only', action='store_true', help='Replay with follower feedback and no cameras')
    parser.add_argument('--recording-plan', type=Path, help='Preloaded trajectory and original driver parameters')
    parser.add_argument("--reset-pose", type=Path, help="Validated dataset first-frame pose JSON; move only after local ARM")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    recording_plan = None if args.recording_plan is None else json.loads(args.recording_plan.read_text())
    if args.live_act and (args.states_only or recording_plan is not None):
        parser.error('--live-act requires live RGB/state observations and no recording plan')
    if recording_plan is not None and not args.states_only:
        parser.error('--recording-plan requires --states-only')
    reset_positions = None if args.reset_pose is None else json.loads(args.reset_pose.read_text())["positions"]
    if recording_plan is not None:
        validated = RecordedReplayGuard(config, reset_positions, recording_plan)
    else:
        validated = (LiveACTGuard(config, reset_positions) if args.live_act else
                     ExecutionGuard(config, reset_positions, states_only=args.states_only))
    config, reset_positions = validated.config, validated.reset_positions
    if not 1 <= args.port <= 65535 or args.session_dir.exists():
        parser.error("Choose an unused port and a new session directory")
    if not sys.stdin.isatty():
        parser.error("A local interactive terminal is required for ARM / Enter-to-stop")
    import rospy
    from sensor_msgs.msg import CompressedImage, JointState
    from std_msgs.msg import Bool, String

    rospy.init_node("rgb_act_executor", anonymous=True, disable_signals=True)
    if rospy.get_param("/use_sim_time", False):
        raise ValueError("Hardware execution requires wall-clock ROS time")
    observations = ObservationBuffer(config["observation_timeout_s"], 0.05, states_only=args.states_only)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), CommandHandler)
    args.session_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    token = secrets.token_hex(32)
    session = args.session_dir / "session.json"
    session.touch(mode=0o600, exist_ok=False)
    session.write_text(json.dumps({"schema_version": 1, "url": "http://127.0.0.1:%d" % args.port,
                                   "token": token, "execute": args.execute, "config": config,
                                   "reset_positions": reset_positions, "states_only": args.states_only,
                                   "routing": "mux", "live_act": args.live_act,
                                   "recording_rules": None if recording_plan is None else recording_plan['rules']}, indent=2) + "\n")
    subscribers, publishers, selected_subscribers = [], {}, {}
    worker = None
    with (args.session_dir / "events.jsonl").open("x") as log:
        trial = Trial(config, observations, rospy, args.execute, publishers, selected_subscribers, log, JointState,
                      reset_positions, states_only=args.states_only, recording_plan=recording_plan, live_act=args.live_act,
                      mux_select=lambda side, topic: select_mux(rospy, side, topic, String))
        server.trial, server.token = trial, token

        def receive(message, key):
            try:
                observations.add(key, message)
            except (ValueError, TypeError, OverflowError) as exc:
                observations.invalidate(key, exc)

        def telemetry(message, side, kind):
            with trial.telemetry_lock:
                if kind == "selected":
                    trial.selected[side] = message.data
                elif kind == "gravity":
                    trial.gravity[side] = (message.data, time.monotonic())
                else:
                    try:
                        trial.diagnostics[side] = (json.loads(message.data), time.monotonic())
                    except (ValueError, TypeError):
                        trial.diagnostics.pop(side, None)

        interrupts = [0]
        handled_interrupts = 0
        previous_sigint = None
        if args.live_act:
            def interrupted(_signum, _frame):
                interrupts[0] += 1
            previous_sigint = signal.signal(signal.SIGINT, interrupted)
        try:
            for key, topic in {**observations.image_topics, **STATE_TOPICS}.items():
                subscribers.append(rospy.Subscriber(topic, CompressedImage if key in IMAGE_TOPICS else JointState,
                                                    receive, callback_args=key, queue_size=2, buff_size=8 * 1024 * 1024))
            for side in STATE_TOPICS:
                prefix = "/robot/arm_%s/" % side
                for kind, suffix, msg_type in (("selected", "joint_cmd_mux/selected", String),
                                                ("diagnostics", "agent_driver_state", String),
                                                ("gravity", "follower_gravity_active", Bool)):
                    sub = rospy.Subscriber(prefix + suffix, msg_type,
                                           lambda msg, s=side, k=kind: telemetry(msg, s, k), queue_size=1)
                    subscribers.append(sub)
                    if kind == "selected":
                        selected_subscribers[side] = sub
                if args.execute:
                    publishers[side] = rospy.Publisher(trial.command_topics[side], JointState, queue_size=1, latch=False)
            worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
            worker.start()
            print(("EXECUTION ENABLED" if args.execute else "DRY RUN: no actuator publishers") +
                  "; DISARMED; session " + str(session), flush=True)
            print(('Type ARM + Enter. Space pauses/resumes; Enter returns slowly to dataset first frame; q exits.'
                   if recording_plan or args.live_act else
                   'Type ARM + Enter for one bounded trial. Enter stops. QUIT exits. Hardware E-stop remains necessary.'), flush=True)
            if args.live_act:
                print('ARM first positions to the dataset first frame, then runs live ACT until stopped. '
                      'Space holds/resumes; Enter returns to first frame and holds; q/Ctrl-C returns to first frame, then exits.', flush=True)
            elif reset_positions is not None:
                print("ARM first moves to the validated start pose (60s maximum), then starts the %.1fs trajectory." %
                      (recording_plan['execution_seconds'] if recording_plan else config["trial_seconds"]), flush=True)
            with ReplayKeyboard() if recording_plan or args.live_act else nullcontext() as keyboard:
                while not rospy.is_shutdown() and not trial.exit_requested:
                    deadline = time.monotonic() + 1.0 / config["rate_hz"]
                    if interrupts[0] > handled_interrupts:
                        if interrupts[0] > 1:
                            trial.stop('Exit return cancelled by Ctrl-C')
                            trial.exit_requested = trial.local_exit_requested = True
                        else:
                            trial.quit()
                        handled_interrupts = interrupts[0]
                    if trial.exit_requested:
                        break
                    if keyboard is not None:
                        for key in keyboard.read():
                            try:
                                trial.replay_key(key)
                            except (ValueError, KeyError, TypeError, OSError) as exc:
                                print('Replay control rejected: ' + str(exc), flush=True)
                    elif select.select([sys.stdin], [], [], 0)[0]:
                        line = sys.stdin.readline()
                        if not line or line.strip().upper() == "QUIT":
                            break
                        if line.strip() == "ARM":
                            try:
                                trial.arm()
                                print(("ARMED: start-pose positioning, then %.1fs trajectory" if reset_positions is not None else
                                       "ARMED for %.1fs") % config["trial_seconds"], flush=True)
                            except (ValueError, KeyError, TypeError, OSError) as exc:
                                print("Cannot arm: " + str(exc), flush=True)
                        else:
                            trial.stop("Local operator stop")
                    trial.tick()
                    time.sleep(max(0, deadline - time.monotonic()))
        finally:
            trial.stop("Bridge exiting")
            if worker is not None:
                server.shutdown()
                worker.join(timeout=1)
            server.server_close()
            for sub in subscribers:
                sub.unregister()
            for pub in publishers.values():
                pub.unregister()
            rospy.signal_shutdown("ACT trial ended")
            if previous_sigint is not None:
                signal.signal(signal.SIGINT, previous_sigint)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("ACT bridge stopped. Motors may remain enabled; use the hardware stop if needed.")
    except (OSError, ValueError, ImportError) as exc:
        raise SystemExit("ACT bridge stopped: " + str(exc))
