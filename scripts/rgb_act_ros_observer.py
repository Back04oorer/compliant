#!/usr/bin/env python3
"""Subscribe to RGB/state ROS topics and serve observations on localhost. No commands."""
import argparse
import base64
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import secrets
import signal
import threading
import time

IMAGE_TOPICS = {
    "observation.image": "/top_camera/image_raw/compressed",
    "observation.left_wrist_image": "/fisheye_left/usb_cam_node/image_raw/compressed",
    "observation.right_wrist_image": "/fisheye_right/usb_cam_node/image_raw/compressed",
}
STATE_TOPICS = {
    "left": "/robot/arm_left/joint_states_single",
    "right": "/robot/arm_right/joint_states_single",
}
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
STATE_NAMES = [arm + "." + joint + "." + field
               for arm in ("left", "right") for field in ("pos", "effort_raw")
               for joint in JOINT_NAMES]
ACTION_NAMES = [arm + "." + joint + ".pos" for arm in ("left", "right") for joint in JOINT_NAMES]
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class NotReady(ValueError):
    pass


class ObservationBuffer:
    def __init__(self, max_age=0.5, max_skew=0.05, states_only=False):
        if not math.isfinite(max_age) or not math.isfinite(max_skew) or min(max_age, max_skew) <= 0:
            raise ValueError("Age and synchronization limits must be finite and positive")
        self.max_age, self.max_skew = max_age, max_skew
        self.image_topics = {} if states_only else IMAGE_TOPICS
        self.streams = {key: deque(maxlen=120) for key in (*self.image_topics, *STATE_TOPICS)}
        self.errors = {}
        self.lock = threading.Lock()

    def add(self, key, message, received=None):
        stamp = float(message.header.stamp.to_sec())
        if not math.isfinite(stamp) or stamp <= 0:
            raise ValueError("Missing or invalid ROS header timestamp")
        if key in IMAGE_TOPICS:
            data = bytes(message.data)
            if not data.startswith(b"\xff\xd8") or len(data) > 4 * 1024 * 1024:
                raise ValueError("Expected a JPEG CompressedImage, at most 4 MiB")
        elif key in STATE_TOPICS:
            if list(message.name) != JOINT_NAMES:
                raise ValueError("Joint names/order differ from the recorded Piper contract")
            if len(message.position) != 7 or len(message.effort) != 7:
                raise ValueError("State needs seven positions and seven raw efforts")
            data = [float(v) for v in (*message.position, *message.effort)]
            if not all(math.isfinite(v) for v in data):
                raise ValueError("Non-finite robot state")
        else:
            raise ValueError("Unknown observation stream: " + key)
        with self.lock:
            if self.streams[key] and stamp <= self.streams[key][-1][0]:
                raise ValueError("Non-increasing ROS timestamp")
            self.streams[key].append((stamp, time.monotonic() if received is None else received, data))
            self.errors.pop(key, None)

    def invalidate(self, key, reason):
        with self.lock:
            self.streams[key].clear()
            self.errors[key] = str(reason)

    def snapshot(self, ros_now, monotonic_now=None):
        now = time.monotonic() if monotonic_now is None else monotonic_now
        if not math.isfinite(ros_now) or ros_now <= 0:
            raise NotReady("ROS clock is unavailable")
        with self.lock:
            missing = [key for key, stream in self.streams.items() if not stream]
            if missing:
                raise NotReady("Missing observations: " + ", ".join(missing) + "; " + str(self.errors))
            reference = min(stream[-1][0] for stream in self.streams.values())
            selected = {key: min(stream, key=lambda sample: abs(sample[0] - reference))
                        for key, stream in self.streams.items()}
        stamps = [sample[0] for sample in selected.values()]
        skew = max(stamps) - min(stamps)
        if skew > self.max_skew:
            raise NotReady("Observation synchronization skew %.4fs exceeds %.4fs" % (skew, self.max_skew))
        for key, (stamp, received, _) in selected.items():
            if now - received > self.max_age or ros_now - stamp > self.max_age:
                raise NotReady("Stale observation: " + key)
            if received > now or stamp - ros_now > self.max_skew:
                raise NotReady("Observation clock is in the future: " + key)
        return {
            "schema_version": 1,
            "prediction_only": True,
            "generated_at_unix": time.time(),
            "observation_stamp": reference,
            "source_stamps": {key: sample[0] for key, sample in selected.items()},
            "max_skew_sec": skew,
            "state_names": STATE_NAMES,
            "observation.state": selected["left"][2] + selected["right"][2],
            "images": {key: base64.b64encode(selected[key][2]).decode("ascii") for key in self.image_topics},
        }


class ObservationHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        token = self.server.shutdown_token
        if self.path != '/quit':
            self.send_error(404)
        elif not token or not secrets.compare_digest(self.headers.get('Authorization', '').encode(), ('Bearer ' + token).encode()):
            self.send_error(403)
        else:
            self.server.exit_requested = True
            self.send_response(200)
            self.send_header('Content-Length', '0')
            self.end_headers()

    def do_GET(self):
        if self.path not in ("/observation", "/health"):
            self.send_error(404)
            return
        try:
            payload = self.server.observations.snapshot(self.server.ros_clock())
            if self.path == "/health":
                payload = {"ready": True, "prediction_only": True,
                           "source_stamps": payload["source_stamps"], "max_skew_sec": payload["max_skew_sec"]}
            status = 200
        except NotReady as error:
            payload = {"ready": False, "prediction_only": True, "error": str(error)}
            status = 503
        encoded = json.dumps(payload, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_args):
        pass


def make_server(observations, port, ros_clock=time.time):
    server = ThreadingHTTPServer(("127.0.0.1", port), ObservationHandler)
    server.observations, server.ros_clock = observations, ros_clock
    server.timeout = 0.2
    server.shutdown_token, server.exit_requested = None, False
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument('--states-only', action='store_true', help='Observe followers without camera subscriptions')
    parser.add_argument('--session-file', type=Path, help='New private launcher session; permits --port 0')
    parser.add_argument("--max-age", type=float, default=0.5, help="Maximum source/receive age in seconds")
    parser.add_argument("--max-skew", type=float, default=0.05, help="Maximum inter-stream timestamp skew")
    parser.add_argument("--top-topic", default=IMAGE_TOPICS["observation.image"])
    parser.add_argument("--left-image-topic", default=IMAGE_TOPICS["observation.left_wrist_image"])
    parser.add_argument("--right-image-topic", default=IMAGE_TOPICS["observation.right_wrist_image"])
    parser.add_argument("--left-state-topic", default=STATE_TOPICS["left"])
    parser.add_argument("--right-state-topic", default=STATE_TOPICS["right"])
    args = parser.parse_args()
    if not (0 if args.session_file else 1) <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    if args.session_file and args.session_file.exists():
        parser.error('Observer session file already exists')
    observations = ObservationBuffer(args.max_age, args.max_skew, states_only=args.states_only)
    topics = dict(zip((*IMAGE_TOPICS, *STATE_TOPICS),
                     (args.top_topic, args.left_image_topic, args.right_image_topic,
                      args.left_state_topic, args.right_state_topic)))
    if args.states_only:
        topics = {key: topics[key] for key in STATE_TOPICS}
    if any(not topic.startswith("/") for topic in topics.values()) or len(set(topics.values())) != len(topics):
        parser.error("Provide distinct absolute ROS topics")

    import rospy
    from sensor_msgs.msg import CompressedImage, JointState

    server = make_server(observations, args.port, lambda: rospy.Time.now().to_sec())
    subscribers = []
    try:
        if args.session_file:
            server.shutdown_token = secrets.token_hex(32)
            args.session_file.touch(mode=0o600, exist_ok=False)
            args.session_file.write_text(json.dumps({'schema_version': 1, 'prediction_only': True,
                'url': 'http://127.0.0.1:%d/observation' % server.server_port,
                'shutdown_token': server.shutdown_token, 'pid': os.getpid()}) + '\n')
            def terminate(_signum, _frame):
                raise KeyboardInterrupt
            signal.signal(signal.SIGTERM, terminate)
        rospy.init_node("rgb_act_prediction_observer", anonymous=True, disable_signals=True)

        def receive(message, key):
            try:
                observations.add(key, message)
            except (ValueError, TypeError, OverflowError) as error:
                observations.invalidate(key, error)
                rospy.logwarn_throttle(5, "Rejected %s: %s", key, error)

        for key, topic in topics.items():
            msg_type = CompressedImage if key in IMAGE_TOPICS else JointState
            subscribers.append(rospy.Subscriber(topic, msg_type, receive, callback_args=key,
                                                queue_size=2, buff_size=8 * 1024 * 1024, tcp_nodelay=True))
        print("PREDICTIONS ONLY: subscriber-only observer at http://127.0.0.1:%d" % server.server_port, flush=True)
        print(json.dumps(topics, indent=2), flush=True)
        while not rospy.is_shutdown() and not server.exit_requested:
            server.handle_request()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        for subscriber in subscribers:
            subscriber.unregister()
        rospy.signal_shutdown("Prediction observer stopped")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, ImportError) as error:
        raise SystemExit("Observer stopped: " + str(error))
