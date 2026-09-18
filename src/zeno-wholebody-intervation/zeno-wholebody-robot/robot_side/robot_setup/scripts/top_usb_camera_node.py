#!/usr/bin/env python3
"""Minimal USB camera ROS node for top camera fallback.

Publishes:
  - image_raw (sensor_msgs/Image, bgr8)
  - image_raw/compressed (sensor_msgs/CompressedImage, jpeg)
  - camera_info (sensor_msgs/CameraInfo)
"""

import shutil
import subprocess
import time

import cv2
import rospy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image


def _parse_video_device(value: str):
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    if text.startswith("/dev/video"):
        suffix = text[len("/dev/video") :]
        if suffix.isdigit():
            return int(suffix)
    return text


def _parse_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off", ""):
        return False
    return default


def _maybe_set_fourcc(cap: cv2.VideoCapture, pixel_format: str) -> None:
    token = (pixel_format or "").strip().lower()
    if token in ("mjpeg", "mjpg"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    elif token in ("yuyv", "yuy2"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))


def _v4l2_pixfmt(pixel_format: str) -> str:
    token = (pixel_format or "").strip().lower()
    if token in ("mjpeg", "mjpg"):
        return "MJPG"
    if token in ("yuyv", "yuy2"):
        return "YUYV"
    return token.upper() if token else "MJPG"


def _fourcc_to_text(value: int) -> str:
    try:
        text = "".join(chr((int(value) >> shift) & 0xFF) for shift in (0, 8, 16, 24))
    except Exception:
        return ""
    return text.strip("\x00").strip()


def _query_v4l2_capture_status(video_device: str):
    if not isinstance(video_device, str) or not video_device.startswith("/dev/"):
        return None
    v4l2_ctl = shutil.which("v4l2-ctl")
    if not v4l2_ctl:
        return None

    try:
        completed = subprocess.run(
            [v4l2_ctl, "-d", video_device, "--get-fmt-video", "--get-parm"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
    except Exception:
        return None

    if completed.returncode != 0:
        return None

    width = 0
    height = 0
    fps = 0.0
    pixel_format = ""
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("Width/Height"):
            _, _, payload = line.partition(":")
            payload = payload.strip()
            if "/" in payload:
                width_text, height_text = payload.split("/", 1)
                try:
                    width = int(width_text.strip())
                    height = int(height_text.strip())
                except ValueError:
                    pass
        elif line.startswith("Pixel Format"):
            start = line.find("'")
            end = line.find("'", start + 1)
            if start >= 0 and end > start:
                pixel_format = line[start + 1 : end].strip()
        elif line.startswith("Frames per second"):
            _, _, payload = line.partition(":")
            token = payload.strip().split(" ", 1)[0]
            try:
                fps = float(token)
            except ValueError:
                pass

    if width <= 0 or height <= 0:
        return None

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "pixel_format": pixel_format,
    }


def _bool_pair(preconfigure_v4l2: bool, configure_capture_properties: bool):
    return (
        bool(preconfigure_v4l2),
        bool(configure_capture_properties),
    )


class TopUsbCameraNode:
    def __init__(self):
        self.video_device_raw = rospy.get_param("~video_device", "/dev/video0")
        self.image_width = int(rospy.get_param("~image_width", 1280))
        self.image_height = int(rospy.get_param("~image_height", 720))
        self.framerate = float(rospy.get_param("~framerate", 30.0))
        self.pixel_format = str(rospy.get_param("~pixel_format", "mjpeg"))
        self.frame_id = str(rospy.get_param("~camera_frame_id", "top_cam_color_optical_frame"))
        self.publish_compressed = _parse_bool(rospy.get_param("~publish_compressed", True), True)
        self.jpeg_quality = int(rospy.get_param("~jpeg_quality", 85))
        self.jpeg_quality = max(10, min(100, self.jpeg_quality))
        self.allow_any_backend = _parse_bool(rospy.get_param("~allow_any_backend", False), False)
        self.preconfigure_v4l2 = _parse_bool(rospy.get_param("~preconfigure_v4l2", True), True)
        self.configure_capture_properties = _parse_bool(
            rospy.get_param("~configure_capture_properties", True), True
        )
        self.startup_open_delay_sec = max(
            0.0, float(rospy.get_param("~startup_open_delay_sec", 0.0))
        )
        self.enforce_actual_resolution = _parse_bool(
            rospy.get_param("~enforce_actual_resolution", False), False
        )
        self.enforce_actual_pixel_format = _parse_bool(
            rospy.get_param("~enforce_actual_pixel_format", False), False
        )
        self.startup_verify_frame_count = max(
            0, int(rospy.get_param("~startup_verify_frame_count", 0))
        )
        self.startup_min_read_fps_ratio = max(
            0.0, float(rospy.get_param("~startup_min_read_fps_ratio", 0.0))
        )
        self.backend_name = "V4L2"

        self.image_pub = rospy.Publisher("image_raw", Image, queue_size=1)
        self.compressed_pub = rospy.Publisher("image_raw/compressed", CompressedImage, queue_size=1)
        self.info_pub = rospy.Publisher("camera_info", CameraInfo, queue_size=1)

        self.cap = None
        self._did_initial_open_delay = False
        self._open_attempt_count = 0
        self._open_capture()

    def _open_capture(self) -> None:
        self._release_capture()
        self._open_attempt_count += 1

        open_preconfigure_v4l2, open_configure_capture_properties = self._select_open_strategy()

        if not self._did_initial_open_delay and self.startup_open_delay_sec > 0.0:
            rospy.loginfo(
                "Top USB camera delaying first open by %.2fs to let the device settle",
                self.startup_open_delay_sec,
            )
            rospy.sleep(self.startup_open_delay_sec)
            self._did_initial_open_delay = True

        rospy.loginfo(
            "Top USB camera open attempt %d: preconfigure_v4l2=%s configure_capture_properties=%s",
            self._open_attempt_count,
            open_preconfigure_v4l2,
            open_configure_capture_properties,
        )

        self._preconfigure_v4l2(open_preconfigure_v4l2)
        v4l2_status_before_open = _query_v4l2_capture_status(self.video_device_raw)

        dev = _parse_video_device(self.video_device_raw)
        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        self.backend_name = "V4L2"
        if (not self.cap.isOpened()) and self.allow_any_backend:
            self.cap = cv2.VideoCapture(dev)
            self.backend_name = "ANY"

        if not self.cap.isOpened():
            rospy.logerr("Failed to open USB camera device: %s", self.video_device_raw)
            self._release_capture()
            return

        if open_configure_capture_properties:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.image_width))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.image_height))
            if self.framerate > 0:
                self.cap.set(cv2.CAP_PROP_FPS, self.framerate)
            # On this camera, applying FOURCC last is the only order that keeps MJPG
            # from being renegotiated back to YUYV by the driver/OpenCV stack.
            _maybe_set_fourcc(self.cap, self.pixel_format)

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        actual_fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC) or 0)
        actual_format = _fourcc_to_text(actual_fourcc)

        measured_fps = None
        if self.startup_verify_frame_count > 0 and self.startup_min_read_fps_ratio > 0.0:
            measured_fps = self._measure_startup_read_fps(self.startup_verify_frame_count)
            min_expected_fps = self.framerate * self.startup_min_read_fps_ratio
            if measured_fps < min_expected_fps:
                self._reject_open(
                    "Top USB camera startup FPS verification failed: "
                    f"measured={measured_fps:.2f} target={self.framerate:.2f} "
                    f"min_ratio={self.startup_min_read_fps_ratio:.2f}",
                )
                return

        v4l2_status = _query_v4l2_capture_status(self.video_device_raw)
        if v4l2_status is None:
            v4l2_status = v4l2_status_before_open
        if v4l2_status is not None:
            actual_width = int(v4l2_status["width"] or actual_width)
            actual_height = int(v4l2_status["height"] or actual_height)
            actual_fps = float(v4l2_status["fps"] or actual_fps)
            actual_format = str(v4l2_status["pixel_format"] or actual_format)

        if (
            self.enforce_actual_resolution
            and actual_width > 0
            and actual_height > 0
            and (actual_width != self.image_width or actual_height != self.image_height)
        ):
            self._reject_open(
                "Top USB camera negotiated unexpected resolution: "
                f"requested={self.image_width}x{self.image_height} "
                f"actual={actual_width}x{actual_height}",
            )
            return

        requested_format = _v4l2_pixfmt(self.pixel_format)
        actual_format_norm = str(actual_format or "").strip().upper()
        if (
            self.enforce_actual_pixel_format
            and requested_format
            and actual_format_norm != requested_format
        ):
            self._reject_open(
                "Top USB camera negotiated unexpected pixel format: "
                f"requested={requested_format} actual={actual_format or 'unknown'}",
            )
            return

        rospy.loginfo(
            "Top USB camera opened: %s backend=%s requested=%dx%d@%.1f %s actual=%dx%d@%.1f %s%s",
            self.video_device_raw,
            self.backend_name,
            self.image_width,
            self.image_height,
            self.framerate,
            self.pixel_format,
            actual_width,
            actual_height,
            actual_fps,
            actual_format or "unknown",
            "" if measured_fps is None else f" measured_read_fps={measured_fps:.2f}",
        )
        if (
            actual_width > 0
            and actual_height > 0
            and (actual_width != self.image_width or actual_height != self.image_height)
        ):
            rospy.logwarn(
                "Top USB camera fallback: requested %dx%d but driver negotiated %dx%d",
                self.image_width,
                self.image_height,
                actual_width,
                actual_height,
            )
        self._open_attempt_count = 0

    def _select_open_strategy(self):
        return _bool_pair(self.preconfigure_v4l2, self.configure_capture_properties)

    def _release_capture(self) -> None:
        if self.cap is None:
            return
        try:
            self.cap.release()
        except Exception:
            pass
        self.cap = None

    def _reject_open(self, reason: str) -> None:
        rospy.logwarn("%s; retrying", reason)
        self._release_capture()

    def _measure_startup_read_fps(self, frame_count: int) -> float:
        if self.cap is None or not self.cap.isOpened() or frame_count <= 0:
            return 0.0

        # Clear stale frames before measuring steady-state delivery.
        warmup_reads = min(5, frame_count)
        for _ in range(warmup_reads):
            ok, _ = self.cap.read()
            if not ok:
                return 0.0

        start = time.perf_counter()
        ok_reads = 0
        for _ in range(frame_count):
            ok, _ = self.cap.read()
            if not ok:
                break
            ok_reads += 1
        elapsed = time.perf_counter() - start
        if elapsed <= 0.0:
            return 0.0
        return ok_reads / elapsed

    def _preconfigure_v4l2(self, enabled: bool) -> None:
        if not enabled:
            return
        if not isinstance(self.video_device_raw, str) or not self.video_device_raw.startswith("/dev/"):
            return
        v4l2_ctl = shutil.which("v4l2-ctl")
        if not v4l2_ctl:
            rospy.logwarn_throttle(30.0, "v4l2-ctl not found; skipping V4L2 preconfiguration")
            return

        pixfmt = _v4l2_pixfmt(self.pixel_format)
        commands = [
            [
                v4l2_ctl,
                "-d",
                self.video_device_raw,
                f"--set-fmt-video=width={self.image_width},height={self.image_height},pixelformat={pixfmt}",
            ]
        ]
        if self.framerate > 0:
            commands.append(
                [v4l2_ctl, "-d", self.video_device_raw, f"--set-parm={int(self.framerate)}"]
            )

        for cmd in commands:
            try:
                completed = subprocess.run(
                    cmd,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5,
                )
            except Exception as exc:
                rospy.logwarn("Top USB camera V4L2 preconfigure failed: %s", exc)
                return

            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "").strip()
                rospy.logwarn(
                    "Top USB camera V4L2 preconfigure command failed (%s): %s",
                    completed.returncode,
                    detail,
                )
                return

    def _publish_frame(self, frame) -> None:
        if frame is None:
            return
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        h, w = frame.shape[:2]
        stamp = rospy.Time.now()

        image_msg = Image()
        image_msg.header.stamp = stamp
        image_msg.header.frame_id = self.frame_id
        image_msg.height = h
        image_msg.width = w
        image_msg.encoding = "bgr8"
        image_msg.is_bigendian = 0
        image_msg.step = w * 3
        image_msg.data = frame.tobytes()

        info_msg = CameraInfo()
        info_msg.header = image_msg.header
        info_msg.height = h
        info_msg.width = w
        info_msg.distortion_model = "plumb_bob"
        info_msg.D = [0.0] * 5
        info_msg.K = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info_msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info_msg.P = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

        self.image_pub.publish(image_msg)
        if self.publish_compressed and self.compressed_pub.get_num_connections() > 0:
            ok, enc = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
            )
            if ok:
                comp_msg = CompressedImage()
                comp_msg.header = image_msg.header
                comp_msg.format = "jpeg"
                comp_msg.data = enc.tobytes()
                self.compressed_pub.publish(comp_msg)
            else:
                rospy.logwarn_throttle(5.0, "Top USB camera JPEG encode failed")
        self.info_pub.publish(info_msg)

    def spin(self) -> None:
        hz = self.framerate if self.framerate > 0 else 30.0
        rate = rospy.Rate(hz)
        fail_count = 0

        while not rospy.is_shutdown():
            if self.cap is None or not self.cap.isOpened():
                self._open_capture()
                rospy.sleep(0.5)
                continue

            ok, frame = self.cap.read()
            if not ok:
                fail_count += 1
                rospy.logwarn_throttle(2.0, "Top USB camera read failed (%d)", fail_count)
                if fail_count >= 10:
                    self._open_capture()
                    fail_count = 0
                rate.sleep()
                continue

            fail_count = 0
            self._publish_frame(frame)
            rate.sleep()

        if self.cap is not None:
            self.cap.release()


def main() -> int:
    rospy.init_node("top_usb_camera_node", anonymous=False)
    node = TopUsbCameraNode()
    node.spin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
