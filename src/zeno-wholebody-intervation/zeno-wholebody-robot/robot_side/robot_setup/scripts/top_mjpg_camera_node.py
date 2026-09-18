#!/usr/bin/env python3
import ctypes
import errno
import fcntl
import mmap
import os
import select
import time
from typing import List, Tuple

import rospy
from sensor_msgs.msg import CameraInfo, CompressedImage


VIDIOC_G_FMT = 0xC0D05604
VIDIOC_S_FMT = 0xC0D05605
VIDIOC_REQBUFS = 0xC0145608
VIDIOC_QUERYBUF = 0xC0585609
VIDIOC_QBUF = 0xC058560F
VIDIOC_DQBUF = 0xC0585611
VIDIOC_STREAMON = 0x40045612
VIDIOC_STREAMOFF = 0x40045613
VIDIOC_S_PARM = 0xC0CC5616

V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_FIELD_NONE = 1
V4L2_MEMORY_MMAP = 1


def _fourcc(token: str) -> int:
    value = token.encode("ascii")
    if len(value) != 4:
        raise ValueError(f"FOURCC must be four ASCII chars, got {token!r}")
    return value[0] | (value[1] << 8) | (value[2] << 16) | (value[3] << 24)


def _fourcc_to_str(value: int) -> str:
    return bytes(
        (
            value & 0xFF,
            (value >> 8) & 0xFF,
            (value >> 16) & 0xFF,
            (value >> 24) & 0xFF,
        )
    ).decode("ascii", errors="replace")


class V4L2PixFormat(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("pixelformat", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("bytesperline", ctypes.c_uint32),
        ("sizeimage", ctypes.c_uint32),
        ("colorspace", ctypes.c_uint32),
        ("priv", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("ycbcr_enc", ctypes.c_uint32),
        ("quantization", ctypes.c_uint32),
        ("xfer_func", ctypes.c_uint32),
    ]


class V4L2FormatUnion(ctypes.Union):
    _fields_ = [
        ("pix", V4L2PixFormat),
        ("raw_data", ctypes.c_uint8 * 200),
        ("align", ctypes.c_uint64),
    ]


class V4L2Format(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("fmt", V4L2FormatUnion)]


class V4L2Fract(ctypes.Structure):
    _fields_ = [("numerator", ctypes.c_uint32), ("denominator", ctypes.c_uint32)]


class V4L2CaptureParm(ctypes.Structure):
    _fields_ = [
        ("capability", ctypes.c_uint32),
        ("capturemode", ctypes.c_uint32),
        ("timeperframe", V4L2Fract),
        ("extendedmode", ctypes.c_uint32),
        ("readbuffers", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 4),
    ]


class V4L2StreamParmUnion(ctypes.Union):
    _fields_ = [
        ("capture", V4L2CaptureParm),
        ("raw_data", ctypes.c_uint8 * 200),
        ("align", ctypes.c_uint64),
    ]


class V4L2StreamParm(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("parm", V4L2StreamParmUnion)]


class V4L2RequestBuffers(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
    ]


class TimeVal(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class V4L2TimeCode(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("frames", ctypes.c_uint8),
        ("seconds", ctypes.c_uint8),
        ("minutes", ctypes.c_uint8),
        ("hours", ctypes.c_uint8),
        ("userbits", ctypes.c_uint8 * 4),
    ]


class V4L2BufferMemory(ctypes.Union):
    _fields_ = [
        ("offset", ctypes.c_uint32),
        ("userptr", ctypes.c_ulong),
        ("planes", ctypes.c_void_p),
        ("fd", ctypes.c_int32),
    ]


class V4L2BufferRequest(ctypes.Union):
    _fields_ = [("request_fd", ctypes.c_int32), ("reserved", ctypes.c_uint32)]


class V4L2Buffer(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("bytesused", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("timestamp", TimeVal),
        ("timecode", V4L2TimeCode),
        ("sequence", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("m", V4L2BufferMemory),
        ("length", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32),
        ("request_fd", V4L2BufferRequest),
    ]


class DirectMjpgCamera:
    def __init__(self) -> None:
        self.video_device = str(rospy.get_param("~video_device", "/dev/video0"))
        self.width = int(rospy.get_param("~image_width", 640))
        self.height = int(rospy.get_param("~image_height", 480))
        self.fps = int(rospy.get_param("~framerate", 30))
        self.frame_id = str(rospy.get_param("~camera_frame_id", "top_cam_color_optical_frame"))
        self.buffer_count = int(rospy.get_param("~buffer_count", 4))
        self.open_retry_sleep_sec = float(rospy.get_param("~open_retry_sleep_sec", 0.5))
        self.select_timeout_sec = float(rospy.get_param("~select_timeout_sec", 1.0))
        self.enforce_resolution = _as_bool(rospy.get_param("~enforce_actual_resolution", True))
        self.enforce_mjpg = _as_bool(rospy.get_param("~enforce_actual_pixel_format", True))

        self.image_pub = rospy.Publisher("image_raw/compressed", CompressedImage, queue_size=2)
        self.info_pub = rospy.Publisher("camera_info", CameraInfo, queue_size=2)
        self.fd = -1
        self.buffers: List[Tuple[mmap.mmap, int]] = []
        self.streaming = False

    def open(self) -> None:
        while not rospy.is_shutdown():
            try:
                self._open_once()
                return
            except Exception as exc:
                self.close()
                rospy.logerr("Failed to open MJPG camera %s: %s", self.video_device, exc)
                time.sleep(self.open_retry_sleep_sec)

    def _open_once(self) -> None:
        self.fd = os.open(self.video_device, os.O_RDWR | os.O_NONBLOCK)
        self._set_format()
        self._set_fps()
        fmt = self._get_format()
        actual_width = int(fmt.fmt.pix.width)
        actual_height = int(fmt.fmt.pix.height)
        actual_fourcc = _fourcc_to_str(int(fmt.fmt.pix.pixelformat))
        if self.enforce_resolution and (actual_width != self.width or actual_height != self.height):
            raise RuntimeError(
                f"resolution mismatch: requested {self.width}x{self.height}, got {actual_width}x{actual_height}"
            )
        if self.enforce_mjpg and actual_fourcc != "MJPG":
            raise RuntimeError(f"pixel format mismatch: requested MJPG, got {actual_fourcc}")
        self._init_mmap()
        self._stream_on()
        rospy.loginfo(
            "Top MJPG camera ready: device=%s actual=%dx%d %s fps=%d buffers=%d",
            self.video_device,
            actual_width,
            actual_height,
            actual_fourcc,
            self.fps,
            len(self.buffers),
        )

    def _set_format(self) -> None:
        fmt = V4L2Format()
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        fmt.fmt.pix.width = self.width
        fmt.fmt.pix.height = self.height
        fmt.fmt.pix.pixelformat = _fourcc("MJPG")
        fmt.fmt.pix.field = V4L2_FIELD_NONE
        try:
            fcntl.ioctl(self.fd, VIDIOC_S_FMT, fmt, True)
        except OSError as exc:
            current = self._get_format()
            if exc.errno != errno.EIO or _fourcc_to_str(int(current.fmt.pix.pixelformat)) != "MJPG":
                raise
            rospy.logwarn("VIDIOC_S_FMT returned EIO, but current device format is already MJPG; continuing.")

    def _get_format(self) -> V4L2Format:
        fmt = V4L2Format()
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        fcntl.ioctl(self.fd, VIDIOC_G_FMT, fmt, True)
        return fmt

    def _set_fps(self) -> None:
        parm = V4L2StreamParm()
        parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        parm.parm.capture.timeperframe.numerator = 1
        parm.parm.capture.timeperframe.denominator = max(1, self.fps)
        try:
            fcntl.ioctl(self.fd, VIDIOC_S_PARM, parm, True)
        except OSError as exc:
            if exc.errno != errno.EIO:
                raise
            rospy.logwarn("VIDIOC_S_PARM returned EIO; continuing with current driver FPS.")

    def _init_mmap(self) -> None:
        req = V4L2RequestBuffers()
        req.count = max(2, self.buffer_count)
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        req.memory = V4L2_MEMORY_MMAP
        fcntl.ioctl(self.fd, VIDIOC_REQBUFS, req, True)
        if req.count < 2:
            raise RuntimeError(f"insufficient V4L2 mmap buffers: {req.count}")

        self.buffers = []
        for index in range(int(req.count)):
            buf = V4L2Buffer()
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            buf.memory = V4L2_MEMORY_MMAP
            buf.index = index
            fcntl.ioctl(self.fd, VIDIOC_QUERYBUF, buf, True)
            mapped = mmap.mmap(
                self.fd,
                int(buf.length),
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
                offset=int(buf.m.offset),
            )
            self.buffers.append((mapped, int(buf.length)))
            self._queue_buffer(index)

    def _queue_buffer(self, index: int) -> None:
        buf = V4L2Buffer()
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = V4L2_MEMORY_MMAP
        buf.index = index
        fcntl.ioctl(self.fd, VIDIOC_QBUF, buf, True)

    def _stream_on(self) -> None:
        buf_type = ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE)
        fcntl.ioctl(self.fd, VIDIOC_STREAMON, buf_type, True)
        self.streaming = True

    def _stream_off(self) -> None:
        if self.fd >= 0 and self.streaming:
            try:
                buf_type = ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE)
                fcntl.ioctl(self.fd, VIDIOC_STREAMOFF, buf_type, True)
            except OSError:
                pass
        self.streaming = False

    def spin(self) -> None:
        rate = rospy.Rate(max(1, self.fps * 2))
        target_period = 1.0 / float(self.fps) if self.fps > 0 else 0.0
        next_publish_time = 0.0
        while not rospy.is_shutdown():
            if self.fd < 0:
                self.open()
                continue
            ready, _, _ = select.select([self.fd], [], [], self.select_timeout_sec)
            if not ready:
                rospy.logwarn_throttle(5.0, "No MJPG frame ready from %s", self.video_device)
                rate.sleep()
                continue

            buf = V4L2Buffer()
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            buf.memory = V4L2_MEMORY_MMAP
            try:
                fcntl.ioctl(self.fd, VIDIOC_DQBUF, buf, True)
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EINTR):
                    continue
                rospy.logerr("VIDIOC_DQBUF failed: %s; reopening camera.", exc)
                self.close()
                continue

            stamp = rospy.Time.now()
            try:
                mapped, _ = self.buffers[int(buf.index)]
                frame = bytes(mapped[: int(buf.bytesused)])
                if frame.startswith(b"\xff\xd8"):
                    now = time.monotonic()
                    if target_period <= 0.0 or now >= next_publish_time:
                        self._publish_frame(frame, stamp)
                        if target_period > 0.0:
                            if next_publish_time <= 0.0 or now - next_publish_time > target_period:
                                next_publish_time = now + target_period
                            else:
                                next_publish_time += target_period
                else:
                    rospy.logwarn_throttle(2.0, "Dropped non-JPEG MJPG buffer: bytesused=%d", int(buf.bytesused))
            finally:
                try:
                    self._queue_buffer(int(buf.index))
                except OSError as exc:
                    rospy.logerr("VIDIOC_QBUF failed: %s; reopening camera.", exc)
                    self.close()
            rate.sleep()

    def _publish_frame(self, frame: bytes, stamp: rospy.Time) -> None:
        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.format = "jpeg"
        msg.data = frame
        self.image_pub.publish(msg)

        info = CameraInfo()
        info.header = msg.header
        info.width = self.width
        info.height = self.height
        self.info_pub.publish(info)

    def close(self) -> None:
        self._stream_off()
        for mapped, _ in self.buffers:
            try:
                mapped.close()
            except Exception:
                pass
        self.buffers = []
        if self.fd >= 0:
            try:
                os.close(self.fd)
            except OSError:
                pass
        self.fd = -1


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def main() -> None:
    rospy.init_node("top_mjpg_camera_node")
    camera = DirectMjpgCamera()
    rospy.on_shutdown(camera.close)
    camera.open()
    camera.spin()


if __name__ == "__main__":
    main()
