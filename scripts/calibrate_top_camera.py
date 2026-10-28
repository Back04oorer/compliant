#!/usr/bin/env python3
"""Launch upstream stereo calibration with Piper's native MJPEG camera access."""
import importlib.util
import logging
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "external/fisheye_calibration/camera_calib_charuco"


def install_camera_source():
    import cv2
    import numpy as np
    from stereo_calibration import controller
    from stereo_calibration.camera import CameraSource, Frame
    from record_session import CAMERA_NODE, ensure_camera_free

    spec = importlib.util.spec_from_file_location("piper_calibration_mjpeg", CAMERA_NODE)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)

    class NativeCameraSource(CameraSource):
        def _run(self):
            camera = None
            try:
                device = self.config.device
                if isinstance(device, int) or str(device).isdigit():
                    device = "/dev/video" + str(device)
                device = str(Path(device).resolve())
                ensure_camera_free(device)
                camera = reader.DirectMjpgCamera.capture_only(
                    device, self.config.width, self.config.height, self.config.fps)
                self.actual = {
                    "backend": "Piper V4L2 MJPEG (native, no resize)",
                    "device": device, "width": camera.width, "height": camera.height,
                    "requested_fps": camera.fps, "timestamp_source": "host_receipt_monotonic",
                }
                sequence, last_frame = 0, time.monotonic()
                while not self.stopping.is_set():
                    jpeg = camera.read_jpeg()
                    if jpeg is None:
                        if time.monotonic() - last_frame > 3:
                            raise OSError("No camera frames for 3 seconds; check the device and reconnect.")
                        continue
                    timestamp = time.monotonic()
                    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if image is None or image.shape[:2] != (camera.height, camera.width):
                        raise ValueError("Decoded frame does not match the selected native camera mode.")
                    self.publish(Frame(sequence, timestamp, image))
                    sequence, last_frame = sequence + 1, timestamp
            except Exception as error:
                logging.getLogger(__name__).exception("Calibration camera stopped")
                self.error = str(error)
            finally:
                if camera is not None:
                    camera.close()
                with self.condition:
                    self.condition.notify_all()

    controller.CameraSource = NativeCameraSource


def main():
    # Installed separately from ROS Python; fail clearly before starting a server.
    try:
        from stereo_calibration.cli import main as upstream_main
        install_camera_source()
    except ImportError as error:
        raise SystemExit("Calibration environment unavailable. Run bash scripts/setup_calibration.sh "
                         "and use external/fisheye_calibration/camera_calib_charuco/.venv/bin/python. "
                         + str(error)) from error
    if sys.argv[1:] == ["--check"]:
        print("Calibration imports and native camera adapter OK; no camera opened.")
        return 0
    upstream_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
