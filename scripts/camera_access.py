#!/usr/bin/env python3
"""Share host camera descriptors with an existing Docker container over a local socket."""
import argparse
import array
import errno
import fcntl
import json
import os
from pathlib import Path
import socket

ROOT = Path(__file__).resolve().parents[1]
SOCKET = ROOT / 'outputs/camera-access/camera.sock'


def capture_devices():
    devices = []
    for entry in sorted(Path('/sys/class/video4linux').glob('video*')):
        device = Path('/dev') / entry.name
        try:
            if device.is_char_device() and (entry / 'index').read_text().strip() == '0':
                devices.append(str(device))
        except OSError:
            continue
    return devices


def request(payload):
    descriptors = []
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as peer:
            peer.settimeout(2)
            peer.connect(str(SOCKET))
            peer.sendall(json.dumps(payload).encode())
            data, ancillary, flags, _ = peer.recvmsg(8192, socket.CMSG_SPACE(4))
        for level, kind, values in ancillary:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                received = array.array('i')
                received.frombytes(values)
                descriptors.extend(received)
        if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
            raise ValueError('Truncated camera access response')
        response = json.loads(data)
        if 'error' in response:
            raise OSError(response['error'])
        for fd in descriptors:
            os.set_inheritable(fd, False)
        return response, descriptors
    except Exception:
        for fd in descriptors:
            os.close(fd)
        raise


def remote_devices():
    if not os.path.isfile('/.dockerenv'):
        return []
    try:
        response, descriptors = request({})
        for fd in descriptors:
            os.close(fd)
        return response['devices']
    except (OSError, ValueError, KeyError):
        return []


def open_camera(device):
    try:
        return os.open(device, os.O_RDWR | os.O_NONBLOCK)
    except OSError as error:
        if error.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
            raise
    response, descriptors = request({'device': device})
    if len(descriptors) != 1:
        for fd in descriptors:
            os.close(fd)
        raise OSError('Host camera access did not return a camera descriptor')
    return descriptors[0]


def serve_peer(peer):
    fd = None
    try:
        peer.settimeout(2)
        payload = json.loads(peer.recv(8192))
        devices = capture_devices()
        if 'device' not in payload:
            peer.sendall(json.dumps({'devices': devices}).encode())
            return
        device = payload['device']
        if device not in devices:
            raise ValueError('Not an available capture camera: ' + str(device))
        fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)
        peer.sendmsg([b'{}'], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', [fd]))])
    except (OSError, ValueError, TypeError) as error:
        try:
            peer.sendall(json.dumps({'error': str(error)}).encode())
        except OSError:
            pass
    finally:
        if fd is not None:
            os.close(fd)


def serve():
    if Path('/.dockerenv').exists():
        raise ValueError('Start camera access on the host, outside Docker')
    SOCKET.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (SOCKET.parent / 'camera.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if SOCKET.exists():
            SOCKET.unlink()
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as server:
            server.bind(str(SOCKET))
            os.chmod(SOCKET, 0o600)
            server.listen(8)
            print('Host camera access ready: ' + str(SOCKET), flush=True)
            try:
                while True:
                    peer, _ = server.accept()
                    with peer:
                        serve_peer(peer)
            finally:
                SOCKET.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serve', action='store_true', required=True)
    parser.parse_args()
    serve()
