"""SoftMimic training dashboard, evaluation replay, and Isaac live proxy."""
import argparse
import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[2]
WEB = Path(__file__).with_name("web")


def public_files():
    # Exact routes prevent exposing repository files, checkpoints or directory listings.
    files = {"/": WEB / "softmimic.html", "/index.html": WEB / "softmimic.html",
             "/style.css": WEB / "style.css"}
    files['/softmimic.html'] = WEB / 'softmimic.html'
    files['/softmimic.js'] = WEB / 'softmimic.js'
    files['/softmimic.css'] = WEB / 'softmimic.css'
    for name in ('replay.html', 'replay.js', 'replay.css', 'soft-live.html', 'soft-live.js', 'vendor/three.module.js', 'vendor/three.core.js'):
        files['/'+name] = WEB / name
    files['/media/replay.json'] = ROOT / 'outputs/piper_compliance/softmimic_official/replay/data.json'
    for name in ('manifest.json', 'force_0.5.json', 'force_1.json', 'force_2.json', 'force_3.json', 'force_5.json'):
        files['/media/rollouts/'+name] = ROOT / 'outputs/piper_compliance/softmimic_official/replay/forces' / name
    for name in ('base_link', 'gripper_base', *[f'link{i}' for i in range(1,9)]):
        files[f'/meshes/{name}.STL'] = ROOT / f'src/piper_ros/src/piper_description/meshes/{name}.STL'
    files['/media/softmimic_progress.json'] = ROOT / 'outputs/piper_compliance/softmimic_official/train/progress.json'
    evaluations = ROOT / 'outputs/piper_compliance/softmimic_official/evaluations'
    for report in evaluations.glob('*/report.json'):
        for name in ('report.json', 'summary.csv', 'xyz_positive.png', 'xyz_negative.png', 'bias_diagnosis.png'):
            files[f'/media/evaluations/{report.parent.name}/{name}'] = report.parent / name
    return files


class DashboardHandler(BaseHTTPRequestHandler):
    def send_json(self, body, status=200):
        encoded = json.dumps(body, separators=(",", ":"), allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(encoded)

    def do_POST(self):
        operations = {f'/api/soft-live/{op}': op for op in ('create', 'step', 'close')}
        operation = operations.get(urlsplit(self.path).path)
        if operation is None:
            self.send_error(501)
            return
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            self.send_json({"error": "Cross-origin simulation commands are not allowed"}, 403)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096 or self.headers.get_content_type() != "application/json":
                raise ValueError("Expected a JSON object of at most 4096 bytes")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object")
            try:
                request = Request('http://127.0.0.1:8766/' + operation, data=json.dumps(payload).encode(),
                                  headers={'Content-Type': 'application/json'})
                with urlopen(request, timeout=12) as response:
                    self.send_json(json.load(response))
            except HTTPError as error:
                self.send_json(json.load(error), error.code)
            except URLError:
                self.send_json({'error': 'Isaac 实时服务未就绪，请稍后重新连接'}, 503)
        except KeyError:
            self.send_json({"error": "Simulation session expired; reconnect to start again"}, 410)
        except (ValueError, TypeError, UnicodeError) as error:
            self.send_json({"error": str(error)}, 400)
        except (RuntimeError, TimeoutError) as error:
            self.send_json({"error": str(error) or "Simulation request timed out"}, 503)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            self.log_error("Live simulation failed; check rendering runtime and policy")
            self.send_json({"error": "Simulation unavailable; check server rendering logs"}, 503)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        route = urlsplit(self.path).path
        run = parse_qs(urlsplit(self.path).query).get('run', [None])[0]
        chart_match = re.fullmatch(r'/api/training-charts/([0-5])\.png', route)
        if chart_match:
            from .training_curves import TAGS, chart_png, training_curves
            try:
                data = training_curves(ROOT / 'outputs/piper_compliance/softmimic_official').read(run)
                index = int(chart_match.group(1))
                body = chart_png(index, tuple(tuple(row) for row in data['series'][TAGS[index]]), data['run'])
            except KeyError:
                self.send_error(404, 'Unknown training run')
                return
            except (ImportError, OSError, ValueError):
                self.send_error(503, 'Training chart temporarily unavailable')
                return
            self.send_response(200)
            self.send_header('Content-Type', 'image/png')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(body)
            return
        if route == '/api/training-curves':
            from .training_curves import training_curves
            try:
                data = training_curves(ROOT / 'outputs/piper_compliance/softmimic_official').read(run)
                self.send_json(data)
            except KeyError:
                self.send_json({'error': '找不到该训练 run'}, 404)
            except ImportError:
                self.send_json({'error': '曲线需要 TensorBoard；请使用 .venv-softmimic/bin/python 启动网页服务'}, 503)
            except (OSError, ValueError) as error:
                self.send_json({'error': f'训练日志暂不可用：{error}'}, 503)
            return
        files = public_files()
        path = files.get(route)
        if path is None or not path.is_file():
            self.send_error(404)
            return
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        requested = self.headers.get("Range")
        if requested:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
            if match and any(match.groups()):
                first, last = match.groups()
                if first:
                    start, end = int(first), min(int(last), end) if last else end
                else:
                    start = max(0, size - int(last))
                    if int(last) == 0:
                        start = size
            else:
                start = size
            if start >= size or start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command != "HEAD":
            try:
                with path.open("rb") as source:
                    source.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        block = source.read(min(65536, remaining))
                        if not block:
                            break
                        self.wfile.write(block)
                        remaining -= len(block)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Browser seeking cancels in-flight media requests.

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        ancestors = "'self'" if urlsplit(self.path).path == '/replay.html' else "'none'"
        self.send_header("Content-Security-Policy", f"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self'; connect-src 'self'; frame-ancestors {ancestors}")
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()
    with ThreadingHTTPServer((args.host, args.port), DashboardHandler) as server:
        print(f"Piper SoftMimic dashboard: http://{args.host}:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
