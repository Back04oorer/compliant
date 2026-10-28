"""SoftMimic dashboard routes and HTTP boundaries, without Isaac startup."""
import http.client
import io
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError, URLError

import pytest

from sim.piper_compliance import dashboard


@pytest.fixture
def viewer(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, 'ROOT', tmp_path)
    output = tmp_path / 'outputs/piper_compliance/softmimic_official'
    (output / 'train').mkdir(parents=True)
    (output / 'train/progress.json').write_text('{"status":"running","iteration":7}')
    (output / 'replay').mkdir()
    (output / 'replay/data.json').write_bytes(b'0123456789')
    evaluation = output / 'evaluations/test_run'
    evaluation.mkdir(parents=True)
    (evaluation / 'report.json').write_text('{"status":"complete"}')
    (evaluation / 'summary.csv').write_text('axis,error\nX,0.01\n')
    server = ThreadingHTTPServer(('127.0.0.1', 0), dashboard.DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection(*server.server_address)

    def request(route, headers=None, method='GET', body=None):
        connection.request(method, route, headers=headers or {}, body=body)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()

    try:
        yield request, 'http://%s:%s' % server.server_address
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join()


def test_dashboard_serves_softmimic_without_legacy_experiment_outputs(viewer):
    get, _ = viewer
    for route in ('/', '/index.html', '/?view=live&run=cma-v3'):
        status, _, body = get(route)
        assert status == 200 and b'/softmimic.js' in body
        assert b'/live.js' not in body and b'/media/xyz_comparison.png' not in body
    for route in ('/softmimic.html', '/softmimic.js', '/softmimic.css', '/style.css',
                  '/replay.js', '/vendor/three.module.js'):
        assert get(route)[0] == 200
    for route in ('/replay.html', '/replay.html?embedded=1'):
        status, headers, _ = get(route)
        assert status == 200
        assert "frame-ancestors 'self'" in headers['Content-Security-Policy']
    for route in ('/', '/softmimic.html', '/soft-live.html'):
        assert "frame-ancestors 'none'" in get(route)[1]['Content-Security-Policy']
    assert json.loads(get('/media/softmimic_progress.json')[2])['iteration'] == 7
    assert json.loads(get('/media/evaluations/test_run/report.json')[2])['status'] == 'complete'
    assert get('/media/evaluations/test_run/summary.csv')[0] == 200
    for route in ('/api/live/status', '/api/experiment', '/live.js', '/live.css', '/app.js',
                  '/media/xyz_comparison.png', '/media/X_policy.mp4', '/media/cma_report.json',
                  '/media/evaluations/test_run/traces.npz', '/media/evaluations/test_run/model_5500.pt',
                  '/media/softmimic_manifest.json', '/meshes/../../etc/passwd', '/../README.md',
                  '/media/../../README.md', '/media/%2e%2e/README.md', '/.git/config',
                  '/media/policy.zip', '/media/'):
        assert get(route)[0] == 404
    assert get('/api/live/step', method='POST')[0] == 501


def test_dashboard_preserves_training_curves_and_run_path_validation(viewer):
    get, _ = viewer
    status, _, body = get('/api/training-curves')
    if status != 200:
        assert status == 503 and 'TensorBoard' in json.loads(body)['error']
        return
    curves = json.loads(body)
    assert curves['run'] == 'train' and curves['progress']['iteration'] == 7
    assert all(not rows for rows in curves['series'].values())
    image_status, image_headers, image_body = get('/api/training-charts/0.png')
    assert image_status == 200 and image_headers['Content-Type'] == 'image/png'
    assert image_body.startswith(b'\x89PNG\r\n\x1a\n')
    assert get('/api/training-charts/6.png')[0] == 404
    assert get('/api/training-charts/../manifest.json')[0] == 404
    assert json.loads(get('/api/training-curves?run=train')[2])['run'] == 'train'
    for run in ('missing', '..%2Ftrain', '%2Fetc'):
        assert get('/api/training-curves?run=' + run)[0] == 404
        assert get('/api/training-charts/0.png?run=' + run)[0] == 404


def test_dashboard_preserves_range_requests_and_head(viewer):
    get, _ = viewer
    route = '/media/replay.json'
    status, headers, body = get(route, {'Range': 'bytes=2-5'})
    assert status == 206 and body == b'2345'
    assert headers['Content-Range'] == 'bytes 2-5/10'
    assert headers['Accept-Ranges'] == 'bytes'
    assert get(route, {'Range': 'bytes=-3'})[2] == b'789'
    assert get(route, {'Range': 'bytes=7-'})[2] == b'789'
    for invalid in ('bytes=50-', 'bytes=5-2', 'bytes=-0', 'bytes=0-1,5-6', 'invalid'):
        assert get(route, {'Range': invalid})[0] == 416
    status, headers, body = get(route, method='HEAD')
    assert status == 200 and headers['Content-Length'] == '10' and not body


def test_isaac_proxy_requires_small_same_origin_json_commands(viewer, monkeypatch):
    request, origin = viewer
    calls = []

    def worker(command, timeout):
        calls.append(command.full_url)
        assert timeout == 12
        payload = json.loads(command.data)
        if payload.get('session') == 'expired':
            raise HTTPError(command.full_url, 410, 'Expired', {}, io.BytesIO(b'{"error":"expired"}'))
        return io.BytesIO(json.dumps({'received': payload}).encode())

    monkeypatch.setattr(dashboard, 'urlopen', worker)

    def post(body, request_origin=origin, content_type='application/json', operation='step'):
        return request('/api/soft-live/' + operation, method='POST', body=body,
                       headers={'Content-Type': content_type, 'Origin': request_origin})

    for operation in ('create', 'step', 'close'):
        status, _, body = post('{"force":[0,2,0]}', operation=operation)
        assert status == 200 and json.loads(body)['received']['force'] == [0, 2, 0]
    assert calls == ['http://127.0.0.1:8766/' + op for op in ('create', 'step', 'close')]
    assert post('{}', 'https://unrelated.example')[0] == 403
    for body in ('[]', 'invalid json', ' ' * 4097, ''):
        assert post(body)[0] == 400
    assert post('{}', content_type='text/plain')[0] == 400
    assert len(calls) == 3
    assert post('{"session":"expired"}')[0] == 410

    def unavailable(*args, **kwargs):
        raise URLError('Worker not running')

    monkeypatch.setattr(dashboard, 'urlopen', unavailable)
    assert post('{}')[0] == 503
