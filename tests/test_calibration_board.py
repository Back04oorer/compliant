"""Integration checks run with the separate calibration environment installed."""
import shutil
import subprocess
import sys

import pytest

from test_entrypoints import ROOT

PRESET = 'stereo-charuco-7x5-38-a4-v1'
PROJECT = ROOT / 'external/fisheye_calibration/camera_calib_charuco'


@pytest.fixture
def board(monkeypatch):
    if not (PROJECT / 'stereo_calibration').is_dir():
        pytest.skip('Run scripts/setup_calibration.sh to install calibration dependencies')
    pytest.importorskip('cv2')
    monkeypatch.syspath_prepend(str(PROJECT))
    from stereo_calibration.config import PRESETS
    return PRESETS[PRESET]


def test_supplied_pdf_matches_detector_geometry(board, tmp_path):
    import cv2
    import numpy as np
    from stereo_calibration.observations import Detector

    assert board.to_dict() == dict(columns=7, rows=5, square_mm=38.0, marker_mm=30.0,
                                   dictionary='DICT_APRILTAG_25h9', legacy=False)
    geometry = board.create()
    assert geometry.getIds().reshape(-1).tolist() == list(range(17))
    assert geometry.getChessboardCorners().shape == (24, 3)
    if not shutil.which('pdftoppm'):
        pytest.skip('Poppler is needed to verify the supplied PDF')
    source = ROOT / 'docs/assets/calibration/stereo-charuco-7x5-38-a4-v1_A4.pdf'
    target = tmp_path / 'board'
    subprocess.run(['pdftoppm', '-scale-to', '1500', '-png', '-singlefile', str(source), str(target)],
                   check=True, capture_output=True)
    image = cv2.imread(str(target.with_suffix('.png')))
    observation, quality = Detector(board).observe(np.hstack([image, image]))
    assert observation.common()[0].tolist() == list(range(24))
    assert [eye['markers'] for eye in quality] == [17, 17]
    expected = geometry.getChessboardCorners()[:, :2] * (image.shape[1] / 297)
    expected += [(297 - 266) / 2 * image.shape[1] / 297,
                 (210 - 190) / 2 * image.shape[0] / 210]
    np.testing.assert_allclose(observation.left_points, expected, atol=2)


@pytest.mark.parametrize('preset', [None, 'existing-20mm', 'a4-30mm', PRESET])
def test_gui_cli_passes_selected_board_to_workspace(board, monkeypatch, tmp_path, preset):
    import uvicorn
    from stereo_calibration import cli, web
    from stereo_calibration.config import PRESETS

    workspaces = []
    monkeypatch.setattr(web, 'create_app', lambda workspace, port: workspaces.append(workspace))
    monkeypatch.setattr(uvicorn, 'run', lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, 'argv', ['calibrate', 'gui', '--no-browser', '--output', str(tmp_path)]
                        + (['--board', preset] if preset else []))
    cli.main()
    assert workspaces[0].board_defaults == PRESETS[preset or PRESET]


def test_web_setup_and_preview_use_new_board(board, tmp_path):
    import cv2
    import numpy as np
    from fastapi.testclient import TestClient
    from stereo_calibration.web import create_app
    from stereo_calibration.workspace import Workspace

    with TestClient(create_app(Workspace(board, output=tmp_path)),
                    base_url='http://127.0.0.1:8765') as client:
        assert client.get('/api/setup').json()['presets'][PRESET] == board.to_dict()
        assert client.get('/api/state').json()['board'] == board.to_dict()
        assert f'value="{PRESET}"' in client.get('/').text
        response = client.get('/api/board.png', params={'preset': PRESET})
        assert response.status_code == 200
        image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_GRAYSCALE)
        corners, ids, *_ = cv2.aruco.CharucoDetector(board.create()).detectBoard(image)
        assert len(ids) == len(corners) == 24
