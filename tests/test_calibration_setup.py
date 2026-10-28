"""Verify patch migration against a disposable checkout, without network or cameras."""
import os
import subprocess

import pytest

from test_entrypoints import ROOT


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], check=True,
                          capture_output=True, text=True).stdout


@pytest.mark.parametrize('installed', [0, 1, 2])
@pytest.mark.parametrize('dirty', [False, True])
def test_setup_migrates_only_bundled_patches(tmp_path, installed, dirty):
    repo = tmp_path / 'external/fisheye_calibration'
    project = repo / 'camera_calib_charuco'
    project.mkdir(parents=True)
    git(repo, 'init')
    files = [project / 'guidance.txt', project / 'board.txt']
    for file in files:
        file.write_text('original\n')
    git(repo, 'add', '.')
    git(repo, '-c', 'user.name=Test', '-c', 'user.email=test@example.com', 'commit', '-m', 'baseline')
    revision = git(repo, 'rev-parse', 'HEAD').strip()
    patches = tmp_path / 'patches'
    patches.mkdir()
    for file, name in zip(files, ['fisheye-bottom-ab.patch', 'fisheye-charuco-7x5-38.patch']):
        file.write_text('updated\n')
        (patches / name).write_text(git(repo, 'diff', '--', str(file)))
        file.write_text('original\n')
    for file in files[:installed]:
        file.write_text('updated\n')
    if dirty:
        files[0].write_text(files[0].read_text() + 'local work\n')
    before = [file.read_bytes() for file in files]
    script = tmp_path / 'scripts/setup_calibration.sh'
    script.parent.mkdir()
    script.write_text((ROOT / 'scripts/setup_calibration.sh').read_text().replace(
        'REVISION=6ea4206ab2b1279171ee04a27990178b9992db57', f'REVISION={revision}'))
    python = project / '.venv/bin/python'
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\nexit 0\n')
    python.chmod(0o755)
    env = dict(os.environ, UV_BIN='/bin/true')
    for _ in range(2):
        result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
        if dirty:
            assert result.returncode != 0 and 'Nothing reset' in result.stderr
            assert [file.read_bytes() for file in files] == before
        else:
            assert result.returncode == 0, result.stderr
            assert [file.read_text() for file in files] == ['updated\n', 'updated\n']
        assert git(repo, 'diff', '--cached') == ''
