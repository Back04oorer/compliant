"""In-place continuation retains prior metadata, checkpoints and TensorBoard logs."""
import json
import os

import pytest

from sim.piper_softmimic.train import prepare_output


def paused_run(tmp_path):
    run=tmp_path/'run';run.mkdir()
    (run/'manifest.json').write_text(json.dumps({'status':'paused'}))
    (run/'progress.json').write_text(json.dumps({'iteration':8001,'status':'paused'}))
    (run/'model_8000.pt').write_bytes(b'checkpoint')
    (run/'events.out.tfevents.previous').write_bytes(b'scalars')
    return run


def test_resume_preserves_existing_artifacts_and_snapshots_metadata(tmp_path):
    run=paused_run(tmp_path)
    before={p.name:p.read_bytes() for p in run.iterdir()}
    snapshot=prepare_output(run,run/'model_8000.pt',True)
    assert {name:(run/name).read_bytes() for name in before}==before
    for name in ('manifest.json','progress.json'):
        assert (snapshot/name).read_bytes()==before[name]
    assert snapshot.parent==run/'resume_history'


def test_existing_directory_still_requires_explicit_in_place(tmp_path):
    run=paused_run(tmp_path)
    with pytest.raises(ValueError,match='Refusing to overwrite'):
        prepare_output(run,run/'model_8000.pt')


def test_resume_rejects_active_process_old_checkpoint_and_different_run(tmp_path):
    run=paused_run(tmp_path)
    (run/'manifest.json').write_text(json.dumps({'status':'paused','pid':os.getpid()}))
    with pytest.raises(ValueError,match='still alive'):
        prepare_output(run,run/'model_8000.pt',True)
    (run/'manifest.json').write_text(json.dumps({'status':'running'}))
    with pytest.raises(ValueError,match='paused or completed'):
        prepare_output(run,run/'model_8000.pt',True)
    (run/'manifest.json').write_text(json.dumps({'status':'paused'}))
    (run/'model_5000.pt').write_bytes(b'older')
    with pytest.raises(ValueError,match='latest checkpoint'):
        prepare_output(run,run/'model_5000.pt',True)
    with pytest.raises(ValueError,match='checkpoint in the output'):
        prepare_output(tmp_path/'other',run/'model_8000.pt',True)
    assert not (run/'resume_history').exists()
