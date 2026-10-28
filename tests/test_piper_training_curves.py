"""Read historical and appended scalars; do not launch the simulator."""
import json
import os

import pytest

from sim.piper_compliance.training_curves import TAGS, TrainingCurves, chart_points, chart_png


def test_smoothing_precedes_sampling_and_preserves_endpoints():
    values = {i: float(i) for i in range(1, 21)}
    full = chart_points(values)
    sampled = chart_points(values, limit=4)
    assert full[1] == pytest.approx([2, 2, 1.1])
    assert len(sampled) == 4 and sampled[0] == full[0] and sampled[-1] == full[-1]
    assert all(row in full for row in sampled)
    assert chart_points({}) == []


def writer(run):
    pytest.importorskip('tensorboard')
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    return EventFileWriter(str(run))


def scalar(log, step, tag, value):
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    log.add_event(Event(step=step, summary=Summary(value=[Summary.Value(tag=tag, simple_value=value)])))
    log.flush()


def progress(run, iteration):
    run.mkdir(parents=True, exist_ok=True)
    (run / 'progress.json').write_text(json.dumps({'status': 'paused', 'iteration': iteration}))


def test_reads_existing_events_then_only_appends_new_updates(tmp_path):
    run = tmp_path / 'train'
    progress(run, 1)
    log = writer(run)
    try:
        scalar(log, 0, TAGS[0], 2)
        scalar(log, 0, TAGS[1], .03)
        scalar(log, 0, TAGS[2], float('nan'))
        reader = TrainingCurves(tmp_path)
        first = reader.read()
        assert first['run'] == 'train' and first['progress']['status'] == 'paused'
        assert first['series'][TAGS[0]] == [[1, 2, 2]]
        assert first['series'][TAGS[1]][0][1] == pytest.approx(.03)
        assert first['series'][TAGS[2]] == []
        assert reader.read() == first
        scalar(log, 1, TAGS[0], 4)
        progress(run, 2)
        second = reader.read()
        assert second['point_counts'][TAGS[0]] == 2
        assert second['series'][TAGS[0]][-1] == pytest.approx([2, 4, 2.2])
        assert second['progress']['iteration'] == 2
    finally:
        log.close()


def test_follows_formal_resume_excludes_smoke_and_keeps_global_iteration(tmp_path):
    progress(tmp_path / 'train', 100)
    progress(tmp_path / 'train_resume_123', 101)
    progress(tmp_path / 'tui_resume_smoke', 999)
    progress(tmp_path / 'optimization' / 'train', 999)
    progress(tmp_path / 'train_debug_not_formal_name', 101)
    # Explicit mtimes make this independent of filesystem clock resolution.
    os.utime(tmp_path / 'train_resume_123' / 'progress.json', (2000, 2000))
    for run in ('train', 'train_debug_not_formal_name'):
        os.utime(tmp_path / run / 'progress.json', (1000, 1000))
    log = writer(tmp_path / 'train_resume_123')
    try:
        scalar(log, 100, TAGS[0], 5)
        data = TrainingCurves(tmp_path).read()
        assert data['run'] == 'train_resume_123'
        assert data['series'][TAGS[0]] == [[101, 5, 5]]
    finally:
        log.close()


def test_empty_events_and_missing_progress(tmp_path):
    pytest.importorskip('tensorboard')
    reader = TrainingCurves(tmp_path)
    with pytest.raises(FileNotFoundError):
        reader.read()
    progress(tmp_path / 'train', 0)
    assert all(not series for series in reader.read()['series'].values())


def test_png_images_cache_snapshots_and_refresh_for_new_values():
    import struct
    pytest.importorskip('matplotlib')
    rows = ((1, .03, .03), (2, .02, .029))
    image = chart_png(1, rows)
    assert image.startswith(b'\x89PNG\r\n\x1a\n')
    assert struct.unpack('>II', image[16:24]) == (1024, 528)
    assert chart_png(1, rows) is image
    assert chart_png(1, (*rows, (3, .01, .0271))) != image
    assert chart_png(0, ()).startswith(b'\x89PNG\r\n\x1a\n')


def test_select_runs_without_mixing_series_and_reject_unknown_paths(tmp_path):
    for name, step, value in [('train', 10, 3), ('smoke_05', 1, 9)]:
        progress(tmp_path / name, step + 1)
        log = writer(tmp_path / name)
        scalar(log, step, TAGS[0], value)
        log.close()
    (tmp_path / 'smoke_05' / 'manifest.json').write_text('{"status":"aborted","num_envs":16}')
    reader = TrainingCurves(tmp_path)
    assert reader.read()['run'] == 'train'
    smoke = reader.read('smoke_05')
    assert smoke['series'][TAGS[0]] == [[2, 9, 9]]
    assert smoke['progress']['status'] == 'aborted'
    assert next(run for run in smoke['runs'] if run['id'] == 'smoke_05')['kind'] == 'validation'
    assert reader.read('train')['series'][TAGS[0]] == [[11, 3, 3]]
    for name in ('missing', '../train', '/etc', 'optimization/train'):
        with pytest.raises(KeyError):
            reader.read(name)
    (tmp_path / 'train_external').symlink_to(tmp_path.parent, target_is_directory=True)
    assert 'train_external' not in [run['id'] for run in reader.runs()]
