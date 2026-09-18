#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Move raw metric depth from parquet into a frame-indexed mmap sidecar. '
            'Use this route when DeFM is trainable; do not cache DeFM tokens for encoder fine-tuning. '
            'The source dataset is never modified.'
        )
    )
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--feature-key', default='observation.fastfs_depth_image')
    parser.add_argument('--dtype', choices=['float32','float16'], default='float32')
    parser.add_argument('--row-group-size', type=int, default=128)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def copy_tree_with_hardlinks(source: Path, output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f'Output exists; pass --overwrite: {output}')
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    source_data = source / 'data'
    source_meta = source / 'meta'

    def copy_file(src: str, dst: str, *, follow_symlinks: bool = True) -> str:
        src_path = Path(src)
        # Parquet shards and metadata are rewritten below. They must be real
        # copies: modifying a hardlink would silently mutate the source
        # depthmetric dataset as well. Large immutable video files can still
        # be hardlinked to avoid duplicating storage.
        if src_path.is_relative_to(source_data) or src_path.is_relative_to(source_meta):
            return shutil.copy2(src, dst, follow_symlinks=follow_symlinks)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst, follow_symlinks=follow_symlinks)
        return dst

    shutil.copytree(source, output, copy_function=copy_file, symlinks=True, ignore=shutil.ignore_patterns('.cache','__pycache__'))


def data_files(root: Path) -> list[Path]:
    files = sorted((root / 'data').glob('chunk-*/file-*.parquet'))
    if not files:
        raise FileNotFoundError(root / 'data')
    return files


def depth_to_array(value: object, expected_shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object:
        arr = np.stack([np.asarray(row, dtype=np.float32) for row in arr], axis=0)
    arr = arr.astype(np.float32, copy=False)
    if arr.shape != expected_shape:
        raise ValueError(f'Expected depth shape {expected_shape}, got {arr.shape}')
    return arr


def update_info(output: Path, feature_key: str, mmap_rel: str, shape: list[int], dtype: str, source_feature: dict) -> None:
    info_path = output / 'meta' / 'info.json'
    info = json.loads(info_path.read_text(encoding='utf-8'))
    features = dict(info.get('features', {}))
    feature = features.pop(feature_key, source_feature)
    info['features'] = features
    feature_mmaps = dict(info.get('feature_mmaps', {})) if isinstance(info.get('feature_mmaps'), dict) else {}
    feature_mmaps[feature_key] = {
        'path': mmap_rel,
        'shape': shape,
        'dtype': dtype,
        'source_feature': feature,
    }
    info['feature_mmaps'] = feature_mmaps
    info_path.write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')

    stats_path = output / 'meta' / 'stats.json'
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text(encoding='utf-8'))
        if isinstance(stats, dict):
            stats.pop(feature_key, None)
        stats_path.write_text(json.dumps(stats, indent=2) + '\n', encoding='utf-8')


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    copy_tree_with_hardlinks(source, output, bool(args.overwrite))

    info = json.loads((output / 'meta' / 'info.json').read_text(encoding='utf-8'))
    total_frames = int(info['total_frames'])
    source_feature = info['features'][args.feature_key]
    frame_shape = tuple(int(x) for x in source_feature['shape'])
    if len(frame_shape) != 2:
        raise ValueError(f'{args.feature_key} must be [H,W], got {frame_shape}')
    np_dtype = np.float16 if args.dtype == 'float16' else np.float32
    mmap_dir = output / 'depth_mmaps'
    mmap_dir.mkdir(parents=True, exist_ok=True)
    mmap_path = mmap_dir / f"{args.feature_key.replace('.', '_')}.{args.dtype}.mmap"
    mmap = np.memmap(mmap_path, dtype=np_dtype, mode='w+', shape=(total_frames, *frame_shape))
    written = np.zeros(total_frames, dtype=np.bool_)

    for i, path in enumerate(data_files(output), start=1):
        df = pd.read_parquet(path)
        if args.feature_key not in df.columns:
            raise KeyError(f'{args.feature_key} missing in {path}')
        if 'index' not in df.columns:
            raise KeyError(f'index missing in {path}')
        indices = df['index'].to_numpy(dtype=np.int64)
        batch = np.empty((len(df), *frame_shape), dtype=np_dtype)
        for row_i, value in enumerate(df[args.feature_key].tolist()):
            batch[row_i] = depth_to_array(value, frame_shape).astype(np_dtype, copy=False)
        mmap[indices] = batch
        written[indices] = True
        stripped = df.drop(columns=[args.feature_key])
        pq.write_table(pa.Table.from_pandas(stripped, preserve_index=False), path, row_group_size=max(1, int(args.row_group_size)))
        print(f'[depth-mmap] {i:03d} {path.relative_to(output)} rows={len(df)}')

    mmap.flush()
    missing = np.flatnonzero(~written)
    if len(missing):
        raise RuntimeError(f'Missing {len(missing)} frames; first={missing[:10].tolist()}')
    update_info(output, args.feature_key, str(mmap_path.relative_to(output)), [total_frames, *frame_shape], args.dtype, source_feature)
    print(f'[depth-mmap] done output={output}')
    print(f'[depth-mmap] mmap={mmap_path} shape={[total_frames, *frame_shape]} dtype={args.dtype}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
