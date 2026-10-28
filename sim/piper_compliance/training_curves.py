"""Read existing TensorBoard scalars without touching the training process."""
import json
import io
import math
import threading
from functools import lru_cache

TAGS = (
    'Train/mean_reward', 'Metrics/target_error_m', 'Metrics/force_error_N',
    'Loss/value_function', 'Loss/surrogate', 'Perf/total_fps',
)
CHART_LABELS = (
    ('Episode return', '', 1), ('Target position error', 'mm', 1000),
    ('Force error', 'N', 1), ('Critic value loss', '', 1),
    ('PPO surrogate loss', '', 1), ('Training throughput', 'steps/s', 1),
)
_plot_lock = threading.Lock()


@lru_cache(maxsize=12)
def chart_png(index, rows, run=''):
    """Render immutable scalar snapshots; cache paused runs and serialize Matplotlib."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator, EngFormatter

    title, unit, scale = CHART_LABELS[index]
    with _plot_lock:
        fig = Figure(figsize=(6.4, 3.3), dpi=160, facecolor='white')
        fig.text(.13, .985, run, fontsize=7, color='#687168', va='top')
        FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        fig.subplots_adjust(left=.13, right=.97, bottom=.19, top=.83)
        color = '#408566'
        if rows:
            steps, raw, average = zip(*rows)
            ax.plot(steps, [value * scale for value in raw], color=color, alpha=.22, linewidth=.8, label='Raw')
            ax.plot(steps, [value * scale for value in average], color=color, linewidth=1.7, label='EMA 0.1')
            ax.scatter([steps[-1]], [average[-1] * scale], color=color, s=12, zorder=3)
            fig.text(.13, .92, f'Latest: {raw[-1] * scale:,.4g} {unit}', fontsize=12, color='#212822', weight='medium')
            fig.text(.97, .92, f'Update {steps[-1]:,}', fontsize=9, color='#687168', ha='right')
            ax.legend(loc='best', frameon=False, fontsize=8)
        else:
            ax.text(.5, .5, 'Waiting for training data', transform=ax.transAxes,
                    ha='center', color='#687168')
        ax.set_xlabel('PPO updates', fontsize=9, color='#687168', labelpad=8)
        ax.set_ylabel(unit or title, fontsize=9, color='#687168', labelpad=8)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(EngFormatter(sep=''))
        ax.tick_params(axis='both', labelsize=8, colors='#687168', length=0, pad=7)
        ax.grid(axis='y', color='#e9ede8', linewidth=.7)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('bottom', 'left'):
            ax.spines[side].set_color('#dce1d8')
        output = io.BytesIO()
        fig.savefig(output, format='png', metadata={'Title': title})
        return output.getvalue()


def chart_points(values, limit=4000):
    """Smooth before sampling, so the EMA retains the same meaning as a run grows."""
    rows = []
    average = None
    for iteration, value in sorted(values.items()):
        average = value if average is None else .1 * value + .9 * average
        rows.append([iteration, value, average])
    if len(rows) <= limit:
        return rows
    # Keep both ends; full-resolution scalars remain in the TensorBoard files.
    return [rows[round(i * (len(rows) - 1) / (limit - 1))] for i in range(limit)]


class TrainingCurves:
    def __init__(self, root):
        self.root = root
        self.lock = threading.Lock()
        self.run = None
        self.loaders = {}
        self.values = {tag: {} for tag in TAGS}

    def runs(self):
        records = []
        for path in self.root.glob('*/progress.json'):
            if path.parent.resolve().parent != self.root.resolve():
                continue
            try:
                progress = json.loads(path.read_text())
                status = progress.get('status', 'unknown')
                manifest = path.parent / 'manifest.json'
                config = {}
                if manifest.exists():
                    config = json.loads(manifest.read_text())
                    status = config.get('status', status)
                records.append({'id': path.parent.name, 'status': status,
                                'num_envs': config.get('num_envs'),
                                'iteration': progress.get('iteration', 0),
                                'kind': 'training' if path.parent.name == 'train' or path.parent.name.startswith('train_') else 'validation',
                                'modified': path.stat().st_mtime_ns})
            except (OSError, ValueError):
                continue
        return sorted(records, key=lambda item: item['modified'], reverse=True)

    def read(self, run=None):
        from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader

        with self.lock:
            runs = self.runs()
            if not runs:
                raise FileNotFoundError('尚无正式训练记录')
            if run is None:
                selected = next((item for item in runs if item['kind'] == 'training'), runs[0])
            else:
                selected = next((item for item in runs if item['id'] == run), None)
                if selected is None:
                    raise KeyError('Unknown training run')
            path = self.root / selected['id'] / 'progress.json'
            progress = json.loads(path.read_text())
            progress['status'] = selected['status']
            if path.parent != self.run:
                self.run = path.parent
                self.loaders = {}
                self.values = {tag: {} for tag in TAGS}
            for event_path in sorted(self.run.glob('events.out.tfevents.*')):
                if event_path not in self.loaders:
                    self.loaders[event_path] = LegacyEventFileLoader(str(event_path))
                for event in self.loaders[event_path].Load():
                    for scalar in event.summary.value:
                        if scalar.tag in self.values and scalar.HasField('simple_value'):
                            value = float(scalar.simple_value)
                            if math.isfinite(value):
                                # RSL event steps start at zero; the page counts completed updates.
                                self.values[scalar.tag][int(event.step) + 1] = value
            return {
                'run': self.run.name, 'runs': runs, 'progress': progress,
                'series': {tag: chart_points(values) for tag, values in self.values.items()},
                'point_counts': {tag: len(values) for tag, values in self.values.items()},
                'ema_alpha': .1, 'display_limit': 4000,
            }


@lru_cache(maxsize=2)
def training_curves(root):
    return TrainingCurves(root)
