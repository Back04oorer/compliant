"use strict";
const $ = id => document.getElementById(id);
const number = value => Number(value).toLocaleString('zh-CN');
const duration = value => Number.isFinite(Number(value)) ? `${(Number(value) / 3600).toFixed(1)} 小时` : '—';
const visibleRuns = ['BalancedPush_v1', 'FirstPPo'];
const requestedRun = new URL(window.location.href).searchParams.get('training_run');
let selectedRun = visibleRuns.includes(requestedRun) ? requestedRun : visibleRuns[0];
if (requestedRun !== selectedRun) {
  const url = new URL(window.location.href); url.searchParams.set('training_run', selectedRun);
  window.history.replaceState(null, '', url);
}
let requestId = 0, timer, displayedRun = null;
const stateNames = {running: '运行中', paused: '已暂停', complete: '已完成', failed: '失败', aborted: '已终止'};
$('run-select').value = selectedRun;
const definitions = [
  ['Train/mean_reward', '平均回合回报'],
  ['Metrics/target_error_m', '目标位置误差 · mm'],
  ['Metrics/force_error_N', '力误差 · N'],
  ['Loss/value_function', 'Critic 价值损失'],
  ['Loss/surrogate', 'PPO 策略损失'],
  ['Perf/total_fps', '训练吞吐量 · 环境步 / 秒']
];
const cards = definitions.map(([tag, title], index) => {
  const card = document.createElement('article'); card.className = 'metric-card';
  const heading = document.createElement('h3'); heading.textContent = title;
  const link = document.createElement('a');
  link.href = `/api/training-charts/${index}.png`;
  link.target = '_blank'; link.rel = 'noopener'; link.title = `打开${title}大图`;
  const image = document.createElement('img');
  image.alt = `${title}随 PPO 更新次数的变化`;
  image.width = 1024; image.height = 528;
  const error = document.createElement('p'); error.hidden = true;
  image.addEventListener('load', () => { error.hidden = true; image.hidden = false; });
  image.addEventListener('error', () => { error.hidden = false; error.textContent = '图片暂不可用，将自动重试。'; });
  link.append(image); card.append(heading, link, error); $('curves').append(card);
  return {tag, index, link, image, error, revision: null};
});
function showRuns(data) {
  $('run-select').replaceChildren();
  for (const id of visibleRuns) {
    const run = data.runs.find(item => item.id === id);
    if (!run) continue;
    const option = document.createElement('option'); option.value = run.id;
    option.textContent = `${run.id} · ${stateNames[run.status] || run.status}`;
    $('run-select').append(option);
  }
  $('run-select').value = selectedRun;
  const current = data.runs.find(run => run.id === data.run);
  $('run-kind').textContent = '正式训练';
  $('run-description').textContent = '固定姿态柔顺控制' + (current?.num_envs ? ` · ${number(current.num_envs)} 并行环境` : '');
}
function showProgress(data) {
  const age = Date.now() / 1000 - data.updated_at_unix;
  $('status').textContent = data.status === 'paused' ? '训练已暂停 · 检查点已保留' :
    data.status === 'complete' ? '训练完成 · 等待评估' :
    data.status === 'failed' ? '训练失败 · 请检查日志' :
    data.status === 'aborted' ? '该 run 已终止' :
    age > 120 ? '进度已超过 2 分钟未更新' : '训练中';
  $('iteration').textContent = `${number(data.iteration)} / ${number(data.total_iterations)}`;
  $('elapsed').textContent = duration(data.elapsed_s);
  $('remaining').textContent = duration(data.estimated_remaining_s);
  $('progress').max = data.total_iterations; $('progress').value = data.iteration;
  $('run-info').textContent = `最后更新：${new Date(data.updated_at_unix * 1000).toLocaleString('zh-CN')}。剩余时间按实测速度估算。`;
}
async function refresh() {
  const currentRequest = ++requestId;
  clearTimeout(timer);
  try {
    const query = selectedRun ? `?run=${encodeURIComponent(selectedRun)}` : '';
    const response = await fetch(`/api/training-curves${query}`, {cache: 'no-store'});
    const data = await response.json();
    if (currentRequest !== requestId) return;
    if (!response.ok) throw new Error(data.error || '训练日志暂不可用');
    showRuns(data);
    showProgress(data.progress);
    $('curve-status').textContent = `${data.run} · ${number(Math.max(...Object.values(data.point_counts)))} 次更新 · 点击图片查看大图`;
    for (const card of cards) {
      if (displayedRun !== data.run) card.image.hidden = true;
      card.link.href = `/api/training-charts/${card.index}.png?run=${encodeURIComponent(data.run)}`;
      const rows = data.series[card.tag] || [];
      const revision = `${data.run}:${data.point_counts[card.tag]}:${JSON.stringify(rows.at(-1))}`;
      if (card.revision !== revision || (card.image.complete && !card.image.naturalWidth)) {
        card.image.src = `${card.link.href}&v=${encodeURIComponent(revision)}`;
        card.revision = revision;
      }
    }
    displayedRun = data.run;
  } catch (error) {
    if (currentRequest !== requestId) return;
    $('curve-status').textContent = `暂时无法更新：${error.message}；稍后自动重试。`;
    $('status').textContent = '所选 run 暂不可用';
  } finally {
    if (currentRequest === requestId) timer = setTimeout(refresh, 15000);
  }
}
$('run-select').addEventListener('change', () => {
  selectedRun = $('run-select').value;
  const url = new URL(window.location.href);
  if (selectedRun) url.searchParams.set('training_run', selectedRun); else url.searchParams.delete('training_run');
  window.history.replaceState(null, '', url);
  for (const card of cards) { card.image.hidden = true; card.revision = null; card.error.hidden = true; }
  $('status').textContent = '读取所选 run…';
  for (const id of ['iteration', 'elapsed', 'remaining']) $(id).textContent = '—';
  $('progress').value = 0;
  refresh();
});
$('rollout-frame').addEventListener('load', () => {
  const frame = $('rollout-frame');
  const body = frame.contentDocument.body;
  body.classList.add('embedded');
  const resize = () => { frame.style.height = `${Math.ceil(body.getBoundingClientRect().height)}px`; };
  new ResizeObserver(resize).observe(body);
  resize();
});
refresh();
