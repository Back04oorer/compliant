"""Exercise the served JavaScript with a small DOM, without starting a browser."""
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize('query,expected',[
    ('','BalancedPush_v1'), ('?training_run=FirstPPo','FirstPPo'),
    ('?training_run=BalancedPush_v1','BalancedPush_v1'),
    ('?training_run=smoke_05','BalancedPush_v1'),
])
def test_only_two_training_runs_display_and_switch_charts(query,expected):
    node=shutil.which('node')
    if not node:pytest.skip('Node is needed to exercise dashboard JavaScript')
    script=Path(__file__).resolve().parents[1]/'sim/piper_compliance/web/softmimic.js'
    harness=r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const [script, query, expected] = process.argv.slice(1);
class Element {
  children = []; listeners = {}; value = ''; complete = false;
  append(...children) { this.children.push(...children); }
  replaceChildren() { this.children = []; }
  addEventListener(event, listener) { this.listeners[event] = listener; }
}
const elements = new Map(), images = [], requests = [];
const get = id => {
  if (!elements.has(id)) elements.set(id, new Element());
  return elements.get(id);
};
const window = {location: {href: 'http://test/' + query}};
window.history = {replaceState: (_, unused, url) => { window.location.href = String(url); }};
const runs = [
  {id: 'train', status: 'paused', kind: 'training'},
  {id: 'BalancedPush_v1', status: 'running', kind: 'validation', num_envs: 4096},
  {id: 'FirstPPo', status: 'paused', kind: 'validation', num_envs: 4096},
  {id: 'smoke_05', status: 'complete', kind: 'validation'},
  {id: 'preflight_model_audit', status: 'aborted', kind: 'validation'}
];
const context = {URL, window, ResizeObserver: class {
  constructor(callback) { this.callback = callback; }
  observe(body) { this.callback(); }
}, clearTimeout: () => {}, setTimeout: () => 1,
  document: {getElementById: get, createElement: tag => {
    const e = new Element(); if (tag === 'img') images.push(e); return e;
  }},
  fetch: async url => {
    requests.push(url); const run = new URL(url, window.location.href).searchParams.get('run');
    return {ok: true, json: async () => ({run, runs, series: {}, point_counts: {reward: 1},
      progress: {status: runs.find(r => r.id === run).status, iteration: 1, total_iterations: 100,
                 elapsed_s: 1, estimated_remaining_s: 99, updated_at_unix: Date.now()/1000}})};
  }
};
(async () => {
  await vm.runInNewContext(fs.readFileSync(script, 'utf8'), context);
  assert.deepEqual(get('run-select').children.map(e => e.value), ['BalancedPush_v1','FirstPPo']);
  assert.equal(get('run-select').value, expected);
  assert.equal(get('run-kind').textContent, '正式训练');
  assert.equal(new URL(window.location.href).searchParams.get('training_run'), expected);
  assert.equal(new URL(requests[0], window.location.href).searchParams.get('run'), expected);
  const frame = get('rollout-frame');
  let embedded = false;
  frame.style = {};
  frame.contentDocument = {body: {
    classList: {add: name => { embedded = name === 'embedded'; }},
    getBoundingClientRect: () => ({height: 1340})
  }};
  frame.listeners.load();
  assert(embedded);
  assert.equal(frame.style.height, '1340px');
  assert.equal(images.length, 6);
  assert(images.every(i => new URL(i.src, window.location.href).searchParams.get('run') === expected));
  get('run-select').value = expected === 'FirstPPo' ? 'BalancedPush_v1' : 'FirstPPo';
  get('run-select').listeners.change();
  await new Promise(setImmediate);
  assert(images.every(i => new URL(i.src, window.location.href).searchParams.get('run') === get('run-select').value));
  assert.deepEqual(get('run-select').children.map(e => e.value), ['BalancedPush_v1','FirstPPo']);
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    subprocess.run([node,'-e',harness,str(script),query,expected],check=True,capture_output=True,text=True)
