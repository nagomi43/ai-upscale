'use strict';
// AI動画アップスケール — ブラウザ内で完結（ONNX Runtime Web + WebCodecs + mp4-muxer）

const $ = (id) => document.getElementById(id);
const ui = {
  drop: $('drop'), file: $('file'), info: $('fileinfo'), kind: $('kind'), res: $('res'),
  start: $('start'), cancel: $('cancel'), bar: $('bar'), status: $('status'),
  compare: $('compare'), before: $('before'), after: $('after'),
};

ort.env.wasm.wasmPaths = new URL('vendor/', location.href).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const state = { file: null, url: null, video: null, w: 0, h: 0, fps: 30, duration: 0, frames: 0,
  sessions: {}, backend: '', running: false, cancel: false, secPerFrame: 0 };

function setStatus(text, cls = '') { ui.status.textContent = text; ui.status.className = cls; }
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function fmtTime(sec) {
  if (!isFinite(sec)) return '—';
  if (sec < 60) return `${Math.max(1, Math.round(sec))}秒`;
  if (sec < 3600) return `${Math.ceil(sec / 60)}分`;
  return `${Math.floor(sec / 3600)}時間${Math.round((sec % 3600) / 60)}分`;
}

// ---------- モデル ----------
async function getSession(kind) {
  if (state.sessions[kind]) return state.sessions[kind];
  const url = `models/${kind}-x4.onnx`;
  const forceCpu = new URLSearchParams(location.search).has('cpu');
  const tries = navigator.gpu && !forceCpu ? [['webgpu'], ['wasm']] : [['wasm']];
  let lastErr;
  for (const eps of tries) {
    try {
      setStatus(`AIモデルを読み込み中…（${eps[0] === 'webgpu' ? 'GPU' : 'CPU'}）`);
      const s = await ort.InferenceSession.create(url, { executionProviders: eps, graphOptimizationLevel: 'all' });
      state.sessions[kind] = s;
      state.backend = eps[0];
      return s;
    } catch (e) { lastErr = e; console.warn('EP failed', eps, e); }
  }
  throw lastErr;
}

// ---------- 動画の読み込み ----------
function waitEvent(el, ev, timeout = 15000) {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => { cleanup(); reject(new Error(`${ev} timeout`)); }, timeout);
    const ok = () => { cleanup(); resolve(); };
    const ng = () => { cleanup(); reject(new Error('この動画形式はブラウザで読めませんでした（MP4 / H.264 がおすすめです）')); };
    function cleanup() { clearTimeout(t); el.removeEventListener(ev, ok); el.removeEventListener('error', ng); }
    el.addEventListener(ev, ok); el.addEventListener('error', ng);
  });
}

async function seekTo(video, t) {
  if (Math.abs(video.currentTime - t) < 1e-6 && video.readyState >= 2) return;
  const p = waitEvent(video, 'seeked');
  video.currentTime = t;
  await p;
}

async function detectFps(video) {
  if (!('requestVideoFrameCallback' in video)) return 30;
  const times = [];
  video.muted = true;
  await new Promise((resolve) => {
    const done = setTimeout(resolve, 2500);
    const cb = (_now, md) => {
      times.push(md.mediaTime);
      if (times.length < 30) video.requestVideoFrameCallback(cb); else { clearTimeout(done); resolve(); }
    };
    video.requestVideoFrameCallback(cb);
    video.play().catch(() => resolve());
  });
  video.pause();
  const diffs = [];
  for (let i = 1; i < times.length; i++) { const d = times[i] - times[i - 1]; if (d > 0.002) diffs.push(d); }
  if (!diffs.length) return 30;
  const fps = 1 / Math.min(...diffs);
  const common = [23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60];
  const snap = common.reduce((a, c) => (Math.abs(c - fps) < Math.abs(a - fps) ? c : a));
  if (Math.abs(snap - fps) / snap > 0.02) return Math.round(fps * 100) / 100;
  return snap;
}

async function loadFile(file) {
  if (state.running) return;
  if (state.url) URL.revokeObjectURL(state.url);
  state.file = file;
  state.url = URL.createObjectURL(file);
  const v = document.createElement('video');
  v.preload = 'auto'; v.muted = true; v.playsInline = true; v.src = state.url;
  ui.start.disabled = true;
  ui.info.classList.remove('hidden');
  ui.info.textContent = `読み込み中: ${file.name}`;
  try {
    await waitEvent(v, 'loadeddata');
    state.video = v;
    state.w = v.videoWidth; state.h = v.videoHeight; state.duration = v.duration;
    setStatus('フレームレートを確認中…');
    state.fps = await detectFps(v);
    state.frames = Math.max(1, Math.floor(state.duration * state.fps + 1e-6));
    ui.info.innerHTML = `<b>${escapeHtml(file.name)}</b>　${state.w}×${state.h}　${state.fps}fps　${fmtTime(state.duration)}（${state.frames}コマ）`;
    await speedTest();
  } catch (e) {
    console.error(e);
    setStatus(e.message || String(e), 'err');
  }
}

function escapeHtml(s) { return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }

// ---------- 1コマの処理 ----------
const srcCanvas = document.createElement('canvas');
const upCanvas = document.createElement('canvas');
const dstCanvas = document.createElement('canvas');

function targetSize() {
  const th = parseInt(ui.res.value, 10);
  let tw = Math.round((state.w * th) / state.h / 2) * 2;
  const w169 = Math.round((th * 16) / 9 / 2) * 2;
  if (Math.abs(tw - w169) <= 4) tw = w169; // 854x480 などは 1920x1080 にそろえる
  return { tw, th };
}

let runLock = Promise.resolve();
async function upscaleCurrentFrame(session) {
  const { w, h } = state;
  srcCanvas.width = w; srcCanvas.height = h;
  const sctx = srcCanvas.getContext('2d', { willReadFrequently: true });
  sctx.drawImage(state.video, 0, 0, w, h);
  const rgba = sctx.getImageData(0, 0, w, h).data;
  const n = w * h;
  const chw = new Float32Array(3 * n);
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    chw[i] = rgba[j] / 255; chw[n + i] = rgba[j + 1] / 255; chw[2 * n + i] = rgba[j + 2] / 255;
  }
  const input = new ort.Tensor('float32', chw, [1, 3, h, w]);
  const prev = runLock;
  let release;
  runLock = new Promise((r) => (release = r));
  await prev;
  let res;
  try { res = await session.run({ input }); } finally { release(); }
  const out = res.output;
  const [, , H, W] = out.dims;
  const od = out.data;
  input.dispose?.();
  upCanvas.width = W; upCanvas.height = H;
  const uctx = upCanvas.getContext('2d');
  const img = uctx.createImageData(W, H);
  const px = img.data; const N = W * H;
  for (let i = 0, j = 0; i < N; i++, j += 4) {
    px[j] = od[i] * 255 + 0.5; px[j + 1] = od[N + i] * 255 + 0.5; px[j + 2] = od[2 * N + i] * 255 + 0.5; px[j + 3] = 255;
  }
  out.dispose?.();
  uctx.putImageData(img, 0, 0);
  const { tw, th } = targetSize();
  dstCanvas.width = tw; dstCanvas.height = th;
  const dctx = dstCanvas.getContext('2d');
  dctx.imageSmoothingEnabled = true; dctx.imageSmoothingQuality = 'high';
  dctx.drawImage(upCanvas, 0, 0, tw, th);
  return dstCanvas;
}

function showPreview() {
  ui.compare.classList.remove('hidden');
  // 比較しやすいよう、中央付近を同じ範囲で切り出して表示
  const cw = Math.min(state.w, Math.round(state.h * 4 / 3)), ch = state.h;
  const sx = Math.round((state.w - cw) / 2);
  ui.before.width = cw * 2; ui.before.height = ch * 2;
  const b = ui.before.getContext('2d'); b.imageSmoothingEnabled = true;
  b.drawImage(srcCanvas, sx, 0, cw, ch, 0, 0, cw * 2, ch * 2);
  const k = dstCanvas.width / state.w;
  ui.after.width = Math.round(cw * k); ui.after.height = Math.round(ch * k);
  ui.after.getContext('2d').drawImage(dstCanvas, sx * k, 0, cw * k, ch * k, 0, 0, ui.after.width, ui.after.height);
}

async function speedTest() {
  if (state.testing) return;
  state.testing = true;
  try {
  const session = await getSession(ui.kind.value);
  setStatus('1コマ試しています…');
  await seekTo(state.video, Math.min(state.duration / 3, state.duration - 0.05));
  await upscaleCurrentFrame(session); // 1回目は準備で遅いので捨てる
  const t0 = performance.now();
  await upscaleCurrentFrame(session);
  state.secPerFrame = (performance.now() - t0) / 1000;
  showPreview();
  const est = state.secPerFrame * state.frames * 1.1;
  const be = state.backend === 'webgpu' ? 'GPU' : 'CPU';
  setStatus(`目安: 1コマ ${state.secPerFrame.toFixed(2)}秒（${be}）→ この動画は 約${fmtTime(est)}。下の比較を見て、よければ「変換スタート」。`);
  ui.start.disabled = false;
  } finally { state.testing = false; }
}

// ---------- 音声 ----------
async function encodeAudio(muxerHolder) {
  let buf;
  try {
    const ab = await state.file.arrayBuffer();
    buf = await new OfflineAudioContext(2, 1, 48000).decodeAudioData(ab);
  } catch (e) { console.warn('no audio', e); return null; }
  if (!buf || !buf.length) return null;
  const channels = Math.min(2, buf.numberOfChannels), sampleRate = buf.sampleRate;
  let cfg = null, codec = null;
  for (const [c, mc, br] of [['mp4a.40.2', 'aac', 192000], ['opus', 'opus', 160000]]) {
    const test = { codec: c, sampleRate, numberOfChannels: channels, bitrate: br };
    try { if ((await AudioEncoder.isConfigSupported(test)).supported) { cfg = test; codec = mc; break; } } catch (_) {}
  }
  if (!cfg) return null;
  return { cfg, codec, buf, channels, sampleRate,
    async run(muxer) {
      const enc = new AudioEncoder({ output: (chunk, meta) => muxer.addAudioChunk(chunk, meta), error: (e) => console.error(e) });
      enc.configure(cfg);
      const block = 4096;
      for (let off = 0; off < buf.length; off += block) {
        const len = Math.min(block, buf.length - off);
        const data = new Float32Array(len * channels);
        for (let c = 0; c < channels; c++) data.set(buf.getChannelData(c).subarray(off, off + len), c * len);
        const ad = new AudioData({ format: 'f32-planar', sampleRate, numberOfFrames: len, numberOfChannels: channels,
          timestamp: Math.round((off / sampleRate) * 1e6), data });
        enc.encode(ad); ad.close();
      }
      await enc.flush(); enc.close();
    } };
}

// ---------- 映像エンコーダの選択 ----------
async function pickVideoConfig(tw, th, fps) {
  const px = tw * th;
  const bitrate = px <= 2.1e6 ? 12e6 : px <= 3.7e6 ? 20e6 : 40e6;
  const avcLevel = px <= 2.1e6 ? '640028' : px <= 3.7e6 ? '640032' : '640033';
  const options = [
    [{ codec: `avc1.${avcLevel}`, avc: { format: 'avc' } }, 'avc'],
    [{ codec: 'vp09.00.41.08' }, 'vp9'],
  ];
  for (const [extra, mux] of options) {
    for (const hw of ['prefer-hardware', 'no-preference']) {
      const cfg = { ...extra, width: tw, height: th, bitrate, framerate: fps, hardwareAcceleration: hw };
      try { if ((await VideoEncoder.isConfigSupported(cfg)).supported) return { cfg, mux }; } catch (_) {}
    }
  }
  throw new Error('このブラウザでは動画の書き出しができません（Chrome / Edge の最新版で開いてください）');
}

// ---------- 変換本体 ----------
async function run() {
  if (!state.video || state.running) return;
  state.running = true; state.cancel = false;
  ui.start.disabled = true; ui.cancel.classList.remove('hidden'); ui.bar.classList.remove('hidden');
  ui.kind.disabled = ui.res.disabled = true;
  let wakeLock = null;
  const { tw, th } = targetSize();
  const base = state.file.name.replace(/\.[^.]+$/, '');
  const outName = `${base}_AI${th === 2160 ? '4K' : th + 'p'}.mp4`;
  let fileHandle = null, writable = null;
  try {
    if (!self.VideoEncoder) throw new Error('このブラウザは対応していません。Chrome か Edge（パソコン版）で開いてください');
    if ('showSaveFilePicker' in window) {
      try {
        fileHandle = await showSaveFilePicker({ suggestedName: outName, types: [{ description: 'MP4動画', accept: { 'video/mp4': ['.mp4'] } }] });
      } catch (e) { if (e.name === 'AbortError') { setStatus('保存先の選択がキャンセルされました'); return; } }
    }
    try { wakeLock = await navigator.wakeLock?.request('screen'); } catch (_) {}

    const session = await getSession(ui.kind.value);
    const { cfg, mux } = await pickVideoConfig(tw, th, state.fps);
    setStatus('音声を準備中…');
    const audio = await encodeAudio();

    let target;
    if (fileHandle) { writable = await fileHandle.createWritable(); target = new Mp4Muxer.FileSystemWritableFileStreamTarget(writable); }
    else target = new Mp4Muxer.ArrayBufferTarget();
    const muxer = new Mp4Muxer.Muxer({
      target,
      video: { codec: mux, width: tw, height: th },
      audio: audio ? { codec: audio.codec, numberOfChannels: audio.channels, sampleRate: audio.sampleRate } : undefined,
      fastStart: fileHandle ? false : 'in-memory',
      firstTimestampBehavior: 'offset',
    });
    if (audio) await audio.run(muxer);

    let encErr = null;
    const venc = new VideoEncoder({ output: (chunk, meta) => muxer.addVideoChunk(chunk, meta), error: (e) => { encErr = e; } });
    venc.configure(cfg);

    const total = state.frames, dt = 1 / state.fps, usPerFrame = 1e6 / state.fps;
    const gop = Math.max(1, Math.round(state.fps * 2));
    const t0 = performance.now();
    for (let i = 0; i < total; i++) {
      if (state.cancel) throw new Error('中止しました');
      if (encErr) throw encErr;
      await seekTo(state.video, Math.min(i * dt + dt / 2, state.duration - 0.001));
      const canvas = await upscaleCurrentFrame(session);
      const frame = new VideoFrame(canvas, { timestamp: Math.round(i * usPerFrame), duration: Math.round(usPerFrame) });
      venc.encode(frame, { keyFrame: i % gop === 0 });
      frame.close();
      while (venc.encodeQueueSize > 4) await sleep(5);
      if (i % 15 === 0 || i === total - 1) showPreview();
      const el = (performance.now() - t0) / 1000, done = i + 1;
      const fps = done / el;
      ui.bar.value = done / total;
      setStatus(`変換中… ${done} / ${total} コマ（${Math.round((done / total) * 100)}%）　${fps.toFixed(2)} コマ/秒　残り約 ${fmtTime((total - done) / fps)}`);
    }
    setStatus('仕上げ中…');
    await venc.flush(); venc.close();
    if (encErr) throw encErr;
    muxer.finalize();
    if (writable) { await writable.close(); writable = null; }
    else {
      const blob = new Blob([target.buffer], { type: 'video/mp4' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob); a.download = outName; a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 60000);
    }
    const took = (performance.now() - t0) / 1000;
    setStatus(`完了しました: ${outName}（${tw}×${th}、${fmtTime(took)}）`, 'ok');
  } catch (e) {
    console.error(e);
    setStatus(e.message || String(e), 'err');
    if (writable) { try { await writable.abort(); } catch (_) {} }
  } finally {
    state.running = false;
    ui.start.disabled = !state.video; ui.cancel.classList.add('hidden');
    ui.kind.disabled = ui.res.disabled = false;
    try { await wakeLock?.release(); } catch (_) {}
  }
}

// ---------- 画面の操作 ----------
ui.drop.addEventListener('click', () => ui.file.click());
ui.drop.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') ui.file.click(); });
ui.file.addEventListener('change', () => { if (ui.file.files[0]) loadFile(ui.file.files[0]); });
['dragenter', 'dragover'].forEach((ev) => document.addEventListener(ev, (e) => { e.preventDefault(); ui.drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((ev) => document.addEventListener(ev, (e) => { e.preventDefault(); ui.drop.classList.remove('over'); }));
document.addEventListener('drop', (e) => { const f = e.dataTransfer?.files?.[0]; if (f) loadFile(f); });
ui.start.addEventListener('click', run);
ui.cancel.addEventListener('click', () => { state.cancel = true; setStatus('中止しています…'); });
ui.kind.addEventListener('change', () => { if (state.video && !state.running) speedTest().catch((e) => setStatus(e.message, 'err')); });
window.addEventListener('beforeunload', (e) => { if (state.running) { e.preventDefault(); e.returnValue = ''; } });
if (!navigator.gpu) setStatus('注意: このブラウザではGPUが使えないため、とても遅くなります。Chrome / Edge の最新版がおすすめです。');
