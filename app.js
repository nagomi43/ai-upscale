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
async function gpuHasF16() {
  try { const a = await navigator.gpu.requestAdapter(); return !!a && a.features.has('shader-f16'); } catch (_) { return false; }
}

async function getSession(kind) {
  if (kind === 'a4k') { state.backend = 'GPU・Anime4K'; return { a4k: true }; }
  if (state.sessions[kind]) return state.sessions[kind];
  const forceCpu = new URLSearchParams(location.search).has('cpu');
  const tries = [];
  if (navigator.gpu && !forceCpu) {
    // 半精度(fp16)に対応したGPUなら、軽い fp16 モデルを使う（約1.5〜2倍速）
    if (await gpuHasF16()) tries.push(['webgpu', `models/${kind}-x4-fp16.onnx`, 'GPU・高速']);
    tries.push(['webgpu', `models/${kind}-x4.onnx`, 'GPU']);
  }
  tries.push(['wasm', `models/${kind}-x4.onnx`, 'CPU']);
  let lastErr;
  for (const [ep, url, label] of tries) {
    try {
      setStatus(`AIモデルを読み込み中…（${label}）`);
      const s = await ort.InferenceSession.create(url, { executionProviders: [ep], graphOptimizationLevel: 'all' });
      // 試しに小さい画像で1回動かし、壊れた結果にならないか確認
      const t = new ort.Tensor('float32', new Float32Array(3 * 16 * 16).fill(0.5), [1, 3, 16, 16]);
      const o = (await s.run({ input: t })).output.data;
      if (!(Math.abs(o[0] - 0.5) < 0.2)) throw new Error('bad output on ' + label);
      state.sessions[kind] = s;
      state.backend = label;
      return s;
    } catch (e) { lastErr = e; console.warn('session failed', label, e); }
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

// ---------- 超高速モード（Anime4K / WebGPU） ----------
let a4k = null;
const A4K_SHADER = `
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;
struct VO { @builtin(position) p: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  var pos = array<vec2f, 3>(vec2f(-1.0, -3.0), vec2f(-1.0, 1.0), vec2f(3.0, 1.0));
  var o: VO;
  o.p = vec4f(pos[i], 0.0, 1.0);
  o.uv = vec2f((pos[i].x + 1.0) * 0.5, (1.0 - pos[i].y) * 0.5);
  return o;
}
@fragment fn fs(v: VO) -> @location(0) vec4f {
  let c = textureSample(t, s, v.uv);
  return vec4f(clamp(c.rgb, vec3f(0.0), vec3f(1.0)), 1.0);
}`;

async function getA4k(tw, th) {
  const { w, h } = state;
  const key = `${w}x${h}>${tw}x${th}`;
  if (a4k && a4k.key === key) return a4k;
  if (!navigator.gpu) throw new Error('超高速モードはGPU（WebGPU）が必要です。Chrome / Edge の最新版で開いてください');
  const A = window['anime4k-webgpu'];
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('GPUが見つかりませんでした');
  const device = a4k?.device || (await adapter.requestDevice());
  const input = device.createTexture({ size: [w, h, 1], format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  const preset = new A.ModeA({ device, inputTexture: input,
    nativeDimensions: { width: w, height: h }, targetDimensions: { width: tw, height: th } });
  const out = preset.getOutputTexture();
  const W = out.width, H = out.height;
  const format = 'rgba8unorm';
  const rt = device.createTexture({ size: [W, H, 1], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const stride = Math.ceil((W * 4) / 256) * 256;
  const readBuf = device.createBuffer({ size: stride * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const module = device.createShaderModule({ code: A4K_SHADER });
  const pipe = device.createRenderPipeline({ layout: 'auto',
    vertex: { module, entryPoint: 'vs' }, fragment: { module, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'triangle-list' } });
  const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const bind = device.createBindGroup({ layout: pipe.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: out.createView() }, { binding: 1, resource: sampler }] });
  a4k = { key, device, input, preset, rt, readBuf, stride, W, H, pipe, bind };
  return a4k;
}

async function runA4k(e, rgba) {
  const { w, h } = state;
  e.device.queue.writeTexture({ texture: e.input }, rgba, { bytesPerRow: w * 4, rowsPerImage: h }, [w, h, 1]);
  const enc = e.device.createCommandEncoder();
  e.preset.pass(enc);
  const pass = enc.beginRenderPass({ colorAttachments: [{ view: e.rt.createView(),
    loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
  pass.setPipeline(e.pipe); pass.setBindGroup(0, e.bind); pass.draw(3); pass.end();
  enc.copyTextureToBuffer({ texture: e.rt }, { buffer: e.readBuf, bytesPerRow: e.stride, rowsPerImage: e.H }, [e.W, e.H, 1]);
  e.device.queue.submit([enc.finish()]);
  await e.readBuf.mapAsync(GPUMapMode.READ);
  const src = new Uint8Array(e.readBuf.getMappedRange());
  upCanvas.width = e.W; upCanvas.height = e.H;
  const uctx = upCanvas.getContext('2d');
  const img = uctx.createImageData(e.W, e.H);
  const rowBytes = e.W * 4;
  for (let y = 0; y < e.H; y++) img.data.set(src.subarray(y * e.stride, y * e.stride + rowBytes), y * rowBytes);
  e.readBuf.unmap();
  uctx.putImageData(img, 0, 0);
  return upCanvas;
}

let runLock = Promise.resolve();
let prevRgba = null;
const SAME_THRESHOLD = 1.0; // 0〜255 の平均差。これ未満なら同じコマとみなす
function frameDiff(a, b) {
  // 全画素を比較。平均の差が小さく、大きく変わった画素がほぼ無いときだけ「同じコマ」
  let sum = 0, big = 0, cnt = 0;
  for (let j = 0; j < a.length; j += 4) {
    const d = Math.abs(a[j] - b[j]) + Math.abs(a[j + 1] - b[j + 1]) + Math.abs(a[j + 2] - b[j + 2]);
    sum += d; if (d > 36) big++; cnt++;
  }
  if (big > 12) return Infinity; // 口パクなど小さな動きも見逃さない
  return sum / cnt / 3;
}
async function upscaleCurrentFrame(session, opts = {}) {
  const { w, h } = state;
  srcCanvas.width = w; srcCanvas.height = h;
  const sctx = srcCanvas.getContext('2d', { willReadFrequently: true });
  sctx.drawImage(state.video, 0, 0, w, h);
  const rgba = sctx.getImageData(0, 0, w, h).data;
  if (opts.skipSame && prevRgba && prevRgba.length === rgba.length && frameDiff(prevRgba, rgba) < SAME_THRESHOLD) {
    return null; // 前のコマとほぼ同じ → AI処理を省略して前の結果を使う
  }
  prevRgba = rgba;
  if (session.a4k) {
    const { tw, th } = targetSize();
    const e = await getA4k(tw, th);
    const c = await runA4k(e, rgba);
    dstCanvas.width = tw; dstCanvas.height = th;
    const dctx = dstCanvas.getContext('2d');
    dctx.imageSmoothingEnabled = true; dctx.imageSmoothingQuality = 'high';
    dctx.drawImage(c, 0, 0, tw, th);
    return dstCanvas;
  }
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
  const be = state.backend;
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
    let skipped = 0;
    prevRgba = null;
    for (let i = 0; i < total; i++) {
      if (state.cancel) throw new Error('中止しました');
      if (encErr) throw encErr;
      await seekTo(state.video, Math.min(i * dt + dt / 2, state.duration - 0.001));
      const res = await upscaleCurrentFrame(session, { skipSame: true });
      if (res === null) skipped++;
      const canvas = dstCanvas;
      const frame = new VideoFrame(canvas, { timestamp: Math.round(i * usPerFrame), duration: Math.round(usPerFrame) });
      venc.encode(frame, { keyFrame: i % gop === 0 });
      frame.close();
      while (venc.encodeQueueSize > 4) await sleep(5);
      if (i % 15 === 0 || i === total - 1) showPreview();
      const el = (performance.now() - t0) / 1000, done = i + 1;
      const fps = done / el;
      ui.bar.value = done / total;
      setStatus(`変換中… ${done} / ${total} コマ（${Math.round((done / total) * 100)}%）　${fps.toFixed(2)} コマ/秒　残り約 ${fmtTime((total - done) / fps)}${skipped ? `　（同じコマを ${skipped} 枚スキップ）` : ''}`);
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
