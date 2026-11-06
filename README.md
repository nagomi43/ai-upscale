# AI動画アップスケール

ブラウザだけで動画をAI（Real-ESRGAN）で 1080p / 1440p / 4K に高画質化するWebアプリです。
処理はすべて開いた人のPCのブラウザ内（WebGPU）で行い、動画はどこにも送信されません。

**使う:** https://nagomi43.github.io/ai-upscale/

## 使い方
1. Chrome または Edge（パソコン版）で上のURLを開く
2. 動画をドラッグ＆ドロップ（MP4がおすすめ）
3. 1コマ試した結果と所要時間の目安が出るので、よければ「変換スタート」
4. 保存先を選ぶと、音声付きのMP4が書き出されます

GPUがうまく動かないときは、URLの最後に `?cpu` を付けるとCPUで処理します（遅くなります）。

## 仕組み
- AIモデル: Real-ESRGAN の動画向け軽量モデルをONNXに変換したもの（`models/`）
  - `anime-x4.onnx` … realesr-animevideov3
  - `real-x4.onnx` … realesr-general-x4v3
- 推論: [ONNX Runtime Web](https://github.com/microsoft/onnxruntime)（WebGPU、半精度対応GPUでは fp16 モデル `*-fp16.onnx`、使えない場合はWASM）
- 高速化: 前のコマとほぼ同じコマ（アニメの2コマ・3コマ打ちなど）はAI処理を省略して前の結果を使う
- 動画の読み込み: `<video>` 要素で1コマずつ取り出し
- 書き出し: WebCodecs（H.264、未対応ならVP9）＋ [mp4-muxer](https://github.com/Vanilagy/mp4-muxer)、音声はAAC（未対応ならOpus）で再エンコード

## ライセンス
- Real-ESRGAN のモデル: BSD-3-Clause（© Xintao Wang）
- ONNX Runtime Web: MIT（© Microsoft）
- mp4-muxer: MIT（© Vanilagy）
