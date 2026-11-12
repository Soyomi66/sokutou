<p align="center"><img src="assets/icon.svg" width="72" height="72" alt=""></p>

<h1 align="center">sokutou</h1>

<p align="center">
  Irodori-TTS（v4.1-Small、v4.1-Small-MF、v4-Large）、Qwen3-TTS（12Hz の 0.6B・1.7B CustomVoice）、Fish Audio Drama 3 を NVIDIA の GPU で速く動かす音声合成サーバー
</p>

<p align="center">
  <a href="#速さ">速さ</a> ·
  <a href="#動かし方">動かし方</a> ·
  <a href="#api">API</a> ·
  <a href="#できないこと">できないこと</a> ·
  <a href="#qwen3-tts">Qwen3-TTS</a> ·
  <a href="#fish-audio-drama-3">Drama 3</a> ·
  <a href="docs/results.md">結果の詳細</a>
</p>

sokutou は [Irodori-TTS](https://github.com/Aratako/Irodori-TTS) v4.1-Small の推論エンジンと、それを差し込んだ音声合成サーバーです。v4.1-Small を MeanFlow で蒸留した v4.1-Small-MF（以下、MF 版）と、約 3.3B のパラメーターを持つ v4-Large にも対応しています。モデルの重みは公式のものをそのまま読み込み、計算の進め方だけを変えています。OpenAI の音声 API と同じ形の `POST /v1/audio/speech` で、参照音声の声で話した WAV を返します。

[Qwen3-TTS](https://github.com/Qwen/Qwen3-TTS) の 12Hz CustomVoice（0.6B と 1.7B）にも対応しています。このモデルは決まった 9 人の話者から選んで話し、参照音声は使いません。自作の CUDA カーネルのエンジン（`qwen3/`）を、Irodori-TTS とは別のサーバーとイメージで動かします。読み込むときに、公式の bf16 の重みから int8 の重みを作ります（bf16 のままも選べます）。

[Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3) にも対応しています。参照音声とその書き起こしを渡すとその声で話し、渡さないと文ごとにモデルが声を選びます。自作の CUDA カーネルのエンジン（`drama3/`）を、ほかのモデルとは別のサーバーとイメージで動かします。重みの使い方には Fish Audio Community License の制限があります（「ライセンス」の節）。Built with Fish Audio.

> sokutou is a faster inference engine for Irodori-TTS v4.1-Small, v4.1-Small-MF and v4-Large on NVIDIA GPUs, served over an OpenAI-style `POST /v1/audio/speech`. On one RTX 6000 Ada the median time for a v4.1-Small sentence is 0.098 s at the model's default 40 steps, against 0.758 s for the official PyTorch implementation and 0.398 s for speech.cpp v0.10.1 on Vulkan, and 0.059 s at 20 steps, against 0.418 s and 0.221 s. The distilled v4.1-Small-MF takes 0.028 s at its 4 steps, against 0.134 s and 0.075 s. The 3.3B v4-Large takes 0.376 s at 40 steps and 0.198 s at 20 steps, against 2.073 s and 1.099 s for the official implementation and 0.846 s and 0.453 s for speech.cpp (F16). It also runs Qwen3-TTS 12Hz CustomVoice 0.6B and 1.7B on its own CUDA kernels: with int8 weights the server takes 0.040 s and 0.058 s per second of audio including HTTP, against 0.081 s and 0.096 s for speech.cpp (Q8_0) and about 1.5 s for the official implementation in bf16. It also runs Fish Audio Drama 3 on its own CUDA kernels (Built with Fish Audio): the int8 server takes a median 0.628 s per sentence including HTTP, against 1.026 s for speech.cpp (Q4_K) and 1.643 s for the official fish-speech implementation. This README is in Japanese.

**同じ GPU、同じステップ数で、v4.1-Small では公式の実装の約 7 倍の速さです。** RTX 6000 Ada で 1 文の合成にかかる時間の中央値は、モデルの既定の 40 ステップで 0.098 秒（公式の実装は 0.758 秒、speech.cpp は 0.398 秒）、20 ステップで 0.059 秒（同じく 0.418 秒、0.221 秒）です。MF 版はモデルの既定の 4 ステップで 0.028 秒です（公式の実装は 0.134 秒、speech.cpp は 0.075 秒）。v4-Large は 40 ステップで 0.376 秒、20 ステップで 0.198 秒です（公式の実装は 2.073 秒と 1.099 秒、speech.cpp は 0.846 秒と 0.453 秒）。Qwen3-TTS では、HTTP を含めた合成の時間が、音声 1 秒あたり 0.6B で 0.040 秒、1.7B で 0.058 秒です（int8。speech.cpp の Q8_0 は 0.081 秒と 0.096 秒、公式の実装は約 1.5 秒）。Drama 3 では、HTTP を含めた 1 文の時間の中央値が 0.628 秒です（int8。speech.cpp の Q4_K は 1.026 秒、公式の実装は 1.643 秒）。

**Irodori-TTS では、重みと API は公式の実装のままです。** 読み込むのは Hugging Face にある公式の重みです。API、入力の確認、24kHz への変換、無音の切り落としは、公式の実装で合成するサーバー（`server/server.py`）と同じコードを使います。Qwen3-TTS と Drama 3 のサーバーも同じエンドポイントとヘッダーで答えます（受け付ける項目は、それぞれの節にあります）。

**公式の実装と数値で突き合わせています。** エンジンを fp32 で動かし、公式の実装と同じ入力から合成すると、ステップごとの出力から最後の音声まで、どの段階の差も公式の実装を GPU と CPU で動かしたときの差より小さくなりました。MF 版（4 ステップ）と v4-Large（20 ステップ）でも同じでした。サーバーで使う BF16 では波形が変わるので、48 文の読み間違いの率（CER）と声の近さを公式の実装と比べました。Qwen3-TTS でも、エンジンを float32 で動かすと、同じ文・同じシードで作った音声のコードが、公式の実装とすべてのフレームで同じになりました（0.6B と 1.7B、12 件ずつ）。Drama 3 では、エンジンの各層を float32 で動かすと、公式の実装の層を float32 にしたものとの SNR が、言語モデルの 40 層すべてで 113 dB 以上になりました。

**公開されているものだけで作れます。** Dockerfile は Docker Hub の python のイメージから始め、版を固定した Irodori-TTS を GitHub から、依存を PyPI と PyTorch の配布元（download.pytorch.org）から取得します。Qwen3-TTS のイメージも、python のイメージ、PyPI、PyTorch の配布元だけから作ります。Drama 3 のイメージは、これに GitHub の fish-speech を加えて作ります。

## 速さ

この節と次の節の値は Irodori-TTS のものです。Qwen3-TTS と Drama 3 の値は、「[Qwen3-TTS](#qwen3-tts)」と「[Fish Audio Drama 3](#fish-audio-drama-3)」の節にあります。

2026-11-11 に、1 枚の RTX 6000 Ada（48GB、電力の上限 300 W）で測りました。48 文の日本語（`data/sentences.json`）をシード 1 と 2 で 1 件ずつ合成し、それを 2 回繰り返した 192 回の中央値です。測る範囲は、合成から 24kHz への変換と前後の無音の切り落としまでで、サーバーが返す `X-Synthesis-Seconds` と同じです。どれも電子透かしを入れず、話す長さ（`seconds`）は同じ値を渡しました。CFG は independent（文 3.0、話者 5.0）で、ステップの前半にかけます。v4-Large も同じ設定です。MF 版には CFG がありません。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/speed-dark.svg">
  <img src="assets/speed-light.svg" width="520" alt="1 文の合成にかかる時間の中央値。v4.1-Small（20 ステップ）は sokutou 0.059 秒、speech.cpp 0.221 秒、公式の実装 0.418 秒。v4.1-Small-MF（4 ステップ）は 0.028 秒、0.075 秒、0.134 秒。v4-Large（20 ステップ）は 0.198 秒、0.453 秒、1.099 秒">
</picture>

| 実装 | v4.1-Small、40 ステップ | v4.1-Small、20 ステップ |
| --- | --- | --- |
| sokutou（BF16） | 0.098 秒 | 0.059 秒 |
| 公式の実装（PyTorch、fp32） | 0.758 秒 | 0.418 秒 |
| speech.cpp v0.10.1（Vulkan、F16） | 0.398 秒 | 0.221 秒 |

| 実装 | v4-Large、40 ステップ | v4-Large、20 ステップ |
| --- | --- | --- |
| sokutou（BF16） | 0.376 秒 | 0.198 秒 |
| 公式の実装（PyTorch、fp32） | 2.073 秒 | 1.099 秒 |
| speech.cpp v0.10.1（Vulkan、F16） | 0.846 秒 | 0.453 秒 |

| 実装 | MF 版、4 ステップ |
| --- | --- |
| sokutou（BF16） | 0.028 秒 |
| 公式の実装（PyTorch、fp32） | 0.134 秒 |
| speech.cpp v0.10.1（Vulkan、F16） | 0.075 秒 |

- sokutou と公式の実装は新しいプロセスで条件を交互に測りました（`scripts/bench_compare.sh`）。sokutou は起動のときの準備だけを済ませ、48 文を前もって通さずに測っています。
- speech.cpp は `speech serve` を立て、同じ本文を HTTP で送って測りました。値は HTTP を含み、音声は 48kHz のまま返り、無音の切り落としはありません。
- v4-Large の speech.cpp の値は、F16 の重み（`irodori-tts-large:f16`）で測ったものです。既定の Q8_0 の重みでは 40 ステップで 0.944 秒、20 ステップで 0.498 秒でした。
- 20 ステップの sokutou をサーバーとして動かし、HTTP で送ったときの `X-Synthesis-Seconds` の中央値は 0.059 秒、HTTP を含めた時間の中央値は 0.061 秒でした。MF 版では 0.030 秒と 0.031 秒、v4-Large の 20 ステップでは 0.196 秒と 0.198 秒でした。

詳しい内訳と測り方は [docs/results.md](docs/results.md) にあります。

## 読み間違いと声の近さ

48 文 × シード 1, 2 の 96 本の音声を faster-whisper（medium）で書き起こして読み間違いの率（CER）を出し、話者照合のモデル（microsoft/wavlm-base-plus-sv）で参照の話者の本物の声との近さを出しました。本物の声の 2 つの区間どうしの近さは 0.952 です。

| 実装 | CER | 声の近さ（平均） |
| --- | --- | --- |
| 公式の実装、v4.1-Small、40 ステップ | 3.11% | 0.916 |
| 公式の実装、v4.1-Small、20 ステップ | 3.11% | 0.916 |
| sokutou、v4.1-Small、40 ステップ | 2.91% | 0.915 |
| sokutou、v4.1-Small、20 ステップ | 2.96% | 0.915 |
| 公式の実装、MF 版（4 ステップ） | 3.57% | 0.908 |
| sokutou、MF 版（4 ステップ） | 4.03% | 0.909 |
| 公式の実装、v4-Large、40 ステップ | 3.26% | 0.920 |
| 公式の実装、v4-Large、20 ステップ | 3.26% | 0.919 |
| sokutou、v4-Large、40 ステップ | 3.36% | 0.920 |
| sokutou、v4-Large、20 ステップ | 3.21% | 0.919 |

- v4.1-Small の公式の実装の音声には電子透かしが入っています。sokutou の音声と、MF 版と v4-Large の公式の実装の音声には入っていません。
- MF 版はシード 1〜10 の 480 本でも比べました。sokutou を fp32 で動かした音声は、480 本すべてで誤りの数が公式の実装と同じでした。BF16 の CER は 4.00% で、公式の実装の 3.69% より高くなりました。ただし、同じ文どうしの差の平均は標準誤差の 1.6 倍で、偶然の差と見分けられるほど大きくありません。
- v4-Large の sokutou の音声で、誤りの数が同じ文の公式の実装の音声と違ったのは、40 ステップと 20 ステップで 1 本ずつでした。sokutou を fp32 で動かした音声（20 ステップ）は、96 本すべてで誤りの数が公式の実装と同じでした。
- 確かめた声は 1 つです（参照音声は 7.9 秒）。speech.cpp の CER と声の近さは測っていません。

## 必要なもの

- Linux（x86-64）、Docker、NVIDIA Container Toolkit
- NVIDIA の GPU と、約 13GB の GPU のメモリー（v4-Large は約 31GB）
- イメージに約 9.4GB、モデルに約 3.4GB のディスク（v4-Large は約 12.8GB）
- Qwen3-TTS には compute capability 8.0 以上の GPU（Ampere 以降）と、約 8.5GB の GPU のメモリー（1.7B は約 11.7GB）
- Qwen3-TTS のイメージに約 9.1GB、モデルに約 2.3GB のディスク（1.7B は約 4.3GB）
- Drama 3 には compute capability 8.9 以上の GPU（Ada 以降）と、約 45 GiB の GPU のメモリー
- Drama 3 のイメージに約 8.7GB、モデルに約 41.5GB のディスク

確かめた環境は RTX 6000 Ada、ドライバー 580.196.07、Ubuntu 22.04、Docker 28.2.2 だけです。エンジンは BF16 で計算するので、ほかの GPU で動くかは確かめていません。

## 動かし方

1. イメージを作ります。初めてのときは、Irodori-TTS の依存（torch など、約 8GB）を取得します。

   ```sh
   git clone https://github.com/kosuketut/sokutou.git
   cd sokutou
   scripts/build_voice_engine.sh
   ```

   最後に、作ったイメージのタグ（`sokutou/server:<commit>`）が出ます。

2. 参照音声を `voices/<声の名前>/` に置きます。書き起こしは要りません。1 つのディレクトリに WAV が 2 つ以上あると、名前順につないで 1 つの参照音声にします（合計 120 秒まで）。`voices/` は git の管理外です。

   ```sh
   mkdir -p voices/me
   cp ~/recordings/me.wav voices/me/ref.wav
   ```

   本人の同意を得た声だけを使ってください（「ライセンス」の節）。

3. サーバーを起動します。`<タグ>` は 1. で出たタグに置き換えます。

   ```sh
   docker run -d --name sokutou --gpus device=0 -p 127.0.0.1:8088:8088 \
     -v sokutou-hf:/root/.cache/huggingface \
     -v "$PWD/voices:/voices:ro" \
     -e HF_HUB_OFFLINE=0 -e IRODORI_NUM_STEPS=20 -e EMPTY_CACHE_EVERY=0 \
     <タグ>
   ```

   MF 版を使うときは、`-e IRODORI_CHECKPOINT=Aratako/Irodori-TTS-v4.1-Small-MF -e IRODORI_NUM_STEPS=4` を足します。

   v4-Large を使うときは、`-e IRODORI_CHECKPOINT=Aratako/Irodori-TTS-v4-Large` を足します。ステップ数は 20 のままで使えます。重みは Gemma の利用規約に従います（「ライセンス」の節）。

   初めて起動するときは、モデル、コーデック、電子透かしのモデル（合わせて約 3.4GB）を Docker のボリューム `sokutou-hf` に取得します。準備ができると、`/health` が 200 を返します。作者のマシンでは、取得を含めて 73 秒、2 回目からは約 30 秒かかりました。v4-Large は初めての起動で合わせて約 12.8GB を取得し、取得を含めて 180 秒で準備ができました。2 回目からは 51 秒でした。

   ```sh
   curl -s http://127.0.0.1:8088/health
   ```

4. 合成します。

   ```sh
   curl -s http://127.0.0.1:8088/v1/audio/speech -H 'Content-Type: application/json' \
     -d '{"input": "こんにちは。今日もよろしくお願いします。"}' -o hello.wav
   ```

サーバーには認証がありません。ポートは `127.0.0.1` に限って公開してください。

## API

Irodori-TTS のサーバーの API です。Qwen3-TTS のサーバーとの違いは「Qwen3-TTS」の節にあります。

| エンドポイント | 中身 |
| --- | --- |
| `POST /v1/audio/speech` | 本文の JSON を受け取り、WAV（PCM16、モノラル、24kHz）を返します。応答のヘッダー `X-Synthesis-Seconds` は合成にかかった秒数、`X-Audio-Seconds` は音声の長さです |
| `GET /health` | 準備ができるまで 503、できたら 200 を返します |
| `GET /v1/audio/voices` | 声の一覧を返します |
| `GET /v1/models` | `irodori-tts` を返します |
| `GET /v1/engine` | エンジンの設定と、合成した回数を返します。`irregular_requests` は、エンジンの外（公式の関数）を通った合成の数です |

### `POST /v1/audio/speech` の本文

| 項目 | 中身 |
| --- | --- |
| `input` | 読む文。200 字まで。文末が「。」などで終わらなければ「。」を足します |
| `voice` | 声の名前。省くと既定の声（名前順で最初の声） |
| `speed` | 話す速さ。0.5〜2.0、既定 1.0 |
| `seed` | 乱数のシード。省くと毎回変わります |
| `seconds` | 話す長さ（秒）。0.5〜30。省くとモデルが予測します。モデルの予測は長めで、余った時間に余計な声が入ることがあります |
| `response_format` | `wav` だけ |

合成は 1 件ずつです。同時に届いた要求は、前の合成が終わるまで待ちます。待つ時間も `X-Synthesis-Seconds` に入ります。

## 環境変数

| 名前 | 中身 |
| --- | --- |
| `IRODORI_CHECKPOINT` | モデル（既定 `Aratako/Irodori-TTS-v4.1-Small`）。MF 版は `Aratako/Irodori-TTS-v4.1-Small-MF` にし、`IRODORI_NUM_STEPS` を 4 にします。v4-Large は `Aratako/Irodori-TTS-v4-Large` です |
| `IRODORI_NUM_STEPS` | 必須。ステップ数で、エンジンと要求の両方に使います。空だと起動しません。確かめたのは 20 と 40 です（MF 版は 4。v4-Large のサーバーは 20 だけ） |
| `EMPTY_CACHE_EVERY` | この回数の合成ごとに GPU のキャッシュを解放します（既定 50）。sokutou では 0 をすすめます。0 でも GPU のメモリーは起動の準備のあとで増えず、50 では解放する回の合成が 13〜47 ミリ秒遅くなります |
| `SOKUTOU_WATERMARK` | 1 にすると電子透かしを残します（既定 0） |
| `HF_HUB_OFFLINE` | イメージの既定は 1（取得しない）。初めて起動するときは 0 にします |
| `VOICE_DIR` | 声のディレクトリ（既定 `/voices`） |
| `DEFAULT_VOICE` | `voice` を省いたときの声 |
| `OUTPUT_SAMPLE_RATE` | 出力のサンプリングレート（既定 24000。モデルの出力は 48kHz） |
| `MAX_CHARS` | 1 件の最大の字数（既定 200） |
| `HOST`、`PORT` | 待ち受け（既定 `0.0.0.0:8088`。コンテナの中） |

`IRODORI_PRECISION` は fp32 のままにします。エンジンが fp32 の重みから BF16 の複製を作ります。

## 仕組み

v4.1-Small の公式の実装では、1 文の時間が、DiT を 40 回通すあいだに CPU が小さなカーネルを 1 つずつ起動する時間で決まっていると考えられました。公式のモジュールのまま CUDA Graph にまとめるだけで、0.78 秒が 0.45 秒になったためです（[docs/baseline.md](docs/baseline.md)）。sokutou は要らない計算を外し、毎回同じになる計算を前もって済ませ、残りを CUDA Graph にまとめて GPU に渡します。

- 文のパディングと空の caption を、文のエンコーダー・KV・attention に入れません。v4.1-Small の 48 文では、文の 256 トークンのうち有効なのは 4〜18 個で、caption の 512 トークンはすべて無効でした。
- 話者の条件とその KV は声ごとに一度だけ、時刻だけで決まる AdaLN の係数は全ステップ分を起動のときに計算します。
- MF 版は CFG がないので、1 ステップで DiT を 1 回だけ通します。AdaLN の係数には時刻に加えてステップの幅の埋め込みも足し、これも全ステップ分を起動のときに計算します。
- 文のエンコード、全ステップ、CFG の合成を CUDA Graph に記録して再生します。潜在の長さと文のトークン数を区切りに合わせ、432 通りの形の 873 個（MF 版は 441 個）のグラフを起動のときにすべて記録します。
- DiT の各層（v4.1-Small は 12 層、v4-Large は 24 層）の重み・行列積・attention を BF16 にし、残差、正規化、CFG の合成は fp32 のまま計算します。
- 小さな処理を自作の CUDA カーネル（CuPy で NVRTC でコンパイルする）にまとめ、v4.1-Small では 1 ステップのカーネルを約 460 個から約 200 個に減らしました。
- v4-Large の文のエンコーダー（T5Gemma 2）には、CUDA Graph に記録できない処理（真偽値のマスクを使った代入）が 1 つあります。sokutou はこれを同じ値を返す処理に差し替えてから記録します。
- 復号（コーデック）は決まった 47 通りの長さに 0 で伸ばして動かします。初めての長さで遅くなることと、GPU のメモリーが増え続けることを防ぎます。
- 電子透かし（SilentCipher）は既定で外します。

要求の設定がエンジンと違うと、エンジンは公式の関数で合成します。サーバーは起動のときの試し合成でそうなったら起動を止め、動いているあいだは `irregular_requests` を数えます。

各段の時間と、公式の実装との突き合わせの結果は [docs/results.md](docs/results.md) にあります。

## できないこと

この節は Irodori-TTS のサーバーのことです。Qwen3-TTS のサーバーでできないことは「Qwen3-TTS」の節にあります。

- 1 つのサーバーで使える声は 1 つです。`voices/` に声が 2 つ以上あると起動しません。
- 使えるモデルは v4.1-Small、MF 版、v4-Large の 3 つです。v2・v3 のモデルは使えません。
- ステップ数のほかの設定（CFG など）は、上の「速さ」の節のものに決まっています。caption（VoiceDesign）と、参照音声を使わない合成はできません。
- GPU のメモリーを約 13GB（nvidia-smi で 12,500 MiB 前後）使い、起動には約 30 秒かかります（うちエンジンの準備は約 8 秒）。MF 版は約 11.7GB で、起動には 28 秒（うちエンジンの準備は 6.2 秒）かかりました。
- v4-Large は約 31GB（nvidia-smi で 29,790 MiB）で、起動には 51 秒（うちエンジンの準備は 12.0 秒）かかりました。
- 出力は公式の実装と同じ値にはなりません（BF16 のため）。読み間違いと声の近さは上の 1 つの声で確かめました。
- v4.1-Small では、長さの違う参照音声（3 秒、15.7 秒、31.4 秒）で起動と速さだけを確かめました。声の質は確かめていません。v4-Large は 7.9 秒の参照音声だけで確かめました。v4-Large のモデルカードは、短い参照音声 1 つでは声の近さがかなり下がるとし、できれば 30 秒以上の参照音声を使うようすすめています。

## Qwen3-TTS

[Qwen3-TTS](https://github.com/Qwen/Qwen3-TTS) の 12Hz CustomVoice のモデル（Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice と Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice）を、自作の CUDA カーネルのエンジン（`qwen3/`）で動かします。このモデルでは、言語モデルが音声のコードを 1 フレーム（0.08 秒）ずつ作り、コーデックがそのコードを 24kHz の波形にします。話者は決まった 9 人から選びます。

### 速さ（Qwen3-TTS）

2026-11-11 に、上と同じ RTX 6000 Ada で測りました。48 文をシード 1 と 2 で 1 件ずつ合成し、それを 2 回繰り返した 192 回の値です（サーバーは 1 回分の 96 回）。話者は Ono_Anna、言語は日本語で、サンプリングの設定はモデルの既定（温度 0.9、top_k 50、repetition_penalty 1.05）です。

このモデルでは話す長さを指定できず、長さはサンプリングで決まります。音声の長さの中央値が実装ごとに違う（3.5〜4.8 秒）ので、合成の時間の合計を音声の長さの合計で割った「音声 1 秒あたり」の時間で比べます。測る範囲は、文を受け取ってから 24kHz の波形ができるまでで、無音の切り落としはしません。

| 実装 | 0.6B、音声 1 秒あたり | 0.6B、1 文の中央値 | 1.7B、音声 1 秒あたり | 1.7B、1 文の中央値 |
| --- | --- | --- | --- | --- |
| sokutou のサーバー（int8、HTTP を含む） | 0.040 秒 | 0.181 秒 | 0.058 秒 | 0.209 秒 |
| speech.cpp v0.10.1（Vulkan、Q8_0、HTTP を含む） | 0.081 秒 | 0.341 秒 | 0.096 秒 | 0.360 秒 |
| sokutou（int8） | 0.039 秒 | 0.176 秒 | 0.058 秒 | 0.207 秒 |
| sokutou（bf16） | 0.056 秒 | 0.270 秒 | 0.087 秒 | 0.312 秒 |
| 公式の実装（qwen-tts 0.1.1、bf16） | 1.526 秒 | 7.066 秒 | 1.549 秒 | 5.476 秒 |

- HTTP を含めて比べると、sokutou のサーバーの時間は speech.cpp の約 5 割（0.6B）と約 6 割（1.7B）です。
- 「sokutou（int8）」と「sokutou（bf16）」は、エンジンを同じプロセスから呼んで測った値です。
- speech.cpp は既定の Q8_0 の重みだけで測りました。sokutou の int8 は Q8_0 と同じく 32 個ごとに尺度を持ちます。bf16 の行は speech.cpp と精度が違います。
- 公式の実装は bf16 と PyTorch の sdpa で動かしました。公式がすすめる flash_attention_2 は入れていません。公式の実装では 1 フレームに約 0.12 秒かかります。
- 1.7B のサーバーは 96 件のうち 1 件（13 字の文で、18 秒を超えて続いた音声）に 500 を返したので、この行は 95 件の値です。

### 読み間違いと声の近さ（Qwen3-TTS）

48 文 × シード 1, 2 の 96 本を、エンジンを直接呼んで作り、上と同じ書き起こしと数え方で CER を出しました。CustomVoice には参照音声がないので、声の近さには公式の実装の音声の埋め込みの平均（重心）とのコサイン類似度を使いました。シード 1 の音声はシード 2 の公式の音声の重心と、シード 2 の音声はシード 1 の重心と比べたので、公式の実装の値がこの測り方での上限の目安になります。

| 実装 | 0.6B、CER | 0.6B、悪い 2 本を除いた CER | 0.6B、声の近さ | 1.7B、CER | 1.7B、悪い 2 本を除いた CER | 1.7B、声の近さ |
| --- | --- | --- | --- | --- | --- | --- |
| 公式の実装（bf16） | 5.71% | 4.34% | 0.956 | 5.96% | 5.09% | 0.945 |
| speech.cpp（Q8_0） | 5.66% | 4.92% | 0.948 | 19.88% | 9.73% | 0.944 |
| sokutou（bf16） | 5.56% | 4.63% | 0.958 | 5.76% | 3.60% | 0.940 |
| sokutou（int8） | 7.03% | 5.66% | 0.953 | 8.56% | 3.81% | 0.944 |

- サンプリングで話すので、同じ文の音声も実装とシードで変わります。話し終えずに続く音声が数本あると、96 本の CER は数 % 動きます。「悪い 2 本を除いた CER」は、CER がいちばん高い 2 本を除いた値です。
- この 96 本には、サーバーなら 500 を返す長さの音声も入っています。0.6B と 1.7B の順に、公式の実装は 1 本ずつ、speech.cpp は 0 本と 3 本、sokutou の int8 は 0 本と 1 本、bf16 は 2 本ずつでした。
- 0.6B の int8 の CER は、96 本では悪い 2 本を除いても公式の実装より高くなりました（5.66% と 4.34%）。シード 1〜6 の 288 本では、int8 が 5.23%（悪い 2 本を除くと 4.74%）、公式の実装が 6.95%（4.98%）でした。
- サンプリングの処理だけが違う 2 つ（torch の演算と、1 つのカーネル。乱数の列も違います）を、0.6B の int8 の 288 本で比べると、CER は 7.41% と 5.23% でした。重みと言語モデルの計算が同じでも、CER はこのくらい動きます。

### 動かし方（Qwen3-TTS）

1. イメージを作ります。初めてのときは、qwen-tts の依存（torch など、約 8GB）を取得します。

   ```sh
   scripts/build_qwen3.sh
   ```

   最後に、作ったイメージのタグ（`sokutou/qwen3-server:<commit>`）が出ます。

2. サーバーを起動します。`<タグ>` は 1. で出たタグに置き換えます。1.7B を使うときは、`-e QWEN3_MODEL=1.7b` を足します。

   ```sh
   docker run -d --name sokutou-qwen3 --gpus device=0 -p 127.0.0.1:18096:18096 \
     -v sokutou-hf:/root/.cache/huggingface -e HF_HUB_OFFLINE=0 <タグ>
   ```

   初めて起動するときは、モデル（0.6B は 2.3GB）を Docker のボリューム `sokutou-hf` に取得します。作者のマシンでは、0.6B は取得を含めて 74 秒で `/health` が 200 を返しました。2 回目からは、0.6B も 1.7B も約 30 秒です。

3. 合成します。

   ```sh
   curl -s http://127.0.0.1:18096/v1/audio/speech -H 'Content-Type: application/json' \
     -d '{"input": "こんにちは。今日もよろしくお願いします。", "voice": "ono_anna"}' -o hello.wav
   ```

### API と環境変数（Qwen3-TTS）

エンドポイント、応答の WAV（PCM16、モノラル、24kHz）、ヘッダーは、Irodori-TTS のサーバーと同じです。`GET /v1/models` は `qwen3-tts` を返します。`GET /v1/engine` は、起動のときの記録と、合成した回数、長さの上限で止めた回数を返します。本文で受け付ける項目は次のとおりです。

| 項目 | 中身 |
| --- | --- |
| `input` | 読む文。200 字まで。最初の入力が 160 トークンを超える文には 400 を返します（ふつうの日本語の 200 字で約 124 トークン） |
| `voice` | 話者。`aiden`、`dylan`、`eric`、`ono_anna`、`ryan`、`serena`、`sohee`、`uncle_fu`、`vivian` のどれかで、大文字と小文字は区別しません。省くと `QWEN3_SPEAKER` |
| `language` | 言語の名前。省くと `QWEN3_LANGUAGE`。公式の実装が受け付けない名前には 400 を返します |
| `seed` | 乱数のシード。同じシードなら同じ音声になり、省くと毎回変わります |
| `response_format` | `wav` だけ |

`speed` と `seconds` は使いません（送っても無視します）。音声が 2.0 + 0.4 × 字数 秒より長くなると、その長さで生成を止めて 500 を返します（話し終えずに続いた音声）。合成は 1 件ずつで、前の合成を待つ時間も `X-Synthesis-Seconds` に入ります。

| 名前 | 中身 |
| --- | --- |
| `QWEN3_MODEL` | `0.6b`（既定）か `1.7b` |
| `QWEN3_WD` | 重みの型。`i8g32`（既定。int8 と 32 個ごとの尺度）か `bf16` |
| `QWEN3_SPEAKER` | `voice` を省いたときの話者（既定 `Ono_Anna`） |
| `QWEN3_LANGUAGE` | `language` を省いたときの言語（既定 `Japanese`） |
| `QWEN3_MAX_PROMPT_TOKENS` | 受け付ける最初の入力の長さの上限（既定 160）。起動のときに、6 からこの長さまでの CUDA Graph を記録します（1 つ約 0.1 秒） |
| `MAX_CHARS` | 1 件の最大の字数（既定 200） |
| `HOST`、`PORT` | 待ち受け（既定 `0.0.0.0:18096`。コンテナの中） |
| `HF_HUB_OFFLINE` | イメージの既定は 1（取得しない）。初めて起動するときは 0 にします |

### 仕組み（Qwen3-TTS）

公式の実装は 1 フレームごとに、code predictor（5 層）で第 1〜15 のコードを 1 つずつ作り、16 個のコードの埋め込みの和を 1 トークンとして言語モデル（28 層）に入れ、次のフレームの第 0 のコードを作ります。1 トークンずつの小さな計算が続くので、カーネルを 1 つずつ起動する時間がかさみ、1 フレームに約 0.12 秒かかります。

- 各層を 5 個の自作の CUDA カーネルにまとめ、1 フレーム分を 1 つの CUDA Graph に記録して再生します。1 フレームには、code predictor の 16 回（言語モデルの最後の隠れ状態を入れる 1 回と、第 1〜15 のコードを作る 15 回）、言語モデルの 1 回、16 回のサンプリングが入ります。
- 1 フレームの時間の大半は重みを読む行列とベクトルの積（GEMV）でした。そこで、読み込むときに重みを int8（32 個ごとの尺度）にして、読む量を減らします。残差と足し算は float32 で計算します。
- code predictor の重み（int8 で約 80 MiB）は 1 フレームに 16 回読むので、L2 キャッシュ（RTX 6000 Ada は 96 MiB）に残るように、読み込みのときに L2 の扱いを指定します。言語モデルの重みのように 1 フレームに 1 回しか読まない重みは、先に L2 から追い出します。
- サンプリング（繰り返しの罰、温度、top_k、引く）を 1 つのカーネルにまとめました。乱数は自作（Philox）なので、同じシードでも公式の実装とは違う音声になります。引いた頻度が公式と同じ分布に合うことは確かめました（[docs/results.md](docs/results.md)）。
- 最初の入力（話者、言語、文）は長さごとに、復号は 18 通りの長さの区切りごとに、CUDA Graph を起動のときに記録します。

1 フレームの時間は、int8 で 0.6B が 2.9 ミリ秒、1.7B が 4.4 ミリ秒です。

### できないこと（Qwen3-TTS）

- 使えるのは 12Hz の CustomVoice の 0.6B と 1.7B だけです。参照音声から声を作るモデル（Base）と、声を文で指定するモデル（VoiceDesign）は使えません。1.7B の話し方の指示（instruct）も使えません。
- 出力は公式の実装と同じ値にはなりません（int8 か bf16 で計算することと、自作の乱数のため）。
- compute capability 8.0 以上の GPU（Ampere 以降）が要ります。確かめた GPU は RTX 6000 Ada だけです。
- 読み間違いと声の近さを確かめたのは、Ono_Anna と日本語だけです。ほかの話者は vivian で合成できることだけを確かめました。
- 話し終えずに続く音声が上の 96 本のうち 0〜2 本ありました。サーバーはこれに 500 を返すので、使う側で別のエンジンに回すなどの扱いが要ります。

## Fish Audio Drama 3

Built with Fish Audio.

[Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3)（fishaudio/drama-3）を、自作の CUDA カーネルのエンジン（`drama3/`）で動かします。言語モデルは 35B のパラメーターを持つ MoE で、1 フレーム（46.4 ミリ秒）ごとに 10 個の音声のコードを作ります。コーデックがそのコードを 44.1kHz の波形にします。参照音声とその書き起こしを渡すとその声で話し、渡さないと文ごとにモデルが声を選びます。公式の実装は [fish-speech](https://github.com/fishaudio/fish-speech) です。モデルと fish-speech は Fish Audio Community License に従い、商用の利用とサービスとしての提供には制限があります（「ライセンス」の節）。

### 速さ（Drama 3）

2026-11-11 に、上と同じ RTX 6000 Ada で測りました。48 文をシード 1 と 2 で 1 件ずつ合成した 96 回の値です。サンプリングの設定は、モデルカードの値（温度 1.0、top_p 0.9、top_k 30）です。「参照あり」は、drama-3 が参照なしで作った 5.8 秒の声を参照音声にした値です（実在の人の声ではありません）。測る範囲は、文を受け取ってから 44.1kHz の波形ができるまでで、無音の切り落としはしません。話す長さはサンプリングで決まるので、Qwen3-TTS と同じく「音声 1 秒あたり」の時間も載せます。

| 実装 | 参照なし、1 文の中央値 | 参照なし、音声 1 秒あたり | 参照あり、1 文の中央値 | 参照あり、音声 1 秒あたり |
| --- | --- | --- | --- | --- |
| sokutou のサーバー（int8、HTTP を含む） | 0.628 秒 | 0.192 秒 | 0.648 秒 | 0.195 秒 |
| speech.cpp v0.10.1（Vulkan、Q4_K、HTTP を含む） | 1.026 秒 | 0.306 秒 | 1.184 秒 | 0.306 秒 |
| speech.cpp v0.10.1（Vulkan、Q8_0、HTTP を含む） | 1.188 秒 | 0.348 秒 | 1.275 秒 | 0.345 秒 |
| sokutou（int8） | 0.620 秒 | 0.191 秒 | 0.641 秒 | 0.194 秒 |
| sokutou（bf16） | 0.806 秒 | 0.233 秒 | 0.951 秒 | 0.263 秒 |
| 公式の実装（fish-speech） | 1.643 秒 | 0.471 秒 | 1.846 秒 | 0.502 秒 |

- HTTP を含めて比べると、sokutou のサーバーは speech.cpp の Q4_K より、参照なしで 1.63 倍、参照ありで 1.83 倍速く答えます。Q8_0 と比べると 1.89 倍と 1.97 倍です。
- 「sokutou（int8）」と「sokutou（bf16）」は、エンジンを同じプロセスから呼んで測った値です。bf16 の行では、最初の入力（参照と文）を CUDA Graph にせずに流しました。
- int8 は、初めて出た長さの最初の入力を CUDA Graph にせずに流し、同じ長さが 2 回目に出たときに記録します（記録は 1 回約 0.5 秒）。同じ 48 文の 2 周目にあたるシード 2 だけの中央値は、参照なしで 0.584 秒、参照ありで 0.611 秒です。
- speech.cpp は `speech serve drama-3`（Q4_K、既定）と `drama-3:q8_0` で立て、本文に `"experts_on_cpu": false` を足して測りました。参照ありでは、同じ参照音声から `speech voice` で作った声のファイルを使いました。speech.cpp は参照音声を符号化する前に、音量を -16 LUFS にそろえます（公式の実装と sokutou は音量を変えません）。
- 公式の実装は fish-speech（commit 0f8a12e）を、fish-speech のサーバーと同じ順に呼んで測りました。重みは bf16 で、エキスパートは FP8 のまま Triton のカーネルで計算します。1 ステップは CUDA Graph にしてあり、1 フレームに 19.6 ミリ秒かかります。
- 長い文（48 文をつないだ 178〜196 字の 3 文を、参照なしと参照ありで 1 回ずつ）もサーバーに送りました。6 回とも 200 を返し、合成に 5.0〜9.9 秒かかりました（音声は 28〜36 秒）。
- 人の声（Irodori-TTS の確かめに使ったのと同じ 7.9 秒の参照音声で、本人の同意を得た声）でも測りました。HTTP の往復の中央値は、sokutou のサーバーが 0.573 秒、speech.cpp の Q4_K が 0.914 秒、Q8_0 が 1.102 秒でした（音声 1 秒あたりは 0.204 秒、0.310 秒、0.354 秒）。公式の実装はプロセスの中で 1.585 秒でした。

### 読み間違いと声の近さ（Drama 3）

上の 96 本を、Irodori-TTS と同じ書き起こしと数え方で調べました。声の近さは、参照音声とのコサイン類似度の平均です。

| 実装 | 参照なし、CER | 参照なし、悪い 2 本を除いた CER | 参照あり、CER | 参照あり、悪い 2 本を除いた CER | 参照あり、声の近さ |
| --- | --- | --- | --- | --- | --- |
| 公式の実装 | 2.04% | 1.75% | 2.40% | 2.11% | 0.967 |
| speech.cpp（Q4_K） | 3.52% | 2.38% | 5.25% | 2.49% | 0.958 |
| speech.cpp（Q8_0） | 3.57% | 2.48% | 2.45% | 1.97% | 0.967 |
| sokutou（bf16） | 3.77% | 2.58% | 2.34% | 2.06% | 0.967 |
| sokutou（int8） | 2.34% | 2.06% | 2.24% | 1.95% | 0.967 |

- サーバーの音声は、同じシードの sokutou（int8）の音声とビット単位で同じでした（参照なし、参照あり、人の声の 96 本ずつ）。
- sokutou（bf16）の参照なしの 3.77% は、シード 1 の 2 本（文頭に息の音が続き、9〜13 秒になった音声）によります。参照なしでシード 3〜8 も測ると、CER は公式の実装が 2.87%、bf16 が 1.95%、int8 が 2.50% でした。公式の実装も、シード 7 で同じ文に CER 90% の音声を出しました。シード 1〜8 の 384 本を合わせた CER は、公式の実装 2.66%、bf16 2.41%、int8 2.46% です。
- サーバーなら 500 を返す長さ（2.0 + 0.4 × 字数 秒）を超えた音声は、参照なしのシード 1〜8 と参照ありのシード 1, 2 の 480 本で、int8 が 0 本、bf16 が 2 本、公式の実装が 1 本でした。speech.cpp はシード 1, 2 の 192 本で、Q4_K が 3 本、Q8_0 が 1 本でした。
- 上の人の声では、CER は公式の実装が 2.34%、sokutou の int8 が 2.19%、bf16 が 2.14%、speech.cpp の Q8_0 が 2.14%、Q4_K が 2.65% でした。参照音声との近さは、公式の実装と sokutou が 0.89、speech.cpp が 0.90〜0.91 でした。speech.cpp は参照音声を符号化する前に音量をそろえます。

### 動かし方（Drama 3）

1. イメージを作ります。初めてのときは、fish-speech と、その依存（torch など）を取得します。

   ```sh
   scripts/build_drama3.sh
   ```

   最後に、作ったイメージのタグ（`sokutou/drama3-server:<commit>`）が出ます。

2. 参照音声を使うときは、声ごとのディレクトリに WAV と、同じ名前の書き起こし（`.lab`、UTF-8）を置きます。書き起こしには WAV で話している文をそのまま書きます。本人の同意を得た声だけを使ってください（「ライセンス」の節）。

   ```sh
   mkdir -p drama3-voices/me
   cp ~/recordings/me.wav drama3-voices/me/ref.wav
   echo 'この録音で話している文をそのまま書きます。' > drama3-voices/me/ref.lab
   ```

   参照音声を使わないときは、このディレクトリは要りません。

3. サーバーを起動します。`<タグ>` は 1. で出たタグに置き換えます。参照音声を使わないときは、`-v` の 2 行目と `-e DRAMA3_VOICE=me` を外します。

   ```sh
   docker run -d --name sokutou-drama3 --gpus device=0 -p 127.0.0.1:18098:18098 \
     -v sokutou-hf:/root/.cache/huggingface \
     -v "$PWD/drama3-voices:/voices:ro" \
     -e HF_HUB_OFFLINE=0 -e DRAMA3_VOICE=me <タグ>
   ```

   初めて起動するときは、モデル（41.5GB）を Docker のボリューム `sokutou-hf` に取得します。作者のマシンでは、取得したあとのキャッシュをつなぐと、`docker run` から `/health` が 200 を返すまで 39 秒でした。取得は、始まることだけを確かめました。GPU のメモリーを約 45 GiB 使うので、48GB の GPU では、ほかのモデルと同じ GPU に載せられません。

4. 合成します。

   ```sh
   curl -s http://127.0.0.1:18098/v1/audio/speech -H 'Content-Type: application/json' \
     -d '{"input": "こんにちは。今日もよろしくお願いします。", "voice": "me"}' -o hello.wav
   ```

### API と環境変数（Drama 3）

エンドポイント、応答の WAV（PCM16、モノラル）、ヘッダーは、Irodori-TTS のサーバーと同じです。WAV のサンプリング周波数は 44.1kHz です。`GET /v1/models` は `drama-3` を返します。`GET /v1/engine` は、起動のときの記録、合成した回数、長さの上限で止めた回数、最初の入力の CUDA Graph を記録した回数を返します。本文で受け付ける項目は次のとおりです。

| 項目 | 中身 |
| --- | --- |
| `input` | 読む文。200 字まで。長い文は fish-speech と同じ規則で区切り、1 区切りずつ作ってつなぎます |
| `voice` | 声の名前（`DRAMA3_VOICES` の中のディレクトリの名前）か `none`。`none` では、文ごとにモデルが声を選びます。省くと `DRAMA3_VOICE` |
| `seed` | 乱数のシード。省くと毎回変わります。同じシードなら同じ音声になります。ただし、声を切り替えた最初の要求では、最初の入力の区切り方が変わるので、少し違う音声になります |
| `response_format` | `wav` だけ |

`speed` と `seconds` は使いません（送っても無視します）。音声が 2.0 + 0.4 × 字数 秒より長くなると、500 を返します。区切りごとの生成はその長さで止め、区切りをまたいだ合計がその長さを超えたら、次の区切りに進まずに止めます。fish-speech が受け付けない入力には 400 を、GPU のメモリーが足りないときは 503 を返します。合成は 1 件ずつで、前の合成を待つ時間も `X-Synthesis-Seconds` に入ります。

| 名前 | 中身 |
| --- | --- |
| `DRAMA3_WD` | エキスパートのほかの重みの型。`i8g32`（既定。int8 と 32 個ごとの尺度）か `bf16`。エキスパートは、どちらでも公式の FP8 のままです |
| `DRAMA3_VOICE` | `voice` を省いたときの声（既定 `none`） |
| `DRAMA3_VOICES` | 声のディレクトリ（既定 `/voices`） |
| `DRAMA3_PREFILL_GRAPHS` | 最初の入力の CUDA Graph を、新しい順にこの数まで持ちます（既定 24。1 つ約 82 MiB。0 なら使いません） |
| `DRAMA3_PREFIX_CACHE` | 1（既定）なら、前の要求と同じ参照の行の状態を使い回します |
| `DRAMA3_TEMPERATURE`、`DRAMA3_TOP_P`、`DRAMA3_TOP_K` | サンプリングの設定（既定 1.0、0.9、30） |
| `MAX_CHARS` | 1 件の最大の字数（既定 200） |
| `HOST`、`PORT` | 待ち受け（既定 `0.0.0.0:18098`。コンテナの中） |
| `HF_HUB_OFFLINE` | イメージの既定は 1（取得しない）。初めて起動するときは 0 にします |

### 仕組み（Drama 3）

公式の実装は 1 フレームごとに、言語モデル（40 層）の最後の隠れ状態から第 0 のコードを引き、tower（共有の 4 層と、コードごとの 9 層）で第 1〜9 のコードを 1 つずつ作ります。10 個のコードは次のフレームの入力として言語モデルに戻ります。言語モデルの各層は MoE で、256 個のエキスパートから選んだ 8 個と、共有のエキスパート 1 個を使います。1 フレームに 19.6 ミリ秒かかります。

- 1 フレーム分の計算を自作の CUDA カーネル（CuPy で NVRTC からコンパイル）にして、1 つの CUDA Graph に記録して再生します。カーネルは、FP8 のエキスパートの行列とベクトルの積（GEMV）、ルーター、Gated DeltaNet の 1 トークンの更新、全注意、tower、出力の層、サンプリングです。
- エキスパートは公式の FP8 の重み（128 × 128 のブロックごとの尺度）をそのまま読みます。そのほかの重みは、読み込むときに int8（32 個ごとの尺度）にします（bf16 のままも選べます）。残差と積の和は float32 で計算します。
- 出力の層は、語彙 252,288 のうち、引く候補になる eos と 4096 個の意味のトークンの行だけを計算します。公式の実装は全部を計算してから候補を選びます。
- サンプリング（top_k、top_p、繰り返しを引き直す RAS）もカーネルにしました。乱数は自作（Philox）なので、同じシードでも公式の実装とは違う音声になります。引いた頻度が公式と同じ分布に合うことは確かめました（[docs/results.md](docs/results.md)）。
- 最初の入力（参照と文）は、公式のモジュールで 128 行ずつ流します。区切りの長さごとに CUDA Graph にし、同じ長さが 2 回目に出たときから記録します。前の要求と参照が同じなら、参照の行を流したあとの Gated DeltaNet の状態を写しておき、次からはそこから先だけを流します（約 200 行で 250 ミリ秒から 33 ミリ秒になりました）。
- 長い文の区切り方とプロンプトの作り方は fish-speech の `generate_long_drama3` のままで、その中で呼ぶ 1 区切り分の生成だけを差し替えます。
- コーデックの復号は公式の実装のままです（1 文で 24〜45 ミリ秒）。読み込むときに、復号で使わない因果マスク（3 GiB）を空けます。

1 フレームの時間は、int8 で 6.7 ミリ秒、bf16 で 8.3 ミリ秒です。

### できないこと（Drama 3）

- compute capability 8.9 以上の GPU（Ada 以降）が要ります。最初の入力に使う公式のモジュールが、FP8 の計算に Triton のカーネルを使うためです。確かめた GPU は RTX 6000 Ada だけです。
- GPU のメモリーを約 45 GiB 使います。48GB の GPU では残りが 2〜3 GiB です。モデルを 2 枚以上の GPU に分けることはできません。
- 出力は公式の実装と同じ値にはなりません（int8 か bf16 で計算することと、自作の乱数のため）。
- Fish Audio のサンプラー（`reference_sampling`）と `epsilon_cutoff` は使えません。
- 複数の話者の場面（`<|speaker:N|>`）と `[laughs]` のような指示は、fish-speech と同じプロンプトのまま渡りますが、音声の質は確かめていません。
- 参照の行の状態を使い回すのは、1 つの声の分だけです。声を切り替えた最初の要求では、参照の行もすべて流します。
- 読み間違いと声の近さを確かめたのは、日本語の 48 文と、参照なしと 2 つの声だけです。

## リポジトリの構成

| 場所 | 中身 |
| --- | --- |
| `engine/` | エンジン。`engine/serve.py` がサーバーにエンジンを差し込みます |
| `server/` | 音声合成サーバー（公式の実装のまま合成する） |
| `qwen3/` | Qwen3-TTS のエンジン（`kernels.py`）、サーバー（`serve.py`）、計測と確かめのスクリプト |
| `drama3/` | Fish Audio Drama 3 のエンジン（`kernels.py`）、サーバー（`serve.py`）、計測と確かめのスクリプト |
| `docker/Dockerfile` | Irodori-TTS のイメージ（official、dev、server の 3 段） |
| `docker/qwen3.Dockerfile` | Qwen3-TTS のイメージ（qwen3-official、qwen3-dev、qwen3-server の 3 段） |
| `docker/drama3.Dockerfile` | Drama 3 のイメージ（drama3-official、drama3-dev、drama3-server の 3 段） |
| `ref/` | 公式の実装を測る・参照を保存するスクリプト |
| `tools/` | 参照との比較（SNR）、速さの集計、HTTP の計測、ロゴとグラフの画像の書き出し（`make_assets.py`） |
| `scripts/` | イメージの中でスクリプトを動かす（`run_in_image.sh`）、イメージを作る、条件を交互に測る |
| `data/sentences.json` | 速さを測る 48 文と、話す長さ |
| `docs/` | 結果（`results.md`）、実装前の記録（`design.md`、`baseline.md`） |

計測と参照の出力は `runs/` に入ります。参照音声から作った潜在変数と音声を含むので、git の管理外にしてあります。

## ライセンス

sokutou のコードは MIT ライセンスです（[LICENSE](LICENSE)）。`engine/` は Irodori-TTS の推論の一部を書き直したもので、実行するときに Irodori-TTS のモジュールを呼ぶので、LICENSE には Irodori-TTS の MIT ライセンスの表示も入れてあります。`qwen3/` は qwen-tts の推論の一部を書き直したもので、実行するときに qwen-tts のモジュールを呼ぶので、LICENSE には qwen-tts の表示も入れてあります。`drama3/` は Drama 3 の推論の一部を書き直したもので、実行するときに fish-speech のモジュールを呼びます。fish-speech のコードはこのリポジトリに入っていません（LICENSE に注記があります）。

モデルの重みはそれぞれの作者のものです。

- Irodori-TTS（コード）と、Irodori-TTS-v4.1-Small と v4.1-Small-MF（重み）は Aratako 氏のもので、MIT ライセンスで提供されています。重みには、モデルカードにある倫理上の制限がかかります。本人の明示的な同意なしに、人の声を複製したり、その人になりすましたりすることと、人を欺く合成音声や誤情報を作ることは禁じられています。
- Irodori-TTS-v4-Large（重み）は Aratako 氏のもので、[Gemma の利用規約（Gemma Terms of Use）](https://ai.google.dev/gemma/terms)に従います。文と caption のエンコーダーが google/t5gemma-2-1b-1b から作られているためです。使うときと配るときは、[Gemma の禁止された使い方の方針（Gemma Prohibited Use Policy）](https://ai.google.dev/gemma/prohibited_use_policy)と、v4.1-Small と同じ倫理上の制限にも従います。
- コーデックの Semantic-DACVAE-Japanese-32dim は Aratako 氏のもので、MIT ライセンスで提供されています。
- Qwen3-TTS の重み（Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice と 1.7B-CustomVoice）と、公式の実装の qwen-tts は Alibaba の Qwen チームのもので、Apache-2.0 で提供されています。
- Fish Audio Drama 3 の重みとコーデック（fishaudio/drama-3）と、公式の実装の fish-speech は Fish Audio（39 AI, INC.）のもので、[Fish Audio Community License 1.0](https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE) で提供されています。研究と非商用の利用は無料です。商用の利用は、年間の売上が 50 万米ドル未満のあいだだけ認められています。他の人が自分の文や声の録音から音声を作れるサービス（API や、音声合成と声の複製を主な目的とするサイトやアプリ）として提供するには、目的や売上にかかわらず、Fish Audio と別の契約が要ります。自分で作る内容（ナレーション、ゲームやアプリのキャラクター、アシスタントの返答など）の音声を作って届けることは、このサービスに当たりません。本人の同意なしにその人の声をまねることと、出力を使ってほかの音声合成のモデルを作ったり学習したりすることは禁じられています。他の人に音声の生成を使わせるときは、この禁止と同じ制限を守らせる約束が要ります。`scripts/build_drama3.sh` で作るイメージには fish-speech が入るので、イメージを配るときは、ライセンスの写しと Notice のファイルを付けます。Built with Fish Audio.
- 電子透かしの SilentCipher は Sony のもので、MIT ライセンスで提供されています。

公式の実装は合成した音声に SilentCipher の電子透かしを自動で入れます。sokutou は既定で入れません。入れるときは `SOKUTOU_WATERMARK=1` にします。この設定のサーバーの音声から透かしを読み出せるかは、確かめていません（[docs/results.md](docs/results.md) の「ほかに確かめたこと」）。

速さの比較には [speech.cpp](https://github.com/nyosegawa/speech.cpp)（MIT ライセンス）を使い、この README の構成も参考にしました。
