# 結果と再現の手順

sokutou の速さ、公式の実装との突き合わせ、読み間違いと声の近さの結果と、それぞれの測り方です。概要は [README](../README.md) の「精度と速度」の節にあります。Qwen3-TTS と Drama 3 の条件と結果は「[Qwen3-TTS](#qwen3-tts)」と「[Fish Audio Drama 3](#fish-audio-drama-3)」の節に、動かし方はモデルごとのページ（[Qwen3-TTS](models/qwen3-tts.md)、[Drama 3](models/drama3.md)）にあります。

## 計測の条件

- 機械: NVIDIA RTX 6000 Ada（48GB、電力の上限 300 W）1 枚、ドライバー 580.196.07、Ubuntu 22.04.5、Docker 28.2.2
- イメージ: `docker/Dockerfile` から作ったもの（Irodori-TTS の commit 7ed4374、torch 2.10.0 + CUDA 12.8）
- モデル: Aratako/Irodori-TTS-v4.1-Small（revision 2b28324）、MF 版の Aratako/Irodori-TTS-v4.1-Small-MF（revision ccc78f5）、v4-Large の Aratako/Irodori-TTS-v4-Large（revision 2e0c554）。コーデックは Aratako/Semantic-DACVAE-Japanese-32dim
- 文: `data/sentences.json` の 48 文と、文ごとの話す長さ（`seconds`）。シードは 1 と 2
- 声: 1 つ（参照音声は 7.9 秒）
- 合成の設定: RF、independent の CFG（文 3.0、話者 5.0）を時刻 0.5〜1.0（ステップの前半）に、caption なし、`sway_coeff` -1、`context_kv_cache` あり。MF 版は CFG がなく、4 ステップです。公式の実装は、MF 版では CFG と RF 版のサンプラーの設定（`sway_coeff` など）を使いません。v4-Large は v4.1-Small と同じ設定です
- 速さの範囲: 合成から、24kHz への変換と前後の無音の切り落としまで（サーバーの `X-Synthesis-Seconds` と同じ）

## 速さ

48 文 × シード 1, 2 の 96 回を 2 回ずつ測り、合わせた 192 回の値です。どれも電子透かしを入れていません。この節の表は v4.1-Small の値で、MF 版と v4-Large はあとの節にあります。

| 実装 | ステップ | 中央値 | 90 パーセンタイル | 最大 | 15 字以下 | 25 字以上 | 生成 | 復号 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sokutou（BF16） | 20 | 0.059 秒 | 0.074 秒 | 0.101 秒 | 0.045 秒 | 0.066 秒 | 37.5 ミリ秒 | 16.2 ミリ秒 |
| sokutou（BF16） | 40 | 0.098 秒 | 0.117 秒 | 0.141 秒 | 0.073 秒 | 0.107 秒 | 75.2 ミリ秒 | 16.2 ミリ秒 |
| 公式の実装（fp32） | 20 | 0.418 秒 | 0.437 秒 | 0.473 秒 | 0.408 秒 | 0.426 秒 | 391.5 ミリ秒 | 18.2 ミリ秒 |
| 公式の実装（fp32） | 40 | 0.758 秒 | 0.782 秒 | 0.823 秒 | 0.750 秒 | 0.770 秒 | 732.5 ミリ秒 | 18.4 ミリ秒 |
| speech.cpp v0.10.1（Vulkan、F16） | 20 | 0.221 秒 | 0.236 秒 | 0.252 秒 | 0.143 秒 | 0.234 秒 | — | — |
| speech.cpp v0.10.1（Vulkan、F16） | 40 | 0.398 秒 | 0.419 秒 | 0.441 秒 | 0.257 秒 | 0.413 秒 | — | — |

- 「15 字以下」と「25 字以上」は、文の長さ（句点を含む字数）で分けた中央値です。「生成」は DiT のステップと CFG の合成（`sample_rf`）、「復号」はコーデックの復号の中央値です。
- sokutou と公式の実装は `scripts/bench_compare.sh` で、4 つの条件を新しいプロセスで交互に 2 回ずつ測りました。sokutou は起動のときの準備だけを済ませ、48 文を前もって通さずに測っています。
- speech.cpp は `speech serve` を 1 つ立て、同じ本文（`sway_coeff` -1、`steps` 40 か 20）を HTTP で送り、40 ステップと 20 ステップを交互に 2 回ずつ測りました。値は HTTP を含みます。音声は 48kHz のまま返り、無音の切り落としはありません。CER と声の近さは測っていません。
- 20 ステップの sokutou をサーバーとして動かし、HTTP で送ったときの `X-Synthesis-Seconds` の中央値は 0.059 秒、HTTP を含めた時間の中央値は 0.061 秒でした。電子透かしを残す設定（`SOKUTOU_WATERMARK=1`）では、`X-Synthesis-Seconds` の中央値は 0.067 秒でした。
- CFG のあるステップを続けて動かすと、GPU は電力の上限（300 W）に当たり、SM のクロックが約 2700 MHz から 1700〜1850 MHz に下がります。電力とクロックの設定は変えずに測りました。

### MF 版

MF 版は同じ測り方で、モデルの既定の 4 ステップを測りました。

| 実装 | ステップ | 中央値 | 90 パーセンタイル | 最大 | 15 字以下 | 25 字以上 | 生成 | 復号 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sokutou（BF16） | 4 | 0.028 秒 | 0.039 秒 | 0.077 秒 | 0.021 秒 | 0.034 秒 | 8.4 ミリ秒 | 15.8 ミリ秒 |
| 公式の実装（fp32） | 4 | 0.134 秒 | 0.147 秒 | 0.198 秒 | 0.124 秒 | 0.139 秒 | 109.3 ミリ秒 | 17.7 ミリ秒 |
| speech.cpp v0.10.1（Vulkan、F16） | 4 | 0.075 秒 | 0.087 秒 | 0.098 秒 | 0.049 秒 | 0.083 秒 | — | — |

- 「生成」は `sample_meanflow` の中央値です。
- speech.cpp は `speech serve irodori-tts-mf` を立て、本文に設定を足さずに送りました（speech.cpp の説明では、MF 版は CFG などの設定を受け付けません）。値は HTTP を含みます。speech.cpp のログにある合成の時間の中央値は 0.073 秒でした。
- MF 版の sokutou をサーバーとして動かしたときは、`X-Synthesis-Seconds` の中央値が 0.030 秒、HTTP を含めた時間の中央値が 0.031 秒でした。

### v4-Large

v4-Large も同じ測り方で、40 ステップと 20 ステップを測りました。

| 実装 | ステップ | 中央値 | 90 パーセンタイル | 最大 | 15 字以下 | 25 字以上 | 生成 | 復号 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sokutou（BF16） | 20 | 0.198 秒 | 0.219 秒 | 0.274 秒 | 0.137 秒 | 0.212 秒 | 173.8 ミリ秒 | 17.1 ミリ秒 |
| sokutou（BF16） | 40 | 0.376 秒 | 0.422 秒 | 0.518 秒 | 0.262 秒 | 0.392 秒 | 350.2 ミリ秒 | 16.7 ミリ秒 |
| 公式の実装（fp32） | 20 | 1.099 秒 | 1.186 秒 | 1.408 秒 | 0.856 秒 | 1.176 秒 | 1069.5 ミリ秒 | 19.0 ミリ秒 |
| 公式の実装（fp32） | 40 | 2.073 秒 | 2.250 秒 | 2.608 秒 | 1.584 秒 | 2.228 秒 | 2040.5 ミリ秒 | 19.1 ミリ秒 |
| speech.cpp v0.10.1（Vulkan、F16） | 20 | 0.453 秒 | 0.494 秒 | 0.623 秒 | 0.328 秒 | 0.482 秒 | — | — |
| speech.cpp v0.10.1（Vulkan、F16） | 40 | 0.846 秒 | 0.914 秒 | 1.114 秒 | 0.613 秒 | 0.894 秒 | — | — |
| speech.cpp v0.10.1（Vulkan、Q8_0） | 20 | 0.498 秒 | 0.535 秒 | 0.568 秒 | 0.315 秒 | 0.525 秒 | — | — |
| speech.cpp v0.10.1（Vulkan、Q8_0） | 40 | 0.944 秒 | 1.006 秒 | 1.058 秒 | 0.598 秒 | 0.993 秒 | — | — |

- speech.cpp は F16 の重み（`speech serve irodori-tts-large:f16`）と既定の Q8_0 の重み（`speech serve irodori-tts-large`）で順にサーバーを立て、v4.1-Small と同じ本文（`sway_coeff` -1、`steps` 40 か 20）を送りました。声のファイルは v4.1-Small で作ったものをそのまま使いました。
- v4-Large の sokutou をサーバーとして動かしたとき（20 ステップ）は、`X-Synthesis-Seconds` の中央値が 0.196 秒、HTTP を含めた時間の中央値が 0.198 秒でした。

## 起動の準備と GPU のメモリー

エンジンはモデルを読み込んだあとで、サーバーが受け付ける入力（0.5〜30 秒、200 字まで）で出てくる形のグラフをすべて記録します。

| 項目 | v4.1-Small、20 ステップ | v4.1-Small、40 ステップ | MF 版、4 ステップ |
| --- | --- | --- | --- |
| エンジンの準備の合計 | 8.0 秒 | 8.2 秒 | 5.9 秒 |
| うち CUDA Graph の記録（432 通りの形） | 4.4 秒 | 4.4 秒 | 2.4 秒 |
| 記録したグラフの数 | 873 個 | 873 個 | 441 個 |
| うち復号の準備（47 通りの長さ） | 2.4 秒 | 2.4 秒 | 2.3 秒 |
| うち CUDA カーネルのコンパイル | 0.25 秒 | 0.25 秒 | 0.25 秒 |
| 準備のあとの GPU のメモリー（GPU 全体の使用量） | 13.1 GB | 13.1 GB | 11.7 GB |
| 192 回の合成のあとの GPU のメモリー | 13.1 GB | 13.1 GB | 11.7 GB |

- GPU のメモリーは `torch.cuda.mem_get_info` の値（1 GB = 10^9 バイト）です。nvidia-smi では 12,500 MiB 前後になります。
- サーバーの起動から `/health` が 200 を返すまでは約 30 秒で、モデルとコーデックの読み込みを含みます。電子透かしを残す設定では、GPU のメモリーは nvidia-smi で 13,277 MiB でした。
- 192 回のあいだに、グラフを途中で記録した回数、公式の関数に戻った回数、準備していない長さで復号した回数は、どれも 0 でした。

v4-Large では、エンジンの準備の合計が 11.8 秒（20 ステップ）と 12.1 秒（40 ステップ）でした。そのうち、CUDA Graph の記録が 8.1 秒と 8.2 秒（同じ 432 通りの形の 873 個）、復号の準備が 2.4 秒、カーネルのコンパイルが 0.23 秒です。GPU のメモリーは準備の前で 21.3 GB と 21.4 GB、準備のあとで 31.2 GB と 31.3 GB で、192 回の合成のあとも変わりませんでした。サーバー（20 ステップ）では、nvidia-smi で 29,790 MiB で、起動から `/health` が 200 を返すまでは 51 秒でした。

## 公式の実装との突き合わせ

エンジンを fp32 で動かし、公式の実装（GPU、fp32）で保存した参照と、同じ入力で比べました。比べ方は 2 つです。

- ステップごと: 各ステップで、参照の `x_t` と参照の条件をエンジンに入れ、出てきた `v` を比べます。実装の誤りを見つけるための比べ方です。
- 通し: 保存した最初のノイズから最後の音声まで、エンジンだけで通して比べます。誤差の積み重なりを見るための比べ方です。

シード 1 の 6 文の、SNR（dB）の最小です（`tools/snr.py`）。差が小さいほど SNR は大きくなります。目安として、公式の実装を GPU と CPU（どちらも fp32、40 ステップ）で動かしたときの差を並べました。

次の表は v4.1-Small の値です。

| 比べたもの | エンジン（fp32）と公式の実装、40 ステップ | 同じく 20 ステップ | 公式の実装の GPU と CPU、40 ステップ |
| --- | --- | --- | --- |
| ステップごとの `v`（CFG のあるステップ、3 つ分） | 123.0 dB | 123.1 dB | 118.8 dB |
| ステップごとの `v`（CFG のないステップ） | 120.0 dB | 119.9 dB | 116.6 dB |
| ステップごとの `v`（CFG を合成した後） | 109.0 dB | 109.4 dB | 106.8 dB |
| 通しの最後の潜在変数 | 96.7 dB | 88.9 dB | 85.0 dB |
| 通しの 24kHz の音声 | 67.0 dB | 65.9 dB | 62.3 dB |

どの段階でも、エンジンと公式の実装の差は、公式の実装を GPU と CPU で動かしたときの差より小さくなりました。エンジンは潜在を区切りの長さまで 0 で伸ばし、伸ばした位置をマスクで attention から外すので、この突き合わせも伸ばした形で動かしました（`--latent-buckets 64,100,128`）。

サーバーで使う BF16 では、ステップごとの `v` の SNR が 46.5 dB（CFG の 3 つ分）と 43.0 dB（CFG のないステップ）になり、波形ははっきり変わります。そのため BF16 の音声は次の節の読み間違いと声の近さで確かめています。

MF 版も同じように比べました（4 ステップ、シード 1 の 6 文の最小）。MF 版には CFG のステップがありません。

| 比べたもの | エンジン（fp32）と公式の実装 | 公式の実装の GPU と CPU |
| --- | --- | --- |
| ステップごとの `v` | 113.8 dB | 112.2 dB |
| 通しの最後の潜在変数 | 93.0 dB | 83.4 dB |
| 通しの 24kHz の音声 | 66.9 dB | 62.4 dB |

MF 版でも、どの段階の差も、公式の実装を GPU と CPU で動かしたときの差より小さくなりました。BF16 では、ステップごとの `v` の SNR は 39.4 dB でした。

v4-Large も同じように比べました（20 ステップ、シード 1 の 6 文の最小）。目安の、公式の実装を GPU と CPU で動かしたときの差も 20 ステップです。公式の実装を同じ GPU で 2 回動かした出力は、ビット単位で同じでした。

| 比べたもの | エンジン（fp32）と公式の実装 | 公式の実装の GPU と CPU |
| --- | --- | --- |
| ステップごとの `v`（CFG のあるステップ、3 つ分） | 126.0 dB | 119.8 dB |
| ステップごとの `v`（CFG のないステップ） | 123.8 dB | 120.0 dB |
| ステップごとの `v`（CFG を合成した後） | 113.1 dB | 108.5 dB |
| 通しの最後の潜在変数 | 100.7 dB | 82.3 dB |
| 通しの 24kHz の音声 | 67.5 dB | 59.2 dB |

v4-Large でも、どの段階の差も、公式の実装を GPU と CPU で動かしたときの差より小さくなりました。BF16 では、ステップごとの `v` の SNR は、CFG のあるステップで 47.8 dB、CFG のないステップで 46.4 dB、CFG を合成した後で 34.8 dB でした。

## 読み間違いと声の近さ

48 文 × シード 1, 2 の 96 本を、サーバーが返すのと同じ形（24kHz、無音を切り落としたもの）で作り、次の 2 つを出しました。

- 読み間違いの率（CER）: faster-whisper（medium、beam 5）で書き起こし、NFKC で正規化して記号を除き、pykakasi でひらがなに直して、元の文と比べた字の誤りの率です。「十個」と「10個」のような表記の違いも誤りに数えます。
- 声の近さ: 話者照合のモデル microsoft/wavlm-base-plus-sv の埋め込みの、コサイン類似度です。参照音声と重ならない、本物の声の 2 つの区間との近さを平均しました。本物の 2 つの区間どうしの近さは 0.952 です。

| 実装 | ステップ | CER | 誤りのある音声 | 声の近さ（平均） | 声の近さ（最小） |
| --- | --- | --- | --- | --- | --- |
| 公式の実装（fp32、透かしあり）、v4.1-Small | 40 | 3.11% | 24 / 96 | 0.916 | 0.702 |
| 公式の実装（fp32、透かしあり）、v4.1-Small | 20 | 3.11% | 23 / 96 | 0.916 | 0.735 |
| sokutou（BF16、透かしなし）、v4.1-Small | 40 | 2.91% | 22 / 96 | 0.915 | 0.720 |
| sokutou（BF16、透かしなし）、v4.1-Small | 20 | 2.96% | 22 / 96 | 0.915 | 0.735 |
| 公式の実装（fp32、透かしなし）、MF 版 | 4 | 3.57% | 24 / 96 | 0.908 | 0.526 |
| sokutou（BF16、透かしなし）、MF 版 | 4 | 4.03% | 26 / 96 | 0.909 | 0.679 |
| 公式の実装（fp32、透かしなし）、v4-Large | 40 | 3.26% | 23 / 96 | 0.920 | 0.673 |
| 公式の実装（fp32、透かしなし）、v4-Large | 20 | 3.26% | 25 / 96 | 0.919 | 0.684 |
| sokutou（BF16、透かしなし）、v4-Large | 40 | 3.36% | 24 / 96 | 0.920 | 0.670 |
| sokutou（BF16、透かしなし）、v4-Large | 20 | 3.21% | 24 / 96 | 0.919 | 0.687 |
| sokutou（fp32、透かしなし）、v4-Large | 20 | 3.26% | 25 / 96 | 0.919 | 0.684 |

speech.cpp の CER と声の近さは測っていません。確かめた声は 1 つだけです。

v4-Large では、BF16 で誤りの数が公式の実装と変わった音声は、20 ステップで 1 本（1 字減った）、40 ステップで 1 本（2 字増えた）でした。fp32 では、96 本すべてで誤りの数が公式の実装と同じでした。

MF 版ではシード 1〜10 の 480 本でも比べました。

| 実装 | CER | 誤りのある音声 | 声の近さ（平均） | 声の近さ（最小） |
| --- | --- | --- | --- | --- |
| 公式の実装（fp32、透かしなし） | 3.69% | 117 / 480 | 0.910 | 0.526 |
| sokutou（fp32） | 3.69% | 117 / 480 | 0.910 | 0.526 |
| sokutou（BF16） | 4.00% | 123 / 480 | 0.910 | 0.562 |

sokutou を fp32 で動かした音声は、480 本すべてで誤りの数が公式の実装と同じでした。BF16 では 20 本で誤りの数が変わり、13 本で増え、7 本で減りました。1 本あたりの差の平均は +0.062 字で、標準誤差（0.039 字）の 1.6 倍です。

## ほかに確かめたこと

- 公開のものだけで作ったイメージ: 作者が前に非公開のファイルから作ったイメージと、パッケージの版がすべて同じで、48 文 × シード 1, 2 の音声も 96 本すべて PCM16 で同じ値になりました。
- 空のモデルのキャッシュからの起動: 取得したのは Aratako/Irodori-TTS-v4.1-Small（2.9GB）、Aratako/Semantic-DACVAE-Japanese-32dim（410MB）、sony/silentcipher（68MB）の 3 つで、取得を含めて 75 秒で準備ができました。音声は取得済みのキャッシュのときと 96 本すべて同じでした。
- 長さの違う参照音声: 3 秒、15.7 秒（WAV 2 つ）、31.4 秒（WAV 4 つ）で、どれも約 30 秒で起動しました。96 回の合成で公式の関数に戻った回数は 0 で、中央値は 0.058〜0.060 秒でした。声の質は確かめていません。
- 入力の範囲の端: 0.5 秒、30 秒、200 字、256 トークンの入力で、落ちず、公式の関数にも戻りませんでした（`engine/edges.py`）。
- 同時の要求: 32 件を 16 本と 8 本の同時の要求で送ると、音声は 1 件ずつ送ったときと同じで、公式の関数に戻った回数は 0 でした。
- 電子透かしを残す設定（`SOKUTOU_WATERMARK=1`）: 起動し、96 回の合成で公式の関数に戻った回数は 0 でした。サーバーの音声から透かしを読み出せるかは確かめていません。エンジンの BF16 の音声に公式の手順で入れた透かしは、24kHz の 96 本のうち 81〜83 本で読み出せ、公式の実装の音声（82 本）と同じくらいでした（`engine/gen_quality.py`）。
- MF 版のサーバー: 48 文 × シード 1, 2 の音声は、`engine/gen_quality.py` で作った音声と 96 本すべて PCM16 で同じ値でした。起動から `/health` が 200 を返すまでは 28 秒でした。シードを送らない 48 件も含めて、公式の関数に戻った回数は 0 でした。
- MF 版の入力の範囲の端: RF 版と同じ入力で、落ちず、公式の関数にも戻りませんでした（`engine/edges.py`）。
- MF 版に対応したあとの RF 版: fp32 の突き合わせ（20 ステップ）の配列 126 個が、対応の前とビット単位で同じでした。サーバーの音声も前のサーバーの音声と 96 本すべて PCM16 で同じ値でした。
- v4-Large のサーバー（20 ステップ）: 48 文 × シード 1, 2 の音声は `engine/gen_quality.py` で作った音声と 96 本すべて PCM16 で同じ値でした。シードを送らない 48 件も含めて、公式の関数に戻った回数は 0 でした。
- v4-Large の空のモデルのキャッシュからの起動: 取得したのは Aratako/Irodori-TTS-v4-Large（12.3GB）、Aratako/Semantic-DACVAE-Japanese-32dim（410MB）、sony/silentcipher（68MB）の 3 つで、Hugging Face のトークンなしで取得できました。取得を含めて 180 秒で準備ができ、音声は取得済みのキャッシュのときと 96 本すべて同じでした。
- v4-Large の入力の範囲の端: v4.1-Small と同じ 11 件の入力で、落ちず、公式の関数にも戻りませんでした（`engine/edges.py`）。T5Gemma 2 のトークナイザーでは、200 字の文は最大 172 トークンでした。
- v4-Large に対応したあとの v4.1-Small と MF 版: qkv のカーネルを変えたあとで、fp32 の突き合わせの配列がどちらも 126 個とも前とビット単位で同じでした。文のエンコーダーの差し替えを足したあとでは、MF 版の BF16 の音声 480 本が前と同じでした。作り直したイメージの v4.1-Small のサーバーの音声も、前のサーバーの音声と 96 本すべて PCM16 で同じ値でした。

## 既定にしていないもの

どちらも出力が変わるので、使っていません。ベンチマークのスクリプト（`ref/bench_engine.py`）で選べます。

- DiT の行列積を FP8（e4m3）にする（`--fp8`）: 20 ステップの中央値が 0.058 秒から 0.047 秒になります。fp32 の参照との `v` の SNR は BF16 の 46.5 / 43.0 dB に対して 25.7 / 22.2 dB です。`torch._scaled_mm`（公開されていない API）を使います。
- 復号を float16 にする（`--half-decode`）: 90 フレームの復号が 13.9 ミリ秒から 9.3 ミリ秒になります（fp32 との SNR は 59.5 dB）。

## Qwen3-TTS

### 条件

- モデル: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice（revision 85e237c）と Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice（revision 0c0e305）。コーデックはモデルのリポジトリにあるもの（speech_tokenizer）
- イメージ: `docker/qwen3.Dockerfile` から作ったもの（qwen-tts 0.1.1、transformers 4.57.3、torch 2.10.0 + CUDA 12.8）
- 公式の実装: qwen-tts の `generate_custom_voice` をそのまま呼びます。bf16 で、attention は PyTorch の sdpa です（公式がすすめる flash_attention_2 は入れていません）
- 話者は Ono_Anna、言語は Japanese です。instruct は使いません
- サンプリング: モデルの既定です（言語モデルは温度 0.9、top_k 50、repetition_penalty 1.05、min_new_tokens 2。code predictor は温度 0.9、top_k 50）
- 文とシード: Irodori-TTS と同じ 48 文と、シード 1 と 2。話す長さ（`seconds`）は使いません
- 速さの範囲: 文を受け取ってから 24kHz の波形ができるまで（最初の入力、生成、復号）。サーバーの `X-Synthesis-Seconds` と同じ範囲です

### 速さ

48 文 × シード 1, 2 の 96 回を 2 回ずつ測った、192 回の値です（サーバーは 1 回の 96 回）。2 回の差は 1% 以内でした。このモデルでは話す長さを指定できず、長さはサンプリングで決まります。音声の長さの中央値が実装ごとに違う（3.5〜4.8 秒）ので、「音声 1 秒あたり」の時間でも比べます。

| 実装（0.6B） | 中央値 | 90 パーセンタイル | 最大 | 15 字以下 | 25 字以上 | 音声の長さの中央値 | 音声 1 秒あたり | 1 フレーム |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sokutou（int8） | 0.176 秒 | 0.253 秒 | 0.354 秒 | 0.141 秒 | 0.223 秒 | 4.52 秒 | 0.0391 秒 | 2.94 ミリ秒 |
| sokutou（bf16） | 0.270 秒 | 0.360 秒 | 0.516 秒 | 0.180 秒 | 0.300 秒 | 4.84 秒 | 0.0561 秒 | 4.29 ミリ秒 |
| sokutou のサーバー（int8） | 0.177 秒 | 0.254 秒 | 0.354 秒 | 0.141 秒 | 0.224 秒 | 4.52 秒 | 0.0392 秒 | — |
| 公式の実装（bf16） | 7.066 秒 | 9.405 秒 | 18.142 秒 | 4.916 秒 | 8.145 秒 | 4.60 秒 | 1.5258 秒 | 121.1 ミリ秒 |
| speech.cpp v0.10.1（Vulkan、Q8_0） | 0.341 秒 | 0.481 秒 | 0.727 秒 | 0.237 秒 | 0.421 秒 | 4.24 秒 | 0.0805 秒 | — |

| 実装（1.7B） | 中央値 | 90 パーセンタイル | 最大 | 15 字以下 | 25 字以上 | 音声の長さの中央値 | 音声 1 秒あたり | 1 フレーム |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sokutou（int8） | 0.207 秒 | 0.281 秒 | 1.066 秒 | 0.135 秒 | 0.240 秒 | 3.60 秒 | 0.0577 秒 | 4.40 ミリ秒 |
| sokutou（bf16） | 0.312 秒 | 0.439 秒 | 1.056 秒 | 0.189 秒 | 0.356 秒 | 3.60 秒 | 0.0868 秒 | 6.72 ミリ秒 |
| sokutou のサーバー（int8） | 0.206 秒 | 0.278 秒 | 0.330 秒 | 0.133 秒 | 0.239 秒 | 3.60 秒 | 0.0575 秒 | — |
| 公式の実装（bf16） | 5.476 秒 | 7.794 秒 | 13.922 秒 | 3.551 秒 | 6.894 秒 | 3.52 秒 | 1.5492 秒 | 122.8 ミリ秒 |
| speech.cpp v0.10.1（Vulkan、Q8_0） | 0.360 秒 | 0.533 秒 | 2.434 秒 | 0.256 秒 | 0.406 秒 | 3.72 秒 | 0.0963 秒 | — |

- 「音声 1 秒あたり」は合成の時間の合計を音声の長さの合計で割った値です。「1 フレーム」は生成の時間をフレーム数で割った値の中央値です。
- 内訳の中央値は sokutou の int8 の 0.6B で、最初の入力が 3.3 ミリ秒、生成が 165.7 ミリ秒、復号が 7.4 ミリ秒でした。公式の実装の 0.6B は、生成が 7011.4 ミリ秒、復号が 30.3 ミリ秒でした。
- サーバーの行は `X-Synthesis-Seconds` の値です。HTTP を含めた時間は、1 文の中央値が 0.181 秒（0.6B）と 0.209 秒（1.7B）、音声 1 秒あたりが 0.0400 秒と 0.0584 秒でした。1.7B のサーバーの 1 件は 500 を返したので、95 件の値です。
- speech.cpp は `speech serve qwen3-tts-0.6b`（1.7B は `qwen3-tts-1.7b`）を立て、`voice` に `ono_anna` を、本文に `{"language": "ja"}` を足して送りました。値は HTTP を含みます。speech.cpp は既定の Q8_0 の重みだけで測りました。sokutou の int8 は Q8_0 と同じく 32 個ごとに尺度を持ちます。
- 2 回の計測のあいだ、GPU 0 にほかのプロセスはありませんでした。

### 起動の準備と GPU のメモリー

- サーバー（int8）の準備の合計は、0.6B が 23.4 秒、1.7B が 23.9 秒でした。そのうち、最初の入力の CUDA Graph の記録（6〜160 トークン）が 17.2 秒と 17.3 秒です。起動から `/health` が 200 を返すまでは、サーバーのイメージで 29〜30 秒でした（0.6B の int8 と bf16、1.7B の int8）。
- GPU 全体の使用量（`torch.cuda.mem_get_info`）は、サーバーで 0.6B が 8.5 GB（bf16 は 8.6 GB）、1.7B が 11.7 GB でした。
- サーバーの 96 回の合成のあいだに、CUDA Graph を途中で記録した回数は 0 でした。

### 公式の実装との突き合わせ

Qwen3-TTS はサンプリングで話すので、波形の SNR ではなく、生成したコードと logits で比べました。

- float32 のコード: エンジンを GPU の float32 で動かし、公式の実装と同じ文・同じシードで合成すると、フレームごとの 16 個のコードが、すべてのフレームで同じになりました（0.6B と 1.7B、6 文 × シード 1, 2 ずつ）。このときのサンプリングには torch の演算の経路を使います（自作のカーネルの乱数は公式と違うため）。
- 教師強制: 公式の実装が出したコードをエンジンに入れ、各ステップの logits を、公式の実装を float32 で動かした基準と比べました（6 文。0.6B は言語モデルが 270 ステップ、code predictor が 3960 回、1.7B は 212 ステップと 3090 回）。次の表の値は温度 0.9 の分布の KL の平均です。

| 実装 | 0.6B、言語モデル | 0.6B、code predictor | 1.7B、言語モデル | 1.7B、code predictor |
| --- | --- | --- | --- | --- |
| 公式の実装（bf16） | 8.7e-4 | 2.3e-3 | 6.1e-4 | 1.8e-3 |
| sokutou（bf16） | 8.2e-5 | 4.2e-4 | 8.7e-5 | 3.3e-4 |
| sokutou（int8） | 4.4e-4 | 1.9e-3 | 3.3e-4 | 1.8e-3 |

sokutou の bf16 は公式の実装の bf16 より基準に近くなりました。int8 は言語モデルでは公式の bf16 より近く、code predictor では同じくらいでした。sokutou を float32 で動かすと、logits の基準との相対差は 6.3e-7 以下でした。

### サンプリングのカーネル

`qwen3/check_sampler.py` で、torch の演算の経路と比べました。

- いちばん確率の高い番号を選ぶ設定で、2000 回すべて torch と同じ番号になりました。繰り返しの罰、抑える番号、min_new_tokens の EOS をかけるところから、そのあとの記録（出た番号、作った数、位置）を書き換えるところまでを比べています。
- 引く設定（温度 0.9、top_k 50）で 20 万回引いた頻度と、softmax(top-k) との全変動距離は 0.004〜0.006 で、torch で同じ回数引いたときと同じくらいでした。top-k の外の番号は引きませんでした。
- 言語モデルの経路で 5 万回引くと、作ったトークンが 5 個のとき、全変動距離はカーネルが 0.0064、torch が 0.0091 で、EOS の頻度は 0.3022 と 0.3054 でした（期待 0.3019）。作ったトークンが 1 個のとき（min_new_tokens の 2 に満たない）は、EOS を 1 回も引きませんでした。抑える番号はどちらも 1 回も引きませんでした。
- 1 フレームの 16 回のサンプリングは、torch の演算で 0.59 ミリ秒、カーネルで 0.07 ミリ秒でした。

### 読み間違いと声の近さ

48 文 × シード 1, 2 の 96 本の値です。音声はエンジンを直接呼んで作りました。書き起こしと CER の数え方は Irodori-TTS と同じです。CustomVoice には参照音声がないので、声の近さには公式の実装の音声の埋め込みの平均（重心）とのコサイン類似度を使いました。シード 1 の音声はシード 2 の公式の音声の重心と、シード 2 の音声はシード 1 の重心と比べたので、公式の実装の値がこの測り方での上限の目安になります。

| 実装 | CER | 誤りのある音声 | CER が 30% を超える音声 | 悪い 2 本を除いた CER | 声の近さ（平均） | 声の近さ（最小） |
| --- | --- | --- | --- | --- | --- | --- |
| 公式の実装（bf16）、0.6B | 5.71% | 31 / 96 | 3 | 4.34% | 0.956 | 0.781 |
| speech.cpp（Q8_0）、0.6B | 5.66% | 33 / 96 | 3 | 4.92% | 0.948 | 0.770 |
| sokutou（bf16）、0.6B | 5.56% | 37 / 96 | 3 | 4.63% | 0.958 | 0.809 |
| sokutou（int8）、0.6B | 7.03% | 38 / 96 | 5 | 5.66% | 0.953 | 0.818 |
| 公式の実装（bf16）、1.7B | 5.96% | 36 / 96 | 5 | 5.09% | 0.945 | 0.702 |
| speech.cpp（Q8_0）、1.7B | 19.88% | 44 / 96 | 8 | 9.73% | 0.944 | 0.589 |
| sokutou（bf16）、1.7B | 5.76% | 24 / 96 | 4 | 3.60% | 0.940 | 0.613 |
| sokutou（int8）、1.7B | 8.56% | 29 / 96 | 3 | 3.81% | 0.944 | 0.671 |

- サンプリングで話すので、同じ文の音声も実装とシードで変わります。話し終えずに続く音声が数本あると、96 本の CER は数 % 動きます。「悪い 2 本を除いた CER」は、CER がいちばん高い 2 本を除いた値です。
- この 96 本には、サーバーなら 500 を返す長さ（2.0 + 0.4 × 字数 秒）の音声も入っています。0.6B と 1.7B の順に、公式の実装は 1 本ずつ、speech.cpp は 0 本と 3 本、sokutou の int8 は 0 本と 1 本、bf16 は 2 本ずつでした。
- speech.cpp の 1.7B にも 0.6B と同じ話者と言語を送っていたことを、計測のスクリプトとサーバーのログで確かめました。

0.6B ではシード 1〜6 の 288 本でも比べました。sokutou の 2 つは、サンプリングの処理だけが違います。

| 実装（0.6B、288 本） | CER | 誤りのある音声 | CER が 30% を超える音声 | 悪い 2 本を除いた CER |
| --- | --- | --- | --- | --- |
| 公式の実装（bf16） | 6.95% | 95 / 288 | 9 | 4.98% |
| sokutou（int8、torch の演算でサンプリング） | 7.41% | 103 / 288 | 13 | 5.75% |
| sokutou（int8、1 つのカーネルでサンプリング） | 5.23% | 98 / 288 | 9 | 4.74% |

sokutou の 2 つでは、フレームの合計は 1 つのカーネルのほうが 2.3% 少なくなりました。並べ替え検定の p は約 0.2 で、偶然の差と見分けられません。

### ほかに確かめたこと

- サーバーの音声: 0.6B の 96 本と 1.7B の 95 本が、エンジンを直接呼んだときと PCM16 で同じ値でした。1.7B の残る 1 本（13 字の文で 18.5 秒まで続く音声）には、91 フレームで生成を止めて 500 を返しました。
- サーバーのイメージ: 0.6B の 48 文 × シード 1 の音声が、上のサーバーの音声と同じでした。空のモデルのキャッシュからも起動しました。取得したのは Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice（2.3GB）だけで、取得を含めて 74 秒で準備ができ、音声は取得済みのキャッシュのときと同じでした。
- 受け付けない入力: 空の文、201 字の文、161 トークンの文（150 字のかな）、wav 以外の形式、公式の実装が受け付けない言語の名前には 400 を、知らない話者には 404 を返しました。1 字の文は合成できました。
- シード: 同じシードでは同じ音声に、シードを送らなければ毎回違う音声になりました。
- bf16 のサーバーと 1.7B のサーバーのイメージ: 48 文 × シード 1 で、0.6B の bf16 は 47 件がエンジンを直接呼んだときと同じ音声で、残る 1 件（話し終えずに続いた音声）には 500 を返しました。1.7B の int8 は 48 件が上のサーバーの音声と同じでした。
- 同時の要求: 0.6B の 48 件を 16 本と 8 本の同時の要求で送ると、音声は 1 件ずつ送ったときと同じでした。前の合成を待つ時間も入るので、`X-Synthesis-Seconds` の最大は 3.29 秒と 1.74 秒でした。
- 確かめていないこと: 1.7B の空のキャッシュからの起動、Ono_Anna と日本語のほかの話者と言語の質、RTX 6000 Ada のほかの GPU。

## Fish Audio Drama 3

### 条件

- モデル: fishaudio/drama-3（revision 78e5ab7）。コーデックはモデルのリポジトリにある codec.pth
- イメージ: `docker/drama3.Dockerfile` から作ったもの（fish-speech 0f8a12e、torch 2.8.0 + CUDA 12.8、transformers 5.19.0）
- 公式の実装: fish-speech の `init_model`（bf16、エキスパートは FP8）と `load_codec_model`（bf16）で読み込み、`generate_long_drama3` で生成し、区切りごとに `from_indices` で復号します。fish-speech のサーバーと同じ順です
- サンプリング: 温度 1.0、top_p 0.9、top_k 30（モデルカードと speech.cpp の既定。fish-speech の既定は 0.7、0.7、30 なので、明示して渡しました）。繰り返しを引き直す RAS は fish-speech の既定のままです
- 参照あり: drama-3 が参照なしで作った 5.8 秒の声（44.1kHz）と、その書き起こし。人の声（Irodori-TTS と同じ 7.9 秒の参照音声、24kHz。本人の同意を得た声）でも測りました。参照の WAV は 44.1kHz に変換してから符号化します
- 文とシード: Irodori-TTS と同じ 48 文と、シード 1 と 2。話す長さ（`seconds`）は使いません
- 速さの範囲: 文を受け取ってから 44.1kHz の波形ができるまで（最初の入力、生成、復号）。サーバーの `X-Synthesis-Seconds` と同じ範囲です
- 公式の出力に電子透かしはありません（fish-speech に透かしのコードはありません）

### 速さ

48 文 × シード 1, 2 の 96 回の値です。「HTTP の往復」はサーバーと speech.cpp に送ってから WAV を受け取り終えるまでの時間です。

| 実装（参照なし） | 中央値 | 90 パーセンタイル | 最大 | 音声の長さの中央値 | 音声 1 秒あたり | 1 フレーム | 最初の入力 | 復号 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sokutou（int8） | 0.620 秒 | 1.170 秒 | 1.368 秒 | 3.60 秒 | 0.191 秒 | 6.74 ミリ秒 | 36 ミリ秒 | 39 ミリ秒 |
| sokutou（bf16） | 0.806 秒 | 1.042 秒 | 2.567 秒 | 3.44 秒 | 0.233 秒 | 8.30 ミリ秒 | 150 ミリ秒 | 45 ミリ秒 |
| sokutou のサーバー（int8、HTTP の往復） | 0.628 秒 | — | — | 3.60 秒 | 0.192 秒 | — | — | — |
| 公式の実装 | 1.643 秒 | 2.146 秒 | 2.682 秒 | 3.48 秒 | 0.471 秒 | 19.59 ミリ秒 | 120 ミリ秒 | 40 ミリ秒 |
| speech.cpp v0.10.1（Q4_K、HTTP の往復） | 1.026 秒 | 1.389 秒 | 6.402 秒 | 3.34 秒 | 0.306 秒 | — | — | — |
| speech.cpp v0.10.1（Q8_0、HTTP の往復） | 1.188 秒 | 1.715 秒 | 9.755 秒 | 3.39 秒 | 0.348 秒 | — | — | — |

| 実装（参照あり） | 中央値 | 90 パーセンタイル | 最大 | 音声の長さの中央値 | 音声 1 秒あたり | 1 フレーム | 最初の入力 | 復号 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sokutou（int8） | 0.641 秒 | 1.057 秒 | 1.541 秒 | 3.69 秒 | 0.194 秒 | 7.04 ミリ秒 | 32 ミリ秒 | 38 ミリ秒 |
| sokutou（bf16） | 0.951 秒 | 1.156 秒 | 1.417 秒 | 3.67 秒 | 0.263 秒 | 8.53 ミリ秒 | 248 ミリ秒 | 39 ミリ秒 |
| sokutou のサーバー（int8、HTTP の往復） | 0.648 秒 | — | — | 3.69 秒 | 0.195 秒 | — | — | — |
| 公式の実装 | 1.846 秒 | 2.270 秒 | 2.852 秒 | 3.69 秒 | 0.502 秒 | 19.58 ミリ秒 | 250 ミリ秒 | 35 ミリ秒 |
| speech.cpp v0.10.1（Q4_K、HTTP の往復） | 1.184 秒 | 1.573 秒 | 4.989 秒 | 3.88 秒 | 0.306 秒 | — | — | — |
| speech.cpp v0.10.1（Q8_0、HTTP の往復） | 1.275 秒 | 1.620 秒 | 2.258 秒 | 3.69 秒 | 0.345 秒 | — | — | — |

- 「1 フレーム」は生成の時間から最初の入力の時間を引き、フレームの数で割った値です。
- sokutou（int8）は、最初の入力の CUDA Graph（24 個まで）と参照の使い回しを使った値です。初めて出た長さの区切りは CUDA Graph にせずに流し、2 回目に出たときに記録します。96 回のうち 13 回で記録しました。シード 2 だけの中央値は、参照なしで 0.584 秒、参照ありで 0.611 秒です。
- sokutou（bf16）は最初の入力を公式のモジュールのまま（CUDA Graph にせず）流した値です。
- 公式の実装はエキスパートを FP8 のまま Triton のカーネルで計算します。1 ステップは CUDA Graph にしてあります。
- speech.cpp は `speech serve drama-3`（Q4_K、既定）と `drama-3:q8_0` で立て、本文に `"experts_on_cpu": false` を足して測りました。参照ありでは、同じ参照音声から `speech voice` で作った声のファイルを使いました。speech.cpp は参照音声を符号化する前に、音量を -16 LUFS にそろえます（公式の実装と sokutou は音量を変えません）。
- 長い文（48 文をつないだ 178〜196 字の 3 文を、参照なしと参照ありで 1 回ずつ）もサーバーに送りました。6 回とも 200 を返し、合成に 5.0〜9.9 秒かかりました（音声は 28〜36 秒）。
- 参照なしでシード 3〜8 の 288 回も測りました。中央値は公式の実装 1.616 秒、bf16 0.763 秒、int8 0.570 秒で、音声 1 秒あたりは 0.465 秒、0.222 秒、0.173 秒でした。
- 人の声（7.9 秒の参照音声）で測った値は次の表です。

| 実装（人の声） | 中央値 | 90 パーセンタイル | 最大 | 音声の長さの中央値 | 音声 1 秒あたり | 1 フレーム | 最初の入力 | 復号 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sokutou（int8） | 0.572 秒 | 0.961 秒 | 1.376 秒 | 3.11 秒 | 0.203 秒 | 7.17 ミリ秒 | 32 ミリ秒 | 30 ミリ秒 |
| sokutou（bf16） | 0.866 秒 | 1.067 秒 | 1.990 秒 | 3.11 秒 | 0.285 秒 | 8.73 ミリ秒 | 250 ミリ秒 | 39 ミリ秒 |
| sokutou のサーバー（int8、HTTP の往復） | 0.573 秒 | — | — | 3.11 秒 | 0.204 秒 | — | — | — |
| 公式の実装 | 1.585 秒 | 2.022 秒 | 3.161 秒 | 3.09 秒 | 0.518 秒 | 19.59 ミリ秒 | 250 ミリ秒 | 39 ミリ秒 |
| speech.cpp v0.10.1（Q4_K、HTTP の往復） | 0.914 秒 | 1.248 秒 | 1.708 秒 | 2.97 秒 | 0.310 秒 | — | — | — |
| speech.cpp v0.10.1（Q8_0、HTTP の往復） | 1.102 秒 | 1.479 秒 | 1.836 秒 | 3.13 秒 | 0.354 秒 | — | — | — |

### 起動の準備と GPU のメモリー

サーバー（int8）の準備には 31 秒かかりました（`GET /v1/engine` の記録）。内訳は、読み込みが 24 秒、声の符号化が 0.6 秒、1 フレームの CUDA Graph の記録が 0.2 秒で、残りは 2 文を試しに合成した時間です。[docs/models/drama3.md](models/drama3.md) の `docker run` の形で起動したときは、コンテナを起動してから `/health` が 200 を返すまで 39 秒でした（準備は 29.5 秒）。GPU のメモリーは、起動のあとで 41.7 GiB、48 文 × 2 を合成したあとで 44.2 GiB、長い文を合成したあとで 45.7 GiB（nvidia-smi で 46,832 / 49,140 MiB）でした。speech.cpp は、Q4_K で約 22 GB、Q8_0 で約 39 GB でした。

### 公式の実装との突き合わせ

35B のモデルを float32 にすると 140GB になるので、Qwen3-TTS のように CPU の float32 でコードの一致を見ることはしていません。代わりに、`drama3/check.py` で次の 4 つを確かめました。参照は、ベンチとサーバーと同じく WAV から符号化しました。

- 層ごと（`layers`）: 参照ありの 193 行のプロンプトを公式の実装で流し、層ごとの入力をエンジンのカーネルに入れて、出力を比べました。基準は公式の層を float32 にしたものです。float32 のカーネルは、言語モデルの 40 層で SNR が最小 113 dB、tower の 13 層で 124〜139 dB でした（生成のときに通る音声の行で 124〜129 dB）。bf16 のカーネルは最小 50 dB で、公式の bf16 の層は最小 15 dB です。
- 教師強制（`teacher`）: 公式の実装が作ったフレームをエンジンに入れ、1 フレームごとのロジットを公式の eager の計算と比べました（4 文、34〜80 フレーム）。bf16 のエンジンと公式の差は、第 0 のコードのロジットの SNR で 29〜35 dB（KL の平均 0.003〜0.009）、第 1〜9 のコードで 34〜48 dB でした。公式の残差を float32 にした設定と公式の差は 24〜35 dB なので、エンジンの差は公式の精度による揺れと同じ大きさです。int8 は、第 0 のコードで 29〜36 dB、第 1〜9 のコードで 33〜46 dB でした。参照ありの bf16 は第 0 のコードで 31〜32 dB でした（残差を float32 にした公式は 27〜28 dB）。
- サンプリング（`sampler`）: 決まったロジットで 4 万回ずつ引き、公式の規則（top_k、top_p、最上位は必ず残す、RAS の引き直し、eos は引き直さない）から求めた確率とカイ二乗検定で比べました。10 通りの p 値は 0.13〜1.00 で、残らないはずの番号は 1 回も出ませんでした。
- 最初の入力（`prefill`）: 区切りごとの CUDA Graph は、eager とビット単位で同じ値を返しました（グラフと eager が混ざる区切り方でも同じ）。参照の使い回しでは区切り方が変わるので、隠れ状態は公式と少し違います（SNR 22〜23 dB）。公式の区切りを 128 行から 64 行にしたときの公式どうしの差は 21〜22 dB です。

サーバーの音声は、同じシードの sokutou（int8）の音声と、参照なし・参照あり・人の声の 96 本ずつすべてでビット単位で同じでした（人の声は、その声だけを置いたサーバーで確かめました。理由は「ほかに確かめたこと」）。

### 読み間違いと声の近さ

Irodori-TTS と同じく、faster-whisper（medium）で書き起こし、ひらがなに直して CER を出しました。声の近さは microsoft/wavlm-base-plus-sv の埋め込みのコサイン類似度で、参照音声との近さと、公式の実装の参照ありの音声の重心との近さの 2 つです。重心との近さは、シード 1 の音声をシード 2 の公式の音声の重心と、シード 2 の音声をシード 1 の重心と比べました。

| 実装（参照なし） | CER | 誤りのある音声 | CER が 30% を超える音声 | 悪い 2 本を除いた CER |
| --- | --- | --- | --- | --- |
| 公式の実装 | 2.04% | 13 | 2 | 1.75% |
| sokutou（bf16） | 3.77% | 21 | 4 | 2.58% |
| sokutou（int8） | 2.34% | 17 | 2 | 2.06% |
| speech.cpp（Q4_K） | 3.52% | 19 | 2 | 2.38% |
| speech.cpp（Q8_0） | 3.57% | 20 | 3 | 2.48% |

| 実装（参照あり） | CER | 誤りのある音声 | CER が 30% を超える音声 | 悪い 2 本を除いた CER | 参照との近さ | 公式の重心との近さ |
| --- | --- | --- | --- | --- | --- | --- |
| 公式の実装 | 2.40% | 19 | 2 | 2.11% | 0.967 | 0.977 |
| sokutou（bf16） | 2.34% | 18 | 2 | 2.06% | 0.967 | 0.978 |
| sokutou（int8） | 2.24% | 17 | 2 | 1.95% | 0.967 | 0.978 |
| speech.cpp（Q4_K） | 5.25% | 19 | 4 | 2.49% | 0.958 | 0.960 |
| speech.cpp（Q8_0） | 2.45% | 16 | 2 | 1.97% | 0.967 | 0.974 |

- サーバーの音声は sokutou（int8）と同じ波形なので、CER も同じです。
- 参照なしのシード 3〜8 の 288 本では、CER は公式の実装 2.87%、bf16 1.95%、int8 2.50% でした。シード 1〜8 の 384 本を合わせると、公式の実装 2.66%、bf16 2.41%、int8 2.46% で、CER が 30% を超える音声は 7 本、6 本、6 本です。bf16 のシード 1 の外れ値 2 本と、公式の実装のシード 7 の外れ値（CER 90%、47.6 秒）は、どれも文頭や途中で話し終えずに続いた音声です。
- サーバーなら 500 を返す長さ（2.0 + 0.4 × 字数 秒）を超えた音声は、参照なしのシード 1〜8 と参照ありのシード 1, 2 の 480 本で、int8 が 0 本、bf16 が 2 本、公式の実装が 1 本でした。speech.cpp はシード 1, 2 の 192 本で、Q4_K が 3 本、Q8_0 が 1 本でした。
- 長い文（178〜196 字の 3 文、シード 1）のサーバーの音声の CER は、参照なしで 1.80%、参照ありで 2.16% でした。

| 実装（人の声） | CER | 誤りのある音声 | CER が 30% を超える音声 | 悪い 2 本を除いた CER | 参照との近さ | 公式の重心との近さ |
| --- | --- | --- | --- | --- | --- | --- |
| 公式の実装 | 2.34% | 17 | 2 | 2.06% | 0.891 | 0.951 |
| sokutou（bf16） | 2.14% | 15 | 2 | 1.85% | 0.891 | 0.950 |
| sokutou（int8） | 2.19% | 16 | 2 | 1.90% | 0.892 | 0.954 |
| speech.cpp（Q4_K） | 2.65% | 22 | 1 | 2.18% | 0.903 | 0.929 |
| speech.cpp（Q8_0） | 2.14% | 16 | 1 | 1.66% | 0.907 | 0.934 |

- 人の声では、speech.cpp の音声が参照音声に近く、公式の実装の音声の重心から遠くなりました。speech.cpp は参照音声を符号化する前に音量を -16 LUFS にそろえるので、モデルに渡る参照の符号が公式の実装と違います。
- 500 を返す長さを超えた音声は、人の声ではどの実装も 0 本でした。

### ほかに確かめたこと

- 参照の符号化の順序: 参照音声の符号は、プロセスで最初に符号化したときだけ、2 回目以降と少し違います（人の声の 1,700 個の符号のうち 289 個）。コーデックの活性化関数（descript-audio-codec の `snake`）が TorchScript で書かれていて、最初の呼び出しと 2 回目以降で計算が変わるためです。`PYTORCH_JIT=0` にすると、順序で変わらなくなりました。サーバーは起動のときに声を名前順に符号化するので、名前順で最初の声だけが `drama3/bench.py --ref` と同じ符号になります。参照ありの声と人の声を両方置いたサーバーでは、2 つめの人の声の音声が sokutou（int8）と 96 本とも違いました（CER 2.24%、参照との近さ 0.889 で、sokutou（int8）と同じくらい）。公式の実装も sokutou も、計測では参照を最初の呼び出しで符号化しました。
- サーバーの長さの上限: 区切りをまたいだフレームの合計が上限を超えたら、生成を止めて 500 を返します。上限を小さくしたサーバーで、短い文（上限 0.3 秒）と長い文（196 字、上限 15 秒）がどちらも 500 になることを確かめました。長い文は、1 つめの区切りの音声が 13.4 秒で上限より短く、2 つめの区切りで合計が上限を超えました。

## 再現の手順

すべて、このリポジトリの直下で動かします。参照と計測の出力は `runs/` に入ります（git の管理外）。

`scripts/run_in_image.sh` は、リポジトリを `/work`、`voices/`（環境変数 `VOICES` で変えられます）を `/voices`、Hugging Face のキャッシュ（`HF_CACHE`、既定 `~/.cache/huggingface`）を読み取り専用でつなぎ、`GPU`（既定 0）の GPU でスクリプトを動かします。その GPU で動いているプロセスがあると止まります。

スクリプトはモデルを取得しないので、先にキャッシュに入れておきます。README の手順でサーバーを起動したなら、`HF_CACHE=sokutou-hf` でそのボリュームを使えます。声の近さに使う microsoft/wavlm-base-plus-sv は、次のコマンドで同じボリュームに取得します。

```sh
docker run --rm -v sokutou-hf:/root/.cache/huggingface -e HF_HUB_OFFLINE=0 \
  --entrypoint /opt/irodori/.venv/bin/python sokutou/official:7ed4374 \
  -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/wavlm-base-plus-sv')"
export HF_CACHE=sokutou-hf
```

この段取りは空のボリュームから試しました（公式の実装で 1 文を合成し、話者照合のモデルを読み込むまで）。下のコマンドは作者のマシンのキャッシュで動かしたもので、この段取りのあとでは動かしていません。

### イメージ

```sh
scripts/build_voice_engine.sh   # sokutou/official:7ed4374、sokutou/dev:7ed4374、sokutou/server:<rev>
```

### 速さ（sokutou と公式の実装）

```sh
scripts/bench_compare.sh bench/share 2 \
  "base:--engine official --steps 40" "base:--engine official --steps 20" \
  "cc:--engine fused --bucket-decode --steps 40" "cc:--engine fused --bucket-decode --steps 20"
```

`base` は `sokutou/official`、`cc` は `sokutou/dev` で動かし、どの条件にも `--precision bf16 --no-watermark` を付けます（公式の実装は fp32 のまま動きます）。最後に `tools/bench_summary.py` が表を出します。

### 速さ（サーバー）

サーバーを README の手順で起動してから（ポートは 18088 などにします）、48 文 × シード 1, 2 を送ります。

```sh
python3 tools/bench_http.py --url http://127.0.0.1:18088 --name http/sokutou-s20 --save-dir bench/http/wav
python3 tools/bench_summary.py runs/bench/http/sokutou-s20.jsonl
```

`tools/bench_http.py` は、:8088 のサーバーには `--allow-production` を付けたときだけ送ります（使っているサーバーを誤って測らないため）。

### 速さ（speech.cpp）

[speech.cpp のリリース](https://github.com/nyosegawa/speech.cpp/releases)の v0.10.1 の Vulkan 版を入れ、参照音声から声のファイルを作り、`speech serve` を立てて送ります。Vulkan のデバイスの番号は `speech devices` で確かめます（nvidia-smi の番号と違うことがあります）。

```sh
speech voice irodori-tts voices/<声の名前>/ref.wav me.voice.gguf
speech serve irodori-tts --device Vulkan0 --port 18092 --add-voice me=me.voice.gguf &
python3 tools/bench_http.py --url http://127.0.0.1:18092 --name speechcpp-s40 --voice me --extra '{"sway_coeff": -1, "steps": 40}'
python3 tools/bench_http.py --url http://127.0.0.1:18092 --name speechcpp-s20 --voice me --extra '{"sway_coeff": -1, "steps": 20}'
```

### 公式の実装との突き合わせ

```sh
scripts/run_in_image.sh ref/dump_reference.py                            # 公式の実装の参照、40 ステップ（runs/ref/gpu/）
scripts/run_in_image.sh ref/dump_reference.py --steps 20 --tag gpu-s20   # 20 ステップ（runs/ref/gpu-s20/）
IMAGE=sokutou/dev:7ed4374 scripts/run_in_image.sh engine/check.py --engine fused --precision fp32 \
  --latent-buckets 64,100,128 --steps 40 --name fused-fp32-pad-s40
python3 tools/snr.py runs/ref/gpu runs/cand/fused-fp32-pad-s40
# 目安: 公式の実装を CPU で動かして、GPU の参照と比べる（40 ステップ）
NO_GPU=1 DOCKER_EXTRA="--cpus 8" scripts/run_in_image.sh ref/run_candidate.py --device cpu --name official-cpu --threads 8
python3 tools/snr.py runs/ref/gpu runs/cand/official-cpu
```

`engine/check.py` の参照はステップ数に合わせて選ばれます（40 なら `runs/ref/gpu`、20 なら `runs/ref/gpu-s20`）。

### 読み間違いと声の近さ

```sh
scripts/run_in_image.sh engine/gen_quality.py --cond official            # 公式の実装、40 ステップ（runs/quality/wav/）
scripts/run_in_image.sh engine/gen_quality.py --cond official-s20        # 公式の実装、20 ステップ
IMAGE=sokutou/dev:7ed4374 scripts/run_in_image.sh engine/gen_quality.py --cond fused-bf16-nowm-bdec       # sokutou、40 ステップ
IMAGE=sokutou/dev:7ed4374 scripts/run_in_image.sh engine/gen_quality.py --cond fused-bf16-nowm-bdec-s20   # sokutou、20 ステップ
scripts/run_in_image.sh tools/score_sim.py                               # 声の近さ（runs/quality/sim.json）
python3 tools/score_asr.py                                               # CER（runs/quality/asr.json）
python3 tools/quality_summary.py                                         # 表（標準出力と runs/quality/summary.json）
```

`tools/score_asr.py` は、faster-whisper と pykakasi を入れた Python で動かします。`tools/score_sim.py` は、Hugging Face のキャッシュにある microsoft/wavlm-base-plus-sv と、参照音声と重ならない本物の声の 2 つの区間（`runs/quality/voice/hold_a.wav`、`hold_b.wav`）を使います。この 2 つの区間は自分で用意します。

### MF 版

`IRODORI_CHECKPOINT` で MF 版を選ぶと、出力の名前に `-mf` が付き、ステップ数の既定が 4 になります。

```sh
export IRODORI_CHECKPOINT=Aratako/Irodori-TTS-v4.1-Small-MF
scripts/bench_compare.sh bench/mf 2 "base:--engine official" "cc:--engine fused --bucket-decode"
scripts/run_in_image.sh ref/dump_reference.py                            # 参照（runs/ref/gpu-mf/）
IMAGE=sokutou/dev:7ed4374 scripts/run_in_image.sh engine/check.py --engine fused --precision fp32 \
  --latent-buckets 64,100,128 --name fused-fp32-pad-mf
python3 tools/snr.py runs/ref/gpu-mf runs/cand/fused-fp32-pad-mf
NO_GPU=1 DOCKER_EXTRA="--cpus 8" scripts/run_in_image.sh ref/run_candidate.py --device cpu --name official-cpu-mf --threads 8
python3 tools/snr.py runs/ref/gpu-mf runs/cand/official-cpu-mf
scripts/run_in_image.sh engine/gen_quality.py --cond official-nowm-mf --seeds 1,2,3,4,5,6,7,8,9,10
IMAGE=sokutou/dev:7ed4374 scripts/run_in_image.sh engine/gen_quality.py --cond fused-bf16-nowm-bdec-mf --seeds 1,2,3,4,5,6,7,8,9,10
speech serve irodori-tts-mf --device Vulkan0 --port 18093 --add-voice me=me.voice.gguf &
python3 tools/bench_http.py --url http://127.0.0.1:18093 --name speechcpp-mf --voice me
```

声のファイル（`me.voice.gguf`）は、RF 版で作ったものをそのまま使えます。

### v4-Large

`IRODORI_CHECKPOINT` で v4-Large を選ぶと、出力の名前に `-large` が付きます。ステップ数の既定は v4.1-Small と同じ 20 です。

```sh
export IRODORI_CHECKPOINT=Aratako/Irodori-TTS-v4-Large
scripts/bench_compare.sh bench/large 2 \
  "base:--engine official --steps 40" "base:--engine official --steps 20" \
  "cc:--engine fused --bucket-decode --steps 40" "cc:--engine fused --bucket-decode --steps 20"
scripts/run_in_image.sh ref/dump_reference.py --steps 20                 # 参照（runs/ref/gpu-large-s20/）
IMAGE=sokutou/dev:7ed4374 scripts/run_in_image.sh engine/check.py --engine fused --precision fp32 \
  --latent-buckets 64,100,128 --name fused-fp32-pad-large-s20
python3 tools/snr.py runs/ref/gpu-large-s20 runs/cand/fused-fp32-pad-large-s20
NO_GPU=1 DOCKER_EXTRA="--cpus 8" scripts/run_in_image.sh ref/run_candidate.py --device cpu --ref gpu-large-s20 \
  --name official-cpu-large-s20 --threads 8
python3 tools/snr.py runs/ref/gpu-large-s20 runs/cand/official-cpu-large-s20
scripts/run_in_image.sh engine/gen_quality.py --cond official-nowm-large-s20
IMAGE=sokutou/dev:7ed4374 scripts/run_in_image.sh engine/gen_quality.py --cond fused-bf16-nowm-bdec-large-s20
speech serve irodori-tts-large:f16 --device Vulkan0 --port 18095 --add-voice me=me.voice.gguf &
python3 tools/bench_http.py --url http://127.0.0.1:18095 --name speechcpp-large-f16-s20 --voice me --extra '{"sway_coeff": -1, "steps": 20}'
```

### Qwen3-TTS

```sh
scripts/build_qwen3.sh   # sokutou/qwen3-official:0.1.1、sokutou/qwen3-dev:0.1.1、sokutou/qwen3-server:<rev>
export PYTHON=/opt/qwen3/.venv/bin/python
# 速さ。1.7B は --model 1.7b。2 回目は名前の -1 を -2 にして、--save-dir を外す
IMAGE=sokutou/qwen3-official:0.1.1 scripts/run_in_image.sh qwen3/bench.py --model 0.6b \
  --name qwen3/official-0.6b-1 --save-dir bench/qwen3/official-0.6b-wav
IMAGE=sokutou/qwen3-dev:0.1.1 scripts/run_in_image.sh qwen3/bench.py --engine kernels --wd i8g32 --model 0.6b \
  --name qwen3/k2-0.6b-i8g32-1 --save-dir bench/qwen3/k2-0.6b-i8g32-wav
python3 qwen3/summary.py speed "公式 0.6B=official-0.6b-1,official-0.6b-2" "int8 0.6B=k2-0.6b-i8g32-1,k2-0.6b-i8g32-2"
# speech.cpp
speech serve qwen3-tts-0.6b --device Vulkan0 --port 18097 &
python3 tools/bench_http.py --url http://127.0.0.1:18097 --name qwen3/speechcpp-0.6b-1 --voice ono_anna --no-seconds \
  --extra '{"language": "ja"}' --save-dir bench/qwen3/speechcpp-0.6b-wav
# サーバー（docs/models/qwen3-tts.md の手順で起動してから）
python3 tools/bench_http.py --url http://127.0.0.1:18096 --name qwen3/http-0.6b-i8g32 --keep-going --save-dir bench/qwen3/http-0.6b-i8g32-wav
# float32 のコード、教師強制、サンプリングのカーネル
IMAGE=sokutou/qwen3-dev:0.1.1 scripts/run_in_image.sh qwen3/check.py --model 0.6b --name qwen3/check-k2-0.6b-fp32 \
  --dtype float32 --engine kernels --sampler torch --modes sample --limit 6 --no-inspect
IMAGE=sokutou/qwen3-dev:0.1.1 scripts/run_in_image.sh qwen3/teacher.py --model 0.6b --variant hf-dynamic --dtype float32 --save-ref
IMAGE=sokutou/qwen3-dev:0.1.1 scripts/run_in_image.sh qwen3/teacher.py --model 0.6b --variant hf-dynamic --dtype bfloat16
IMAGE=sokutou/qwen3-dev:0.1.1 scripts/run_in_image.sh qwen3/teacher.py --model 0.6b --variant kernels --wd i8g32 --dtype bfloat16
IMAGE=sokutou/qwen3-dev:0.1.1 scripts/run_in_image.sh qwen3/check_sampler.py
# 読み間違いと声の近さ
python3 qwen3/quality_rows.py qwen3-official-0.6b=official-0.6b-wav/official-0.6b-1 \
  qwen3-k2-0.6b-i8g32=k2-0.6b-i8g32-wav/k2-0.6b-i8g32-1 qwen3-speechcpp-0.6b=speechcpp-0.6b-wav/speechcpp-0.6b-1
QUALITY_DIR=runs/quality-qwen3 python3 tools/score_asr.py qwen3-official-0.6b qwen3-k2-0.6b-i8g32 qwen3-speechcpp-0.6b
unset PYTHON
scripts/run_in_image.sh qwen3/score_sim.py --ref qwen3-official-0.6b qwen3-official-0.6b qwen3-k2-0.6b-i8g32 qwen3-speechcpp-0.6b
python3 qwen3/summary.py quality qwen3-official-0.6b qwen3-k2-0.6b-i8g32 qwen3-speechcpp-0.6b
```

`tools/score_asr.py` は、Irodori-TTS と同じく faster-whisper と pykakasi を入れた Python で動かします。`qwen3/score_sim.py` は Irodori-TTS のイメージ（`scripts/run_in_image.sh` の既定）で動かし、Hugging Face のキャッシュにある microsoft/wavlm-base-plus-sv を使います。

### Drama 3

```sh
scripts/build_drama3.sh   # sokutou/drama3-official:0f8a12e、sokutou/drama3-dev:0f8a12e、sokutou/drama3-server:<rev>
export PYTHON=/opt/fish-speech/.venv/bin/python
# 速さ。参照ありは --ref <runs/ の下の WAV> --ref-text "<書き起こし>" を足し、名前を -ref-1 にする
IMAGE=sokutou/drama3-official:0f8a12e scripts/run_in_image.sh drama3/bench.py --name drama3/official-1 \
  --save-dir bench/drama3/wav-official-1
IMAGE=sokutou/drama3-dev:0f8a12e scripts/run_in_image.sh drama3/bench.py --name drama3/kernels-1 --engine kernels \
  --drop-codec-masks --save-dir bench/drama3/wav-kernels-1
# int8。参照ありでは --prefix-cache も足す
IMAGE=sokutou/drama3-dev:0f8a12e scripts/run_in_image.sh drama3/bench.py --name drama3/kernels-i8-2 --engine kernels \
  --wd i8g32 --prefill-graphs 24 --drop-codec-masks --save-dir bench/drama3/wav-kernels-i8-2
python3 drama3/summary.py speed "公式=official-1" "bf16=kernels-1" "int8=kernels-i8-2"
# speech.cpp（参照ありは speech voice drama-3:q8_0 <WAV> --transcript "<書き起こし>" <声>.gguf で声のファイルを作り、
# serve に --add-voice <声>=<声>.gguf、bench_http.py に --voice <声> を渡す）
speech serve drama-3 --device Vulkan0 --port 18097 &
python3 tools/bench_http.py --url http://127.0.0.1:18097 --name drama3/speechcpp-q4k-1 --voice none --no-seconds \
  --extra '{"experts_on_cpu": false}' --save-dir bench/drama3/wav-speechcpp-q4k-1
# サーバー（docs/models/drama3.md の手順で起動してから）
python3 tools/bench_http.py --url http://127.0.0.1:18098 --name drama3/http-i8-2 --voice none --no-seconds \
  --save-dir bench/drama3/wav-http-i8-2
# 公式の実装との突き合わせ（参照ありの確かめは --ref と --ref-text を足す。teacher の --teacher は公式の bench.py の出力）
IMAGE=sokutou/drama3-dev:0f8a12e scripts/run_in_image.sh drama3/check.py layers --ref <WAV> --ref-text "<書き起こし>"
IMAGE=sokutou/drama3-dev:0f8a12e scripts/run_in_image.sh drama3/check.py teacher --teacher bench/drama3/wav-official-1/official-1
IMAGE=sokutou/drama3-dev:0f8a12e scripts/run_in_image.sh drama3/check.py teacher --wd i8g32 --teacher bench/drama3/wav-official-1/official-1
IMAGE=sokutou/drama3-dev:0f8a12e scripts/run_in_image.sh drama3/check.py sampler
IMAGE=sokutou/drama3-dev:0f8a12e scripts/run_in_image.sh drama3/check.py prefill --ref <WAV> --ref-text "<書き起こし>"
# 読み間違いと声の近さ（声の近さは参照ありの条件だけ）
python3 drama3/quality_rows.py drama3-official-ref=wav-official-ref-1/official-ref-1 \
  drama3-kernels-i8-ref=wav-kernels-i8-ref-2/kernels-i8-ref-2
QUALITY_DIR=runs/quality-drama3 python3 tools/score_asr.py drama3-official-ref drama3-kernels-i8-ref
PYTHON=/opt/qwen3/.venv/bin/python IMAGE=sokutou/qwen3-dev:0.1.1 scripts/run_in_image.sh drama3/score_sim.py \
  --ref drama3-official-ref --ref-wav <WAV> drama3-official-ref drama3-kernels-i8-ref
python3 drama3/summary.py quality drama3-official-ref drama3-kernels-i8-ref
unset PYTHON
```

`drama3/check.py` は、結果を `runs/check/drama3/` に書きます。`drama3/score_sim.py` は microsoft/wavlm-base-plus-sv を使うので、transformers の入った Qwen3-TTS のイメージで動かします。
