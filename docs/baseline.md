# 公式の実装のプロファイルと基準

> エンジンを作る前（2026-11-10）に、公式の実装を測った記録です。エンジンの結果は [results.md](results.md) にあります。「再現の手順」のコマンドは、今のリポジトリで動く形に直してあります。

独自のエンジンが超えるべき基準として、公式の実装の 1 文の合成時間とその内訳、CUDA Graph にまとめたときの時間、speech.cpp の時間、公式どうしの出力の差を測りました。

計測日: 2026-11-10。GPU 0（RTX 6000 Ada）で、音声合成サーバーと同じイメージ（今の `sokutou/official:7ed4374` と同じ中身。公式の実装のコミット 7ed4374、torch 2.10.0+cu128、fp32）を動かしました。合成の引数は音声合成サーバーと同じです（`ref/common.py`）。文は作者の前の計測と同じ 48 文で、`seconds` も同じ値を渡しました（`data/sentences.json`）。行列積の TF32 は無効、cuDNN の TF32 は有効でした（イメージの既定）。

## 1 文の合成時間

測る範囲は、合成から、24kHz への変換と前後の無音の切り落としまでです（音声合成サーバーの `X-Synthesis-Seconds` と同じ範囲）。48 文 × シード 1, 2 の 96 回の値です。

| 条件 | 中央値 | 90 パーセンタイル | 最大 | 15 字以下 | 25 字以上 | 生成（`sample_rf`） | 復号 | 電子透かし |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 公式の実装 | 0.775 秒 | 0.800 秒 | 0.928 秒 | 0.763 秒 | 0.786 秒 | 0.741 秒 | 0.018 秒 | 0.008 秒 |
| CUDA Graph | 0.449 秒 | 0.532 秒 | 0.679 秒 | 0.339 秒 | 0.516 秒 | 0.417 秒 | 0.016 秒 | 0.008 秒 |
| CUDA Graph、パディングなし | 0.389 秒 | 0.469 秒 | 0.571 秒 | 0.284 秒 | 0.456 秒 | 0.356 秒 | 0.016 秒 | 0.007 秒 |
| CUDA Graph、パディングなし、TF32 | 0.237 秒 | 0.264 秒 | 0.316 秒 | 0.203 秒 | 0.254 秒 | 0.209 秒 | 0.016 秒 | 0.007 秒 |
| speech.cpp v0.10.1（Vulkan、F16 の重み） | 0.394 秒 | 0.417 秒 | 0.439 秒 | 0.259 秒 | 0.411 秒 | — | — | — |

「15 字以下」と「25 字以上」は、文の長さで分けた中央値です（句点を含む字数）。

- 「CUDA Graph」は公式のモジュールのまま、KV の準備と 40 ステップを 1 つの CUDA Graph に記録して再生したものです（`ref/graphs_baseline.py`）。グラフは潜在の長さごとに記録しました。48 文で 36 通りの長さがあり、1 つの記録に約 1.6 秒かかりました。記録は計測の前にすべて済ませています。別の回（`ref/graphs_baseline.py`）でも、中央値は 0.445 秒でした。
- CUDA Graph にすると、文が長いほど時間がかかるようになりました。CPU の待ちがなくなり、GPU の計算量で時間が決まるようになったためと考えています。
- 「CUDA Graph、パディングなし」は、そのうえで、パディングした文のトークンと空の caption を attention と KV から外したものです。グラフは潜在の長さと文のトークン数の組ごとに記録し、44 通りになりました。
- 「TF32」は、そのうえで行列積の TF32 を有効にしたものです（`torch.backends.cuda.matmul.allow_tf32`）。DiT 以外の行列積（ModernBERT など）にも効いています。出力の波形は公式の実装とはっきり変わります（「公式どうしの出力の差」の最後）。
- 1 文ごとに 2 秒の間を空けて測ると（シード 1 の 48 回、`--gap 2`）、中央値は CUDA Graph、パディングなしで 0.376 秒、TF32 で 0.240 秒でした。続けて測ったときとほとんど変わりません。
- speech.cpp は同じ GPU 0 で `speech serve` を立て、同じ 48 文、同じ `seconds`、シード 1, 2 を HTTP で送って測りました（`tools/bench_http.py`）。`sway_coeff` は本番と同じ -1 にし、ほかは speech.cpp の既定（40 ステップ、independent の CFG、文 3.0、話者 5.0、`cfg_min_t` 0.5）で、本番と同じです。時間は HTTP を含めた値で、speech.cpp のログにある合成の時間は中央値 0.392 秒でした。speech.cpp は電子透かしを入れず、48kHz のまま返し、無音の切り落としもしません。このマシンの Vulkan の番号は nvidia-smi と逆で、GPU 0 は `Vulkan1` です。測っているあいだ、GPU 1 の使用メモリは変わりませんでした。

## 内訳

公式の実装の 1 文の内訳です。48 文 × シード 1 で、区間の前後で GPU と同期して測りました。公式のループは毎ステップ同期するので、同期を足しても時間はほとんど変わりません（この回の 1 文の中央値は 0.778 秒）。

| 段階 | 中央値 |
| --- | --- |
| 条件のエンコード（`encode_conditions`。ModernBERT を文 256 トークンと caption 512 トークンで 2 回、話者のエンコーダー） | 33.6 ミリ秒 |
| KV の準備（`build_context_kv_cache`、2 回） | 9.0 ミリ秒 |
| DiT を 3 つ分通す 1 回（CFG のステップ） | 17.1 ミリ秒 |
| DiT を 1 つ分通す 1 回 | 17.9 ミリ秒 |
| DiT の 40 回の合計 | 0.70 秒 |
| ループのそのほか（CFG の合成、x の更新、同期） | 7.0 ミリ秒 |
| 復号（`decode_latent`） | 15.8 ミリ秒 |
| 電子透かし（SilentCipher） | 7.4 ミリ秒 |

形は次のとおりでした。

- 文のトークン: 256（有効なのは 4〜18 個。残りはパディング）
- caption のトークン: 512（caption は空なので、すべて無効）
- 話者の条件: 50 トークン（参照音声 7.9 秒から）
- 潜在の長さ: 41〜137（中央値 89。25Hz で 32 次元）

## 文のエンコードだけの時間

文のエンコード（ModernBERT と、そのあとの正規化）だけを、48 文で測りました（`ref/time_text_encode.py`）。PyTorch のまま、1 文ごとに 7 回動かし、最初の 1 回を除いた中央値をとっています。

| 文のトークン | 中央値 | 最小 | 最大 |
| --- | --- | --- | --- |
| 256（公式と同じくパディングする） | 12.9 ミリ秒 | 12.7 ミリ秒 | 13.2 ミリ秒 |
| 有効なトークンだけ（4〜18 個） | 11.8 ミリ秒 | 11.5 ミリ秒 | 12.0 ミリ秒 |

- トークンを 256 から十数個に減らしても、時間は 1 ミリ秒ほどしか短くなりません。DiT と同じく、CPU がカーネルを起動する時間で決まっていると考えています。
- 有効なトークンだけ通した結果は、256 トークンで通した結果の有効な行と比べて、SNR が最小 126.4 dB・中央値 127.7 dB でした。

## 1 回通すときの GPU の動き

中くらいの長さの 1 文（「次の配信は土曜日の20時からです。」、潜在の長さ 90）を torch.profiler で測りました。プロファイラーは CPU 側の時間を長くするので、ここでは時間ではなく、カーネルの数と GPU の計算時間だけを見ています。

| | 3 つ分（CFG のステップ） | 1 つ分 |
| --- | --- | --- |
| 1 回のカーネルの数 | 1,109 | 1,095 |
| 1 回の GPU の計算時間 | 10.4 ミリ秒 | 5.7 ミリ秒 |

1 回の時間（17〜18 ミリ秒）のうち、GPU が計算しているのは 3 つ分で 6 割、1 つ分で 3 割です。残りは、CPU がカーネルを起動し終えるのを GPU が待っている時間です。1 文の合成で `cudaLaunchKernel` は 47,487 回、`cudaStreamSynchronize` は 234 回呼ばれていました。

GPU の計算時間の内訳（20 回の合計を 1 回あたりにした値）は次のとおりです。

- 3 つ分: fp32 の行列積（`ampere_sgemm_128x64_tn`）5.8 ミリ秒、attention（fp32 の memory-efficient attention）1.2 ミリ秒、AdaLN の小さな行列の計算 0.9 ミリ秒、行列積の split-K の足し合わせ 0.4 ミリ秒、要素ごとの処理と連結が合わせて約 1.8 ミリ秒。
- 1 つ分: 行列積が 2.6 ミリ秒（`ampere_sgemm_64x32` と `128x64`）、attention 1.2 ミリ秒、残りは小さな処理。

attention には、パディングした文のトークン（約 250）と空の caption（512）を含む約 900 個のキーが入っています。有効なキーは潜在の長さと文のトークンと話者の 50 を足した約 150 個です。

## 行列積だけの速さ

DiT の 12 層分の行列積（1 層に 8 個。重みは 12 層それぞれ別で、計 268M 個）だけを、1 つの CUDA Graph にして続けて回しました（`ref/gemm_bench.py`、cuBLAS、30 回の平均）。重みは L2 キャッシュ（96 MiB）に収まらないので、毎回 GPU のメモリから読みます。

| 行の数 | fp32 | TF32 | BF16 の重み |
| --- | --- | --- | --- |
| 90（1 つ分） | 2.9 ミリ秒（重みの読み込み 368 GB/s） | 1.8 ミリ秒（590 GB/s） | 1.0 ミリ秒（528 GB/s） |
| 270（3 つ分） | 8.3 ミリ秒（17.3 TFLOPS） | 3.4 ミリ秒（42.6 TFLOPS） | 1.7 ミリ秒（84.6 TFLOPS） |
| 4096 | 118.9 ミリ秒（18.5 TFLOPS） | 34.6 ミリ秒（63.4 TFLOPS） | 23.5 ミリ秒（93.4 TFLOPS） |

- fp32 の行列積は、行の数を 4096 に増やしても約 19 TFLOPS でした。行の数が少ないことは、効率が低い主な理由ではありませんでした。
- 測っているあいだ、GPU の消費電力は中央値 293 W（最大 302 W）で、SM のクロックは中央値 1,035 MHz（最小 795 MHz）まで下がっていました。nvidia-smi が示す制限の理由は電力の上限（SW Power Cap）です。GPU の性能から計算した FP32 の約 91 TFLOPS は 2.5 GHz のときの値で、続けて計算すると、そのクロックは出ません。
- 公式の実装のプロファイルでは、3 つ分の行列積は 1 回 5.8 ミリ秒で、ここで測った 8.3 ミリ秒より短い時間でした。プロファイルのときは GPU が 4 割ほど待っていたので、クロックが下がりにくかったためと考えています（確かめていません）。
- 入力が同じ行列積をまとめても（1 層に 4 個）、cuBLASLt に替えても、速くなりませんでした。

## 公式どうしの出力の差

保存した参照（GPU、シード 1、6 文）に対して、同じ入力で動かした結果を SNR（dB）で比べました（`tools/snr.py`）。

- 同じ GPU でもう一度動かすと、すべての段階で一致しました（誤差 0）。CUDA Graph にまとめたときも一致しました。
- CPU（fp32）で同じノイズを入れて動かしたときの差は、次のとおりです。6 文の最小と中央値です。

| 比べたもの | 最小 | 中央値 |
| --- | --- | --- |
| 文の状態（有効なトークン） | 125.3 dB | 126.5 dB |
| 話者の状態 | 113.5 dB | 113.5 dB |
| ステップごとに参照の x を入れたときの v（CFG の 3 つ分） | 118.8 dB | 119.1 dB |
| 同じく 1 つ分 | 116.6 dB | 116.8 dB |
| 同じく CFG を合成した後の v | 106.8 dB | 108.0 dB |
| 参照の潜在変数を入れたときの復号 | 62.7 dB | 63.6 dB |
| 参照の音声を入れたときの電子透かし | 71.6 dB | 71.7 dB |
| 最初のノイズから通したときの最後の潜在変数 | 85.0 dB | 96.9 dB |
| 同じく 24kHz の最終の出力 | 62.3 dB | 63.2 dB |

- CFG を合成すると差が大きくなるのは、合成の式（v = 9c0 − 3c1 − 5c2。c0〜c2 は 3 つ分の出力）で、3 つの出力の誤差がそれぞれ 9 倍、3 倍、5 倍になって足されるためと考えています。
- 復号の差が大きいのは、GPU の復号が cuDNN の畳み込みに TF32 を使っているためと考えています（確かめていません）。
- CUDA Graph でパディングを外したときは、最後の潜在変数が最小 88.3 dB・中央値 112.7 dB、24kHz の出力が最小 67.3 dB・中央値 68.1 dB でした。どちらも CPU と GPU の差より小さい差です。
- そのうえで行列積の TF32 を有効にすると、最後の潜在変数が最小 27.3 dB・中央値 45.6 dB、24kHz の出力が最小 14.5 dB・中央値 34.3 dB でした。CPU と GPU の差よりずっと大きく、波形がはっきり変わります。聞いて分かる違いかどうかは確かめていません。TF32 を使うかどうかは、SNR ではなく、読み間違いと声の近さで決める必要があります。

## 再現の手順

```sh
scripts/run_in_image.sh ref/profile_official.py            # 合成時間、内訳、プロファイル（runs/profile/）
# ここの記録は 40 ステップ。bench_engine.py の既定は 20 ステップになった（2026-11-11）ので、--steps 40 を付ける
scripts/run_in_image.sh ref/bench_engine.py --engine official --steps 40
scripts/run_in_image.sh ref/bench_engine.py --engine graphs --steps 40
scripts/run_in_image.sh ref/bench_engine.py --engine graphs-trim --steps 40
scripts/run_in_image.sh ref/bench_engine.py --engine graphs-trim --tf32 --steps 40
scripts/run_in_image.sh ref/bench_engine.py --engine graphs-trim --seeds 1 --gap 2 --steps 40   # 間を空けたとき（--tf32 も同じ）
# speech.cpp v0.10.1（Vulkan 版）。デバイスの名前は `speech devices` で確かめる（nvidia-smi の番号と違うことがある）
speech voice irodori-tts voices/<声の名前>/ref.wav me.voice.gguf
speech serve irodori-tts --device <Vulkan のデバイス> --port 18092 --add-voice me=me.voice.gguf &
python3 tools/bench_http.py --url http://127.0.0.1:18092 --name speechcpp --voice me --extra '{"sway_coeff": -1}'
python3 tools/bench_summary.py runs/bench/official.jsonl runs/bench/graphs.jsonl runs/bench/graphs-trim.jsonl runs/bench/graphs-trim-tf32.jsonl runs/bench/speechcpp.jsonl
scripts/run_in_image.sh ref/gemm_bench.py                  # 行列積だけの速さ（runs/profile/gemm_bench.json）
scripts/run_in_image.sh ref/time_text_encode.py            # 文のエンコードだけの時間（runs/profile/text_encode.jsonl）
scripts/run_in_image.sh ref/dump_reference.py              # 参照の保存（runs/ref/gpu/）
NO_GPU=1 DOCKER_EXTRA="--cpus 8" scripts/run_in_image.sh ref/run_candidate.py --device cpu --name official-cpu --threads 8
python3 tools/snr.py runs/ref/gpu runs/cand/official-cpu
scripts/run_in_image.sh ref/check_graphs.py --trim --tf32 --name graphs-trim-tf32
python3 tools/snr.py runs/ref/gpu runs/cand/graphs-trim-tf32
```
