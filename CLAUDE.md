# sokutou

Irodori-TTS v4.1-Small を公式の PyTorch の実装より速く合成する推論エンジン。エンジンの設定は RF 20 ステップと independent の CFG（前半 10 ステップ、文 3.0・caption 3.0・話者 5.0）。20 ステップは 2026-11-11 に決めた。計測と確かめのスクリプトの既定は `ref/common.py` の `DEFAULT_STEPS`。速さの目標は、同じ GPU・同じ 48 文で speech.cpp（v0.10.1、Vulkan、F16、40 ステップ）より速くすること。MF 版（v4.1-Small-MF、4 ステップ、CFG なし）にも対応した（2026-11-11）。v4-Large（`Aratako/Irodori-TTS-v4-Large`）にも対応した（2026-11-11）。GPU のメモリーを約 31GB 使い、重みは Gemma の利用規約に従う。対応するモデルは、speech.cpp が対応している音声合成のモデルに広げている（順と進み具合は `TASKS.md`）。計測と確かめのスクリプトは `IRODORI_CHECKPOINT` でモデルを選び、`ref/common.py` の `CHECKPOINTS` にないモデルは動かさない。

Qwen3-TTS 12Hz の 0.6B・1.7B CustomVoice にも対応した（2026-11-11、`qwen3/`）。Irodori とは依存の版が違うので、イメージは `docker/qwen3.Dockerfile`（`scripts/build_qwen3.sh`）で作り、スクリプトは `PYTHON=/opt/qwen3/.venv/bin/python IMAGE=sokutou/qwen3-dev:0.1.1 scripts/run_in_image.sh qwen3/<スクリプト>` で動かす。モデルは `--model 0.6b|1.7b`（`qwen3/bench.py` の `MODELS`）で選ぶ。サーバーは `qwen3/serve.py`（既定のポート 18096、重みは int8）。サンプリングで話す長さが変わるので、速さは音声 1 秒あたりでも比べ、確かめは float32 でのコードの一致（`qwen3/check.py`）と、CER と声の近さ（`qwen3/summary.py`）で行う。

Fish Audio Drama 3（`fishaudio/drama-3`）にも対応した（2026-11-11、`drama3/`）。公式の実装（fish-speech 0f8a12e、torch 2.8.0）のイメージは `docker/drama3.Dockerfile`（`scripts/build_drama3.sh`）で作り、スクリプトは `PYTHON=/opt/fish-speech/.venv/bin/python IMAGE=sokutou/drama3-dev:0f8a12e scripts/run_in_image.sh drama3/<スクリプト>` で動かす。サーバーは `drama3/serve.py`（既定のポート 18098、重みは int8）で、GPU のメモリーを約 45 GiB 使う（48 GiB の GPU で、残りは 2〜3 GiB）。確かめは、層ごと・教師強制・サンプリング・最初の入力（`drama3/check.py`）と、CER と声の近さ（`drama3/summary.py`）で行う。
- ライセンスの表示は speech.cpp と同じ形にした（2026-11-11 に利用者が決めた）。`drama3/` は MIT のままにする。fish-speech のコードは写さず、実行するときに呼ぶ。fish-speech・重み・コーデックは Fish Audio Community License なので、リポジトリに入れない（イメージを作るときに取得する）。
- README で Drama 3 を紹介するところには「Built with Fish Audio」を添え、FACL の要約は README の「ライセンス」の節と LICENSE の注記に書く。FACL の写しと Notice のファイルはリポジトリに置かず、イメージを配るときに付ける。
- Drama 3 の参照には書き起こしが要る。WAV と同じ名前の書き起こし（.lab）を `runs/dev/voices/<声>/` に置き、`VOICES=runs/dev/voices` で渡す（`voices/` の WAV も、ここに写して使う。書き起こしは参照の派生物なので git に入れない）。

GPU の割り当てと本番のサーバーの決まりは、このマシンだけの `CLAUDE.local.md` にある。そのファイルがないとき（別のマシンや worktree）は、GPU を使う前とサーバーを動かす前に、どの GPU とポートを使ってよいかを利用者に聞く。

## 守ること

- 学習（train.py、LoRA、Speaker Inversion）はしない。
- 参照音声（`voices/`）とその派生物は git に入れない。派生物は、参照の潜在変数・話者の状態・合成した音声・書き起こし・それらを含む参照の保存（`runs/`）。潜在変数は復号すると元の声に戻る。README と docs にも、音声の例や文ごとの書き起こしは載せない（集計した値だけ）。
- 電子透かし（SilentCipher）は外してよい（2026-11-11 に決めた）。外すときは `engine/bucket.py` の `disable_watermark` を使う。速さを前の記録と比べるときは、透かしの有無をそろえる（公式の実装とエンジンの 11-10 までの記録は、すべて透かしあり。speech.cpp は透かしを入れない。透かしなしの記録はファイル名に `-nowm` が付く）。

## 公式の実装

- イメージ `sokutou/official:7ed4374`（`docker/Dockerfile` の official の段）の `/opt/irodori`（コミット 7ed4374、torch 2.10+cu128、fp32）。読むときは `docker cp` で scratchpad に出す。イメージの中のコードは書き換えず、関数を包んで値を取り出す。
- サーバーの合成の引数は `server/server.py` の `SamplingRequest`。`ref/common.py` が同じ引数を組み立てる。
- TF32 は、行列積で無効、cuDNN（コーデックの畳み込み）で有効（イメージの既定）。突き合わせるときはこれに合わせる。
- 公式のノイズは CUDA の `torch.Generator` で作る。移植した側には、保存したノイズを入力として渡して比べる。

## 確かめ方

- 参照は `ref/dump_reference.py` で `runs/ref/` に保存し、`tools/snr.py` で比べる。
- 比べ方は 2 通り。ステップごとに公式の `x_t` を入れて `v` を比べる（実装の誤りを見つける）。最初のノイズから最後まで通して比べる（誤差の積み重なりを見る）。
- SNR の良し悪しは、公式どうしの差（同じ GPU で 2 回、GPU と CPU）を下限の目安にして判断する。値は `docs/baseline.md` にある。
- 速さは `data/sentences.json` の 48 文 × シード 1 と 2 で測る（範囲は、合成から 24kHz への変換と前後の無音の切り落としまで。サーバーの `X-Synthesis-Seconds` と同じ）。
- 速さを前の記録と比べるときは、ステップ数をそろえる。名前に `-s<数>` のない記録は 40 ステップ。speech.cpp の 0.394 秒と公式の約 0.78 秒は 40 ステップで、公式の 20 ステップは約 0.44 秒。20 ステップの fp32 の参照は `runs/ref/gpu-s20`。
- TF32 にすると、出力の波形は公式とはっきり変わる（24kHz の SNR が 14〜34 dB）。精度を下げた出力は、SNR ではなく、読み間違いと声の近さで確かめる。

## 参照先

- 使い方と結果の要約: `README.md`、`docs/results.md`
- 実装前の方式の比較: `docs/design.md`
- 公式の実装のプロファイルと基準の数値: `docs/baseline.md`
- 進み具合: `TASKS.md`（終えたらチェックし、見つけた作業を足す）
