# 参照と候補のファイルの形

公式の実装の段階ごとの入出力（参照）と、移植した側の出力（候補）は、どちらも npz（numpy）で保存する。候補が同じ名前と形で書けば、`tools/snr.py` で比べられる。

- 参照: `runs/ref/<tag>/<case>/`。`ref/dump_reference.py` が書く。`<case>` は `<文の番号>_s<シード>`（例 `28_s1`）。
- 候補: `runs/cand/<name>/<case>/`。公式の実装を候補にするときは `ref/run_candidate.py` が書く。

参照音声から作った値と合成した音声を含むので、どちらも git に入れない（`runs/` は `.gitignore` 済み）。

## 参照のファイル

| ファイル | 中身（形。L は潜在の長さ、B は一度に通す数） |
| --- | --- |
| `inputs.npz` | `text_ids`、`text_mask`（1, 256）。`caption_ids`、`caption_mask`（1, 512。caption は空なのでマスクはすべて false）。`ref_latent`（1, 197, 32）、`ref_mask`。`noise`（1, L, 32。最初の x）。`t_schedule`（41） |
| `cond.npz` | 条件のエンコードの出力。`text_state`（1, 256, 512）、`text_mask`、`speaker_state`（1, 50, 768）、`speaker_mask`、`caption_state`（1, 512, 512）、`caption_mask` |
| `kv.npz` | cond の分の KV の射影。`kv_l<層>_<j>`（層は 0 と 11。j は 0 文の K、1 文の V、2 話者の K、3 話者の V、4・5 caption）。形は（1, トークン数, 20, 64） |
| `steps.npz` | `x`（40, 1, L, 32。各ステップのモデルの入力）、`t`（40）、`v`（40, 1, L, 32。CFG を合成した後）、`v_raw_cfg`（20, 3, L, 32。CFG のステップのモデルの出力。並びは cond、文を外したもの、話者を外したもの）、`v_raw_nocfg`（20, 1, L, 32）、`cfg_steps`、`x_final`（1, L, 32） |
| `blocks_s00.npz`、`blocks_s20.npz` | ステップ 0（CFG あり、B=3）と 20（CFG なし、B=1）の、各層の出力 `block<層>_out` と、層 0 の中間（`b0_attention_adaln_h`、`_gate`、`b0_attention`、`b0_mlp_adaln_h`、`_gate`、`b0_mlp`） |
| `audio.npz` | `decode_in`（復号の入力、1, L, 32）、`decode_out`（48kHz）、`wm_in`、`wm_out`（電子透かしの前後、48kHz）、`out24k`（24kHz に変換して前後の無音を切り落とした最終の出力） |
| `meta.json` | 文、`seconds`、シード、CFG の強さ、形、TF32 の設定、torch の版と公式の実装のコミット |

## 候補のファイル

比べ方は 2 通りある。

- ステップごとに参照の入力を入れる（teacher forcing）。実装の誤りを段階ごとに見つける。
  - `cond.npz`: `inputs.npz` から計算した条件。文と話者は、マスクが有効な行だけを比べるので、パディングを持たない形（1, 有効なトークン数, 次元）でもよい。そのときは `text_mask` と `speaker_mask` も同じ長さで書く。
  - `kv.npz`: 参照の `cond.npz` から計算した KV（参照と同じ名前。有効な行だけを比べる）。
  - `steps_tf.npz`: 参照の `x[i]` と参照の条件から計算した `v_raw_cfg`、`v_raw_nocfg`、`v`。
  - `blocks_s00.npz`、`blocks_s20.npz`: 参照の `x` から計算した各層の出力（参照と同じ名前）。
  - `audio_tf.npz`: 参照の `decode_in` を復号した `decode_out`、参照の `wm_in` に透かしを入れた `wm_out`。
- 最初のノイズから最後まで通す。誤差の積み重なりを見る。
  - `free.npz`: `x`（各ステップの入力。なくてもよい）、`x_final`、`decode_out`、`wm_out`、`out24k`。

## SNR の目安

公式どうしの差（`docs/baseline.md`）を下限の目安にする。同じ GPU で 2 回動かすと、すべての段階で一致する（誤差 0）。GPU と CPU では、ステップごとの v は 116〜119 dB、最後の x は 85〜97 dB、最終の出力は 62〜63 dB になる。
