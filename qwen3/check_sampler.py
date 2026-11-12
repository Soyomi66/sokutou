"""qwen3/sampler.py の 1 カーネルのサンプリングを、torch の演算の経路（qwen3/fast.py の _process と _sample）と比べる。

  PYTHON=/opt/qwen3/.venv/bin/python IMAGE=sokutou/qwen3-dev:0.1.1 scripts/run_in_image.sh qwen3/check_sampler.py

1. 確率がいちばん高い番号を選ぶ設定（do_sample=False）で、processor（繰り返しの罰、抑える番号、min_new_tokens の EOS）を
   かけたあとの番号が torch と同じか（乱数を使わないので、完全に一致するはず）
2. 引く設定（温度 0.9、top_k 50）で、同じ logits から多数回引いた頻度が、softmax(top-k(logits / 温度)) と合うか。
   torch の経路で同じ回数引いたときのずれ（全変動距離）を、偶然のずれの目安にする
2b. 言語モデルの経路（processor をかけたあとに引く）で、同じく頻度を見る。EOS の頻度と、抑える番号を引いていないことも見る
3. 1 回の時間（CUDA Graph に 16 回分を記録して再生）
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fast import _sample  # noqa: E402
from sampler import FusedSampler  # noqa: E402


def process(scores, seen, suppress, ngen, min_new, eos, p):
    pen = torch.where(scores < 0, scores * p, scores / p)
    scores = torch.where(seen, pen, scores)
    scores = scores.masked_fill(suppress, -float("inf"))
    if ngen < min_new:
        scores = scores.clone()
        scores[:, eos] = -float("inf")
    return scores


def main() -> None:
    dev = torch.device("cuda")
    fs = FusedSampler(dev)
    out = torch.zeros(1, dtype=torch.int64, device=dev)
    g = torch.Generator(device=dev).manual_seed(0)
    # 1. argmax（processor あり、言語モデルの形 V=3072）
    V, eos = 3072, 2150
    bad = 0
    for trial in range(2000):
        logits = torch.randn((1, V), device=dev, generator=g) * 3
        if trial % 3 == 0:
            logits[0, eos] = logits.max() + 1  # EOS がいちばん高い場合
        seen = torch.rand((1, V), device=dev, generator=g) < 0.05
        suppress = torch.zeros((1, V), dtype=torch.bool, device=dev)
        suppress[:, V - 1024:] = True
        suppress[:, eos] = False
        ngen_v = trial % 4
        ngen = torch.tensor([ngen_v], device=dev)
        seen_k = seen.clone().view(torch.uint8)
        ref = torch.argmax(process(logits, seen, suppress, ngen_v, 2, eos, 1.05), dim=-1)
        id0 = torch.zeros(1, dtype=torch.int64, device=dev)
        pos = torch.zeros(1, dtype=torch.int64, device=dev)
        fs(logits.view(-1).contiguous(), out, False, 1.0, 0,
           talker=dict(seen=seen_k, rep_pen=1.05, suppress=suppress, ngen=ngen, min_new=2, eos=eos, id0=id0, pos=pos))
        torch.cuda.synchronize()
        ok = int(out) == int(ref) and int(id0) == int(ref) and int(ngen) == ngen_v + 1 and int(pos) == 1 \
            and bool(seen_k[0, int(ref)]) and int(seen_k.sum()) == int(seen.sum()) + (0 if bool(seen[0, int(ref)]) else 1)
        bad += not ok
    print(f"1. argmax と後始末: 2000 回中 {2000 - bad} 回が torch と一致")

    # 2. 分布（code predictor の形 V=2048、温度 0.9、top_k 50）
    N = 200000
    for V, scale in ((2048, 2.0), (2048, 4.0), (3072, 1.0)):
        logits = (torch.randn((1, V), device=dev, generator=g) * scale).contiguous()
        sc = logits / 0.9
        kth = torch.topk(sc, 50)[0][..., -1, None]
        p = torch.softmax(sc.masked_fill(sc < kth, -float("inf")), dim=-1)[0]
        fs.set_seed(1234)
        res = torch.zeros(N, dtype=torch.int64, device=dev)
        lv = logits.view(-1)
        for n in range(N):
            fs(lv, res.data_ptr() + 8 * n, True, 0.9, 50)
        torch.manual_seed(1234)
        ref = torch.cat([_sample(logits, True, 0.9, 50).view(-1) for _ in range(N)])
        torch.cuda.synchronize()
        hf = torch.bincount(res, minlength=V).double() / N
        ht = torch.bincount(ref, minlength=V).double() / N
        out_k = int((hf[p == 0] > 0).sum())
        print(f"2. V={V} 尺度 {scale}: 全変動距離 fused {0.5 * float((hf - p).abs().sum()):.4f}、"
              f"torch {0.5 * float((ht - p).abs().sum()):.4f}（{N} 回）; top-k の外を引いた番号 {out_k}; "
              f"最大確率 {float(p.max()):.3f}")

    # 2b. 言語モデルの経路（繰り返しの罰、抑える番号、min_new_tokens の EOS）で引いた頻度。EOS が出る確率を高めにした logits で、
    #     作ったトークンが min_new_tokens 以上のとき（EOS を引ける）と未満のとき（EOS は 0）を見る。
    #     カーネルは seen・ngen・pos を書き換えるので、1 回ごとに元に戻す
    V, eos, N2 = 3072, 2150, 50000
    logits = (torch.randn((1, V), device=dev, generator=g) * 2).contiguous()
    logits[0, eos] = float(logits.max()) - 0.5
    seen0 = torch.zeros((1, V), dtype=torch.bool, device=dev)
    seen0[0, torch.topk(logits[0, :V - 1024], 20).indices] = True  # 確率の高い番号のいくつかを「出た」ことにする
    suppress = torch.zeros((1, V), dtype=torch.bool, device=dev)
    suppress[:, V - 1024:] = True
    suppress[:, eos] = False
    for ngen_v in (5, 1):
        sc = process(logits, seen0, suppress, ngen_v, 2, eos, 1.05) / 0.9
        kth = torch.topk(sc, 50)[0][..., -1, None]
        p = torch.softmax(sc.masked_fill(sc < kth, -float("inf")), dim=-1)[0]
        seen_k = seen0.clone().view(torch.uint8)
        ngen = torch.zeros(1, dtype=torch.int64, device=dev)
        pos = torch.zeros(1, dtype=torch.int64, device=dev)
        id0 = torch.zeros(1, dtype=torch.int64, device=dev)
        res = torch.zeros(N2, dtype=torch.int64, device=dev)
        fs.set_seed(99)
        tk = dict(seen=seen_k, rep_pen=1.05, suppress=suppress, ngen=ngen, min_new=2, eos=eos, id0=id0, pos=pos)
        for n in range(N2):
            seen_k.copy_(seen0.view(torch.uint8))
            ngen.fill_(ngen_v)
            fs(logits.view(-1), res.data_ptr() + 8 * n, True, 0.9, 50, talker=tk)
        torch.manual_seed(99)
        ref = torch.cat([_sample(process(logits, seen0, suppress, ngen_v, 2, eos, 1.05), True, 0.9, 50).view(-1)
                         for _ in range(N2)])
        torch.cuda.synchronize()
        hf = torch.bincount(res, minlength=V).double() / N2
        ht = torch.bincount(ref, minlength=V).double() / N2
        print(f"2b. 言語モデルの経路 ngen={ngen_v}（min_new_tokens 2）: 全変動距離 fused {0.5 * float((hf - p).abs().sum()):.4f}、"
              f"torch {0.5 * float((ht - p).abs().sum()):.4f}（{N2} 回）; EOS の頻度 fused {float(hf[eos]):.4f}・"
              f"torch {float(ht[eos]):.4f}・期待 {float(p[eos]):.4f}; 抑える番号を引いた回数 {int(((res >= V - 1024) & (res != eos)).sum())}")

    # 3. 時間（16 回分を 1 つのグラフに）
    logits = (torch.randn((1, 2048), device=dev, generator=g) * 3).contiguous()
    lv = logits.view(-1)
    codes = torch.zeros((1, 16), dtype=torch.int64, device=dev)
    for name in ("torch", "fused"):
        def body():
            for i in range(16):
                if name == "torch":
                    codes[:, i:i + 1].copy_(_sample(logits, True, 0.9, 50))
                else:
                    fs(lv, codes.data_ptr() + 8 * i, True, 0.9, 50)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            body()
        torch.cuda.current_stream().wait_stream(s)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            import cupy as cp

            with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
                body()
        gr.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            gr.replay()
        torch.cuda.synchronize()
        print(f"3. {name}: 16 回で {(time.perf_counter() - t0) / 200 * 1000:.3f} ms")


if __name__ == "__main__":
    main()
