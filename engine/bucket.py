"""初めての形の文でも CUDA Graph の記録を待たないエンジン（engine/fast.py の FastSampler を元にする）。

FastSampler は (潜在の長さ, 文のトークン数) ごとに 40 ステップ全体を 1 つのグラフに記録するので、
初めての形の文は記録のぶん遅くなる。BucketSampler は次の 3 つで、起動のときにすべての形を記録しておく。

1. 形をまとめる（バケット）: 潜在の長さと文のトークン数を、決まった区切りまで詰め物で伸ばす。
   詰め物の位置はキーのマスクで attention から外し、出力からも捨てる。DiT の中で位置をまたぐ処理は
   attention だけ（ほかは位置ごとの線形層と正規化）なので、有効な位置の計算は数学的に変わらない。
   RoPE の位置も変わらない（有効な位置は前に詰めてある）。
2. ステップごとのグラフ: CFG ありの steps_per_graph ステップと、CFG なしの steps_per_graph ステップを、
   形ごとに 1 つずつ記録して繰り返し再生する。ステップの番号は GPU の上のテンソルに持ち、AdaLN の係数と
   x の更新の幅をグラフの中で表から引き、番号もグラフの中で進める。x は決まったバッファをその場で更新する。
3. 文のエンコード（ModernBERT）と文の K・V の射影は、ステップとは別のグラフにする。こちらも文のトークン数を
   区切りまでパディングし（公式は 256 までパディングする）、マスクで外す。区切りはステップのグラフより細かい。
   256 トークンで通すと 7.8 ミリ秒かかるので、短い文は短い区切りで通す。

グラフの外に置くバッファ（x、ステップの番号、文の状態、条件の K・V、マスク）は記録の前に確保する。
グラフはメモリのプールを共有するので、ほかのグラフの再生をまたいで読む値は、プールの外に置く必要がある。

話者のトークン数は形の一部なので、起動のときの記録（prewarm）は声の長さごとに行う。
"""

from __future__ import annotations

import time

import torch

from engine.fast import FastSampler, Voice, matmul_tf32, patch_runtime
from irodori_tts.rf import draw_initial_noise


def default_latent_buckets(max_len: int = 750) -> list[int]:
    """潜在の長さの区切り。256 まで 4 刻み、512 まで 8 刻み、その先は 16 刻みで、最後は max_len（30 秒 = 25Hz で 750）。
    1 文の時間は長さに対して階段のように増える（行列積のカーネルが切り替わる長さで 6〜14 ミリ秒増える。
    測った範囲では 68、88、132、152、196 の手前）。区切りを 4 刻みにすると、詰め物で段をまたがない。"""
    out = list(range(16, 257, 4)) + list(range(264, 513, 8)) + list(range(528, max_len, 16)) + [max_len]
    return sorted(set(b for b in out if b <= max_len))


def default_text_buckets(max_len: int = 256) -> list[int]:
    """ステップのグラフが持つ、文の K・V の区切り（公式のトークナイザーの上限 256 まで）。
    詰め物のキーが増えても 1 文で 1〜3 ミリ秒しか変わらないので、粗くする。"""
    return [b for b in (24, 64, 128) if b < max_len] + [max_len]


def default_text_encode_buckets(max_len: int = 256) -> list[int]:
    """文のエンコードのグラフの区切り。"""
    return [b for b in (16, 24, 32, 48, 64, 96, 128, 192) if b < max_len] + [max_len]


def pick_bucket(buckets: list[int], n: int) -> int | None:
    for b in buckets:
        if n <= b:
            return b
    return None


class _TextCtx:
    """文のバケットごとの、条件の K・V のバッファ（文 Tb 個、話者 S 個の順）。"""

    def __init__(self, sampler: "BucketSampler", tb: int, n_spk: int):
        shape = (len(sampler.ctx_weights), 1, tb + n_spk, sampler.heads, sampler.head_dim)
        self.tb = tb
        self.k = torch.zeros(shape, device=sampler.device, dtype=sampler.wdtype)
        self.v = torch.zeros(shape, device=sampler.device, dtype=sampler.wdtype)
        self.voice: Voice | None = None


class _Shape:
    """(潜在のバケット, 文のバケット, 話者のトークン数) ごとのバッファとグラフ。"""

    def __init__(self, sampler: "BucketSampler", lb: int, ctx: _TextCtx, n_spk: int):
        self.lb, self.tb, self.n_spk, self.ctx = lb, ctx.tb, n_spk, ctx
        self.x = torch.zeros((1, lb, sampler.model.cfg.patched_latent_dim), device=sampler.device, dtype=sampler.model.dtype)
        self.mask3 = torch.ones((3, 1, 1, lb + ctx.tb + n_spk), dtype=torch.bool, device=sampler.device)
        self.mask1 = self.mask3[0:1]
        self.filled: tuple[int, int] | None = None
        self.g_cfg = None
        self.g_nocfg = None


class BucketSampler(FastSampler):
    def __init__(self, model, *, precision: str = "bf16", latent_buckets: list[int] | None = None,
                 text_buckets: list[int] | None = None,
                 text_encode_buckets: list[int] | None = None, steps_per_graph: int = 1, cudnn_attention: bool = False,
                 warm: str = "first", **kw):
        super().__init__(model, precision=precision, **kw)
        self.latent_buckets = sorted(latent_buckets or default_latent_buckets())
        self.text_buckets = sorted(text_buckets or default_text_buckets())
        self.text_encode_buckets = sorted(text_encode_buckets or default_text_encode_buckets())
        self.core.cudnn_attention = cudnn_attention
        self.n_cfg = sum(self.use_cfg)
        assert self.use_cfg == [True] * self.n_cfg + [False] * (self.num_steps - self.n_cfg), "CFG のステップが前半に並んでいない"
        self.k = int(steps_per_graph)
        assert self.n_cfg % self.k == 0 and (self.num_steps - self.n_cfg) % self.k == 0
        assert warm in ("first", "all")
        self.warm = warm
        self._warmed = False
        self.shapes: dict[tuple[int, int, int], _Shape] = {}
        self.ctxs: dict[tuple[int, int], _TextCtx] = {}
        self.text_graphs: dict[int, dict] = {}
        self.late_captures = 0
        self.prewarm_sec = 0.0
        with torch.inference_mode():
            self.idx = torch.zeros((1,), dtype=torch.long, device=self.device)
            # 公式が掛ける (t_next − t)（fp32）の表。
            self.dt_t = (self.t_schedule[1:] - self.t_schedule[:-1]).contiguous()

    # ---- 計算の本体（記録するグラフと、確かめの両方がこれを使う） ----

    def _body(self, sh: _Shape, cfg: bool):
        """1 ステップ。sh.x をその場で更新し、ステップの番号を 1 進める。戻り値は (CFG を合成した後の v, モデルの出力)。"""
        coef = self.coef.index_select(0, self.idx)[0]
        dt = self.dt_t.index_select(0, self.idx)
        x = sh.x
        freqs = self.freqs[:, : sh.lb]
        if cfg:
            raw = self.core(x.expand(3, -1, -1), coef, sh.ctx.k, sh.ctx.v, sh.mask3, freqs)
            c0, c1, c2 = raw[0:1], raw[1:2], raw[2:3]
            v = c0 + self.scales[0] * (c0 - c1)
            v = v + self.scales[1] * (c0 - c2)
        else:
            raw = self.core(x, coef, sh.ctx.k, sh.ctx.v, sh.mask1, freqs)
            v = raw
        x.add_(v * dt)
        self.idx.add_(1)
        return v, raw

    def _steps(self, sh: _Shape, cfg: bool) -> None:
        with matmul_tf32(self.precision == "tf32"):
            for _ in range(self.k):
                self._body(sh, cfg)

    def text_kv(self, text_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """文の状態 (1, T, 次元) から、各層の K・V (層, 1, T, heads, head_dim) を作る。"""
        ks, vs = [], []
        with matmul_tf32(self.precision == "tf32"):
            s = text_state.to(self.wdtype)
            for li in range(len(self.ctx_weights)):
                k, v = self._project_kv(s, li, "text")
                ks.append(k)
                vs.append(v)
        return torch.stack(ks), torch.stack(vs)

    def _text_body(self, tg: dict) -> None:
        """公式と同じ形（パディングあり）で文をエンコードし、K・V を作って、決まったバッファに書く。"""
        model = self.model
        with matmul_tf32(False):  # 文のエンコーダーは公式と同じ fp32
            if model.pretrained_text_backbone is None:
                state = model.text_encoder(tg["ids"], tg["mask"])
            else:
                state = model.text_encoder(model.pretrained_text_backbone, tg["ids"], tg["mask"])
            state = model.text_norm(state)
        tg["state"].copy_(state)
        k, v = self.text_kv(state)
        tg["k"].copy_(k)
        tg["v"].copy_(v)

    # ---- 記録 ----

    def _capture(self, fn, *args):
        if self.warm == "all" or not self._warmed:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                fn(*args)  # 温め
            torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            fn(*args)
        return graph

    def _text_graph(self, width: int) -> dict:
        tg = self.text_graphs.get(width)
        if tg is None:
            t0 = time.perf_counter()
            n_layers = len(self.ctx_weights)
            tg = {
                "ids": torch.zeros((1, width), dtype=torch.long, device=self.device),
                "mask": torch.ones((1, width), dtype=torch.bool, device=self.device),
                "state": torch.zeros((1, width, self.model.cfg.text_dim), device=self.device, dtype=self.model.dtype),
                "k": torch.zeros((n_layers, 1, width, self.heads, self.head_dim), device=self.device, dtype=self.wdtype),
                "v": torch.zeros((n_layers, 1, width, self.heads, self.head_dim), device=self.device, dtype=self.wdtype),
            }
            if self.use_graph:
                warmed, self._warmed = self._warmed, False  # 文のグラフは 1 つだけなので、いつも温める
                tg["graph"] = self._capture(self._text_body, tg)
                self._warmed = warmed
                torch.cuda.synchronize()
            self.text_graphs[width] = tg
            self.capture_sec.append(time.perf_counter() - t0)
        return tg

    def _shape(self, lb: int, tb: int, n_spk: int) -> _Shape:
        key = (lb, tb, n_spk)
        sh = self.shapes.get(key)
        if sh is None:
            t0 = time.perf_counter()
            ctx = self.ctxs.get((tb, n_spk))
            if ctx is None:
                ctx = self.ctxs[(tb, n_spk)] = _TextCtx(self, tb, n_spk)
            sh = _Shape(self, lb, ctx, n_spk)
            if self.use_graph:
                self.idx.zero_()
                if self.n_cfg:  # MF 版には CFG のステップがない
                    sh.g_cfg = self._capture(self._steps, sh, True)
                sh.g_nocfg = self._capture(self._steps, sh, False)
                self._warmed = True
                torch.cuda.synchronize()
                sh.x.zero_()
                self.idx.zero_()
            self.shapes[key] = sh
            self.capture_sec.append(time.perf_counter() - t0)
        return sh

    @torch.inference_mode()
    def prewarm(self, voice: Voice) -> dict:
        """この声の長さで、すべてのバケットの組を記録する。起動のときに一度呼ぶ。"""
        t0 = time.perf_counter()
        for te in self.text_encode_buckets:
            self._text_graph(te)
        for tb in self.text_buckets:
            for lb in self.latent_buckets:
                self._shape(lb, tb, voice.tokens)
        torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        self.prewarm_sec += sec
        return {"prewarm_sec": sec, "shapes": len(self.shapes), "graphs": self.graph_count()}

    def graph_count(self) -> int:
        return (1 + bool(self.n_cfg)) * len(self.shapes) + len(self.text_graphs) if self.use_graph else 0

    # ---- 1 文の準備と実行 ----

    def bucket_for(self, seq_len: int, n_text: int) -> tuple[int, int] | None:
        lb, tb = pick_bucket(self.latent_buckets, seq_len), pick_bucket(self.text_buckets, n_text)
        return None if lb is None or tb is None else (lb, tb)

    def _fill(self, sh: _Shape, seq_len: int, n_text: int, voice: Voice, tk: torch.Tensor, tv: torch.Tensor) -> None:
        """条件の K・V とマスクを、この文のものにする。tk・tv は (層, 1, 文のトークン数以上, heads, head_dim)。"""
        ctx, tb = sh.ctx, sh.tb
        if ctx.voice is not voice:
            for li, (k, v) in enumerate(voice.kv):
                ctx.k[li, :, tb:].copy_(k)
                ctx.v[li, :, tb:].copy_(v)
            ctx.voice = voice
        w = min(tb, tk.shape[2])
        ctx.k[:, :, :w].copy_(tk[:, :, :w])
        ctx.v[:, :, :w].copy_(tv[:, :, :w])
        if sh.filled != (seq_len, n_text):
            # 並びは公式と同じ: cond、文を外したもの、話者を外したもの。詰め物の位置はどの行でも外す。
            m, lb = sh.mask3, sh.lb
            m.zero_()
            m[:, :, :, :seq_len] = True
            m[0, :, :, lb:lb + n_text] = True
            m[2, :, :, lb:lb + n_text] = True
            m[0:2, :, :, lb + tb:] = True
            sh.filled = (seq_len, n_text)

    def _load_x(self, sh: _Shape, x0: torch.Tensor) -> None:
        if x0.shape[1] != sh.lb:
            sh.x.zero_()
        sh.x[:, : x0.shape[1]].copy_(x0)

    def _run_steps(self, sh: _Shape) -> None:
        self.idx.zero_()
        if self.use_graph:
            for _ in range(self.n_cfg // self.k):
                sh.g_cfg.replay()
            for _ in range((self.num_steps - self.n_cfg) // self.k):
                sh.g_nocfg.replay()
        else:
            for i in range(self.num_steps // self.k):
                self._steps(sh, i * self.k < self.n_cfg)

    def sample(self, **kw) -> torch.Tensor:
        """公式の sample_euler_rf_cfg（MF 版は sample_euler_meanflow）と同じ引数で呼べる。"""
        if kw.get("model") is not self.model or not self._supported(kw):
            self.fallbacks += 1
            return self.official(**kw)
        model = self.model
        seq_len = int(kw["sequence_length"])
        tmask = kw["text_mask"]
        n_text = int(tmask.sum())
        bucket = self.bucket_for(seq_len, n_text)
        if (bucket is None or n_text == 0 or not bool(tmask[:, :n_text].all())
                or pick_bucket(self.text_encode_buckets, n_text) is None):
            self.fallbacks += 1
            return self.official(**kw)
        if kw.get("initial_noise") is not None:
            x0 = kw["initial_noise"].to(device=self.device, dtype=model.dtype)
        else:  # 公式と同じく、詰め物のない長さで乱数を引く
            x0, _, _ = draw_initial_noise(int(kw.get("seed", 0)), (1, seq_len, model.cfg.patched_latent_dim),
                                          device=self.device, dtype=model.dtype)
        voice = self.voice_for(kw["ref_latent"], kw["ref_mask"])
        key = (bucket[0], bucket[1], voice.tokens)
        # 文のエンコードの区切り。トークナイザーが返した幅（公式は 256）より長くはしない。
        width = min(pick_bucket(self.text_encode_buckets, n_text), int(kw["text_input_ids"].shape[1]))
        if self.use_graph and (key not in self.shapes or width not in self.text_graphs):
            self.late_captures += 1
        tg = self._text_graph(width)
        sh = self._shape(*key)
        tg["ids"].copy_(kw["text_input_ids"][:, :width])
        tg["mask"].copy_(tmask[:, :width])
        if self.use_graph:
            tg["graph"].replay()
        else:
            self._text_body(tg)
        self.last_text_state = tg["state"][:, :n_text]
        self._fill(sh, seq_len, n_text, voice, tg["k"], tg["v"])
        self._load_x(sh, x0)
        self._run_steps(sh)
        return sh.x[:, :seq_len].clone()

    # ---- 確かめ用（engine/check.py） ----

    def tf_prepare(self, seq_len: int, text_state: torch.Tensor, voice: Voice) -> _Shape:
        """参照の条件（有効なトークンだけの文の状態）で、ステップごとの確かめの準備をする。グラフは使わない。"""
        n_text = int(text_state.shape[1])
        lb, tb = self.bucket_for(seq_len, n_text)
        ctx = _TextCtx(self, tb, voice.tokens)
        sh = _Shape(self, lb, ctx, voice.tokens)
        tk, tv = self.text_kv(text_state)
        self._fill(sh, seq_len, n_text, voice, tk, tv)
        return sh

    def tf_step(self, sh: _Shape, x: torch.Tensor, i: int):
        """参照の x を入れて 1 ステップ。戻り値は (v, モデルの出力)。どちらも有効な位置だけ。"""
        seq_len = x.shape[1]
        self._load_x(sh, x)
        self.idx.fill_(i)
        with matmul_tf32(self.precision == "tf32"):
            v, raw = self._body(sh, self.use_cfg[i])
        return v[:, :seq_len], raw[:, :seq_len]


def install_bucket(rt, *, precision: str = "bf16", **kw) -> BucketSampler:
    """公式の runtime にエンジンを差し込む。復号・電子透かし・後処理は公式のまま。"""
    sampler = BucketSampler(rt.model, precision=precision, **kw)
    patch_runtime(sampler)
    return sampler


def prewarm_codec(rt, seconds: tuple[float, ...] = (0.5, 2.0, 4.0, 8.0)) -> dict:
    """電子透かしと復号を、起動のときに合成した入力で一度ずつ通す。

    電子透かしは、約 2.9 秒より長い音声を初めて通すときに一度だけ約 0.16 秒かかる（測定:
    runs/experiments/decode-prewarm.json）。これを起動の時間に移す。復号は最も短い長さだけを通す。
    30 秒の復号を通すと PyTorch が約 8.7GB を確保したままになるので、長い入力は通さない。
    復号が初めての長さで 8〜14 ミリ秒遅くなる分は、ここでは消せない（runs/experiments/decode-cache.txt）。
    """
    t0 = time.perf_counter()
    sr = int(rt.codec.sample_rate)
    g = torch.Generator().manual_seed(0)
    with torch.inference_mode():
        for sec in seconds if rt.watermarker.ready else ():
            audio = torch.randn(1, int(round(sec * sr)), generator=g) * 0.1
            rt.watermarker.encode_batch([audio], sample_rate=sr)
        frames = -(-int(min(seconds) * sr) // int(rt.codec.model.hop_length))
        z = torch.randn(1, frames, rt.model_cfg.latent_dim, generator=g).to(rt.codec_device)
        rt.codec.decode_latent(z).cpu()
    torch.cuda.synchronize()
    return {"prewarm_codec_sec": time.perf_counter() - t0}


def disable_watermark(rt) -> None:
    """電子透かし（SilentCipher）を外す（2026-11-11 に利用者が外してよいと決めた）。

    公式の runtime は `rt.watermarker.ready`（モデルが読み込めているか）が偽なら透かしの段階を飛ばし、
    音声をそのまま返す（inference_runtime.py の _finish_request）。ここでは透かしのモデルへの参照を捨てる。
    GPU の使用量は、起動の準備のあとで 10.87 → 9.85GB、48 文 × シード 2 のあとで 11.47 → 11.24GB だった
    （runs/bench/nowm/*.info.json）。公式は飛ばしたときに「watermark is unavailable」の警告を
    SamplingResult.messages に足す。
    """
    rt.watermarker.model = None
    torch.cuda.empty_cache()
