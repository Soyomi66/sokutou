"""1 段目のエンジンの中心: サンプリングのループを自前で持ち、重い計算は公式のモジュールを呼ぶ。

公式の `sample_euler_rf_cfg` の代わりに `FastSampler.sample` を runtime に差し込んで使う
（`irodori_tts.inference_runtime.sample_euler_rf_cfg` を置き換える。復号・電子透かし・後処理は公式のまま）。

公式と違うところ（どれも数学的には同じ計算。docs/design.md の「速くするところ」1〜4）:
1. パディングした文のトークンと空の caption を、ModernBERT・KV・attention に入れない。
2. AdaLN の係数（時刻だけで決まる）を、40 ステップ × 12 層 × 2 個ぶん、読み込みのときに計算しておく。
3. 話者の条件と、その KV の射影を、声（参照音声の潜在変数）ごとに一度だけ計算する。
4. 文のエンコード・文の KV・40 ステップ・CFG の合成を、1 つの CUDA Graph に記録して再生する。
   グラフは (潜在の長さ, 文のトークン数, 話者のトークン数) ごとに記録する。

精度（precision）:
- "fp32": 公式と同じ。
- "tf32": DiT（KV の射影と 40 ステップ）の行列積だけを TF32 にする。`allow_tf32` は記録と温めのあいだ、
          DiT の区間でだけ有効にする。cuBLAS の計算の種類は記録のときに決まるので、再生のときは
          フラグが無効でも TF32 で計算される。文のエンコーダーは同じグラフの中でもフラグを無効にして記録する。
- "bf16": 12 層のブロックの重み（attention と MLP の線形層、文と話者の KV の射影）を BF16 の複製にし、
          その行列積、attention、SwiGLU を BF16 で計算する。残差の流れ、AdaLN、q と k の正規化と RoPE、
          in_proj、out_norm、out_proj、CFG の合成、x の更新は fp32 のまま。
          文と話者のエンコーダー、コーデック、電子透かしは触らない。

本番の設定（independent の CFG、文と話者、caption は空、linear の時刻）だけを扱う。
ほかの設定で呼ばれたら、公式の関数をそのまま呼ぶ。

MF 版（Irodori-TTS-v4.1-Small-MF。MeanFlow で蒸留した版）も扱う。公式の `sample_euler_meanflow` と同じく、
時刻の表は linspace(1, 0)、AdaLN の係数は時刻の埋め込みに区間の長さ（delta = t − t_next）の埋め込みを足したもの、
CFG はなし（蒸留に含まれる）。DiT の 1 回の計算は RF 版と同じなので、表を作り替えるだけで残りはそのまま使う。
差し込む先は `sample_euler_meanflow`（`patch_runtime`）。
"""

from __future__ import annotations

import contextlib
import os
import time

import torch
import torch.nn.functional as F

import irodori_tts.model as M
from irodori_tts.attention import masked_sdpa
from irodori_tts.meanflow import sample_euler_meanflow
from irodori_tts.rf import draw_initial_noise, sample_euler_rf_cfg

PRECISIONS = ("fp32", "tf32", "bf16")


@contextlib.contextmanager
def matmul_tf32(enabled: bool):
    """行列積の TF32 を、この区間でだけ切り替える。"""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def make_text_backbone_capturable(model) -> None:
    """文のエンコーダーの中で、CUDA Graph に記録できない処理を、同じ値を返す形に差し替える。

    v4-Large の T5Gemma 2（transformers の T5Gemma2TextScaledWordEmbedding）は、埋め込みのあとで eoi のトークンの位置を
    真偽値のマスクで書き換える。マスクでの代入は要素の数を CPU に問い合わせるので、記録の途中では動かない。
    同じ値を torch.where で作る forward に差し替える（イメージの中のコードは書き換えない）。
    v4.1-Small の ModernBERT には当てはまる部品がないので、何もしない。"""
    backbone = getattr(model, "pretrained_text_backbone", None)
    if backbone is None:
        return
    for mod in backbone.modules():
        if type(mod).__name__ != "T5Gemma2TextScaledWordEmbedding" or getattr(mod, "_sokutou_capturable", False):
            continue

        def forward(input_ids: torch.Tensor, _m=mod) -> torch.Tensor:
            emb = F.embedding(input_ids, _m.weight, _m.padding_idx, _m.max_norm, _m.norm_type, _m.scale_grad_by_freq,
                              _m.sparse) * _m.embed_scale.to(_m.weight.dtype)
            eoi = (input_ids == _m.eoi_token_index)[..., None]
            return torch.where(eoi, _m.eoi_embedding.to(emb.dtype), emb)

        mod.forward = forward
        mod._sokutou_capturable = True


class Voice:
    """声ごとに一度だけ計算する値: 話者の状態と、各層の K・V の射影。"""

    def __init__(self, ref_latent: torch.Tensor | None, state: torch.Tensor, kv: list[tuple[torch.Tensor, torch.Tensor]]):
        self.ref_latent = ref_latent
        self.state = state  # (1, S, speaker_dim)。有効なトークンだけ
        self.kv = kv  # 層ごとの (K, V)。(1, S, heads, head_dim)。エンジンの計算の型

    @property
    def tokens(self) -> int:
        return int(self.state.shape[1])


class DitCore:
    """DiT の 12 層を 1 回通す本体。公式の DiffusionBlock.forward と数学的に同じ計算で、違いは次のとおり。
    - AdaLN の係数は、前計算した表（coef）から読む。
    - 条件の K・V は、前計算したもの（有効なトークンだけ）をつなぐ。
    - 同じ入力にかける線形層（wq・wk・wv・gate と、MLP の w1・w3）は、重みをつないで 1 回の行列積にする。
    - q と k の正規化と RoPE は、まとめて 1 回で計算する。正規化は fp32。
    - 小さな処理は、カーネルの数が少ない書き方にする（vector_norm、addcmul、複素数の表への掛け算）。
    - 重みが BF16 のとき、残差の流れと正規化は fp32 のまま、行列積・attention・SwiGLU を BF16 で計算する。
    重みは公式のモジュールから読み、行列積の型（fp32 か BF16）でつないだ複製を持つ。
    """

    def __init__(self, model, wdtype: torch.dtype):
        self.in_proj, self.out_norm, self.out_proj = model.in_proj, model.out_norm, model.out_proj
        att0 = model.blocks[0].attention
        self.heads, self.head_dim, self.dim = att0.heads, att0.head_dim, att0.dim
        self.wdtype = wdtype
        self.cast = wdtype != model.dtype
        self.layers = []
        dev = model.device
        for block in model.blocks:
            att, mlp = block.attention, block.mlp
            self.layers.append({
                "w_qkvg": torch.cat([att.wq.weight, att.wk.weight, att.wv.weight, att.gate.weight]).to(wdtype).contiguous(),
                "w_o": att.wo.weight.to(wdtype).contiguous(),
                "w_13": torch.cat([mlp.w1.weight, mlp.w3.weight]).to(wdtype).contiguous(),
                "w_2": mlp.w2.weight.to(wdtype).contiguous(),
                "qk_weight": torch.stack([att.q_norm.weight, att.k_norm.weight]).float().contiguous(),  # (2, heads, head_dim)
                "qk_eps": torch.full((1,), float(att.q_norm.eps), device=dev),
                "a_eps": torch.full((1,), float(block.attention_adaln.eps), device=dev),
                "m_eps": torch.full((1,), float(block.mlp_adaln.eps), device=dev),
                "hidden": int(mlp.w1.weight.shape[0]),
            })
            assert att.q_norm.eps == att.k_norm.eps
        # マスクつきの attention を、公式と同じ優先順（cuDNN が先）で呼ぶか。False なら PyTorch の既定の順で呼ぶ。
        # cuDNN は新しい形ごとに計算の計画を作るので、形が多いと記録が遅くなる（engine/bucket.py は False にする）。
        self.cudnn_attention = True
        self.inv_dim = 1.0 / self.dim
        self.inv_head_dim = 1.0 / self.head_dim

    def _adaln(self, h_res: torch.Tensor, eps: torch.Tensor, scale1: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """h_res × rsqrt(mean(h_res²) + eps) × (1 + scale) + shift。出力は行列積の型。"""
        n = torch.linalg.vector_norm(h_res, dim=-1, keepdim=True)
        r = torch.rsqrt(torch.addcmul(eps, n, n, value=self.inv_dim))  # (B, L, 1)
        h = h_res * r
        if self.cast:
            return torch.addcmul(shift, h, scale1, out=torch.empty_like(h, dtype=self.wdtype))
        return torch.addcmul(shift, h, scale1)

    def __call__(self, x, coef, ctx_k, ctx_v, mask, freqs, return_blocks: bool = False):
        """x: (B, L, 32)。coef: (層, 6, 1, 1, D)。ctx_k・ctx_v: (層, 1, S, heads, head_dim)。
        mask: (B, 1, 1, L + S) か None。freqs: (1, L, 1, 1, head_dim / 2) の複素数。"""
        bsz, seq_len = x.shape[0], x.shape[1]
        heads, head_dim, dim, wd = self.heads, self.head_dim, self.dim, self.wdtype
        half = heads // 2
        h_res = self.in_proj(x)
        outs = []
        for li, p in enumerate(self.layers):
            c = coef[li]
            h = self._adaln(h_res, p["a_eps"], c[0], c[1])
            o = F.linear(h, p["w_qkvg"])  # (B, L, 4D): q、k、v、gate
            qk = o[..., : 2 * dim].reshape(bsz, seq_len, 2, heads, head_dim)
            n = torch.linalg.vector_norm(qk, dim=-1, keepdim=True, dtype=torch.float32)
            r = torch.rsqrt(torch.addcmul(p["qk_eps"], n, n, value=self.inv_head_dim))
            qk = (qk * r).mul_(p["qk_weight"])  # fp32
            # RoPE: 前半の head だけを回す（公式の _apply_rotary_half）。新しく作ったテンソルなので、その場で掛ける。
            torch.view_as_complex(qk[:, :, :, :half].unflatten(-1, (head_dim // 2, 2))).mul_(freqs)
            if self.cast:
                qk = qk.to(wd)
            q, k = qk[:, :, 0], qk[:, :, 1]
            v = o[..., 2 * dim: 3 * dim].reshape(bsz, seq_len, heads, head_dim)
            k = torch.cat([k, ctx_k[li].expand(bsz, -1, -1, -1)], dim=1)
            v = torch.cat([v, ctx_v[li].expand(bsz, -1, -1, -1)], dim=1)
            if mask is None or not self.cudnn_attention:
                y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                                   attn_mask=mask).transpose(1, 2)
            else:
                y = masked_sdpa(q, k, v, mask)
            y = y.reshape(bsz, seq_len, dim) * torch.sigmoid(o[..., 3 * dim:])
            h_res = torch.addcmul(h_res, c[2], F.linear(y, p["w_o"]))
            h = self._adaln(h_res, p["m_eps"], c[3], c[4])
            o = F.linear(h, p["w_13"])
            y = F.silu(o[..., : p["hidden"]]) * o[..., p["hidden"]:]
            h_res = torch.addcmul(h_res, c[5], F.linear(y, p["w_2"]))
            if return_blocks:
                outs.append(h_res)
        out = self.out_proj(self.out_norm(h_res))
        if return_blocks:
            return out, outs
        return out


class FastSampler:
    def __init__(self, model, *, precision: str = "fp32", num_steps: int = 40, cfg_min_t: float = 0.5,
                 cfg_max_t: float = 1.0, scale_text: float = 3.0, scale_speaker: float = 5.0,
                 graph_text: bool = True, use_graph: bool = True):
        if precision not in PRECISIONS:
            raise ValueError(f"precision は {PRECISIONS} のどれか: {precision!r}")
        if torch.backends.cuda.matmul.allow_tf32:
            raise RuntimeError("allow_tf32 が有効なまま作ろうとした。エンジンは DiT の区間でだけ自分で有効にする。")
        flow = str(model.cfg.flow_parameterization).strip().lower()
        self.meanflow = flow == "meanflow"
        if flow not in ("rf_velocity", "meanflow") or (model.delta_cond_module is not None) != self.meanflow:
            raise ValueError(f"RF 版と MF 版だけを扱う（flow_parameterization={flow!r}）。")
        # 扱わない要求で呼ぶ公式の関数（runtime に差し込む前の元の関数）。
        self.official = sample_euler_meanflow if self.meanflow else sample_euler_rf_cfg
        self.model = model
        make_text_backbone_capturable(model)
        self.precision = precision
        self.num_steps = int(num_steps)
        self.cfg_min_t, self.cfg_max_t = float(cfg_min_t), float(cfg_max_t)
        self.scales = (float(scale_text), float(scale_speaker))
        self.graph_text = graph_text
        self.use_graph = use_graph
        self.device = model.device
        self.heads = model.blocks[0].attention.heads
        self.head_dim = model.blocks[0].attention.head_dim
        # 行列積と attention の型。bf16 のときは、ブロックの重みを BF16 に複製する（元の fp32 の重みは残す）。
        self.wdtype = torch.bfloat16 if precision == "bf16" else model.dtype
        self.voices: list[Voice] = []
        self.graphs: dict[tuple, dict] = {}
        self.pool = torch.cuda.graph_pool_handle() if use_graph else None
        self.capture_sec: list[float] = []
        self.fallbacks = 0
        with torch.inference_mode():
            self._precompute()

    # ---- 読み込みのときの前計算（fp32、TF32 は無効） ----

    def _precompute(self) -> None:
        model, n = self.model, self.num_steps
        if self.meanflow:
            # 公式の sample_euler_meanflow と同じ時刻の表（GPU の fp32 の linspace(1, 0)）。CFG は使わない。
            self.t_schedule = torch.linspace(1.0, 0.0, n + 1, device=self.device, dtype=torch.float32)
            self.use_cfg = [False] * n
        else:
            # 公式と同じ時刻の表（GPU の linspace、(1 − u) × 0.999）。
            u = torch.linspace(0.0, 1.0, n + 1, device=self.device)
            self.t_schedule = (1.0 - u) * 0.999
            ts = self.t_schedule.cpu()
            self.use_cfg = [self.cfg_min_t <= float(ts[i]) <= self.cfg_max_t for i in range(n)]
        # x の更新の幅。公式は fp32 の 0 次元のテンソル (t_next − t) を掛ける。同じ値を Python の数にしておく。
        self.dt = [float((self.t_schedule[i + 1] - self.t_schedule[i]).item()) for i in range(n)]
        # AdaLN の係数。公式の LowRankAdaLN.forward の前半と同じ計算を、ステップごとに 1 回だけ行う。
        self.adaln: list[list[tuple[torch.Tensor, ...]]] = []
        for i in range(n):
            t = self.t_schedule[i].reshape(1).to(model.dtype)
            t_embed = M.get_timestep_embedding(t, model.cfg.timestep_embed_dim).to(dtype=model.dtype)
            cond = model.cond_module(t_embed)
            if self.meanflow:  # 公式の forward_with_encoded_conditions と同じく、区間の長さの埋め込みを足す
                delta = (self.t_schedule[i] - self.t_schedule[i + 1]).reshape(1)
                d_embed = M.get_timestep_embedding(delta, model.cfg.timestep_embed_dim).to(dtype=model.dtype)
                cond = cond + model.delta_cond_module(d_embed)
            cond = cond[:, None, :]
            per_layer = []
            for block in model.blocks:
                coeffs = []
                for ada in (block.attention_adaln, block.mlp_adaln):
                    shift, scale, gate = cond.chunk(3, dim=-1)
                    shift = ada.shift_up(ada.shift_down(F.silu(shift))) + shift
                    scale = ada.scale_up(ada.scale_down(F.silu(scale))) + scale
                    gate = ada.gate_up(ada.gate_down(F.silu(gate))) + gate
                    coeffs += [(1.0 + scale).contiguous(), shift.contiguous(), torch.tanh(gate).contiguous()]
                per_layer.append(tuple(coeffs))
            self.adaln.append(per_layer)
        # (ステップ, 層, 6, 1, 1, D)。6 個は attention の (1 + scale, shift, tanh(gate)) と MLP の同じ 3 つ。
        self.coef = torch.stack([torch.stack([torch.stack(c) for c in per_layer]) for per_layer in self.adaln]).contiguous()
        # RoPE の表（公式と同じ複素数の表）。最大の長さまで先に作る。形は (1, 長さ, 1, 1, head_dim / 2)。
        self.freqs = model._rope_freqs(2048, self.device)[None, :, None, None, :].contiguous()
        self.core = DitCore(model, self.wdtype)
        # 条件の K・V の射影の重み（行列積の型）と、K の正規化。
        self.ctx_weights = []
        for block in model.blocks:
            att = block.attention
            self.ctx_weights.append({
                "text": torch.cat([att.wk_text.weight, att.wv_text.weight]).to(self.wdtype).contiguous(),
                "speaker": torch.cat([att.wk_speaker.weight, att.wv_speaker.weight]).to(self.wdtype).contiguous(),
                "k_norm": att.k_norm,
            })

    # ---- 声 ----

    def _project_kv(self, state: torch.Tensor, li: int, which: str) -> tuple[torch.Tensor, torch.Tensor]:
        """条件の状態 (1, S, 次元) から、層 li の K（正規化あり）と V を作る。公式の project_context_kv と同じ。"""
        p = self.ctx_weights[li]
        dim = self.heads * self.head_dim
        o = F.linear(state, p[which])
        k = p["k_norm"](o[..., :dim].reshape(1, -1, self.heads, self.head_dim))
        v = o[..., dim:].reshape(1, -1, self.heads, self.head_dim)
        return k, v

    def make_voice(self, speaker_state: torch.Tensor, speaker_mask: torch.Tensor,
                   ref_latent: torch.Tensor | None = None) -> Voice:
        """話者の状態（公式の encode_conditions の出力）から、各層の K・V を作る。"""
        assert speaker_state.shape[0] == 1
        state = speaker_state[:, speaker_mask[0]].contiguous()  # 有効なトークンだけ
        kv = []
        with matmul_tf32(self.precision == "tf32"):
            s = state.to(self.wdtype)
            for li in range(len(self.ctx_weights)):
                k, v = self._project_kv(s, li, "speaker")
                kv.append((k.contiguous(), v.contiguous()))
        return Voice(ref_latent, state, kv)

    def voice_for(self, ref_latent: torch.Tensor, ref_mask: torch.Tensor) -> Voice:
        """参照音声の潜在変数に対する声。中身が同じなら、前に計算したものを返す。"""
        for v in self.voices:
            if v.ref_latent is not None and v.ref_latent.shape == ref_latent.shape and torch.equal(v.ref_latent, ref_latent):
                return v
        model = self.model
        with matmul_tf32(False):  # 話者のエンコーダーは公式と同じ fp32
            lat, mask = M.patch_sequence_with_mask(seq=ref_latent, mask=ref_mask, patch_size=model.cfg.speaker_patch_size)
            state = model.speaker_norm(model.speaker_encoder(lat, mask))
            state, mask = model._prepend_masked_mean_token(state, mask)
        voice = self.make_voice(state, mask, ref_latent.clone())
        self.voices.append(voice)
        return voice

    # ---- 計算の部品（記録するグラフと、ステップごとの確かめの両方がこれを使う） ----

    def encode_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        """有効なトークンだけの文を、公式の ModernBERT と射影と正規化に通す。常に fp32、TF32 なし。"""
        model = self.model
        mask = torch.ones_like(text_ids, dtype=torch.bool)
        with matmul_tf32(False):
            if model.pretrained_text_backbone is None:
                state = model.text_encoder(text_ids, mask)
            else:
                state = model.text_encoder(model.pretrained_text_backbone, text_ids, mask)
            return model.text_norm(state)

    def build_context(self, text_state: torch.Tensor, spk_kv: list[tuple[torch.Tensor, torch.Tensor]]):
        """各層の条件の K・V（文、話者の順につないだもの）。戻り値は (K, V)。形は (層, 1, 文 + 話者, heads, head_dim)。"""
        ks, vs = [], []
        s = text_state.to(self.wdtype)
        for li, (k_spk, v_spk) in enumerate(spk_kv):
            k_text, v_text = self._project_kv(s, li, "text")
            ks += [k_text, k_spk]
            vs += [v_text, v_spk]
        n_layers, total = len(spk_kv), text_state.shape[1] + spk_kv[0][0].shape[1]
        k = torch.cat(ks, dim=1).reshape(n_layers, 1, total, self.heads, self.head_dim)
        v = torch.cat(vs, dim=1).reshape(n_layers, 1, total, self.heads, self.head_dim)
        return k, v

    def cfg_mask(self, latent_len: int, n_text: int, n_spk: int) -> torch.Tensor:
        """CFG の 3 つ分のキーのマスク。並びは公式と同じ: cond、文を外したもの、話者を外したもの。"""
        m = torch.ones((3, 1, 1, latent_len + n_text + n_spk), dtype=torch.bool, device=self.device)
        m[1, :, :, latent_len:latent_len + n_text] = False
        m[2, :, :, latent_len + n_text:] = False
        return m

    def forward(self, x: torch.Tensor, step: int, ctx, mask: torch.Tensor | None, trace: dict | None = None) -> torch.Tensor:
        """DiT を 1 回通す。x: (B, L, 32)。mask が None なら、すべてのキーを見る。"""
        freqs = self.freqs[:, : x.shape[1]]
        if trace is None:
            return self.core(x, self.coef[step], ctx[0], ctx[1], mask, freqs)
        out, blocks = self.core(x, self.coef[step], ctx[0], ctx[1], mask, freqs, True)
        for li, h in enumerate(blocks):
            trace[f"block{li:02d}_out"] = h
        return out

    def step(self, x: torch.Tensor, i: int, ctx, mask3: torch.Tensor):
        """1 ステップ。戻り値は (次の x, CFG を合成した後の v, モデルの出力そのまま)。"""
        if self.use_cfg[i]:
            raw = self.forward(x.expand(3, -1, -1), i, ctx, mask3)
            c0, c1, c2 = raw[0:1], raw[1:2], raw[2:3]
            v = c0 + self.scales[0] * (c0 - c1)
            v = v + self.scales[1] * (c0 - c2)
        else:
            raw = self.forward(x, i, ctx, None)
            v = raw
        return x + v * self.dt[i], v, raw

    def _loop(self, st: dict) -> None:
        """記録する本体。st の静的なテンソルだけを読み書きする。"""
        text_state = self.encode_text(st["text_ids"]) if self.graph_text else st["text_state"]
        st["text_state_out"] = text_state
        with matmul_tf32(self.precision == "tf32"):
            ctx = self.build_context(text_state, st["spk_kv"])
            x = st["x"]
            for i in range(self.num_steps):
                x, _, _ = self.step(x, i, ctx, st["mask3"])
        st["out"] = x

    # ---- 呼び出し口 ----

    def _supported(self, kw: dict) -> bool:
        if self.meanflow:
            return self._supported_meanflow(kw)
        return (
            kw.get("cfg_guidance_mode") == "independent"
            and kw.get("latent_mask") is None
            and kw.get("truncation_factor") is None and kw.get("rescale_k") is None
            and kw.get("speaker_kv_scale") is None and kw.get("t_schedule_mode", "linear") == "linear"
            and kw.get("speaker_uncond_mode") == "mask" and kw.get("cfg_scale") is None
            and kw.get("speaker_state_override") is None
            and kw["text_input_ids"].shape[0] == 1 and kw.get("ref_latent") is not None
            and int(kw.get("num_steps", 40)) == self.num_steps
            and float(kw["cfg_min_t"]) == self.cfg_min_t and float(kw["cfg_max_t"]) == self.cfg_max_t
            and (float(kw["cfg_scale_text"]), float(kw["cfg_scale_speaker"])) == self.scales
            and (kw.get("caption_mask") is None or not bool(kw["caption_mask"].any()))
            and bool(kw["ref_mask"].any())
        )

    def _supported_meanflow(self, kw: dict) -> bool:
        """MF 版の要求（sample_euler_meanflow の引数）のうち、エンジンが扱うもの。CFG の引数は MF 版にはない。"""
        return (
            kw.get("latent_mask") is None
            and kw.get("speaker_state_override") is None and kw.get("speaker_mask_override") is None
            and kw.get("speaker_uncond_mode", "mask") == "mask"
            and kw["text_input_ids"].shape[0] == 1 and kw.get("ref_latent") is not None
            and int(kw.get("num_steps", 4)) == self.num_steps
            and (kw.get("caption_mask") is None or not bool(kw["caption_mask"].any()))
            and bool(kw["ref_mask"].any())
        )

    def sample(self, **kw) -> torch.Tensor:
        """公式の sample_euler_rf_cfg（MF 版は sample_euler_meanflow）と同じ引数で呼べる。"""
        if kw.get("model") is not self.model or not self._supported(kw):
            self.fallbacks += 1
            return self.official(**kw)
        model = self.model
        seq_len = int(kw["sequence_length"])
        if kw.get("initial_noise") is not None:
            x0 = kw["initial_noise"].to(device=self.device, dtype=model.dtype)
        else:
            x0, _, _ = draw_initial_noise(int(kw.get("seed", 0)), (1, seq_len, model.cfg.patched_latent_dim),
                                          device=self.device, dtype=model.dtype)
        tmask = kw["text_mask"]
        n_text = int(tmask.sum())
        assert n_text > 0 and bool(tmask[:, :n_text].all()), "文のマスクが前詰めでない"
        text_ids = kw["text_input_ids"][:, :n_text]
        voice = self.voice_for(kw["ref_latent"], kw["ref_mask"])
        return self.run(x0, text_ids, voice)

    def run(self, x0: torch.Tensor, text_ids: torch.Tensor, voice: Voice) -> torch.Tensor:
        seq_len, n_text = int(x0.shape[1]), int(text_ids.shape[1])
        if not self.use_graph:
            st = {"x": x0, "text_ids": text_ids, "spk_kv": voice.kv, "mask3": self.cfg_mask(seq_len, n_text, voice.tokens)}
            if not self.graph_text:
                st["text_state"] = self.encode_text(text_ids)
            self._loop(st)
            self.last_text_state = st["text_state_out"]
            return st["out"]
        key = (seq_len, n_text, voice.tokens)
        g = self.graphs.get(key)
        if g is None:
            t0 = time.perf_counter()
            st = {"x": x0.clone(), "text_ids": text_ids.clone(),
                  "spk_kv": [(k.clone(), v.clone()) for k, v in voice.kv],
                  "mask3": self.cfg_mask(seq_len, n_text, voice.tokens)}
            if not self.graph_text:
                st["text_state"] = self.encode_text(text_ids).clone()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._loop(st)  # 温め（同じ入力で一度通す）
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                self._loop(st)
            torch.cuda.synchronize()
            g = {"graph": graph, "st": st, "voice": voice}
            self.graphs[key] = g
            self.capture_sec.append(time.perf_counter() - t0)
        st = g["st"]
        if g["voice"] is not voice:  # 別の声が来たら、話者の K・V を入れ替える
            for (dk, dv), (sk, sv) in zip(st["spk_kv"], voice.kv):
                dk.copy_(sk)
                dv.copy_(sv)
            g["voice"] = voice
        st["x"].copy_(x0)
        if self.graph_text:
            st["text_ids"].copy_(text_ids)
        else:
            st["text_state"].copy_(self.encode_text(text_ids))
        g["graph"].replay()
        self.last_text_state = st["text_state_out"]
        return st["out"].clone()


def runtime_sampler_name(model) -> str:
    """runtime がこのモデルで呼ぶサンプラーの名前（inference_runtime の中の名前）。"""
    return "sample_euler_meanflow" if str(model.cfg.flow_parameterization).strip().lower() == "meanflow" else "sample_euler_rf_cfg"


def patch_runtime(sampler) -> None:
    """runtime のサンプラーを、エンジンの sample に置き換える（RF 版と MF 版で置き換える関数が違う）。"""
    import irodori_tts.inference_runtime as IR

    setattr(IR, runtime_sampler_name(sampler.model), sampler.sample)
    patch_tail()


def patch_tail() -> None:
    """文末を探す関数を、同じ番号を返す速いものに差し替える（engine/tail.py）。
    環境変数 SOKUTOU_OFFICIAL_TAIL=1 なら、公式の関数のままにする。"""
    if os.environ.get("SOKUTOU_OFFICIAL_TAIL", "0") != "1":
        from engine.tail import install_fast_tail

        install_fast_tail()


def install(rt, *, precision: str = "fp32", **kw) -> FastSampler:
    """公式の runtime にエンジンを差し込む。復号・電子透かし・後処理は公式のまま。"""
    sampler = FastSampler(rt.model, precision=precision, **kw)
    patch_runtime(sampler)
    return sampler
