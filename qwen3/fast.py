"""Qwen3-TTS（12Hz、CustomVoice）を速く動かすエンジンの最初の版。公式のモジュール（qwen-tts 0.1.1）をそのまま使う。

公式の実装（Qwen3TTSForConditionalGeneration.generate）は、1 フレームごとに次を Hugging Face の generate で動かす。
  1. code predictor（5 層）に [言語モデルの最後の隠れ状態, 第 0 のコードの埋め込み] を入れ、第 1〜15 のコードを 1 つずつ作る
  2. 16 個のコードの埋め込みの和（と tts_pad の埋め込み）を言語モデル（28 層）に 1 トークン入れ、次の第 0 のコードを作る
generate の中では小さなカーネルが 1 つずつ起動され、1 フレーム（0.08 秒の音声）に約 0.12 秒かかる。

このエンジンは、言語モデルと code predictor の KV キャッシュを決まった大きさ（transformers の StaticCache）にし、
1 フレーム分（code predictor の 15 回、言語モデルの 1 回、そのあいだのサンプリング）を 1 つの CUDA Graph に記録して再生する。
プロンプト（話者・言語・文の埋め込み）は公式の generate に作らせ、言語モデルに渡す直前で取り出す。
サンプリングは Hugging Face の generate と同じ処理（repetition_penalty、suppress_tokens、min_new_tokens、temperature、
top_k、softmax、torch.multinomial）を、同じ順に torch の演算で書いたもの。top_p は 1.0（既定）だけを受け付ける。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers import StaticCache


class _Captured(Exception):
    def __init__(self, kwargs):
        self.kwargs = kwargs


@dataclass
class Sampling:
    """公式の generate（Qwen3TTSForConditionalGeneration.generate）が言語モデルの generate に渡す設定。"""
    do_sample: bool = True
    temperature: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05
    min_new_tokens: int = 2
    max_new_tokens: int = 2048
    sub_do_sample: bool = True
    sub_temperature: float = 0.9
    sub_top_k: int = 50

    @classmethod
    def from_kwargs(cls, kw: dict) -> "Sampling":
        """取り出した引数から作る。ここで扱わない設定（top_p < 1 など）が来たら止める。"""
        for k in ("top_p", "subtalker_top_p"):
            if kw.get(k, 1.0) != 1.0:
                raise ValueError(f"{k}={kw[k]} は扱わない（1.0 だけ）")
        return cls(do_sample=bool(kw["do_sample"]), temperature=float(kw["temperature"]), top_k=int(kw["top_k"]),
                   repetition_penalty=float(kw["repetition_penalty"]), min_new_tokens=int(kw["min_new_tokens"]),
                   max_new_tokens=int(kw["max_new_tokens"]), sub_do_sample=bool(kw["subtalker_dosample"]),
                   sub_temperature=float(kw["subtalker_temperature"]), sub_top_k=int(kw["subtalker_top_k"]))


def _multinomial(probs: torch.Tensor) -> torch.Tensor:
    """torch.multinomial(probs, 1) と同じ計算（ATen の 1 個だけ引くときの近道: argmax(p / q)、q ~ Exp(1)）。
    torch.multinomial は確率の検査で .item() を呼ぶので CUDA Graph に記録できない。乱数の使い方は同じ。"""
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True)


def _sample(scores: torch.Tensor, do_sample: bool, temperature: float, top_k: int) -> torch.Tensor:
    """Hugging Face の _sample の最後（warper と softmax と multinomial、または argmax）。scores は float32、(1, V)。"""
    if not do_sample:
        return torch.argmax(scores, dim=-1, keepdim=True)
    if temperature != 1.0:
        scores = scores / temperature
    if top_k:
        kth = torch.topk(scores, min(top_k, scores.shape[-1]))[0][..., -1, None]
        scores = scores.masked_fill(scores < kth, -float("inf"))
    probs = torch.softmax(scores, dim=-1)
    return _multinomial(probs)


class Qwen3Fast:
    def __init__(self, tts, sampling: Sampling | None = None, max_cache: int = 1024, compile: bool = False,
                 graphs: bool = True):
        self.tts = tts
        self.model = tts.model
        self.talker = self.model.talker
        self.cp = self.talker.code_predictor
        cfg = self.model.config.talker_config
        self.cfg = cfg
        self.s = sampling or Sampling()
        self.G = cfg.num_code_groups
        self.V = cfg.vocab_size
        self.eos = cfg.codec_eos_token_id
        self.max_cache = max_cache
        dev, dt = self.talker.device, self.talker.dtype
        self.device, self.dtype = dev, dt
        H = cfg.hidden_size
        self.tcache = StaticCache(config=self.talker.config, max_cache_len=max_cache)
        self.ccache = StaticCache(config=self.cp.config, max_cache_len=self.G)
        # 第 0 のコードで抑える番号（公式の generate の suppress_tokens: 語彙の最後の 1024 個のうち EOS 以外）
        sup = torch.zeros((1, self.V), dtype=torch.bool, device=dev)
        sup[:, self.V - 1024:] = True
        sup[:, self.eos] = False
        self.suppress = sup
        # グラフの入出力（同じ場所を使い続ける）
        self.id0 = torch.zeros((1, 1), dtype=torch.long, device=dev)
        self.past_hidden = torch.zeros((1, 1, H), dtype=dt, device=dev)
        self.pad_embed = torch.zeros((1, 1, H), dtype=dt, device=dev)
        self.pos = torch.zeros((1,), dtype=torch.long, device=dev)
        self.seen = torch.zeros((1, self.V), dtype=torch.bool, device=dev)
        self.ngen = torch.zeros((1,), dtype=torch.long, device=dev)
        self.codes = torch.zeros((1, self.G), dtype=torch.long, device=dev)
        self.next_id0 = torch.zeros((1, 1), dtype=torch.long, device=dev)
        self.cp_pos = [torch.arange(2, device=dev)] + [torch.tensor([i], device=dev) for i in range(2, self.G)]
        self.graph: torch.cuda.CUDAGraph | None = None
        self.stats: dict = {}
        # compile=True: 1 フレーム分を torch.compile（inductor）で変換してから CUDA Graph に記録する。
        # 正規化・RoPE・活性化・足し算などの小さな演算が少数のカーネルにまとまる。乱数は torch のもの
        # （fallback_random）を使い、公式と同じ数を引く。計算の順が変わるので、bfloat16 の値は少し変わる
        self.compile = compile
        # graphs=True: 最初の入力と復号も CUDA Graph にする（_prefill、decode）。False なら HF のモジュールを Python から動かす
        self.graphs = graphs and self.device.type == "cuda"
        self._pf: dict = {}
        self._dec: dict = {}
        self._pool = torch.cuda.graph_pool_handle() if self.graphs else None
        self._frame_fn = self._frame
        if compile:
            import torch._inductor.config as ic

            ic.fallback_random = True
            self._frame_fn = torch.compile(self._frame, fullgraph=True, dynamic=False)

    # --- プロンプト ---------------------------------------------------------------------------------------------
    def prompt(self, text: str, speaker: str, language: str, **gen) -> dict:
        """公式の generate_custom_voice を、言語モデルの generate を呼ぶ直前まで動かし、その引数を返す。
        gen は generate_custom_voice の生成の引数（do_sample など）。"""
        orig = self.talker.generate

        def capture(**kw):
            raise _Captured(kw)

        self.talker.generate = capture
        try:
            self.tts.generate_custom_voice(text=text, speaker=speaker, language=language, **gen)
        except _Captured as c:
            kw = c.kwargs
        finally:
            del self.talker.generate
        assert self.talker.generate.__func__ is orig.__func__
        tth, pad = kw["trailing_text_hidden"], kw["tts_pad_embed"]
        # 文をプロンプトに入れる形（non_streaming_mode=True）では、どのステップでも tts_pad の埋め込みを足す
        assert tth.shape[1] == 1 and torch.equal(tth, pad), "non_streaming_mode=True の形だけを扱う"
        assert bool(kw["attention_mask"].all()), "1 文ずつだけを扱う（詰め物なし）"
        return kw

    # --- 1 フレーム ---------------------------------------------------------------------------------------------
    def _talker_scores(self, hidden_last: torch.Tensor) -> torch.Tensor:
        """言語モデルの最後の隠れ状態から、第 0 のコードの scores（processor を通したもの、float32）を作る。"""
        return self._process(self.talker.codec_head(hidden_last)[:, -1, :].to(torch.float32))

    def _process(self, scores: torch.Tensor) -> torch.Tensor:
        """第 0 のコードの logits（float32、(1, V)）に、公式の generate と同じ processor をかける。"""
        p = self.s.repetition_penalty
        if p != 1.0:
            pen = torch.where(scores < 0, scores * p, scores / p)
            scores = torch.where(self.seen, pen, scores)
        scores = scores.masked_fill(self.suppress, -float("inf"))
        # min_new_tokens: 作ったトークンがその数になるまで EOS を出さない
        block = (self.ngen < self.s.min_new_tokens).view(1, 1)
        eos_col = scores[:, self.eos:self.eos + 1]
        scores = torch.cat([scores[:, :self.eos], torch.where(block, torch.full_like(eos_col, -float("inf")), eos_col),
                            scores[:, self.eos + 1:]], dim=-1)
        return scores

    def _frame(self) -> None:
        """id0（第 0 のコード）と past_hidden から、このフレームの 16 個のコードと次の id0 を作る。"""
        s = self.s
        emb0 = self.talker.get_input_embeddings()(self.id0)  # (1, 1, H)
        cp_emb = self.cp.get_input_embeddings()
        x = self.cp.small_to_mtp_projection(torch.cat([self.past_hidden, emb0], dim=1))
        out = self.cp.model(inputs_embeds=x, past_key_values=self.ccache, cache_position=self.cp_pos[0], use_cache=True)
        logits = self.cp.lm_head[0](out.last_hidden_state)[:, -1, :].to(torch.float32)
        code = _sample(logits, s.sub_do_sample, s.sub_temperature, s.sub_top_k)  # (1, 1)
        codes = [self.id0, code]
        embs = [emb0, cp_emb[0](code)]
        for g in range(1, self.G - 1):
            e = self.cp.small_to_mtp_projection(cp_emb[g - 1](code))
            out = self.cp.model(inputs_embeds=e, past_key_values=self.ccache, cache_position=self.cp_pos[g], use_cache=True)
            logits = self.cp.lm_head[g](out.last_hidden_state)[:, -1, :].to(torch.float32)
            code = _sample(logits, s.sub_do_sample, s.sub_temperature, s.sub_top_k)
            codes.append(code)
            embs.append(cp_emb[g](code))
        self.codes.copy_(torch.cat(codes, dim=1))
        x = torch.cat(embs, dim=1).sum(1, keepdim=True) + self.pad_embed
        out = self.talker.model(inputs_embeds=x, past_key_values=self.tcache, cache_position=self.pos, use_cache=True)
        hidden = out.last_hidden_state
        nxt = _sample(self._talker_scores(hidden), s.do_sample, s.temperature, s.top_k)
        self.next_id0.copy_(nxt)
        self.past_hidden.copy_(hidden[:, -1:, :])
        self.seen.scatter_(1, nxt, True)
        self.ngen.add_(1)
        self.pos.add_(1)
        self.id0.copy_(nxt)

    def _after_prefill(self, hidden: torch.Tensor) -> None:
        """最初の入力（プロンプト）のあとに呼ぶ。自作のカーネルの版（qwen3/kernels.py）が、隠れ状態を写すのに使う。"""

    def capture(self) -> float:
        """1 フレーム分を CUDA Graph に記録する。記録の前に、同じ処理を別のストリームで 3 回動かしてキャッシュを用意する。"""
        t0 = time.perf_counter()
        saved = [t.clone() for t in (self.id0, self.past_hidden, self.pos, self.seen, self.ngen)]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):
                self._frame_fn()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.no_grad():
            self._frame_fn()
        for t, v in zip((self.id0, self.past_hidden, self.pos, self.seen, self.ngen), saved):
            t.copy_(v)
        self.graph = g
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    # --- 1 文 ---------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def generate_codes(self, kw: dict, max_frames: int | None = None) -> torch.Tensor:
        """プロンプトから、フレームごとの 16 個のコード（frames, 16）を作る。公式の generate と同じ順に乱数を使う
        （最初に第 0 のコードを 1 回、そのあとフレームごとに code predictor の 15 回と第 0 のコードの 1 回）。
        max_frames を渡すと、終わりのトークンが出なくてもその数で止める（サーバーが、話し終えずに続く失敗を早く止めるため。
        止めたかは stats["frame_limited"]）。"""
        step = self.graph.replay if self.graph is not None else self._frame  # 記録していなければ 1 つずつ動かす
        if Sampling.from_kwargs(kw) != self.s:
            raise ValueError(f"記録したときの設定 {self.s} と違う: {Sampling.from_kwargs(kw)}")
        embeds = kw["inputs_embeds"]
        L = embeds.shape[1]
        # 公式は max_new_tokens 個のトークン（最初の 1 個を含む）で止まる。フレームはその 1 個少ない数まで
        limit = min(self.s.max_new_tokens - 1, self.max_cache - L)
        if limit <= 0:
            raise ValueError(f"プロンプトが長すぎる（{L} トークン、キャッシュ {self.max_cache}）")
        cap = limit if max_frames is None else min(limit, max_frames)
        self.pad_embed.copy_(kw["tts_pad_embed"])
        self._prefill(embeds)
        frames = []
        nid = int(self.id0.item())
        while nid != self.eos and len(frames) < cap:
            step()
            frames.append(self.codes.clone())
            nid = int(self.next_id0.item())
        self.stats["prompt_tokens"] = L
        self.stats["stopped"] = nid == self.eos
        self.stats["cache_limited"] = nid != self.eos and len(frames) == limit and limit < self.s.max_new_tokens - 1
        self.stats["frame_limited"] = nid != self.eos and cap < limit and len(frames) == cap
        return torch.cat(frames, dim=0) if frames else torch.zeros((0, self.G), dtype=torch.long, device=self.device)

    def _prefill_body(self, embeds: torch.Tensor, cpos: torch.Tensor) -> None:
        """最初の入力（プロンプト全体）を言語モデルに通し、最初の第 0 のコードを引いて、フレームの状態を用意する。"""
        self.seen.zero_()
        self.ngen.zero_()
        out = self.talker.model(inputs_embeds=embeds, past_key_values=self.tcache, cache_position=cpos, use_cache=True)
        hidden = out.last_hidden_state
        first = _sample(self._talker_scores(hidden), self.s.do_sample, self.s.temperature, self.s.top_k)
        self.id0.copy_(first)
        self.past_hidden.copy_(hidden[:, -1:, :])
        self._after_prefill(hidden)
        self.seen.scatter_(1, first, True)
        self.ngen.fill_(1)
        self.pos.fill_(embeds.shape[1])

    def _prefill(self, embeds: torch.Tensor) -> None:
        """graphs が有効なら、プロンプトの長さごとに記録した CUDA Graph で _prefill_body を動かす（初めての長さは記録してから）。
        HF のモジュールを Python から動かすと、カーネルの起動だけで 1 文あたり約 40 ms かかるため。"""
        L = embeds.shape[1]
        if not self.graphs:
            self._prefill_body(embeds, torch.arange(L, device=self.device))
            return
        if L not in self._pf:
            self._pf[L] = self._capture_prefill(L)
        g, buf, _ = self._pf[L]
        buf.copy_(embeds)
        g.replay()

    def _capture_prefill(self, L: int):
        """長さ L の最初の入力を記録する。記録の前の試し実行で乱数とフレームの状態が変わらないよう、保存して戻す。"""
        t0 = time.perf_counter()
        buf = torch.zeros((1, L, self.cfg.hidden_size), dtype=self.dtype, device=self.device)
        cpos = torch.arange(L, device=self.device)
        rng = torch.cuda.get_rng_state(self.device)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(2):
                self._prefill_body(buf, cpos)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._pool), torch.no_grad():
            self._prefill_body(buf, cpos)
        torch.cuda.synchronize()
        torch.cuda.set_rng_state(rng, self.device)
        self.stats["prefill_capture_sec"] = self.stats.get("prefill_capture_sec", 0.0) + time.perf_counter() - t0
        return g, buf, cpos

    def warm_prefill(self, lengths) -> float:
        """よく使う長さの最初の入力を、先に記録しておく。かかった秒数を返す。"""
        t0 = time.perf_counter()
        for L in lengths:
            if L not in self._pf:
                self._pf[L] = self._capture_prefill(L)
        return time.perf_counter() - t0

    # --- 復号 ---------------------------------------------------------------------------------------------------
    DEC_BUCKETS = (16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128, 160, 192, 224, 256, 300, 325)

    def decode(self, codes: torch.Tensor):
        """コード（frames, 16）を 24kHz の波形（numpy の float32）にする。公式の speech_tokenizer.decode と同じ計算で、
        graphs が有効なら、コードの後ろを 0 で詰めて決まった長さ（DEC_BUCKETS）にし、長さごとに記録した CUDA Graph で
        復号してから元の長さで切る。復号器は因果的（畳み込みは左だけを詰め、注意も前だけを見る）なので、後ろを詰めても
        前の部分の計算は変わらない。300 フレームを超えるときは、公式（chunked_decode）と同じく 300 フレームずつ、
        左に 25 フレームの文脈をつけて区切る。"""
        tok = self.model.speech_tokenizer
        if not self.graphs:
            wavs, sr = tok.decode([{"audio_codes": codes}])
            return wavs[0], sr
        dec = tok.model.decoder
        up = int(dec.total_upsample)
        c = torch.clamp(codes, min=0).t().unsqueeze(0)  # (1, 16, frames)
        F = c.shape[-1]
        parts = []
        start = 0
        while start < F:  # Qwen3TTSTokenizerV2Decoder.chunked_decode と同じ区切り方
            end = min(start + 300, F)
            ctx = 25 if start - 25 > 0 else start
            seg = c[..., start - ctx:end]
            parts.append(self._decode_segment(seg)[..., ctx * up:])
            start = end
        wav = torch.cat(parts, dim=-1).view(-1)[: F * int(tok.model.decode_upsample_rate)]
        return wav.to(torch.float32).cpu().numpy(), int(tok.model.get_output_sample_rate())

    def _decode_segment(self, seg: torch.Tensor) -> torch.Tensor:
        n = seg.shape[-1]
        B = next(b for b in self.DEC_BUCKETS if b >= n)
        if B not in self._dec:
            self._dec[B] = self._capture_decode(B)
        g, buf, out = self._dec[B]
        buf.zero_()
        buf[..., :n].copy_(seg)
        g.replay()
        up = int(self.model.speech_tokenizer.model.decoder.total_upsample)
        return out[..., : n * up].clone()

    def _capture_decode(self, B: int):
        t0 = time.perf_counter()
        dec = self.model.speech_tokenizer.model.decoder
        buf = torch.zeros((1, self.G, B), dtype=torch.long, device=self.device)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.inference_mode():
            for _ in range(2):
                dec(buf)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._pool), torch.inference_mode():
            out = dec(buf)
        torch.cuda.synchronize()
        self.stats["decode_capture_sec"] = self.stats.get("decode_capture_sec", 0.0) + time.perf_counter() - t0
        return g, buf, out

    def warm_decode(self, buckets=None) -> float:
        t0 = time.perf_counter()
        for B in buckets or self.DEC_BUCKETS:
            if B not in self._dec:
                self._dec[B] = self._capture_decode(B)
        return time.perf_counter() - t0

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def synth(self, text: str, speaker: str, language: str, max_frames: int | None = None, **gen):
        t0 = time.perf_counter()
        self.stats = {}
        kw = self.prompt(text, speaker, language, **gen)
        self._sync()
        t1 = time.perf_counter()
        codes = self.generate_codes(kw, max_frames=max_frames)
        self._sync()
        t2 = time.perf_counter()
        wav, sr = self.decode(codes)
        self._sync()
        t3 = time.perf_counter()
        self.stats.update({"prompt": t1 - t0, "generate": t2 - t1, "decode": t3 - t2})
        return wav, sr, codes
