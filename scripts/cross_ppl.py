# scripts/cross_ppl.py
import argparse, json, math, os, sys, time, logging
from typing import Dict, List, Tuple

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

FINISH_TOKEN = "<END>"

# -------------------- Logging helpers --------------------
def setup_logging(level: str = "INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("cross_ppl")

# -------------------- Core utils --------------------
def unwrap_logprob(val):
    # vLLM >= 0.5 returns a Logprob object; older returns float
    return float(getattr(val, "logprob", val))

def build_formatter(tokenizer, max_hist_tokens: int):
    def truncate_solution_tokens(text: str) -> str:
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) <= max_hist_tokens:
            return text
        tail = ids[-max_hist_tokens:]
        return tokenizer.decode(tail, skip_special_tokens=True)

    def chat_history(problem: str, solution_so_far: str) -> str:
        system = (
            f"You are a helpful assistant. "
            f"Answer step by step and output the final answer within \\boxed{{}} and then output {FINISH_TOKEN}."
        )
        user = f"Problem:\n{problem}\n\n"
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt + truncate_solution_tokens(solution_so_far)
    return chat_history, truncate_solution_tokens

def score_span(logger: logging.Logger, llm: LLM, tokenizer, full_prompt: str, span_text: str) -> Tuple[float, int]:
    """
    Teacher-force `span_text` appended to `full_prompt`. Return (sum_logprob, token_count).
    """
    sp = SamplingParams(max_tokens=1, logprobs=1, prompt_logprobs=True, temperature=0.0, top_p=1.0)
    full = full_prompt + span_text
    out = llm.generate([full], sp)[0]  # sync; single item

    # Map the scored span within prompt tokens
    hist_ids = tokenizer(full_prompt, add_special_tokens=False).input_ids
    full_ids = tokenizer(full,        add_special_tokens=False).input_ids
    span_len = len(full_ids) - len(hist_ids)
    if span_len <= 0:
        return 0.0, 0

    start = len(out.prompt_token_ids) - span_len
    end   = len(out.prompt_token_ids)

    logp_sum, tok_cnt = 0.0, 0
    for i in range(start, end):
        d = out.prompt_logprobs[i]
        if d is None:
            continue
        tok_id = out.prompt_token_ids[i]

        val = d.get(tok_id)
        if val is None:
            # try by token text
            try:
                tok_text = tokenizer.convert_ids_to_tokens([tok_id])[0]
            except Exception:
                tok_text = None
            if tok_text is not None:
                val = d.get(tok_text)
        if val is None:
            # last resort: scan dict for matching shapes
            for k, v in d.items():
                if hasattr(k, "id") and k.id == tok_id: val = v; break
                if tok_text and hasattr(k, "text") and k.text == tok_text: val = v; break
                if tok_text and hasattr(v, "decoded_token") and v.decoded_token == tok_text: val = v; break
        if val is None:
            continue

        try:
            logp = unwrap_logprob(val)
        except Exception as e:
            logger.debug(f"Failed to unwrap logprob at i={i}: {e}")
            continue
        logp_sum += logp
        tok_cnt += 1

    return logp_sum, tok_cnt

def cross_ppl_for_turns(
    logger: logging.Logger,
    judge_llm: LLM,
    judge_tok,
    turns: List[Dict],
    problem: str,
    other_id: int,
    chat_history_fn,
    truncate_fn,
    every_debug_piece: bool = False,
) -> Dict[str, float]:
    sol_text = ""
    total_logp, total_tok = 0.0, 0

    for step, t in enumerate(turns):
        piece = " " + t["piece"]
        hist = chat_history_fn(problem, truncate_fn(sol_text))
        if t["who"] == other_id:
            lp, n = score_span(logger, judge_llm, judge_tok, hist, piece)
            total_logp += lp
            total_tok  += n
            if every_debug_piece:
                logger.debug(f"  step={step} who={t['who']} tok_scored={n} lp_sum={lp:.4f} piece={repr(t['piece'][:40])}")
        sol_text += piece

    if total_tok == 0:
        return {"ppl": float("nan"), "nll": float("nan"), "tok": 0}
    nll = - total_logp / total_tok
    return {"ppl": math.exp(nll), "nll": nll, "tok": total_tok}

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input JSONL from generator")
    ap.add_argument("--out", dest="outp", required=True, help="output JSONL with cross-PPL")
    ap.add_argument("--model_a", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--model_b", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--max-model-len", dest="max_model_len", type=int, default=1024)
    ap.add_argument("--gpu-mem-util", dest="gpu_mem_util", type=float, default=0.75)
    ap.add_argument("--max-hist-tokens", dest="max_hist_tokens", type=int, default=512)
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    ap.add_argument("--quantization", default=None, help="awq|gptq|bitsandbytes|None")
    ap.add_argument("--csv", dest="csv", help="optional: also write a CSV rollup")
    ap.add_argument("--limit", type=int, default=None, help="score only first N rows (debug)")
    ap.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    ap.add_argument("--every", type=int, default=10, help="log every N rows")
    ap.add_argument("--debug-pieces", action="store_true", help="DEBUG log each scored piece")
    args = ap.parse_args()

    logger = setup_logging(args.log_level)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Input sanity
    if not os.path.exists(args.inp):
        logger.error(f"Input not found: {args.inp}")
        sys.exit(2)
    total_lines = sum(1 for _ in open(args.inp, "r", encoding="utf-8"))
    logger.info(f"Input rows: {total_lines}  |  limit={args.limit}  |  out={args.outp}")

    # ---------- PASS 1: Load A, score A_on_B for all rows ----------
    t0 = time.time()
    logger.info(f"Loading tokenizer A: {args.model_a}")
    tok_a = AutoTokenizer.from_pretrained(args.model_a, trust_remote_code=args.trust_remote_code)
    logger.info("Loading model A...")
    llm_a = LLM(model=args.model_a,
                dtype=args.dtype,
                trust_remote_code=args.trust_remote_code,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_mem_util,
                quantization=args.quantization)
    logger.info(f"Model A loaded in {time.time()-t0:.2f}s")
    hist_a, trunc_a = build_formatter(tok_a, args.max_hist_tokens)

    logger.info("PASS 1: scoring A_on_B")
    rows = []
    with open(args.inp, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit is not None and i >= args.limit:
                logger.info(f"Hit --limit {args.limit}; stopping early (pass 1).")
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON at line {i+1}: {e}")
                continue
            problem = row.get("problem", "")
            turns   = row.get("turns", [])
            if not turns:
                logger.warning(f"Row {i}: no 'turns' found; skipping.")
                continue

            res_a = cross_ppl_for_turns(logger, llm_a, tok_a, turns, problem, other_id=2,
                                        chat_history_fn=hist_a, truncate_fn=trunc_a,
                                        every_debug_piece=args.debug_pieces)
            row["_tmp_A_on_B"] = res_a
            rows.append(row)
            if (i + 1) % args.every == 0:
                logger.info(f"[PASS1 {i+1}/{total_lines}] A_on_B ppl={res_a['ppl']:.3f} tok={res_a['tok']}")

    # Free A completely
    logger.info("Unloading model A and clearing CUDA cache...")
    del llm_a, tok_a
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    # ---------- PASS 2: Load B, score B_on_A, write output ----------
    t0 = time.time()
    logger.info(f"Loading tokenizer B: {args.model_b}")
    tok_b = AutoTokenizer.from_pretrained(args.model_b, trust_remote_code=args.trust_remote_code)
    logger.info("Loading model B...")
    llm_b = LLM(model=args.model_b,
                dtype=args.dtype,
                trust_remote_code=args.trust_remote_code,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_mem_util,
                quantization=args.quantization)
    logger.info(f"Model B loaded in {time.time()-t0:.2f}s")
    hist_b, trunc_b = build_formatter(tok_b, args.max_hist_tokens)

    logger.info("PASS 2: scoring B_on_A and writing output")
    written = 0
    with open(args.outp, "w", encoding="utf-8") as out_f:
        for i, row in enumerate(rows):
            if args.limit is not None and i >= args.limit:
                logger.info(f"Mirroring --limit {args.limit} in pass 2; stopping.")
                break
            problem = row.get("problem", "")
            turns   = row.get("turns", [])
            res_b = cross_ppl_for_turns(logger, llm_b, tok_b, turns, problem, other_id=1,
                                        chat_history_fn=hist_b, truncate_fn=trunc_b,
                                        every_debug_piece=args.debug_pieces)
            res_a = row.pop("_tmp_A_on_B")
            row["metrics_offline"] = {
                "cross_ppl_A_on_B": res_a["ppl"],
                "cross_nll_A_on_B": res_a["nll"],
                "cross_tok_A_on_B": res_a["tok"],
                "cross_ppl_B_on_A": res_b["ppl"],
                "cross_nll_B_on_A": res_b["nll"],
                "cross_tok_B_on_A": res_b["tok"],
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if written % args.every == 0:
                logger.info(f"[PASS2 wrote {written}] "
                            f"A_on_B ppl={res_a['ppl']:.3f} tok={res_a['tok']} | "
                            f"B_on_A ppl={res_b['ppl']:.3f} tok={res_b['tok']}")

    logger.info(f"Done. Wrote {written} rows to {args.outp}")

        # ---- Micro-average perplexity (always print) ----
    def compute_micro_avgs(out_path: str):
        total_nll_a = total_tok_a = 0
        total_nll_b = total_tok_b = 0
        with open(out_path, "r", encoding="utf-8") as rf:
            for line in rf:
                o = json.loads(line)
                m = o.get("metrics_offline", {})
                nll_a = m.get("cross_nll_A_on_B")
                tok_a = m.get("cross_tok_A_on_B", 0)
                nll_b = m.get("cross_nll_B_on_A")
                tok_b = m.get("cross_tok_B_on_A", 0)
                if nll_a is not None and math.isfinite(nll_a) and tok_a > 0:
                    total_nll_a += nll_a * tok_a
                    total_tok_a += tok_a
                if nll_b is not None and math.isfinite(nll_b) and tok_b > 0:
                    total_nll_b += nll_b * tok_b
                    total_tok_b += tok_b
        micro_ppl_a = math.exp(total_nll_a / total_tok_a) if total_tok_a > 0 else float("nan")
        micro_ppl_b = math.exp(total_nll_b / total_tok_b) if total_tok_b > 0 else float("nan")
        return micro_ppl_a, micro_ppl_b, total_tok_a, total_tok_b

    micro_ppl_a, micro_ppl_b, total_tok_a, total_tok_b = compute_micro_avgs(args.outp)
    logger.info(
        "Micro-averaged perplexity (token-weighted): "
        f"A_on_B={micro_ppl_a:.4f} over {total_tok_a} tokens | "
        f"B_on_A={micro_ppl_b:.4f} over {total_tok_b} tokens"
    )

    # CSV rollup
    if args.csv:
        import csv
        total_nll_a = 0.0
        total_tok_a = 0
        total_nll_b = 0.0
        total_tok_b = 0
        for r in rows[: (args.limit or len(rows))]:
            m = r.get("metrics_offline")
            # if pass 2 didn't run (or you change flow), we can recompute from tmp;
            # but here we wrote metrics_offline, so reload from file if needed.
        # Re-read output to be sure we aggregate what we wrote
        total_nll_a = total_tok_a = total_nll_b = total_tok_b = 0
        with open(args.outp, "r", encoding="utf-8") as rf:
            for line in rf:
                obj = json.loads(line)
                m = obj.get("metrics_offline", {})
                if math.isfinite(m.get("cross_nll_A_on_B", float("nan"))):
                    total_nll_a += m["cross_nll_A_on_B"] * m.get("cross_tok_A_on_B", 0)
                    total_tok_a += m.get("cross_tok_A_on_B", 0)
                if math.isfinite(m.get("cross_nll_B_on_A", float("nan"))):
                    total_nll_b += m["cross_nll_B_on_A"] * m.get("cross_tok_B_on_A", 0)
                    total_tok_b += m.get("cross_tok_B_on_A", 0)

        micro_ppl_a = math.exp(total_nll_a / total_tok_a) if total_tok_a > 0 else float("nan")
        micro_ppl_b = math.exp(total_nll_b / total_tok_b) if total_tok_b > 0 else float("nan")

        with open(args.csv, "w", newline="") as cf:
            w = csv.writer(cf)
            w.writerow(["micro_ppl_A_on_B", micro_ppl_a])
            w.writerow(["micro_ppl_B_on_A", micro_ppl_b])
            w.writerow(["total_tok_A_on_B", total_tok_a])
            w.writerow(["total_tok_B_on_A", total_tok_b])
        logger.info(f"Wrote rollup CSV -> {args.csv}")
        logger.info(f"Micro-averaged PPLs: A_on_B={micro_ppl_a:.4f}, B_on_A={micro_ppl_b:.4f}")

if __name__ == "__main__":
    main()
