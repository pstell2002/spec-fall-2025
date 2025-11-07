# src/ex.py
import argparse
import asyncio
import json
import os
import time

import ray
import torch
from datasets import load_dataset
from tqdm import tqdm
from score_math500 import score_file

# Stop strings and explicit "I'm done" token
STOP_SENTINEL = "</w>"
FINISH_TOKEN = "<END>"  # ask the model to emit this when the solution is done


# ---------- Ray Actor (one GPU per actor) ----------
@ray.remote(num_gpus=1)
class RayAsyncTextEngine:
    def __init__(self, model_name: str, gpu_mem_util: float,
                 dtype: str, max_model_len: int, trust_remote_code: bool):
        # Local imports inside the actor
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        from transformers import AutoTokenizer

        print(f"[{model_name}] Initializing on GPU...")
        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_mem_util,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            max_model_len=max_model_len,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model_name = model_name
        print(f"[{model_name}] Ready!")

    def _format_chat(self, problem: str, solution_so_far: str) -> str:
        """
        Use the model's chat template so Instruct models follow directions.
        We instruct it to output exactly ONE next word, then the sentinel </w>,
        or <END> when the solution is finished.
        """

        system = (f"You are a careful math solver. Answer step by step and output the final answer within \\boxed{{}} and then output {FINISH_TOKEN}.")

        user = (
            f"Problem:\n{problem}\n\n"
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt = prompt + solution_so_far  # continue from the solution so far
        # print(prompt)
        # raise NotImplementedError
        return prompt

    async def next_word(self, problem: str, solution_so_far: str,
                        temperature: float, top_p: float,
                        max_tokens_word: int, seed: int) -> str:
        from vllm import SamplingParams

        prompt = self._format_chat(problem, solution_so_far)

        # Stop ONLY on the private sentinel; we add the space ourselves to advance context.
        sp = SamplingParams(
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            max_tokens=max_tokens_word,     # per-step safety cap
            # stop=[STOP_SENTINEL],
            seed=seed,
        )

        final = None
        async for out in self.engine.generate(prompt, sp, request_id=f"{self.model_name}_{seed}"):
            final = out

        if final is None or not final.outputs:
            return ""
    
        piece = final.outputs[0].text or ""
        piece = piece.strip()


        # If model claims it's finished, it should have output <END> first
        if piece.strip() == FINISH_TOKEN:
             return FINISH_TOKEN

        # Reduce to a single "word-like" token for safety
        if piece:
            piece = piece.split()[0]
            return piece
        if not piece:
            return ""  


# ---------- Runner ----------
async def run(args):
    # Initialize Ray so it can see the GPUs srun reserved
    if not ray.is_initialized():
        ray.init(num_gpus=torch.cuda.device_count())
    print(f"Ray initialized with {torch.cuda.device_count()} GPUs visible")

    # Resolve output path (optionally timestamped)
    out_path = args.output
    if args.append_timestamp:
        root, ext = os.path.splitext(out_path)
        out_path = f"{root}_{int(time.time())}{ext or '.jsonl'}"
    compact_path = None
    if args.also_compact:
        os.makedirs(args.compact_dir, exist_ok=True)
        compact_path = os.path.join(args.compact_dir, os.path.basename(out_path))

    # Spin up two actors (one per GPU)
    print("Loading Model A...")
    eng_a = RayAsyncTextEngine.remote(
        args.model_a, args.gpu_mem_util, args.dtype, args.max_model_len, args.trust_remote_code
    )

    print("Loading Model B...")
    eng_b = RayAsyncTextEngine.remote(
        args.model_b, args.gpu_mem_util, args.dtype, args.max_model_len, args.trust_remote_code
    )

    # Give engines a moment to fully warm up
    await asyncio.sleep(3)
    print("Both models loaded!")

    # Dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    ds = ds.select(range(min(args.max_samples, len(ds))))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    written = 0

    # Distinct base seeds so models don’t mirror each other
    base_seed_a = args.seed
    base_seed_b = args.seed + 100_000

    with open(out_path, "w", encoding="utf-8") as out_f, \
     (open(compact_path, "w", encoding="utf-8") if compact_path else open(os.devnull, "w")) as out_fc:
        for i in tqdm(range(len(ds)), desc="Processing"):
            row = ds[i]
            problem = row["problem"]

            # Two buffers:
            #  - words_context: list of accepted words (for stats/anti-loop)
            #  - solution_buffer: the exact string fed back to models (always ends with a space after each word)
            #  - turns: saved for tracking, with who=1/2 and piece text
            words_context = []
            solution_buffer = ""   # <- IMPORTANT: we manage spaces here
            turns = []
            turn = 0
            dup_streak = 0
            prev_piece = None
            max_context_chars = 4000

            def recent_count(piece: str, window: int = 30) -> int:
                if not piece:
                    return 0
                recent = words_context[-window:]
                return sum(1 for w in recent if w == piece)

            # cap by words and rough char budget
            while len(words_context) < args.max_words and len(solution_buffer) < max_context_chars:
                # alternate engines: Model 1 on even turns, Model 2 on odd turns
                who = 1 if (turn % 2 == 0) else 2
                actor = eng_a if who == 1 else eng_b
                step_seed = (base_seed_a if who == 1 else base_seed_b) + turn

                ref = actor.next_word.remote(
                    problem, solution_buffer,   # feed buffer with trailing spaces
                    args.temperature, args.top_p,
                    args.max_tokens_per_word, step_seed
                )
                piece = ray.get(ref).strip()

                # Finish condition
                if piece == FINISH_TOKEN:
                    # don't add <END> to the context, just stop
                    break

                # Accept the piece
                words_context.append(piece)
                turns.append({"who": who, "piece": piece})
                # CRITICAL: add word + trailing space for the next turn’s context
                solution_buffer = solution_buffer + " " + piece
                prev_piece = piece
                turn += 1

            # Human-readable combined string with 1./2. prefixes for tracking
            generation_display = " ".join(
                (f"{t['piece']}" for t in turns)
            ).strip()

            # write one JSONL row per problem
            obj = {
                "idx": i,
                "unique_id": row.get("unique_id"),
                "subject": row.get("subject"),
                "level": row.get("level"),
                "problem": problem,
                "gold_answer": row["answer"],
                "model": f"ALT_WORD({args.model_a} ⟷ {args.model_b})",
                "prompt": "chat_template_one_word_turns",
                "generation": generation_display,
                "turns": turns,  # structured trail of who said what
                "sampling": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_words": args.max_words,
                    "max_tokens_per_word": args.max_tokens_per_word,
                    "seed": args.seed,
                    "finish_token": FINISH_TOKEN,
                },
            }
            pretty_line = json.dumps(obj, ensure_ascii=False, indent=4)
            out_f.write(pretty_line + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())

            # Compact twin (jq -c equivalent): separators remove spaces, no indent
            if compact_path:
                out_fc.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
                out_fc.flush()
                os.fsync(out_fc.fileno())
            written += 1

    print(f"Wrote {written} items -> {out_path}")

    try:
        # Ensure PYTHONPATH includes src/ when you run this file, e.g.:
        # PYTHONPATH=src:$PYTHONPATH python src/ex.py ...
        target_path = compact_path if compact_path else out_path
        acc, lo, hi, correct, total, csv_path = score_file(
            preds_path=target_path,
            csv_out=None,           # None -> auto name
            float_precision=6,
            prefer_extracted=True,  # use your \boxed extractor
            conf_level=0.95,
            env_debug=False,
        )
        print(f"[postrun] Accuracy {correct}/{total} = {acc:.3%} "
            f"(95% CI: {lo:.3%}–{hi:.3%}); CSV: {csv_path}")
    except Exception as e:
        print(f"[postrun] scoring failed: {e}")
    ray.shutdown()


# ---------- CLI ----------
def main():
    if not torch.cuda.is_available():
        raise SystemExit("No GPU detected.")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_a", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--model_b", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-samples", dest="max_samples", type=int, default=100)
    ap.add_argument("--max-words", dest="max_words", type=int, default=500)
    ap.add_argument("--max-tokens-per-word", dest="max_tokens_per_word", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", dest="top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dtype", type=str, default="float16")
    ap.add_argument("--output", type=str, default="outputs/alt_vllm_ray.jsonl")
    ap.add_argument("--append_timestamp", action="store_true", default=True,
                    help="Append UNIX timestamp to output filename.")
    ap.add_argument("--trust_remote_code", action="store_true", default=True)
    ap.add_argument("--max-model-len", dest="max_model_len", type=int, default=4096,
                    help="Target context budget; enforced by vLLM engine.")
    ap.add_argument("--gpu-mem-util", dest="gpu_mem_util", type=float, default=0.90)
    ap.add_argument("--also-compact", action="store_true", default=True, help="Also emit a compact one-line-per-object JSONL alongside the pretty file.")
    ap.add_argument("--compact-dir", type=str, default="outputs/compact", help="Directory to write the compact JSONL copy.")

    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
