#!/usr/bin/env python3
# score_math500.py
# Robust grading for MATH-500-style generations.

import argparse
import csv
import json
import os
import re
import sys
import unicodedata
import inspect
from typing import List, Dict, Optional, Tuple

from tqdm import tqdm

# ---- Required: math_verify must be importable ----
from math_verify import parse as mv_parse, verify as mv_verify, LatexExtractionConfig, ExprExtractionConfig

# Optional: SymPy fallback
try:
    import sympy as sp
    from sympy.parsing.latex import parse_latex as sp_parse_latex
    SYMPY_OK = True
except Exception:
    SYMPY_OK = False


# ============= Extraction helpers =============

BOXED_IN_BRACKETS_RE = re.compile(
    r"""\\\[\s*\\boxed\s*\{(?P<inner>.*?)\}\s*\\\]""",
    flags=re.DOTALL,
)
BOX_PATTERNS = [
    r"\\boxed\s*\{([^{}]|{[^{}]*})*\}",
    r"\\boxed\s*\\left\s*\((.*?)\\right\s*\)",
    r"\\boxed\s*\((.*?)\)",
]
MATH_SPAN_PATTERNS = [
    r"\$\$(.*?)\$\$",
    r"\$(.*?)\$",
    r"\\\[(.*?)\\\]",
    r"\\\((.*?)\\\)",
]
PAREN_TAIL_PATTERNS = [
    r"\(\s*[^()]*\s*\)\s*$",
]

def _last_full_match(text: str, patterns: List[str]) -> Optional[str]:
    found = None
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.DOTALL):
            found = m.group(0)
    return found

def _last_group_match(text: str, patterns: List[str]) -> Optional[str]:
    last = None
    for pat in patterns:
        matches = list(re.finditer(pat, text, flags=re.DOTALL))
        if matches:
            last = matches[-1].group(1)
    return last

def _strip_wrappers(s: str) -> str:
    s = re.sub(r"^\\boxed\s*\{(.*)\}\s*$", r"\1", s, flags=re.DOTALL)
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = re.sub(r"^\\\((.*)\\\)$", r"\1", s, flags=re.DOTALL)
    s = re.sub(r"^\\\[(.*)\\\]$", r"\1", s, flags=re.DOTALL)
    return s.strip()

def extract_answer(text: str) -> str:
    """Prefer \[ \boxed{ ... } \], then boxed variants, then last math span, then trailing (...)."""
    if not text:
        return ""
    m = BOXED_IN_BRACKETS_RE.search(text)
    if m:
        return _strip_wrappers(m.group("inner"))
    box_full = _last_full_match(text, BOX_PATTERNS)
    if box_full:
        return _strip_wrappers(box_full)
    math_inner = _last_group_match(text, MATH_SPAN_PATTERNS)
    if math_inner:
        return _strip_wrappers(math_inner)
    tail = _last_full_match(text, PAREN_TAIL_PATTERNS)
    if tail:
        return tail.strip()
    return text.strip()


# ============= Sanitization & utilities =============

ZERO_WIDTH = {
    "\u200b", "\u200c", "\u200d", "\u2060", "\ufe0e", "\ufe0f",
    "\u180e", "\u2061", "\u2062", "\u2063", "\u2064"
}

def sanitize_math(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in s if ch not in ZERO_WIDTH and unicodedata.category(ch) != "Cf")
    s = s.replace(r"\left", "").replace(r"\right", "")
    return s.strip()

def strip_text_wrappers(s: str) -> str:
    # \text{Evelyn} -> Evelyn
    m = re.fullmatch(r"\\text\{(.+)\}", s)
    return m.group(1).strip() if m else s

def looks_like_tuple(s: str) -> bool:
    return bool(re.fullmatch(r"\(\s*.+\s*,\s*.+\s*\)", s))

def split_tuple(s: str) -> Optional[Tuple[str, str]]:
    m = re.fullmatch(r"\(\s*(.+?)\s*,\s*(.+?)\s*\)", s)
    if not m:
        return None
    return m.group(1), m.group(2)

def has_degree(s: str) -> bool:
    return r"^\circ" in s or r"^\circ" in s.replace(" ", "")

def normalize_degrees(s: str) -> Optional[float]:
    # "90^\circ" -> 90.0
    m = re.fullmatch(r"\s*([+-]?\d+(?:\.\d+)?)\s*\^\\circ\s*", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


# ============= math_verify compatibility =============

# some versions don't accept numeric_precision
_MV_HAS_PREC = "numeric_precision" in inspect.signature(mv_parse).parameters

def mv_parse_compat(expr: str, cfg, float_precision: int):
    if _MV_HAS_PREC:
        return mv_parse(expr, extraction_config=cfg, numeric_precision=float_precision)
    else:
        return mv_parse(expr, extraction_config=cfg)

def parse_with_fallback(expr: str, float_precision: int) -> object:
    last_err = None
    for cfg in (LatexExtractionConfig(), ExprExtractionConfig()):
        try:
            return mv_parse_compat(expr, cfg, float_precision)
        except Exception as e:
            last_err = e
    raise last_err if last_err else RuntimeError("parse failed")

# ============= SymPy fallback comparators =============

def sympy_expr(expr: str):
    if not SYMPY_OK:
        raise RuntimeError("sympy unavailable")
    # try LaTeX parse then sympify
    e = expr.replace(r"\left", "").replace(r"\right", "")
    try:
        return sp_parse_latex(e)
    except Exception:
        return sp.sympify(e)

def sympy_eq(a: str, b: str, tol: float = 1e-9) -> bool:
    if not SYMPY_OK:
        return False
    try:
        ea = sympy_expr(a)
        eb = sympy_expr(b)
        diff = sp.simplify(ea - eb)
        if diff == 0:
            return True
        return abs(sp.N(diff)) <= tol
    except Exception:
        return False


# ============= High-level comparator =============

def compare_answers(gold: str, pred: str, float_precision: int, debug: bool = False) -> bool:
    """Layered strategy: math_verify -> tuple handling -> degrees -> text -> sympy -> strict string."""
    g = sanitize_math(gold)
    p = sanitize_math(pred)

    # 1) math_verify direct try
    try:
        gp = parse_with_fallback(g, float_precision)
        pp = parse_with_fallback(p, float_precision)
        if mv_verify(gp, pp):
            return True
    except Exception as e:
        if debug:
            print(f"[mv fail] gold={g!r} pred={p!r} err={e}", file=sys.stderr)

    # 2) ordered pair / tuple (compare componentwise with mv/sympy/string)
    if looks_like_tuple(g) and looks_like_tuple(p):
        gt = split_tuple(g); pt = split_tuple(p)
        if gt and pt:
            g1, g2 = gt; p1, p2 = pt
            if compare_answers(g1, p1, float_precision, debug=debug) and compare_answers(g2, p2, float_precision, debug=debug):
                return True

    # 3) degrees: "90^\circ" vs "90" (or numeric)
    g_deg = normalize_degrees(g)
    p_deg = normalize_degrees(p)
    if g_deg is not None and p_deg is not None:
        return abs(g_deg - p_deg) <= 10 ** (-float_precision)
    if g_deg is not None:
        # gold has degrees; try parse pred numeric
        try:
            if SYMPY_OK:
                pv = float(sp.N(sympy_expr(p)))
            else:
                pv = float(p)
            return abs(g_deg - pv) <= 10 ** (-float_precision)
        except Exception:
            pass

    # 4) text answers: \text{Evelyn} vs Evelyn
    if g.startswith(r"\text{") or p.startswith(r"\text{"):
        gt = strip_text_wrappers(g)
        pt = strip_text_wrappers(p)
        if gt.lower().strip() == pt.lower().strip():
            return True

    # 5) SymPy fallback algebraic equivalence
    if sympy_eq(g, p):
        return True

    # 6) Final strict-ish normalization compare
    def norm(s: str) -> str:
        s = s.strip()
        s = s.replace(" ", "")
        s = s.rstrip(".")
        return s

    return norm(g) == norm(p)


# ============= Reader for JSONL or pretty-printed =============

def read_json_records(path: str) -> List[Dict]:
    """
    Reads either compact JSONL (one obj per line) or pretty-printed objects back-to-back.
    Scans the whole file and extracts balanced {...} while respecting strings/escapes.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()

    rows: List[Dict] = []
    n = len(data)
    i = 0

    while i < n:
        while i < n and data[i] != '{':
            i += 1
        if i >= n:
            break

        start = i
        depth = 0
        in_str = False
        esc = False

        while i < n:
            ch = data[i]
            if in_str:
                if esc:
                    esc = False
                else:
                    if ch == '\\':
                        esc = True
                    elif ch == '"':
                        in_str = False
                i += 1
                continue

            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    i += 1
                    chunk = data[start:i].strip()
                    try:
                        rows.append(json.loads(chunk))
                    except json.JSONDecodeError:
                        chunk2 = chunk.rstrip(", \n\r\t")
                        try:
                            rows.append(json.loads(chunk2))
                        except Exception:
                            pass
                    break
            i += 1

    return rows


# ============= Main =============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="Path to generations (JSONL or pretty-printed)")
    ap.add_argument("--csv-out", default=None, help="Optional path for per-item CSV")
    ap.add_argument("--float-precision", type=int, default=6, help="Numeric precision for comparisons")
    ap.add_argument("--prefer-extracted", action="store_true",
                    help="Extract final answer (e.g., \\[\\boxed{...}\\]) from 'generation' before verifying.")
    args = ap.parse_args()

    debug = bool(os.environ.get("SCORE_DEBUG"))

    preds = read_json_records(args.preds)

    correct = 0
    total = 0
    rows_for_csv: List[Dict] = []

    for obj in tqdm(preds, total=len(preds)):
        gold_raw_orig = (obj.get("gold_answer") or "").strip()
        pred_full_orig = (obj.get("generation") or "").strip()

        gold_extracted = sanitize_math(extract_answer(gold_raw_orig))
        pred_extracted = sanitize_math(extract_answer(pred_full_orig) if args.prefer_extracted else pred_full_orig)

        try:
            is_correct = compare_answers(gold_extracted, pred_extracted, args.float_precision, debug=debug)
        except Exception as e:
            if debug:
                print(f"[compare fail] id={obj.get('unique_id')} gold={gold_extracted!r} pred={pred_extracted!r} err={e}",
                      file=sys.stderr)
            is_correct = False

        total += 1
        correct += int(is_correct)

        rows_for_csv.append({
            "idx": obj.get("idx"),
            "unique_id": obj.get("unique_id"),
            "subject": obj.get("subject"),
            "level": obj.get("level"),
            "model": obj.get("model"),
            "is_correct": int(is_correct),
            "gold": gold_raw_orig,
            "gold_extracted": gold_extracted,
            "prediction": pred_full_orig,
            "prediction_extracted": pred_extracted,
        })

    acc = correct / max(total, 1)
    print(f"Accuracy: {correct}/{total} = {acc:.3%}")

    out_csv = args.csv_out or args.preds.replace(".jsonl", "_scored.csv")
    if rows_for_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(rows_for_csv[0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows_for_csv:
                w.writerow(r)
        print(f"Wrote per-item scores -> {out_csv}")
    else:
        print("No rows to write.")


if __name__ == "__main__":
    main()
