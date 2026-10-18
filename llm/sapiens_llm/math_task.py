"""MATH, as in the finetuning_es project: level 3-5 MATH train problems, the MATH-500 test set,
Qwen's math instruction and a math-verify correctness reward (1 if the last \\boxed{} answer is
mathematically equal to the reference, else 0)."""
import pathlib

import pandas as pd

DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "math"
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def load(split):
    return pd.read_parquet(DATA / f"{split}.parquet").to_dict(orient="records")


def prompt_ids(tokenizer, problem):
    messages = [{"role": "user", "content": f"{problem}\n{INSTRUCTION}"}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tokenizer(text)["input_ids"]


def reward(text, answer):
    from math_verify import parse, verify
    gold = parse(f"\\boxed{{{answer}}}")
    try:
        return float(bool(gold) and verify(gold, parse(text)))
    except Exception:
        return 0.0
