#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MedGemma vLLM smoke test")
    p.add_argument("--model", default="google/medgemma-1.5-4b-it")
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    p.add_argument("--max-num-seqs", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    prompt = (
        "[SYSTEM]\n"
        "Return exactly one valid JSON object and nothing else.\n\n"
        "[QUESTION]\n"
        'Return {"answer":"ok"}.\n\n'
        "[FORMAT]\n"
        '{"answer":"string"}'
    )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
    )
    outputs = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=args.max_tokens))
    text = outputs[0].outputs[0].text
    print(json.dumps({"text": text}, ensure_ascii=False))


if __name__ == "__main__":
    main()
