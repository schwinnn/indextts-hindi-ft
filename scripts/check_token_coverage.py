#!/usr/bin/env python3
"""
统计 IndexTTS-2.5 的 text_embedding 里，
印地语语料用到的 token 有多少比例落在未训练行上。

用法：
    python scripts/check_token_coverage.py \
        --manifest data/hi/utterances_ivr.jsonl \
        --model-dir ../index-tts/checkpoints \
        --repo     ../index-tts \
        --normalizer module:hindi.normalizer:normalize_hindi \
        --case none \
        --samples 5000
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",   required=True)
    p.add_argument("--model-dir",  default=None)
    p.add_argument("--repo",       default=None)
    p.add_argument("--normalizer", default="module:hindi.normalizer:normalize_hindi")
    p.add_argument("--case",       default="none")
    p.add_argument("--samples",    type=int, default=5000,
                   help="最多取多少条计算（0 = 全部）")
    p.add_argument("--norm-threshold", type=float, default=0.5,
                   help="低于此范数视为未训练（默认 0.5）")
    return p.parse_args()


def main():
    args = parse_args()

    # bootstrap
    sys.path.insert(0, str(Path(args.repo or "../index-tts-2.5-ft").resolve()))
    from itts25ft import env
    from itts25ft.lang import resolve
    from itts25ft.textfront import TextFrontend, TextFrontendConfig
    from itts25ft.modeling import build_gpt

    repo     = env.bootstrap(args.repo)
    model_dir = env.find_model_dir(args.model_dir, repo)
    cfg      = env.load_model_config(model_dir)

    slot     = resolve("hi")
    frontend = TextFrontend(model_dir, slot,
                            TextFrontendConfig(normalizer=args.normalizer, case=args.case))

    import torch
    print("模型加载中（CPU）…")
    model = build_gpt(cfg, model_dir, device="cpu")
    model.eval()

    norms = model.text_embedding.weight.detach().float().norm(dim=1).cpu()
    capacity = norms.shape[0]
    print(f"text_embedding 大小: {capacity} 行")
    print(f"已训练行（norm >= {args.norm_threshold}）: "
          f"{(norms >= args.norm_threshold).sum().item()}")
    print(f"未训练行（norm < {args.norm_threshold}）:  "
          f"{(norms < args.norm_threshold).sum().item()}")

    # 读 manifest
    with open(args.manifest, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]
    if args.samples and len(records) > args.samples:
        records = random.sample(records, args.samples)
    print(f"\n抽取 {len(records)} 条计算 token 分布…")

    token_counts: dict[int, int] = {}
    for rec in records:
        try:
            ids = frontend.encode(str(rec.get("text", "")))
        except Exception:
            continue
        for tid in ids:
            token_counts[tid] = token_counts.get(tid, 0) + 1

    total_occ   = sum(token_counts.values())
    trained_occ = sum(c for tid, c in token_counts.items()
                      if tid < capacity and norms[tid].item() >= args.norm_threshold)
    untrained_occ = total_occ - trained_occ

    print(f"\n── 结果 ──────────────────────────────────")
    print(f"唯一 token 数:         {len(token_counts)}")
    print(f"总出现次数:            {total_occ:,}")
    print(f"未训练行出现次数:      {untrained_occ:,}  "
          f"({100*untrained_occ/max(total_occ,1):.1f}%)")
    print(f"已训练行出现次数:      {trained_occ:,}  "
          f"({100*trained_occ/max(total_occ,1):.1f}%)")

    print(f"\n参考值：")
    print(f"  土耳其语未训练行占比约 37%  → --train-text-rows 效果显著")
    print(f"  天城文预计远高于此         → 需要 --train-text-rows")

    # tokenizer 效率
    report = frontend.efficiency_report([str(r.get("text","")) for r in records[:500]])
    print(f"\ntokens/char: {report['tokens_per_char']:.3f}  "
          f"tokens/word: {report['tokens_per_word']:.2f}")
    print("（>0.5 tokens/char 表示 BPE 把这门语言切碎，需要更多数据和步数）")


if __name__ == "__main__":
    main()
