#!/usr/bin/env python3
"""
kapturecx/Chaashini 印地语分片 → utterances.jsonl

Chaashini 的转写是 ASR 生成的，需要靠质量字段过滤。
默认阈值比录音棚数据宽松，仅用于第一阶段训练。

用法：
    python scripts/make_manifest_chaashini.py \
        --input  data/chaashini/hi \
        --output data/hi/utterances_chaashini.jsonl \
        --min-dnsmos 3.0 --min-asr-conf 0.85 \
        --max-music-prob 0.1 --max-noise-prob 0.3
"""
from __future__ import annotations
import argparse, glob, io, json, re
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--audio-dir", default=None)
    p.add_argument("--min-duration",  type=float, default=1.0)
    p.add_argument("--max-duration",  type=float, default=25.0)
    p.add_argument("--min-cps",       type=float, default=3.0)
    p.add_argument("--max-cps",       type=float, default=30.0)
    p.add_argument("--min-dnsmos",    type=float, default=3.0,
                   help="dnsmos_p808 下限（Chaashini 的总体质量分）")
    p.add_argument("--min-asr-conf",  type=float, default=0.85,
                   help="asr_confidence 下限")
    p.add_argument("--max-music-prob",type=float, default=0.1)
    p.add_argument("--max-noise-prob",type=float, default=0.3)
    p.add_argument("--min-snr",       type=float, default=15.0,
                   help="snr_db 下限")
    p.add_argument("--max-same-text", type=int,   default=3)
    p.add_argument("--shards",        type=int,   default=0,
                   help="只处理前 N 个分片（0 = 全部）")
    return p.parse_args()


def main():
    args = parse_args()
    inp = Path(args.input)
    files = sorted(glob.glob(str(inp / "*.parquet"))) if inp.is_dir() else [str(inp)]
    if args.shards:
        files = files[:args.shards]
    print(f"分片数: {len(files)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    same_text: dict[str, int] = defaultdict(int)
    seen_ids:  dict[str, int] = {}
    stats = dict(total=0, kept=0, drop_dur=0, drop_cps=0, drop_quality=0,
                 drop_same=0, drop_text=0, drop_audio=0)

    with open(out_path, "w", encoding="utf-8") as fout:
        for fpath in files:
            df = pd.read_parquet(fpath)
            # 只保留印地语
            if "language" in df.columns:
                df = df[df["language"].str.lower().isin(["hi", "hindi"])]
            if df.empty:
                continue

            for _, row in df.iterrows():
                stats["total"] += 1

                text = str(row.get("text", "")).strip()
                if len(text) < 2:
                    stats["drop_text"] += 1
                    continue

                # 质量过滤
                def _f(col, default=None):
                    v = row.get(col)
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        return default
                    return float(v)

                dnsmos    = _f("dnsmos_p808", default=99.0)
                asr_conf  = _f("asr_confidence", default=1.0)
                music     = _f("music_prob", default=0.0)
                noise     = _f("noise_prob", default=0.0)
                snr       = _f("snr_db", default=99.0)

                if (dnsmos   < args.min_dnsmos    or
                    asr_conf < args.min_asr_conf  or
                    music    > args.max_music_prob or
                    noise    > args.max_noise_prob or
                    snr      < args.min_snr):
                    stats["drop_quality"] += 1
                    continue

                dur = _f("duration_s", default=0.0)
                if dur == 0.0 and HAS_SF:
                    av = row.get("audio")
                    if isinstance(av, dict) and "array" in av:
                        sr = av.get("sampling_rate", 24000)
                        dur = len(av["array"]) / max(sr, 1)
                if dur == 0.0:
                    stats["drop_audio"] += 1
                    continue
                if dur < args.min_duration or dur > args.max_duration:
                    stats["drop_dur"] += 1
                    continue

                cps = len(text) / dur
                if cps < args.min_cps or cps > args.max_cps:
                    stats["drop_cps"] += 1
                    continue

                key = " ".join(text.lower().split())
                if same_text[key] >= args.max_same_text:
                    stats["drop_same"] += 1
                    continue
                same_text[key] += 1

                # speaker_id 仅在同一 source_id 内有效
                spk_raw = str(row.get("speaker_id", "")).strip()
                src_id  = str(row.get("source_id",  "")).strip()
                spk = f"{src_id}__{spk_raw}" if spk_raw and src_id else spk_raw or "spk_unknown"

                uid_base = re.sub(r"[^\w\-]", "_",
                    str(row.get("id", Path(fpath).stem + f"_{stats['total']}")))
                if uid_base in seen_ids:
                    seen_ids[uid_base] += 1
                    uid = f"{uid_base}_{seen_ids[uid_base]}"
                else:
                    seen_ids[uid_base] = 0
                    uid = uid_base

                if args.audio_dir:
                    wav_path = Path(args.audio_dir) / (uid + ".wav")
                    wav_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        av = row.get("audio")
                        if isinstance(av, dict) and "array" in av:
                            sr = av.get("sampling_rate", 24000)
                            sf.write(str(wav_path), np.asarray(av["array"]), sr)
                        elif isinstance(av, dict) and "bytes" in av:
                            a, sr = sf.read(io.BytesIO(av["bytes"]))
                            sf.write(str(wav_path), a, sr)
                        audio_stored = str(wav_path)
                    except Exception:
                        stats["drop_audio"] += 1
                        continue
                else:
                    audio_stored = fpath + ":" + uid

                record = {
                    "id": uid, "audio": audio_stored,
                    "text": text, "speaker": spk,
                    "lang": "hi", "duration": round(dur, 3),
                }
                for extra in ("gender", "genre", "dnsmos_p808", "snr_db"):
                    v = row.get(extra)
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        record[extra] = v

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["kept"] += 1

    print(json.dumps(stats, indent=2))
    print(f"→ 输出: {out_path}  ({stats['kept']} 条)")


if __name__ == "__main__":
    main()
