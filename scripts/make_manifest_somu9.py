#!/usr/bin/env python3
"""
somu9/iisc_mono_hindi_female → utterances.jsonl

用法：
    python scripts/make_manifest_somu9.py \
        --input  data/somu9 \
        --output data/hi/utterances_somu9.jsonl \
        --max-per-speaker 15.0
"""
from __future__ import annotations
import argparse, glob, io, json, re
from pathlib import Path

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
    p.add_argument("--min-duration", type=float, default=1.5)
    p.add_argument("--max-duration", type=float, default=25.0)
    p.add_argument("--min-cps",      type=float, default=3.0)
    p.add_argument("--max-cps",      type=float, default=28.0)
    # 单声音数据：强制设上限，避免把训练偏向这一个人
    p.add_argument("--max-per-speaker", type=float, default=15.0,
                   help="该说话人最多保留多少小时（默认 15h）")
    return p.parse_args()


def main():
    args = parse_args()
    inp = Path(args.input)
    files = sorted(glob.glob(str(inp / "*.parquet"))) if inp.is_dir() else [str(inp)]
    print(f"分片数: {len(files)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    max_sec  = args.max_per_speaker * 3600
    kept_sec = 0.0
    seen_ids: dict[str, int] = {}
    stats = dict(total=0, kept=0, drop_dur=0, drop_cps=0, drop_cap=0, drop_audio=0)

    with open(out_path, "w", encoding="utf-8") as fout:
        for fpath in files:
            df = pd.read_parquet(fpath)
            for _, row in df.iterrows():
                stats["total"] += 1
                if kept_sec >= max_sec:
                    stats["drop_cap"] += 1
                    continue

                text = str(row.get("text", "")).strip()
                if len(text) < 2:
                    stats["drop_audio"] += 1
                    continue

                dur = 0.0
                if "duration" in row.index and pd.notna(row["duration"]):
                    try:
                        dur = float(row["duration"])
                    except Exception:
                        pass
                if dur == 0.0 and HAS_SF:
                    av = row.get("audio")
                    if isinstance(av, dict):
                        if "array" in av:
                            sr = av.get("sampling_rate", 48000)
                            dur = len(av["array"]) / max(sr, 1)
                        elif "bytes" in av:
                            try:
                                a, sr = sf.read(io.BytesIO(av["bytes"]))
                                dur = len(a) / sr
                            except Exception:
                                pass
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

                spk = str(row.get("speaker_id", "hindi_female_spk001")).strip()
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
                            sr = av.get("sampling_rate", 48000)
                            sf.write(str(wav_path), np.asarray(av["array"]), sr)
                        elif isinstance(av, dict) and "bytes" in av:
                            a, sr = sf.read(io.BytesIO(av["bytes"]))
                            sf.write(str(wav_path), a, sr)
                        elif isinstance(av, bytes):
                            a, sr = sf.read(io.BytesIO(av))
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
                    "gender": "female",
                }
                if "domain" in row.index and pd.notna(row["domain"]):
                    record["domain"] = str(row["domain"])

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept_sec += dur
                stats["kept"] += 1

    print(json.dumps(stats, indent=2))
    print(f"\n实际保留: {kept_sec/3600:.2f} 小时")
    print(f"→ 输出: {out_path}")


if __name__ == "__main__":
    main()
