#!/usr/bin/env python3
"""
kalita11/Hindi-TTS-corpus → utterances.jsonl

用法：
    python scripts/make_manifest_kalita.py \
        --input  data/kalita11 \
        --output data/hi/utterances_kalita.jsonl \
        --max-duration 20.0 \
        --max-per-speaker 10.0
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
    p.add_argument("--min-duration", type=float, default=1.5)
    p.add_argument("--max-duration", type=float, default=20.0)
    p.add_argument("--min-cps",      type=float, default=3.0)
    p.add_argument("--max-cps",      type=float, default=28.0)
    p.add_argument("--max-per-speaker", type=float, default=0.0,
                   help="每个 speaker_id 最多保留多少小时（0=不限）。"
                        "用于防止 SYSPIN 单人占比过高")
    p.add_argument("--exclude-source", nargs="*", default=[],
                   help="排除某来源，例如 --exclude-source IndicTTS")
    return p.parse_args()


def save_wav(audio_val, path: Path) -> bool:
    if not HAS_SF:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(audio_val, dict):
            if "array" in audio_val:
                sr  = audio_val.get("sampling_rate", 24000)
                arr = np.asarray(audio_val["array"])
                sf.write(str(path), arr, sr)
                return True
            if "bytes" in audio_val:
                audio, sr = sf.read(io.BytesIO(audio_val["bytes"]))
                sf.write(str(path), audio, sr)
                return True
        elif isinstance(audio_val, bytes):
            audio, sr = sf.read(io.BytesIO(audio_val))
            sf.write(str(path), audio, sr)
            return True
    except Exception:
        pass
    return False


def main():
    args = parse_args()
    inp = Path(args.input)
    files = sorted(glob.glob(str(inp / "*.parquet"))) if inp.is_dir() else [str(inp)]
    print(f"分片数: {len(files)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 每说话人已保留秒数
    spk_seconds: dict[str, float] = defaultdict(float)
    max_spk_sec = args.max_per_speaker * 3600 if args.max_per_speaker > 0 else float("inf")

    same_text: dict[str, int] = defaultdict(int)
    seen_ids:  dict[str, int] = {}
    stats = dict(total=0, kept=0, drop_dur=0, drop_cps=0,
                 drop_spk_cap=0, drop_source=0, drop_text=0, drop_audio=0)

    with open(out_path, "w", encoding="utf-8") as fout:
        for fpath in files:
            df = pd.read_parquet(fpath)
            for _, row in df.iterrows():
                stats["total"] += 1

                # 过滤来源
                src = str(row.get("source", "")).strip()
                if any(ex.lower() in src.lower() for ex in args.exclude_source):
                    stats["drop_source"] += 1
                    continue

                text = str(row.get("text", "")).strip()
                if len(text) < 2:
                    stats["drop_text"] += 1
                    continue

                dur = 0.0
                for dc in ("duration_s", "duration"):
                    if dc in row.index and pd.notna(row[dc]):
                        try:
                            dur = float(row[dc])
                        except Exception:
                            pass
                        break
                if dur == 0.0 and HAS_SF:
                    av = row.get("audio")
                    if av is not None:
                        if isinstance(av, dict) and "array" in av:
                            sr = av.get("sampling_rate", 1)
                            dur = len(av["array"]) / max(sr, 1)
                        elif isinstance(av, dict) and "bytes" in av:
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

                spk = str(row.get("speaker_id", row.get("speaker", "spk_unknown"))).strip()
                if spk_seconds[spk] + dur > max_spk_sec:
                    stats["drop_spk_cap"] += 1
                    continue

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
                    if not save_wav(row.get("audio"), wav_path):
                        stats["drop_audio"] += 1
                        continue
                    audio_stored = str(wav_path)
                else:
                    audio_stored = fpath + ":" + uid

                record = {
                    "id": uid, "audio": audio_stored,
                    "text": text, "speaker": spk,
                    "lang": "hi", "duration": round(dur, 3),
                }
                for extra in ("gender", "style", "source", "speaking_rate_cps"):
                    if extra in row.index and pd.notna(row[extra]):
                        record[extra] = row[extra]

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                spk_seconds[spk] += dur
                stats["kept"] += 1

    print(json.dumps(stats, indent=2))
    total_h = sum(spk_seconds.values()) / 3600
    print(f"\n实际保留: {total_h:.2f} 小时，{len(spk_seconds)} 个 speaker_id")
    print(f"→ 输出: {out_path}")


if __name__ == "__main__":
    main()
