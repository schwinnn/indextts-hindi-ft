#!/usr/bin/env python3
"""
IndicVoices-R 印地语分片 → utterances.jsonl

用法：
    python scripts/make_manifest_ivr.py \
        --input  data/indicvoices_r/Hindi/Hindi \
        --output data/hi/utterances_ivr.jsonl \
        --min-duration 1.0 --max-duration 25.0 \
        --min-cps 3.0 --max-cps 30.0 \
        --max-same-text 3
"""
from __future__ import annotations
import argparse, glob, io, json, os, re, unicodedata
from pathlib import Path
from collections import defaultdict

import re as _re

import pandas as pd
import numpy as np


def _parse_float(val) -> float | None:
    """IndicVoices-R 的 cer/snr 列值有时是 'tensor(0.0248)' 格式，解析成 float。"""
    if val is None:
        return None
    s = str(val).strip()
    m = _re.match(r"tensor\(([^)]+)\)", s)
    if m:
        s = m.group(1)
    try:
        return float(s)
    except ValueError:
        return None

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="parquet 分片目录或单个文件")
    p.add_argument("--output", required=True, help="输出 jsonl 路径")
    p.add_argument("--audio-dir", default=None, help="wav 落盘目录（不指定则不落盘）")
    p.add_argument("--min-duration", type=float, default=1.0)
    p.add_argument("--max-duration", type=float, default=25.0)
    p.add_argument("--min-cps",      type=float, default=3.0,
                   help="最低字符速率；3.0 适合天城文口语，低于此为静音/噪声")
    p.add_argument("--max-cps",      type=float, default=30.0)
    p.add_argument("--max-same-text", type=int,  default=3)
    p.add_argument("--shards",        type=int,  default=0,
                   help="只处理前 N 个分片（0 = 全部）")
    p.add_argument("--use-normalized", action="store_true", default=True,
                   help="优先使用 normalized 列而非 verbatim（默认开启）")
    return p.parse_args()


def get_text(row, prefer_normalized: bool) -> str:
    if prefer_normalized and "normalized" in row.index and pd.notna(row["normalized"]):
        return str(row["normalized"]).strip()
    for col in ("text", "verbatim", "transcript"):
        if col in row.index and pd.notna(row[col]):
            return str(row[col]).strip()
    return ""


def decode_audio_duration(audio_val) -> float:
    """从 parquet 音频字段解码时长（秒），失败返回 0。"""
    if not HAS_SF:
        return 0.0
    try:
        if isinstance(audio_val, dict):
            if "array" in audio_val:
                sr  = audio_val.get("sampling_rate", 1)
                arr = np.asarray(audio_val["array"])
                # 立体声：行数是采样点数
                n   = arr.shape[0] if arr.ndim >= 1 else len(arr)
                return n / max(sr, 1)
            if "bytes" in audio_val:
                audio, sr = sf.read(io.BytesIO(audio_val["bytes"]))
                return audio.shape[0] / sr
        elif isinstance(audio_val, bytes):
            audio, sr = sf.read(io.BytesIO(audio_val))
            return audio.shape[0] / sr
    except Exception:
        pass
    return 0.0


def _to_mono(arr: np.ndarray) -> np.ndarray:
    """立体声 → 单声道（平均两声道）。"""
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    return arr


def save_wav(audio_val, path: Path) -> bool:
    if not HAS_SF:
        return False
    try:
        import soundfile as sf2
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(audio_val, dict):
            if "array" in audio_val:
                sr  = audio_val.get("sampling_rate", 48000)
                arr = _to_mono(np.asarray(audio_val["array"]))
                sf2.write(str(path), arr, sr)
                return True
            if "bytes" in audio_val:
                audio, sr = sf2.read(io.BytesIO(audio_val["bytes"]))
                sf2.write(str(path), _to_mono(audio), sr)
                return True
        elif isinstance(audio_val, bytes):
            audio, sr = sf2.read(io.BytesIO(audio_val))
            sf2.write(str(path), _to_mono(audio), sr)
            return True
    except Exception:
        pass
    return False


def main():
    args = parse_args()
    inp = Path(args.input)
    if inp.is_dir():
        files = sorted(glob.glob(str(inp / "*.parquet")))
    else:
        files = [str(inp)]
    if args.shards:
        files = files[:args.shards]
    print(f"分片数: {len(files)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    same_text: dict[str, int] = defaultdict(int)
    stats = dict(total=0, kept=0, drop_dur=0, drop_cps=0,
                 drop_text=0, drop_same=0, drop_audio=0)
    seen_ids: dict[str, int] = {}

    with open(out_path, "w", encoding="utf-8") as fout:
        for fpath in files:
            df = pd.read_parquet(fpath)
            for _, row in df.iterrows():
                stats["total"] += 1

                text = get_text(row, args.use_normalized)
                if len(text) < 2:
                    stats["drop_text"] += 1
                    continue

                # 时长
                dur = 0.0
                for dc in ("duration", "duration_s"):
                    if dc in row.index and pd.notna(row[dc]):
                        try:
                            dur = float(row[dc])
                        except Exception:
                            pass
                        break
                if dur == 0.0:
                    dur = decode_audio_duration(row.get("audio"))
                if dur == 0.0:
                    stats["drop_audio"] += 1
                    continue
                if dur < args.min_duration or dur > args.max_duration:
                    stats["drop_dur"] += 1
                    continue

                # 字符速率
                cps = len(text) / dur
                if cps < args.min_cps or cps > args.max_cps:
                    stats["drop_cps"] += 1
                    continue

                # 重复文本
                key = " ".join(text.lower().split())
                if same_text[key] >= args.max_same_text:
                    stats["drop_same"] += 1
                    continue
                same_text[key] += 1

                # speaker_id
                spk = ""
                for sc in ("speaker_id", "speaker", "spk"):
                    if sc in row.index and pd.notna(row[sc]):
                        spk = str(row[sc]).strip()
                        break
                if not spk:
                    spk = "spk_unknown"

                # uid
                uid_base = str(row.get("id", "")) or Path(fpath).stem + f"_{stats['total']}"
                uid_base = re.sub(r"[^\w\-]", "_", uid_base)
                if uid_base in seen_ids:
                    seen_ids[uid_base] += 1
                    uid = f"{uid_base}_{seen_ids[uid_base]}"
                else:
                    seen_ids[uid_base] = 0
                    uid = uid_base

                # 音频路径
                if args.audio_dir:
                    wav_path = Path(args.audio_dir) / (uid + ".wav")
                    audio_val = row.get("audio")
                    if audio_val is not None and not save_wav(audio_val, wav_path):
                        stats["drop_audio"] += 1
                        continue
                    audio_stored = str(wav_path)
                else:
                    # 不落盘，存原始 parquet 路径作备注（训练脚本不用这个字段）
                    audio_stored = fpath + ":" + uid

                record = {
                    "id":       uid,
                    "audio":    audio_stored,
                    "text":     text,
                    "speaker":  spk,
                    "lang":     "hi",
                    "duration": round(dur, 3),
                }
                # 把 gender/snr/cer 带上，方便后续过滤
                # cer/snr 可能是 'tensor(0.025)' 格式，统一解析成 float
                for extra in ("gender", "age_group", "area", "scenario"):
                    if extra in row.index and pd.notna(row[extra]):
                        record[extra] = str(row[extra])
                for fextra in ("snr", "cer", "speaking_rate",
                               "utterance_pitch_mean", "utterance_pitch_std"):
                    if fextra in row.index:
                        v = _parse_float(row[fextra])
                        if v is not None:
                            record[fextra] = round(v, 4)

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["kept"] += 1

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\n→ 输出: {out_path}  ({stats['kept']} 条)")


if __name__ == "__main__":
    main()
