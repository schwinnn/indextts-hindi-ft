#!/usr/bin/env python3
"""
对任意 IndicVoices-R / kalita11 / somu9 / Chaashini 分片做全面探查。
用法：python inspect_shard.py <path/to/shard.parquet>
"""
import sys, io
import pandas as pd
import numpy as np

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False
    print("[警告] soundfile 未安装，音频字段只做元数据探查")

PARQUET = sys.argv[1] if len(sys.argv) > 1 else \
    "data/indicvoices_r/Hindi/Hindi/train-00000-of-00099.parquet"

print(f"\n{'='*64}")
print(f"文件: {PARQUET}")
print('='*64)

df = pd.read_parquet(PARQUET)

# ── 1. 列名 / 类型 / 样本 ────────────────────────────────────────
print(f"\n[1] 列名 / 类型  ({len(df.columns)} 列, {len(df)} 行)")
for col in df.columns:
    dtype = str(df[col].dtype)
    val = df[col].iloc[0]
    if isinstance(val, bytes):
        sample = f"<bytes {len(val):,}>"
    elif isinstance(val, dict):
        sample = "{" + ", ".join(f"{k}:{type(v).__name__}" for k, v in list(val.items())[:4]) + "}"
    else:
        sample = repr(str(val)[:80])
    print(f"  {col:<32} {dtype:<12}  {sample}")

# ── 2. 文本样本 ──────────────────────────────────────────────────
text_col = (next((c for c in df.columns if c == "normalized"), None)
         or next((c for c in df.columns if "norm" in c.lower()), None)
         or next((c for c in df.columns if "text" in c.lower()), None))
print(f"\n[2] 文本样本  (列: {text_col})")
if text_col:
    for i, t in enumerate(df[text_col].dropna().head(5)):
        print(f"  [{i}] {str(t)[:120]}")
else:
    print("  未找到文本列")

# ── 3. 说话人 ────────────────────────────────────────────────────
spk_col = next((c for c in df.columns if "speaker" in c.lower()), None)
if spk_col:
    n_spk = df[spk_col].nunique()
    top = df[spk_col].value_counts().head(8)
    print(f"\n[3] 说话人 (列: {spk_col}): {n_spk} 个唯一值")
    for spk, cnt in top.items():
        print(f"  {spk}: {cnt} 条")
else:
    print("\n[3] 未找到 speaker 列")

# ── 4. 性别 ──────────────────────────────────────────────────────
gen_col = next((c for c in df.columns if "gender" in c.lower()), None)
if gen_col:
    print(f"\n[4] 性别 (列: {gen_col})")
    print(df[gen_col].value_counts().to_string(header=False))

# ── 5. 时长 ──────────────────────────────────────────────────────
dur_col = next((c for c in df.columns if "duration" in c.lower()), None)
if dur_col:
    d = pd.to_numeric(df[dur_col], errors="coerce").dropna()
    total_h = d.sum() / 3600
    print(f"\n[5] 时长 (列: {dur_col})")
    print(f"  本分片合计:  {total_h:.2f} h  ({d.sum():.0f} s, {len(d)} 条)")
    print(f"  均值: {d.mean():.2f}s  中位: {d.median():.2f}s")
    print(f"  P1:  {d.quantile(.01):.2f}s  P5: {d.quantile(.05):.2f}s  "
          f"P95: {d.quantile(.95):.2f}s  P99: {d.quantile(.99):.2f}s")
    print(f"  < 1s: {(d<1).sum()}  1–25s: {((d>=1)&(d<=25)).sum()}  "
          f"> 25s (会被丢): {(d>25).sum()}")
    n_shards = int(PARQUET.split("-of-")[-1].replace(".parquet","")) if "-of-" in PARQUET else 99
    print(f"  → 外推全量 ({n_shards} 分片): 约 {total_h*n_shards:.0f} 小时")
else:
    print("\n[5] 无 duration 列，将从音频实测第 1 条时长")

# ── 6. 音频字段 ───────────────────────────────────────────────────
audio_col = next((c for c in df.columns if "audio" in c.lower()), None)
if audio_col and HAS_SF:
    sample_audio = df[audio_col].iloc[0]
    print(f"\n[6] 音频字段 (列: {audio_col})  类型: {type(sample_audio).__name__}")
    raw = None
    if isinstance(sample_audio, dict):
        print(f"  子键: {list(sample_audio.keys())}")
        raw = sample_audio.get("bytes") or sample_audio.get("path")
        if "array" in sample_audio:
            arr = np.asarray(sample_audio["array"])
            sr  = sample_audio.get("sampling_rate", 0)
            dur = len(arr) / sr if sr else 0
            print(f"  采样率: {sr} Hz  时长: {dur:.2f}s  "
                  f"声道: {'立体声' if arr.ndim>1 else '单声道'}  dtype: {arr.dtype}")
            if not dur_col:
                print(f"  → 外推全量 (99 分片): 约 {dur*len(df)*99/3600:.0f} 小时")
            raw = None
    elif isinstance(sample_audio, bytes):
        raw = sample_audio
    if raw is not None and isinstance(raw, bytes):
        try:
            audio, sr = sf.read(io.BytesIO(raw))
            dur = len(audio) / sr
            print(f"  采样率: {sr} Hz  时长: {dur:.2f}s  "
                  f"声道: {'立体声' if audio.ndim>1 else '单声道'}  dtype: {audio.dtype}")
            if not dur_col:
                n_shards = int(PARQUET.split("-of-")[-1].replace(".parquet","")) if "-of-" in PARQUET else 99
                print(f"  → 外推全量 ({n_shards} 分片): 约 {dur*len(df)*n_shards/3600:.0f} 小时")
        except Exception as e:
            print(f"  sf.read 失败: {e}")
elif audio_col:
    print(f"\n[6] 音频列 {audio_col} 存在（soundfile 未安装，跳过解码）")

# ── 7. 质量字段 ───────────────────────────────────────────────────
quality_keys = ["snr", "cer", "pitch", "rate", "dnsmos", "asr_conf",
                "music_prob", "noise_prob", "speech_prob", "bandwidth"]
q_cols = [c for c in df.columns if any(k in c.lower() for k in quality_keys)]
if q_cols:
    print(f"\n[7] 质量字段")
    for c in q_cols:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().sum() > 10:
            print(f"  {c:<30} min={v.min():.3f}  mean={v.mean():.3f}  "
                  f"P5={v.quantile(.05):.3f}  P95={v.quantile(.95):.3f}  max={v.max():.3f}")
        else:
            print(f"  {c:<30} {df[c].value_counts().head(3).to_dict()}")

# ── 8. 字符速率 ────────────────────────────────────────────────────
if dur_col and text_col:
    d_num = pd.to_numeric(df[dur_col], errors="coerce")
    t_len = df[text_col].astype(str).str.len()
    cps   = (t_len / d_num).replace([np.inf, -np.inf], np.nan).dropna()
    print(f"\n[8] 字符速率 (chars/s)  基于 {len(cps)} 条")
    print(f"  P1={cps.quantile(.01):.1f}  P5={cps.quantile(.05):.1f}  "
          f"中位={cps.median():.1f}  P95={cps.quantile(.95):.1f}  P99={cps.quantile(.99):.1f}")
    print(f"  建议 --min-cps / --max-cps 参考: "
          f"{cps.quantile(.01):.1f} ~ {cps.quantile(.99):.1f}")

# ── 9. 额外字段（scenario / task / verbatim）─────────────────────
extra = [c for c in df.columns if any(k in c.lower()
         for k in ["scenario","task","verbatim","style","domain","area","state"])]
if extra:
    print(f"\n[9] 其他分类字段")
    for c in extra[:8]:
        vc = df[c].value_counts().head(5)
        print(f"  {c}: {vc.to_dict()}")

print(f"\n{'='*64}\n完成\n{'='*64}\n")
