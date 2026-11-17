# IndexTTS-2.5 Hindi Fine-tune

在 [IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5) 上增加印地语（Hindi / Devanagari）支持，保留原版中英日西阿五种语言和跨语言克隆能力。

基于 [oberkayb/index-tts-2.5-ft](https://github.com/oberkayb/index-tts-2.5-ft) 框架，参照土耳其语（WER 1.32%）和德语微调的经验。

## 状态

- [x] 探查脚本（IndicVoices-R、kalita11、somu9、Chaashini）
- [x] Manifest 转换脚本（4 个数据源）
- [x] 印地语文本前端（`normalize_hindi`）
- [x] `modeling.py` 天城文种子映射补丁
- [x] 训练配置 `configs/hindi.yaml`
- [ ] 阶段 1 训练（Chaashini + IndicVoices-R，目标 ~200h）
- [ ] 阶段 2 训练（录音棚数据精调）
- [ ] 评估

## 数据来源

| 数据集 | 时长 | 说话人 | 质量 | 授权 | 阶段 |
|---|---|---|---|---|---|
| kalita11/Hindi-TTS-corpus | ~122h | 实际约 6 声音 | 录音棚 | CC-BY-4.0 | 2 |
| somu9/iisc_mono_hindi_female | ~55h | 1 女 | 录音棚 Neumann | CC-BY-4.0 | 2 |
| ai4bharat/indicvoices_r（印地语） | ~75-137h | 多人 | 增强口语 | CC-BY-4.0 | 1+2+验证集 |
| kapturecx/Chaashini（印地语） | ~1394h | ASR diarized | 公开录音 | Apache-2.0 | 1 |

## 仓库结构

```
indextts-hindi-ft/
├── scripts/
│   ├── inspect_shard.py       # 探查单个 parquet 分片
│   ├── make_manifest_ivr.py   # IndicVoices-R → utterances.jsonl
│   ├── make_manifest_kalita.py
│   ├── make_manifest_somu9.py
│   ├── make_manifest_chaashini.py
│   └── check_token_coverage.py  # 统计未训练 embedding 行占比
├── hindi/
│   ├── normalizer.py          # normalize_hindi 文本前端
│   └── devanagari_translit.py # 天城文→拉丁映射（seed_translit 补丁）
├── patches/
│   └── modeling_seed_translit.patch  # 应用到 index-tts-2.5-ft/itts25ft/modeling.py
├── configs/
│   └── hindi.yaml             # 训练配置
└── README.md
```

## 快速开始

```bash
# 0. 克隆两个上游仓库
git clone https://github.com/index-tts/index-tts.git
git clone https://github.com/oberkayb/index-tts-2.5-ft.git
git clone <this-repo> indextts-hindi-ft

# 1. 安装依赖
cd index-tts && uv sync --all-extras
uv pip install -r ../index-tts-2.5-ft/requirements.txt
uv pip install num2words indic-transliteration

# 2. 下载模型
hf download IndexTeam/IndexTTS-2.5 --local-dir checkpoints

# 3. 应用天城文种子映射补丁
cd ../index-tts-2.5-ft
git apply ../indextts-hindi-ft/patches/modeling_seed_translit.patch

# 4. 探查数据（先下一个分片）
python ../indextts-hindi-ft/scripts/inspect_shard.py \
    data/indicvoices_r/Hindi/Hindi/train-00000-of-00099.parquet

# 5. 语言槽位检查
python scripts/check_setup.py --lang hi --load-model \
    --normalizer module:hindi.normalizer:normalize_hindi --case none

# 6. 生成 manifest
python ../indextts-hindi-ft/scripts/make_manifest_ivr.py \
    --input data/indicvoices_r/Hindi/Hindi \
    --output data/hi/utterances_ivr.jsonl

# 7. 训练（见 configs/hindi.yaml）
python scripts/train.py --config ../indextts-hindi-ft/configs/hindi.yaml \
    --normalizer module:hindi.normalizer:normalize_hindi --case none
```
