# 临界分词 / Critical Tokenization — 实证研究

把"临界分词"理论假设拆成 11 条可证伪命题，分三阶段检验：统计临界态 → 双语神经 LM → 动态分词 / V–N 律。
所有数字均可复现，全部代码 + 结果 JSON + 图表 + 论文都在本仓库。

## 入口文档
- **`PAPER.md`** — 完整论文（摘要 / 11 命题 / 方法 / 三阶段结果 / V–N 律 / 结论）
- `VERIFICATION_REPORT.md` — 统计阶段逐条裁决
- `llm_verify/LLM_VERIFICATION_REPORT.md` — 双语 LLM 验证
- `llm_verify/DYN_VERIFICATION_REPORT.md` — 动态分词证伪

## 一句话结论
- ✅ α≈1（Zipf 临界）是稳健的好分词区域，落在语言内禀构词单元（英=子词，中=字），满足幂律长程相关的临界判据。
- ⚠️ 是否尖锐最优**语言依赖**（中文 char 双轴最优，英文是宽平台）。
- 🔑 真正机理是**算力—信息双轴权衡**；最优词表随规模亚线性增长 **V_opt ∝ N^{2/3}**（实测指数 0.666）。
- ❌ 分形维 D、MDL 最小=临界、动态>静态（最简形态）、15-20% 普适效率——被下调/证伪。

## 复现
```bash
# 统计阶段（Python 3.14 + numpy/scipy/matplotlib）
python run_all.py

# LLM 阶段（torch + flash-linear-attention, GPU 推荐）
cd llm_verify
python build_caches.py en && python train_compare.py en
python build_caches.py cn && python train_compare.py cn
python dyn_experiment.py          # 动态分词验证
python n_sweep.py                 # V–N 律扫描
```
> token 缓存（`*.pt`，~1.2GB）已从仓库排除，由上述脚本自动重建。

## 模型
FRSMASH v3.6（SSM + 线性注意力，fla 加速） · NVIDIA RTX 4090 D · PyTorch 2.12 + CUDA 12.6
