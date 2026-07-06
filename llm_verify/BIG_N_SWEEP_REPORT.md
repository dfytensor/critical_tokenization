# V–N 律扩展到 50M / 100M / 150M — 报告（600 步干净版）

## 你诊断对了：是显存，不是热降频
- 100M 在 batch=32 时 ~300 步后降速 27×：**PyTorch caching allocator 碎片化**（chunk_hgrn/gla 的中间 buffer 碎片化分配器，到阈值退化成慢路径），非 OOM、非热降频。
- 修复：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（治碎片化）+ **micro-batch 8→4、grad-accum=8（有效 batch 32 不变）** + eval batch 64→16。修复后显存稳定 ~5G、**全程无降速无 OOM**，三个规模全部 600 步跑完。

## 600 步结果（clean, micro4/accum8, single seed）

| N | bpe6000 | bpe10000 | bpe15846 | V_opt |
|---|---|---|---|---|
| 50M | 1.740 | 1.717 | **1.698** | ~16k |
| 100M | **1.681** | 1.683 | 1.781 | ~6k |
| 150M | 2.019 | 1.667 | **1.635** | ~16k |

## 关键诚实结论：**测不稳，不能下判断**

V_opt 在 50→100→150M 上 **非单调**（~16k → ~6k → ~16k），且同一 N 内相邻 V 出现 **0.3+ BPC 的跳变**（150M 的 bpe6000=2.019 vs bpe10000=1.667；100M 的 bpe15846=1.781 vs bpe6000=1.681）。这是**单种子 + 严重欠训练**造成的不稳定性：

- 600 步对 150M 仅 ~0.1% 的 Chinchilla 最优算力；
- 大词表的 embedding（如 V=16001 时 2VH≈20M 参数）在 600 步内**学不动**，被系统性惩罚 → V_opt 在每个 N 上不规则漂移。

→ **50-150M 的 V_opt 这次测不出可信值**。它既不是"持续升到 16k"（250 步时的假象），也不是"在 6k 饱和"——就是噪声。

## 对 V–N 律的最终态度

- **4-32M 段的 V∝N^{2/3}（0.666）依然干净可信**（那里 1000 步算力 adequate）。
- **≥50M 无法用本会话的算力干净延伸**：欠训练把大词表系统性打偏，掩盖了真值。
- 这**不证伪** V–N 律在更大 N 上成立——只是说明：**要测它，必须用 Chinchilla-token 训练量 + 多种子**。这是论文"档一①"的真正门槛（也是这次撞墙的根因）。

## 方法论副产品
- **`expandable_segments:True` + 小 micro-batch + accum** 是在这套 fla-SSM 上避免长训练显存碎片化/降速的可用配方，后续大规模实验应默认开启。

## 复现
```
python big_n_sweep.py     # expandable_segments + micro4/accum8, 可续跑增量存盘
python analyze_big_n.py   # 出图 big_n_sweep.png
```
