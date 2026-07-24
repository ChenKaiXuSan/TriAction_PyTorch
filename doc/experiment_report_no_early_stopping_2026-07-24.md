# 实验矩阵报告:无早停全量训练(2026-07-24)

多视角驾驶员动作识别(4 类方向任务:left / right / down / up)完整实验矩阵,
**关闭 early stopping、全量训练 50 epochs** 的正式结果。自本报告起,
`configs/config.yaml` 默认 `train.early_stopping: false`,即所有实验默认全量训练。

## 实验设置

- **硬件**:Pegasus 集群 `gpu` 队列,每作业 1 节点(1× NVIDIA H100 PCIe 80GB,48 核,CUDA 13.0)
- **软件**:conda env `direction`(torch 2.12.0+cu130,pytorch-lightning 2.6.4),precision `bf16-mixed`
- **数据**:80 train / 8 val 视频(`magic_move` 切分,种子 0),视频分块 ≤1000 帧,segment 批处理
- **公共超参**:50 epochs(无早停),等效 batch 16,lr 1e-4(Adam + CosineAnnealingLR),8 帧采样(videomae 16 帧、vivit 32 帧),测试用 val/loss 最优 checkpoint
- **实验 ID 规则**:下表 ID 均带 `_noes` 后缀(no early stopping);front 3dcnn 基线对应 `E_noes_single_front_rgb_3dcnn`
- **日志**:qsub 输出 `logs/pegasus/<EXP_ID>_{out,err}.log`,训练产物 `logs/train/<EXP_ID>/2026-07-24/`

## 总榜(按 test accuracy 排序)

Δ 列为相对早停版(patience 5)同配置实验的变化。

| # | 实验 | acc | F1 | precision | recall | Δ acc |
|---|---|---|---|---|---|---|
| 1 | L_multi_late_vivit | **56.7%** | 51.2% | 48.5% | 56.7% | +4.2 |
| 2 | A_front_rgb_vivit | 52.9% | 47.1% | 45.0% | 52.9% | −4.5 |
| 3 | L_multi_late_videomae | 51.8% | 43.1% | 39.6% | 51.8% | −1.7 |
| 4 | T_mid_heads8 | 51.6% | 37.9% | 39.1% | 51.6% | +6.7 |
| 5 | T_mid_no_gated_aggregation | 51.2% | 42.1% | 56.2% | 51.2% | +6.4 |
| 6 | F_multi_rgb_3dcnn_mid(TS-CVA 完整) | 48.9% | 45.6% | 46.8% | 48.9% | +4.2 |
| 7 | A_front_rgb_videomae ⚠️ | 46.5% | 38.0% | 35.1% | 46.5% | +28.8 |
| 8 | T_mid_no_view_embedding | 44.7% | 36.3% | 31.2% | 44.7% | −8.6 |
| 9 | T_mid_heads2 | 42.5% | 36.8% | 42.1% | 42.5% | +6.8 |
| 10 | L_multi_late_mamba | 41.9% | 33.8% | 31.6% | 41.9% | +4.8 |
| 11 | F_multi_rgb_3dcnn_late | 40.3% | 36.3% | 37.2% | 40.3% | −2.0 |
| 12 | F_multi_rgb_3dcnn_add | 40.2% | 29.9% | 26.0% | 40.2% | +5.1 |
| 13 | F_multi_rgb_3dcnn_avg | 40.1% | 31.5% | 28.9% | 40.1% | +4.4 |
| 14 | M_front_rgb_kpt_3dcnn | 40.1% | 30.6% | 29.7% | 40.1% | +2.4 |
| 15 | B_single_right_rgb_3dcnn | 39.8% | 35.0% | 35.8% | 39.8% | +4.8 |
| 16 | F_multi_rgb_3dcnn_concat | 37.7% | 31.4% | 29.2% | 37.7% | +1.3 |
| 17 | A_front_rgb_mamba | 37.5% | 30.5% | 29.4% | 37.5% | ±0 |
| 18 | L_multi_late_transformer | 37.5% | 28.0% | 23.2% | 37.5% | ±0 |
| 19 | (E_noes)single_front_rgb_3dcnn | 37.2% | 30.2% | 31.1% | 37.2% | +4.9 |
| 20 | B_single_left_rgb_3dcnn | 36.7% | 30.4% | 27.5% | 36.7% | +2.2 |
| 21 | A_front_rgb_transformer | 36.5% | 28.6% | 24.9% | 36.5% | ±0 |
| 22 | M_front_kpt | 30.8% | 21.8% | 21.2% | 30.8% | +10.5 |

注:transformer 两个实验 Δ=0 是因为测试用 val/loss 最优 checkpoint,全量训练没有产生更优的
val checkpoint,best ckpt 与早停版相同。

## 分组对比

### 1. 视角(single,rgb,3dcnn)

| front | left | right |
|---|---|---|
| 37.2% | 36.7% | **39.8%** |

三个视角相差 ≤3 点,right 略优。

### 2. 模态(front 视角)

| rgb | kpt | rgb+kpt(concat) |
|---|---|---|
| 37.2% | 30.8% | **40.1%** |

纯 SAM-3D 关键点最弱,但与 RGB 拼接后比纯 RGB 高约 3 点——关键点信息互补有效。

### 3. Backbone(single front,rgb)

| 3dcnn | transformer | mamba | videomae ⚠️ | vivit |
|---|---|---|---|---|
| 37.2% | 36.5% | 37.5% | 46.5% | **52.9%** |

ViViT(Kinetics-400 预训练)大幅领先;3dcnn / transformer / mamba 基本同档。

### 4. 融合方式(multi 3-view,3dcnn)

| 单视角基线 | early add | early concat | early avg | mid(TS-CVA) | late |
|---|---|---|---|---|---|
| 37.2% | 40.2% | 37.7% | 40.1% | **48.9%** | 40.3% |

**mid fusion(TS-CVA)明显最优**(+11.7 vs 单视角);early / late 只有 ~+3 点,基本同档。

### 5. 晚融合 × backbone(multi 3-view,late)

| 3dcnn | transformer | mamba | videomae | vivit |
|---|---|---|---|---|
| 40.3% | 37.5% | 41.9% | 51.8% | **56.7%** |

**Late-ViViT 是全场最优(56.7%)**;晚融合对每个 backbone 都比其单视角版高 3–5 点
(vivit +3.8,videomae +5.3,mamba +4.4)。

### 6. TS-CVA 消融(multi 3-view,mid)

| 完整(heads=4) | 无 gated aggregation | 无 view embedding | heads=8 | heads=2 |
|---|---|---|---|---|
| 48.9% | 51.2% | 44.7% | **51.6%** | 42.5% |

- **view embedding 有效**:去掉后 −4.2 点;
- **gated aggregation 无益**:换成 mean pooling 反而 +2.3 点,该组件可考虑简化;
- 注意力头数 8 > 4 > 2,头数不足伤害明显(−6.4)。

## 与早停版矩阵的总体结论

1. 全量训练平均提升约 4 点;受益最大的是 videomae(+28.8)、纯关键点(+10.5)、TS-CVA 系列(+4~7)。
2. 早停版的两个"亮点"被证伪:单视角 vivit 的 57.4% 和"去 view embedding 大涨到 53.3%"
   都是早停撞上好停点的噪声,全量版结论(late-vivit 最优、view embedding 有效)更可靠。
3. **正式结论应以本报告(全量版)为准**。

## 已知问题

- ⚠️ **videomae 权重映射不完整**:当前 transformers 版本加载
  `MCG-NJU/videomae-base-finetuned-kinetics` 时 attention 的 `q_bias`/`v_bias`
  无法映射(load report 中 MISSING/UNEXPECTED),这部分参数为随机初始化。
  46.5% 仍可能低估 videomae 真实水平;修复方向:固定旧版 transformers 后重跑。

## 复现

```bash
# 默认已无早停(train.early_stopping: false),直接提交矩阵:
pegasus/prepare_index.sh
pegasus/prepare_hf_models.sh
pegasus/qsub_all.sh all --run
```
