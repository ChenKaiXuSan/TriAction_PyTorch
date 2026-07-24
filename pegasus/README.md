# Pegasus 集群配置与实验矩阵

本文件夹集中存放 TriAction 在 **Pegasus 集群**上运行所需的全部机器相关内容：Hydra 路径覆盖（`pegasus.yaml`）、环境引导（`setup_env.sh`）、NQSV 任务脚本（`run_*.sh`）与批量提交工具（`qsub_all.sh`）。仓库根目录的 `configs/config.yaml` 保持不动，里面的默认路径属于另一台机器。脚本风格参考 `ClinicalGait-CrossAttention_ASD_PyTorch/pegasus/`。

## 机器概况（2026-07 已验证）

- 登录节点 `pegasus03`，**无 GPU**；计算节点通过 NQSV 调度器提交（`qsub`，位于 `/opt/nec/nqsv/bin`），gpu 队列单作业 24h 上限（smoke 作业 2h）。
- 计算节点**无外网**，Hugging Face 权重必须先在登录节点缓存（见下文一次性准备）。
- conda 环境是 **`direction`**（torch 2.12.0+cu130 / pytorch-lightning 2.6.4 / hydra 1.3.2）。本机没有 CLAUDE.md 里提到的 `drivefusion` 环境——那是另一台机器的。用 `TRIACTION_CONDA_ENV=...` 可覆盖。
- `/work/SKIING/chenkaixu` 与 `/work/1/SKIING/chenkaixu` 是同一位置，脚本统一使用前者。

## 数据位置（已逐项核对）

数据根：`/work/SKIING/chenkaixu/data/drive`（`TRIACTION_DATA_ROOT=...` 覆盖）

| 内容 | 路径 | 状态 |
|---|---|---|
| 标注 | `label/`：88 个 `person_XX_(day\|night)_(high\|low)_h265.json` | 22 人 × 4 环境，命名与代码正则一致 |
| 视频 | `videos_split/{01..22}/{昼多い,昼少ない,夜多い,夜少ない}/{front,left,right}.mp4` | 无缺失（另有不使用的 `drive_view.mp4`） |
| SAM-3D 关键点 | `sam3d_body_results_right/{person}/{环境}/{front,left,right}/NNNNNN_sam3d_body.npz` | 层级与 `cross_validation.py` 期望一致 |
| start/mid/end 标注 | `annotation/split_mid_end/mini.json` | **比默认布局多一层 `annotation/`**，`pegasus.yaml` 与 `run_common.sh` 已各自处理 |
| split 索引缓存 | `index_mapping/` | 旧 K-fold 的 `fold_*.json` 已弃用；新代码用 `index_single_magicmove.json`，缺失时自动生成 |

## 交互式运行（调试）

```bash
conda activate direction
python -m project.main --config-dir pegasus --config-name pegasus [其他 overrides 照常写]
python -m project.eval --config-dir pegasus --config-name pegasus
```

`pegasus.yaml` 通过 `defaults: [/config, _self_]` 继承 `configs/config.yaml`，只覆盖 `paths.*`。注意路径同时写在 `pegasus.yaml`（交互式）和 `run_common.sh`（qsub 作业）两处，改数据位置时两处一起改。

## 实验矩阵总览（22 个正式实验 + 1 个 smoke）

去重原则：同一配置只训练一次，多张对比表复用同一个 run。

| 组 | 实验 ID | view | 输入 | backbone | 融合 | 备注 |
|---|---|---|---|---|---|---|
| S0 | `S0_smoke_single_front_rgb_3dcnn` | single front | rgb | 3dcnn | – | 1 epoch 冒烟测试 |
| B 视角 | `B_single_front_rgb_3dcnn` | single front | rgb | 3dcnn | – | 也是 A 组 3dcnn / M 组 rgb 基线 |
| B 视角 | `B_single_left_rgb_3dcnn` | single left | rgb | 3dcnn | – | |
| B 视角 | `B_single_right_rgb_3dcnn` | single right | rgb | 3dcnn | – | |
| M 模态 | `M_front_kpt` | single front | kpt | – | – | SAM-3D 关键点 |
| M 模态 | `M_front_rgb_kpt_3dcnn` | single front | rgb+kpt | 3dcnn | modality concat | |
| A 骨干 | `A_front_rgb_transformer` | single front | rgb | transformer | – | |
| A 骨干 | `A_front_rgb_mamba` | single front | rgb | mamba | – | |
| A 骨干 | `A_front_rgb_videomae` | single front | rgb | videomae | – | HF, batch 4 / accum 4 / 16 帧 |
| A 骨干 | `A_front_rgb_vivit` | single front | rgb | vivit | – | HF, batch 2 / accum 8 / 32 帧, `refs/pr/3` |
| F 融合 | `F_multi_rgb_3dcnn_add` | multi 3-view | rgb | 3dcnn | early add | |
| F 融合 | `F_multi_rgb_3dcnn_concat` | multi 3-view | rgb | 3dcnn | early concat | |
| F 融合 | `F_multi_rgb_3dcnn_avg` | multi 3-view | rgb | 3dcnn | early avg | |
| F 融合 | `F_multi_rgb_3dcnn_mid` | multi 3-view | rgb | 3dcnn | mid (TS-CVA) | 也是 T 组完整模型（heads=4） |
| F 融合 | `F_multi_rgb_3dcnn_late` | multi 3-view | rgb | 3dcnn | late | 也是 L 组 3dcnn 基线 |
| L 晚融合 | `L_multi_late_transformer` | multi 3-view | rgb | transformer | late | |
| L 晚融合 | `L_multi_late_mamba` | multi 3-view | rgb | mamba | late | |
| L 晚融合 | `L_multi_late_videomae` | multi 3-view | rgb | videomae | late (logit_mean) | HF 配置同 A 组 |
| L 晚融合 | `L_multi_late_vivit` | multi 3-view | rgb | vivit | late (logit_mean) | HF 配置同 A 组 |
| T 消融 | `T_mid_no_gated_aggregation` | multi 3-view | rgb | 3dcnn | mid | 门控聚合 → mean pooling |
| T 消融 | `T_mid_no_view_embedding` | multi 3-view | rgb | 3dcnn | mid | 去掉视角嵌入 |
| T 消融 | `T_mid_heads8` | multi 3-view | rgb | 3dcnn | mid | 注意力头数 8 |
| T 消融 | `T_mid_heads2` | multi 3-view | rgb | 3dcnn | mid | 注意力头数 2 |

公共设置：50 epochs、precision `bf16-mixed`（H100 原生 bfloat16）、非 HF 实验 batch 16 / accum 1 / 8 帧、`train.gpu=[0]`（每作业 1 GPU 节点）。所有参数可用环境变量覆盖后 qsub，如 `MAX_EPOCHS=30 qsub pegasus/run_f_multi_mid.sh`（可用变量见 `run_common.sh` 顶部）。

## 批量提交步骤

```bash
cd /work/SKIING/chenkaixu/code/TriAction_PyTorch

# 0. 一次性准备（登录节点，有网络）
pegasus/prepare_index.sh        # 预生成共享 train/val 索引，避免并发作业竞争重建
pegasus/prepare_hf_models.sh    # 仅 videomae/vivit 作业需要：装 transformers + 下载权重

# 1. 冒烟测试（先单独跑通）
pegasus/qsub_all.sh smoke --run

# 2. 正式矩阵（默认 dry-run，确认后加 --run）
pegasus/qsub_all.sh                  # 查看将提交什么
pegasus/qsub_all.sh all --run        # 提交全部 22 个作业
pegasus/qsub_all.sh fusion --run     # 或按组提交
```

## 环境与路径细节

- HF 缓存：`HF_HOME=/work/SKIING/chenkaixu/hf_cache`；计算节点无外网，HF 作业设置 `HF_HUB_OFFLINE=1`，权重必须先用 `prepare_hf_models.sh` 缓存。
- 日志：qsub stdout/stderr 在 `logs/pegasus/<EXP_ID>_{out,err}.log`；训练日志/checkpoint 照旧在 `logs/train/<EXP_ID>/<date>/<time>/`。

## 注意事项

- 索引缓存：`${DATA_ROOT}/index_mapping/index_single_magicmove.json` 只在缺失时生成。改了数据路径或切分逻辑后要先删掉再重新 `prepare_index.sh`。
- 单个作业超 24h 时：用 `MAX_EPOCHS=30 qsub pegasus/run_xxx.sh` 减 epoch，或先用 smoke 作业估算单 epoch 用时。
- 作业名（`#PBS -N`）限制较短，与实验 ID 不完全一致；对照上表查询。
