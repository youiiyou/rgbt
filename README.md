# NuanceID shared-text baseline

这是 NuanceID 后续实验的干净主干：保留 IRRA 的 CLIP 全局检索思路，只做视频与可见光/红外任务所必需的改动。VCM 和 BUPTCampus 使用同一套训练、模型、评估和记录接口，运行时只切换数据集名称。

## 固定 baseline

- 输入：每条 tracklet 采样 6 帧，模型边界张量为 `[B, T, C, H, W]`。
- 采样：把有序帧序列分成 6 段并取每段中点；不足 6 帧时等间隔重复。
- 视觉：同一个 CLIP ViT 编码每帧，取每帧 CLS，再沿时间维做均值池化。
- 文本：RGB 来源描述经一个共享 CLIP 文本编码器得到唯一文本特征；baseline 没有 prompt、adapter 或 modality routing。
- 训练：RGB 与真实 IR tracklet，损失固定为 `SDM + ID`。
- BUPT：使用 `train + train_auxiliary`，完全排除 `FakeIR`。
- 评估：同一 checkpoint 分别跑 RGB、IR、mixed gallery，默认同时报告 text-to-video 与 video-to-text 的 R1/R5/R10、mAP、mINP。

```text
caption ── CLIP text encoder ─────────────────────── shared text feature

frame 1 ─┐
frame 2 ─┤
   ...   ├─ shared CLIP ViT ─ per-frame CLS ─ mean ─ video feature
frame 6 ─┘

training: SDM(shared text, video) + ID(shared classifier)
testing : cosine similarity, independently for RGB / IR / mixed gallery
```

本分支刻意不包含 MLM、灰度 IR、IR-only、text-guided pooling、residual adapter、DSNL 或 CMNA。旧实验已保存在 `codex/archive-pre-nuanceid`，不能当作本 baseline 的已验证组成。

## 当前外部数据契约

数据位于 `/data/ydl/datasets`，不会复制进 Git。

| 数据集 | train IDs | train tracklets | queries | RGB gallery | IR gallery | mixed gallery |
|---|---:|---:|---:|---:|---:|---:|
| VCM | 500 | 2961 | 1261 | 1261 | 1261 | 2522 |
| BUPT | 2004 | 9008 | 1076 | 2422 | 2422 | 4844 |

标注校验值：

- VCM `VCM.json`: `db2d5c9eec7f4c79911ae892bb4e7125e8b94f9ab56f60087f7f7517efccd3fe`
- BUPT `BUPT.json`: `81e46435b916f361b00a4976736983622fd29b660a89d84569304e55a74180e5`

VCM 目录树负责 tracklet、相机、模态和帧顺序，JSON 负责 RGB 来源 captions 与 query 标记。IR 训练 tracklet 复用同身份自然排序第一台 RGB 相机的 caption。

BUPT 使用官方 `train.txt`、`train_auxiliary.txt`、`query.txt`、`gallery.txt`。`RGB/IR` 训练行展开为一条真实 RGB 和一条真实 IR tracklet；query caption 始终来自相同 `(PID, camera)` 的 RGB JSON record。正式运行会同时快照 JSON 和四个 protocol 文件。

## 先验证再训练

所有 Python 命令使用现有 Conda 环境：

```bash
conda run -n rgbt python -m unittest discover -s tests -v

conda run -n rgbt python scripts/check_dataset.py VCM --load-batch
conda run -n rgbt python scripts/check_dataset.py BUPT --load-batch
```

BUPT 第一次启动需要扫描约 9000 条真实 tracklet，在当前磁盘上约需 2–3 分钟；期间没有日志不代表卡死。

真实 CLIP 单 batch 前向/反向：

```bash
DEVICE=cuda CUDA_DEVICE=0 ./smoke_baseline.sh VCM
DEVICE=cuda CUDA_DEVICE=0 ./smoke_baseline.sh BUPT
```

托管执行环境若看不到 CUDA，请在自己的 GPU 终端运行以上命令。

## 训练两个 baseline

两条命令只有数据集名称不同，其余固定配置完全一致：

```bash
CUDA_DEVICE=0 ./run_baseline.sh VCM
CUDA_DEVICE=0 ./run_baseline.sh BUPT
```

默认输出到 `/data/ydl/experiments/nuanceid/<DATASET>/...`。可用环境变量修改：

```bash
OUTPUT_ROOT=/path/to/experiments DATA_ROOT=/data/ydl/datasets \
CUDA_DEVICE=0 ./run_baseline.sh VCM
```

脚本固定 30 epochs、batch size 8、学习率 `5e-6`、warmup 1 epoch、seed 1、single caption、random sampler、6 帧、`sdm+id`。额外命令行参数放在数据集名称后，可显式覆盖默认值。

每次训练目录会保存：

```text
configs.yaml
annotations/<VCM.json|BUPT.json>
protocols/*.txt                 # 仅 BUPT
best.pth
last.pth
training_summary.json
train_log.txt
TensorBoard event file
```

配置中记录源码 commit/dirty 状态、seed、帧数、采样策略、训练模态、FakeIR 策略、标注路径与 SHA256、query 数量、gallery 计数和参数量。`training_summary.json` 记录总耗时、最佳 epoch、进程数与 GPU 峰值显存。

## 三种 gallery 评估

```bash
CUDA_DEVICE=0 ./evaluate_all.sh /path/to/run/configs.yaml
```

也可把 checkpoint 作为第二个参数。脚本独立运行 RGB、IR、mixed 三次，生成：

```text
eval_rgb.json
eval_ir.json
eval_mixed.json
eval_summary.json
eval_summary.csv
```

汇总器会拒绝 checkpoint、query 数、标注 SHA、seed 或关键 baseline 配置不一致的三个结果。

## 标注重建

如外部 caption 文件被有意修改，可重建规范 JSON：

```bash
./json_vcm.sh
./json_bupt.sh
```

重建后必须重新运行数据检查，并把新的 SHA256 与 query 数写进实验说明；不能把不同 query 快照的结果放在同一表里直接比较。

## 代码入口

- `train.py`：训练编排、快照输入、保存配置与 checkpoint。
- `test.py`：加载训练配置和快照，评估单个 gallery。
- `datasets/`：两个正式数据协议、六帧采样、tracklet transform 和 DataLoader。
- `model/`：CLIP 主干、六帧均值池化、SDM/ID。
- `processor/`：epoch 循环与推理入口。
- `solver/`：optimizer 与 warmup scheduler。
- `utils/`：指标、配置、日志、checkpoint 和实验元数据。
- `scripts/`：数据检查、标注构建、真实 smoke、结果汇总。
- `tests/`：不依赖外部大数据的协议与数值单元测试。

逐文件与逐张量说明见 [docs/CODE_WALKTHROUGH.md](docs/CODE_WALKTHROUGH.md)。

## 旧 VCM checkpoint

历史配置没有 `caption_source` 字段时，测试入口按 `legacy` 解析目录内 `caption.txt`。这仅用于复现旧 checkpoint；新的正式训练强制 `caption_source=json` 并快照标注。当前 provisional 根目录对应 1260 个 query，而正式 VCM JSON 对应 1261 个 query，二者必须分表报告。

## 来源

模型主干和 SDM/ID 基于 IRRA（CVPR 2023）代码。项目 LICENSE 保留原 MIT 授权。NuanceID 后续模块应从本 baseline commit 新建独立分支，每次只改变一个研究变量。
