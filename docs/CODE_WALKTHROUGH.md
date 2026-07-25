# Baseline 代码解读

这份文档按一次样本从磁盘进入指标表的顺序解释代码。先掌握这条主线，再做 adapter、DSNL 或 CMNA，会比从单个函数倒推容易很多。

## 1. 一次训练从哪里开始

入口是 `train.py`：

1. `utils/options.py` 解析配置。
2. 固定随机种子并解析 CPU/CUDA。
3. `utils/experiment.py` 把 caption JSON 快照到本次输出目录；BUPT 还会快照四个 protocol 文件。
4. `datasets/build.py` 创建训练 loader，以及 mixed-gallery 验证 loader。
5. `model/build.py` 创建共享文本 baseline。
6. `solver/build.py` 创建 optimizer 和 scheduler。
7. `processor/processor.py` 执行 epoch 循环，以 mixed-gallery R1 选择 `best.pth`，结束时保存 `last.pth`。

最重要的原则是：配置与数据快照先固定，模型才开始训练。以后对比方法时，两个实验必须使用相同 commit、JSON 快照、protocol 快照、seed 和超参数。

## 2. 数据层

### `datasets/video_utils.py`

这是两个数据集共用的底层契约：

- `natural_key`：保证 `D2` 排在 `D10` 前面。
- `collect_frames`：只收集合法图像，排除 caption mosaic，并按文件名自然排序。
- `uniform_sample_frames`：长序列做 6 段中点采样；短序列等间隔重复。
- `load_caption_records`：校验 JSON schema、RGB 来源、video 类型和 `(split, pid, camera)` 唯一性。
- `build_eval_dict`：生成 evaluator 统一需要的 PID、frame paths、modality、caption 列表。

采样只看帧顺序和配置，不使用身份标签，因此不会泄漏测试真值。

### `datasets/vcm.py`

VCM 的目录树是 tracklet 真值，JSON 是 caption/query 真值。

- 扫描 `Train/<PID>/{rgb,ir}/<camera>` 与 `Test/...`。
- JSON 模式严格比较所有 RGB `(split, PID, camera)`；少一条、多一条或路径不一致都会报错。
- RGB 训练 tracklet 使用自己的相机 caption。
- IR 训练 tracklet 使用同身份自然排序第一台 RGB 相机的 caption。
- 测试 query 是 JSON 中 `split=test, is_query=true` 的第一条 caption。
- 同一 query 列表原样复用于 RGB、IR、mixed gallery。

`legacy` 模式只服务旧 VCM checkpoint。它从目录内读取 `caption.txt`，不应拿来启动新的正式实验。

### `datasets/bupt.py`

BUPT 同时使用官方 protocol 与 RGB caption JSON：

- `train.txt + train_auxiliary.txt` 共同组成训练身份。
- 三个身份集合 `train / train_auxiliary / test` 必须互斥。
- `RGB/IR` 行展开到真实 `RGB` 和真实 `IR` 文件夹。
- gallery 行必须明确是 `RGB` 或 `IR`。
- query 行用于确定 query 身份/相机，但文本仍取相同 `(PID, camera)` 的 RGB JSON caption。
- 文件名正则精确解析 `PID_MODALITY_CAMERA_TRACKLET_FRAME.jpg`。
- 代码从不扫描 `FakeIR`，检查脚本和测试也会验证没有 `FakeIR` 路径泄漏。

### `datasets/bases.py`

训练样本统一为：

```text
(pid, image_id, [6 frame paths], RGB-derived caption, modality)
```

`ImageTextDataset.__getitem__` 读取六张图，并返回：

```text
pids         [scalar]
image_ids    [scalar]
images       [T, C, H, W]
caption_ids  [77]
modalities   [scalar]   # 0=RGB, 1=IR
```

DataLoader 的 `collate` 后得到 `[B, T, C, H, W]`。同一 tracklet 的所有帧会重放相同 Python、NumPy 和 PyTorch RNG 状态，因此随机翻转、裁剪、擦除在时间维保持空间一致。

`ImageDataset` 用于 gallery，返回 `(pid, modality, images)`；`TextDataset` 用于 query，返回 `(pid, token_ids)`。

### `datasets/build.py`

这里把数据对象、transform、sampler 和 DataLoader 接起来。

- 默认正式脚本使用 random sampler，与现有六帧 IRRA 配置一致。
- identity sampler 仍可显式选择，并支持当前五字段样本。
- 训练时只构建 mixed gallery 作为 checkpoint 选择集。
- 最终 RGB/IR/mixed 结果由 `evaluate_all.sh` 对同一 checkpoint 独立产生。

## 3. 模型层

### `model/clip_model.py`

这是从 IRRA 继承的 OpenAI CLIP 实现和权重加载代码。`build_CLIP_from_openai_pretrained` 会载入 `ViT-B/16`，并把原始方形位置编码插值到 `384×128` 对应的 patch 网格。

这个文件属于基础设施，后续研究通常不应直接在里面堆新模块；优先在 `model/build.py` 中组合 CLIP 输出。

### `model/build.py`

`IRRA` 是 baseline 模型：

```text
images [B,T,C,H,W]
  -> reshape [B*T,C,H,W]
  -> CLIP image tokens [B*T,L,D]
  -> reshape [B,T,L,D]
  -> each frame token 0 (CLS) [B,T,D]
  -> mean over T [B,D]
```

文本走同一个 CLIP text encoder，取 end-of-text token 位置的特征 `[B,D]`。这里没有根据 modality 改写文本，所以叫 shared-text baseline。

分类器对 image feature 和 text feature 共享。`modalities` 会一直保留在 batch 中，为后续 modality-adaptive 方法提供接口，但 baseline 前向不会使用它改变文本。

### `model/objectives.py`

- `compute_similarity`：先做 L2 normalize，再计算 cosine similarity matrix。
- `compute_sdm`：保持 IRRA 的 Similarity Distribution Matching；同 PID 样本共同构成正匹配分布。
- `compute_id`：image 与 text 两侧交叉熵的平均。

总损失由 `processor` 收集所有名称以 `_loss` 结尾的输出后相加。

## 4. 训练循环

`processor/processor.py` 每个 iteration 做：

```text
batch -> device -> model(batch) -> sum losses
      -> finite check -> backward -> optimizer.step
```

每个 epoch 后 scheduler 更新。到 `eval_period` 时，主进程对 mixed gallery 评估；R1 变好则保存 `best.pth`。多进程会在评估后同步，避免其他 rank 提前进入下一 epoch。训练结束保存 `last.pth`。

TensorBoard 与日志记录总损失、SDM、ID、image/text 分类准确率、学习率和 temperature。

## 5. Optimizer 与学习率

`solver/build.py` 为每个参数建立 parameter group：

- CLIP 主干使用基础学习率。
- bias 使用 `bias_lr_factor` 与独立 weight decay。
- ID classifier 使用 `lr_factor` 倍学习率，因为它是随机初始化层。

`solver/lr_scheduler.py` 提供 warmup 后的 cosine/step/linear/poly/exp。正式 baseline 固定 cosine、1 epoch warmup、30 epochs。

## 6. 评估与指标

入口 `test.py` 先加载训练时保存的 `configs.yaml`，再验证 annotation/protocol 快照 SHA256。这样外部 live JSON 后续被修改，也不会悄悄改变测试 query 数。

`utils/metrics.py`：

1. 编码全部 query text features。
2. 编码选定 gallery 的 video features，并保留每个候选的 modality。
3. 计算 cosine similarity。
4. 对每个 query 排序并计算 R1/R5/R10、mAP、mINP。
5. 默认将矩阵转置，再计算 video-to-text 指标。

mixed gallery 只使用候选自身的 modality metadata 做计数；shared-text baseline 不做路由，也绝不使用候选身份作为打分 oracle。

`evaluate_all.sh` 连续调用三次 `test.py`。`scripts/summarize_results.py` 会检查三份结果是否来自相同 checkpoint、commit、seed、query 快照和 baseline 配置。

## 7. 实验记录相关文件

- `utils/experiment.py`：设备解析、Git 状态、JSON/protocol 快照与 SHA 校验、结果元数据。
- `utils/iotools.py`：图片读取、YAML 配置和 JSON 结果读写。
- `utils/checkpoint.py`：保存/恢复模型、optimizer、scheduler；兼容旧 DDP 的 `module.` 前缀。
- `utils/logger.py`：控制台与训练/测试日志。
- `utils/meter.py`：移动统计。
- `utils/comm.py`：分布式 rank、barrier 与归约工具。
- `utils/simple_tokenizer.py`：CLIP BPE tokenizer。

## 8. 工具与测试

- `scripts/build_caption_json.py`：从外部 caption 文件重建 VCM/BUPT 规范 JSON。
- `scripts/check_dataset.py`：在真实数据上检查固定计数、SHA、FakeIR、query 复用和 batch shape。
- `scripts/smoke_baseline.py`：真实 CLIP 单 batch 前向/反向和 finite gradient 检查。
- `tests/test_video_utils.py`：帧采样、自然排序、同 tracklet augmentation、identity sampler。
- `tests/test_datasets.py`：合成 VCM/BUPT 的 split、caption、modality、FakeIR 和 coverage 契约。
- `tests/test_model_metrics.py`：时间均值、SDM finite loss、双向指标。
- `tests/test_experiment.py`：标注/protocol 快照与历史配置兼容。
- `tests/test_build_caption_json.py`：caption JSON builder。

## 9. 后续实验怎样分支

不要直接改 baseline 默认路径。每个研究变量从固定 baseline commit 新建分支，并新增显式配置：

1. shared text -> modality-adaptive text；其余不变。
2. uniform mean ->普通 temporal baseline；其余不变。
3. DSNL only。
4. CMNA only。
5. full NuanceID。

每次先跑单元测试、两个数据检查和对应数据集的真实 GPU smoke，再开始长训练。结果比较必须同时看 RGB、IR、mixed，且保留 query 数与 annotation SHA。
