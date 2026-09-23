# SmolVLA-ICL 实现备忘录

本文只记录当前代码中的稳定契约和仍需真实数据验证的事项。架构动机与注意力设计见
`smolvla_and_smolvla_icl_design.md`，集群操作见 `jubail_smolvla_icl_workflow.md`。

## 1. 当前训练数据链路

训练前按以下顺序生成数据产物：

1. `libero_manifest.py` 固定 Dataset revision、camera，以及 train/val/test 与
   demo/query 的 episode 划分。
2. `pairing_builder.py` 使用独立、冻结的 SigLIP matcher，为每个
   `train/query + val/query` episode 选择一条同 task Demo，并生成逐帧
   `local_anchor` sidecar。一个 epoch 内配对不变，不允许 Query 与自身配对。
3. `global_cache_builder.py` 为 sidecar 实际引用的 Demo 离线缓存冻结 S3D 的
   clip feature。State、时间和 mask 仍保留，供可训练 Global 路径使用。
4. `make_smolvla_icl_train_eval_datasets()` 在创建 worker 前执行全量 preflight，
   然后复用 LeRobot `LeRobotDataset` 构造 Query、Demo 子集。
5. `SmolVLAICLQueryDataset` 返回标准 Query 样本和轻量 `DemoSampleRef`；collator
   在 batch 内按 `demo_id` 去重 Global Demo，并按 `(demo_id, local_anchor)` 去重
   Local 窗口。

训练样本不会保存整条 Demo RGB。Dataset split 和 Demo/Query 身份只由 Manifest
决定，不再叠加 LeRobot 通用 `eval_split`。

## 2. 三条视觉路径的边界

### Matcher

- 使用与训练模型分离的冻结 SigLIP snapshot。
- Demo pooled embedding 可以离线复用，只负责 DTW 和 `local_anchor`。
- Query 只能看到当前与历史 observation；不读取未来 Query 帧。
- Matcher feature 不进入 Action Loss，也不作为 Local Encoder 输入。

### Global Demo

- 离线只缓存冻结 S3D 的 clip feature。
- State Encoder、RGB/State fusion、Temporal Aggregator 和 Task Queries 在线训练。
- batch 内相同 `demo_id` 只加载一次，再用 `sample_to_demo` 映射回各 Query。

### Local Demo

- sidecar 只给出 anchor，Dataset 在线读取其附近 48 帧 raw RGB、State 和时间。
- RGB 在 Dataset、collator 和 `LocalDemoBatch` 中始终是 CPU `uint8`。
- forward 中按 `local_vision_encode_batch_size` 把图像 microbatch 异步搬到 GPU，
  转为浮点并送入 Query 共用的可训练 SmolVLA Vision Encoder。
- 每帧 spatial tokens 立即经可学习 pooling 压成一个向量；输出保留计算图，
  Action Loss 可以更新共享视觉编码器。
- 训练期不缓存 Local visual tokens，避免切断梯度或使用过期特征。

`LocalDemoBatch.pin_memory()` 配合 DataLoader 的 `pin_memory=True`；State、时间、
位置和 mask 等小张量可以一次搬到模型设备，完整 RGB 不会整体进入 GPU。

## 3. 模型侧接口

包级公共 API 与其他 LeRobot policy 保持一致，仅暴露：

- `SmolVLAICLConfig`
- `SmolVLAICLPolicy`
- `make_smolvla_icl_pre_post_processors`

训练调用保持标准 `policy(batch)`。SmolVLA-ICL 的 collator 注入两个显式字段：

- `smolvla_icl.global_demo: GlobalDemoBatch`
- `smolvla_icl.local_demo: LocalDemoBatch`

rollout 使用 `policy.set_demo(...)` 注册完整 Demo。该路径会建立冻结 matcher cache、
Global 条件和按 anchor 读取 Local window 所需的状态；每次重新规划仅编码当前 Query
与当前 Local window。训练和 rollout 使用不同的结构化输入，不使用“万能字典”兼容层。

## 4. 缓存身份与失效规则

Global cache identity 包含：

- Dataset repo/revision、camera 和 State key；
- 会影响 clip feature 的 S3D 与 clip 配置；
- S3D snapshot；
- State normalization 统计。

目录级 `cache_manifest.json` 还绑定当前数据 Manifest，并列出所有 Demo entry。
训练启动时检查 sidecar 所需 Demo 是否完整、episode 长度是否一致、Tensor 形状及
identity 是否有效。改变数据 revision、camera、S3D 权重、clip 划分或归一化统计后，
必须重建 Global cache。

## 5. 已明确的研究边界

- Demo 与 Query 必须 task 相同、robot embodiment/State 定义相同、camera/view 兼容。
- 当前 DTW 假设 layout、轨迹和拓扑近似；不处理“直接下降”和“先侧移再下降”这类
  路径拓扑明显不同的配对。
- DTW 只提供阶段语义，不复制 Demo action；但错误 anchor 仍会提供错误 Local 语义。
- LIBERO 首版使用 Franka Panda；State/Action 的真实维度由 LeRobot metadata 检查，
  再补齐到 SmolVLA 的固定维度。

## 6. 仍需真实数据或 GPU 闭合的验证

以下项目不应继续用兜底分支掩盖，而应在 LIBERO 数据和训练机器上直接验证：

1. Manifest、sidecar 和 Global cache CLI 对真实 LIBERO revision 的全量运行。
2. 单 batch 的 Query/Global/Local shape、dtype、mask 与 episode 边界检查。
3. Action Loss 到 Local Encoder、共享 Vision Encoder、Demo Expert 和两条 cross-attention
   gate 的非零有限梯度。
4. `uint8` CPU Local RGB + pinned memory + microbatch 编码后的峰值 GPU 显存与吞吐。
5. 训练恢复时 epoch-indexed pairing 的确定性，以及多 worker/DDP 下没有重复缓存构建。
6. validation 只使用 `val/query`，test split 不参与训练、归一化统计或模型选择。

这些验证通过前，代码链路已闭合，但不能宣称真实 LIBERO 训练已经完成验证。
