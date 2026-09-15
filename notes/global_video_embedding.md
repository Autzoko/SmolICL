# SmolVLA-ICL：Global Video Embedding 设计

## 1. 目标与功能边界

Global Embedding 从一条完整 Demo 的 RGB 视频和机械臂 State 序列中提取任务级表示，回答：

- Demo 在执行什么任务、操作什么对象；
- 任务由哪些主要子阶段组成，以及大致先后顺序；
- 任务的目标状态和整体运动模式是什么。

它不负责精确判断机器人当前处于 Demo 的哪一帧或哪一步。Global 与 Local 的职责应分开：

- **Global Task Tokens**：描述完整任务，只与 VLM 分支进行 Cross-Attention；
- **Local Demo Chunk**：提供已完成阶段匹配的局部参考，只与 Action Expert 进行 Cross-Attention；
- 二者可以共享部分视频骨干，但应保留独立的投影、位置编码和注意力路径。

## 2. 表征要求

Global Embedding 应满足：

1. **全局任务语义**：区分操作对象、目标位置和任务类型；
2. **时序信息**：对任务步骤的先后顺序敏感；
3. **RGB + State 融合**：视觉描述外部交互，State 补充关节、末端位姿和夹爪状态；
4. **速度鲁棒性**：同一任务在不同执行速度和采样率下表示接近；
5. **适度视角鲁棒性**：容忍固定视角附近的小幅扰动；
6. **可区分性**：相似场景中的不同任务不能得到近似表示；
7. **固定输出形状**：变长 Demo 输出固定数量的 Task Tokens；
8. **轻量且可缓存**：加载 Demo 时计算一次，不能显著增加每个 action chunk 的在线延迟。

## 3. 输入与输出

将完整 Demo 划分为按时间排序的 $K$ 个片段，每个片段采样 $L$ 帧：

\[
V_{\mathrm{demo}}\in
\mathbb{R}^{B\times K\times L\times3\times224\times224},
\qquad
S_{\mathrm{demo}}\in
\mathbb{R}^{B\times K\times L\times D_s}.
\]

其中 $B$ 是 batch size，$D_s$ 是 State 维度。还应保留时间戳和有效位 Mask，以支持变长 Demo 和不同采样率。

不建议把长 Demo 直接平均成单个向量。建议生成 $N_G=4\sim8$ 个 Global Task Tokens：

\[
G\in\mathbb{R}^{B\times N_G\times D_E}.
\]

首版可令 $D_E=720$，与 SmolVLA Action Expert 的 hidden dimension 一致。

## 4. 通用编码流程

对第 $k$ 个视频和 State 片段分别编码：

\[
v_k=E_{\mathrm{video}}(V_k)\in\mathbb{R}^{D_v},
\qquad
s_k=E_{\mathrm{state}}(S_k)\in\mathbb{R}^{D_s'}.
\]

State Encoder 可以先使用轻量 GRU 或 Temporal MLP。片段级融合并加入时间编码：

\[
x_k=W_f[v_k;s_k]+e_{\mathrm{time}}(k/K).
\]

将有序片段特征交给两层轻量 Temporal Transformer，并用可学习 Task Queries 压缩：

\[
G=\operatorname{TemporalAggregator}
\left([q_1,\ldots,q_{N_G};x_1,\ldots,x_K]\right)_{1:N_G}.
\]

~~~text
完整 Demo RGB ──> K 个有序视频片段 ──> 预训练视频编码器 ──> 片段特征
                                                               │
完整 Demo State ─> 对齐的 State 片段 ──> State Encoder ────────┤
                                                               v
                                              融合 + 时间位置编码
                                                               │
                                          轻量 Temporal Aggregator
                                                               │
                                               Global Task Tokens
                                                               │
                                             Cross-Attention with VLM
~~~

若视频骨干能输出时序特征图，应优先只池化空间维，保留若干 temporal tokens；最终时序压缩交给 Temporal Aggregator，而不是过早做全时空平均。

## 5. 候选视频编码器

> 名称说明：此前提到的 “MoViNet-40” 按 **MoViNet-A0** 记录。公开 MoViNet 系列使用 A0、A1、A2 等命名，没有标准的 MoViNet-40 型号。

| 候选模型 | 主要机制 | 优点 | 局限与实现注意点 | 计划定位 |
|---|---|---|---|---|
| **S3D** | 可分离 3D 时空卷积 | 约 8.3M 参数；TorchVision 原生；接入简单；运动建模明确 | Kinetics 与机器人操作存在域差异；全局池化可能损失阶段顺序 | 第一候选、轻量基线 |
| **MoViNet-A0** | 面向移动端的 2+1D/3D 视频卷积，可流式推理 | 约 3.1M 参数；计算量最低 | 官方预训练生态以 TensorFlow 为主；PyTorch 权重转换成本较高 | 第二候选、极致轻量实验 |
| **VideoMAE-Small** | Tubelet + Video Transformer，自监督掩码重建 | 全局视频表示自然；长程时序和迁移能力值得测试 | 更重；注意力随 token 数增长；许可证需确认 | 第三候选、表征能力实验 |
| **Swin3D-T** | 层次化窗口式时空 Self-Attention | 约 28.2M 参数；TorchVision 原生；表征较强 | 约 43.88 GFLOPs，显存和延迟更高 | 第四候选、精度上界 |

计划依次测试：

1. S3D；
2. MoViNet-A0；
3. VideoMAE-Small；
4. Swin3D-T。

为保证公平，应使用相同的 Demo 采样、State Encoder、Temporal Aggregator、输出 token 数和训练策略，只替换视频骨干及必要预处理。

## 6. 训练策略

第一阶段冻结预训练视频骨干，只训练：

- State Encoder；
- RGB/State Fusion；
- Temporal Aggregator 和 Task Queries；
- 投影层、Demo Expert 和新增 Cross-Attention。

模型跑通后，再以较小学习率解冻视频骨干最后一至两个 stage。除 action flow-matching loss 外，可加入任务级对比损失：

\[
\mathcal{L}_{\mathrm{global}}
=-\log
\frac{\exp(\operatorname{sim}(G_i,G_i^+)/\tau)}
{\sum_j\exp(\operatorname{sim}(G_i,G_j)/\tau)}.
\]

$G_i^+$ 是相同任务的另一条 Demo，不同任务作为负样本。这使 Global Tokens 学到任务身份，而不是相机背景或机器人外观。

还应进行 **Demo swap / hard negative** 测试：保持当前 observation 不变，替换成另一任务的 Demo。如果动作预测几乎不变，说明模型仍在忽略 Global Demo 信息。

## 7. 评测标准

- seen-task 和 unseen-composition 的任务成功率；
- 同任务 Demo 检索准确率和不同任务的特征可分性；
- 打乱 Demo 时间顺序后的性能下降；
- 正确 Demo 与错误 Demo 对动作输出的影响；
- Global Encoding 一次性延迟、峰值内存和缓存大小；
- 冻结/解冻骨干、RGB-only/RGB+State 的差异。

首版选择标准是在 unseen-task 成功率、Demo 依赖性和计算开销之间取得最好平衡，而不是只追求视频分类准确率。

## 8. 参考资料

- [TorchVision S3D](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.video.s3d.html)
- [TorchVision 视频模型对比](https://docs.pytorch.org/vision/main/models)
- [MoViNets](https://openaccess.thecvf.com/content/CVPR2021/papers/Kondratyuk_MoViNets_Mobile_Video_Networks_for_Efficient_Video_Recognition_CVPR_2021_paper.pdf)
- [TensorFlow Model Garden 视频模型](https://github.com/tensorflow/models/blob/master/official/vision/README.md)
- [VideoMAE-Small checkpoint](https://huggingface.co/MCG-NJU/videomae-small-finetuned-kinetics)
