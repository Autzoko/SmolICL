# SmolVLA 与 SmolVLA-ICL 结构笔记

## 1. SmolVLA Baseline

### 1.1 输入和输出

SmolVLA 的条件输入是当前视觉、语言指令和机器人 State：

\[
X_{\mathrm{prefix}}=
[X_{\mathrm{vision}},X_{\mathrm{language}},X_{\mathrm{state}}].
\]

动作侧输入是加入 flow-matching 噪声的 action chunk 和时间步：

\[
X_{\mathrm{suffix}}=E_a(A_t)+E_\tau(t).
\]

默认 action chunk 长度为 50，State 和 Action 最多补齐到 32 维。Action Expert 预测条件速度场：

\[
\hat u_\theta(A_t,t\mid O),
\]

并通过多步数值积分从噪声恢复动作序列。

### 1.2 VLM 与 Action Expert

- **VLM 分支**：编码 Vision、Language、State，hidden size 为 960；
- **Action Expert 分支**：由 VLM 文本模型配置缩放而来，宽度系数为 0.75，hidden size 为 720；
- 两套 hidden 在对应层共同参与注意力计算，但保留各自的归一化、MLP 和残差路径；
- 因此它不是简单的“VLM 完成后再运行 Expert”，而是层间持续交换条件信息的双分支结构。

### 1.3 Self-Attention 与 Cross-Attention

SmolVLA 在 Expert 层交替使用两类注意力：

- **联合 Self-Attention**：Prefix 与动作 token 联合参与注意力，动作 token 之间交换信息，建立 action chunk 的时间连续性；
- **Cross-Attention**：Action Hidden 作为 Query，VLM Prefix Hidden 作为 Key/Value，使动作反复读取视觉、语言和 State 条件。

\[
\text{Self-Attention}
\Rightarrow
\text{动作内部协调和时序连续性},
\]

\[
\text{Cross-Attention}
\Rightarrow
\text{动作与场景、指令和 State 重新对齐}.
\]

### 1.4 Prefix KV Cache

在一次 action chunk 的多步 flow-matching 去噪中，Vision、Language 和 State 条件保持不变，因此可以缓存每层 Prefix 的 Key/Value：

\[
K_l^{\mathrm{prefix}},V_l^{\mathrm{prefix}}.
\]

缓存的是当前一次策略调用的条件 hidden 投影，并不是整段任务历史。下一次重新观测并生成新的 action chunk 时，Prefix 与 KV Cache 都应更新。

### 1.5 残差结构

每个分支在注意力和 MLP 后保留自己的残差流：

\[
H'_l=H_l+\operatorname{Attention}(\operatorname{Norm}(H_l)),
\]

\[
H_{l+1}=H'_l+\operatorname{MLP}(\operatorname{Norm}(H'_l)).
\]

跨分支信息通过注意力作为增量注入，而不是覆盖原 hidden。新增 Demo 条件时也应保留这种设计。

## 2. SmolVLA-ICL

### 2.1 目标与组成

SmolVLA-ICL 在保留 SmolVLA 视觉—语言—动作能力的基础上，引入 RGB 或 RGB+State Demo，以根据示范执行：

- seen tasks；
- 由已学习原语重新组合形成的 unseen tasks。

核心模块包括：

1. 原 SmolVLA VLM；
2. 原 Action Expert；
3. 新增 Demo Expert；
4. Global Task Tokens；
5. 经过阶段对齐的 Local Demo Chunk。

### 2.2 Global 与 Local 分工

两类 Demo 信息同时使用，但不混用：

- **Global Task Tokens**：由完整 Demo 产生，描述任务身份、对象、目标和阶段顺序，只与 VLM 做 Cross-Attention；
- **Local Demo Chunk**：围绕当前匹配阶段提取，描述当前附近的状态变化和未来参考，只与 Action Expert 做 Cross-Attention。

\[
H_V'=H_V+
\operatorname{CrossAttn}(Q=H_V,K/V=G),
\]

\[
H_A'=H_A+
\operatorname{CrossAttn}(Q=H_A,K/V=H_D^{\mathrm{local}}).
\]

这样形成非对称的信息瓶颈：VLM 从 Global Demo 理解任务，Action Expert 从 Local Demo 获取阶段相关参考。

### 2.3 Demo Expert

Demo Expert 的 Transformer 配置由 VLM 配置缩放生成，构造方式与 Action Expert 一致，但参数独立。它将 Demo RGB+State tokens 转换为可供 Action Expert 读取的 hidden：

\[
H_D^{(0)}=E_D(D_{\mathrm{local}}),
\]

\[
H_D^{(l+1)}
=H_D^{(l)}+F_D^{(l)}(H_D^{(l)}).
\]

Demo Hidden 不应在中间使用一次后立即丢弃。推荐逐层保留和更新，让 Action Expert 在多个深度读取，并通过残差保护原始 Demo 信息。

### 2.4 推荐的信息交互

每个宏观 block 可以采用：

1. 各分支内部 Self-Attention；
2. VLM 从 Global Task Tokens 读取任务语义；
3. Demo Expert 更新 Local Demo Hidden；
4. Action Expert 从 VLM Hidden 读取场景条件；
5. Action Expert 从 Local Demo Hidden 读取阶段参考；
6. 各分支执行残差和 MLP 更新。

~~~text
Current Vision + Language + State
                 │
                VLM <──── Cross-Attention ──── Global Task Tokens
                 │
          Scene-conditioned Hidden
                 │
                 ├──── Cross-Attention ────> Action Expert
                 │                              ▲
                 │                              │
Local RGB + State Chunk ──> Demo Expert ────────┘
                                                │
                                          Action Chunk
~~~

VLM—Action 路径继续采用 SmolVLA 原有的 Self/Cross-Attention 思想；Demo 路径是在此基础上增加条件源，而不是完全重写 backbone。

### 2.5 阶段对齐

完整 Demo 可能持续几十秒，而当前控制只处于其中一个阶段。Local Demo Chunk 应先由外部模块定位：

\[
\hat j_t=\operatorname{Align}(O_{1:t},D_{1:N}).
\]

首版采用 RGB+State 的在线受限 DTW，再围绕 $\hat j_t$ 提取 40% 历史和 60% 当前/未来：

\[
D_{\mathrm{local}}=D[\hat j_t-40:\hat j_t+60].
\]

外部对齐提供长视频搜索先验，Cross-Attention 在候选窗口内部执行软选择。二者分别解决全局检索和局部细粒度对应。

### 2.6 防止忽略 Demo

仅把 Demo token 接入注意力并不能保证模型使用 Demo。应加入：

- unseen task composition，使当前 observation 或语言本身不足以决定动作；
- 相同 observation 配对不同 Demo 的反事实训练；
- Demo swap / hard negatives；
- Global task-level 对比损失；
- Local 阶段一致性或排序辅助损失；
- 独立的 Demo Cross-Attention 残差门控及门值监控。

核心检验为：

\[
\Delta_A=
\|f(O,D_{\mathrm{correct}})
-f(O,D_{\mathrm{wrong}})\|.
\]

如果替换 Demo 后动作几乎不变，说明模型学会了跳过 Demo。

## 3. 第一版实现原则

- 保持 SmolVLA 原接口和预训练权重加载逻辑；
- Global Encoder、Stage Alignment、Demo Expert 分模块实现；
- 第一版冻结视频骨干和 VLM，训练新投影、Demo Expert 与注意力层；
- DTW 首先作为模型外的非可微模块；
- **chunk_size** 可以保持 50，但实际执行更少动作后重新观测和规划；
- 对新增 token 使用显式 Mask、相对时间编码和统一维度；
- 先验证 Demo 是否真正被利用，再增加更复杂的交互层。

## 4. 相关笔记

- [Global Video Embedding](./global_video_embedding.md)
- [Local Demo Chunk 与阶段对齐](./local_demo_chunking.md)
