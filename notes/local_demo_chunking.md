# SmolVLA-ICL：Local Demo Chunk 与阶段对齐

## 1. 目标与边界

Local Demo Chunk 用于回答：

> 当前机器人对应 Demo 的哪个阶段，以及接下来应参考 Demo 的哪一段局部轨迹？

其职责不同于 Global Task Tokens：

- Global 描述“完整任务是什么”，只供 VLM 读取；
- Local 描述“当前阶段附近怎么做”，只供 Action Expert 读取；
- 阶段对齐首先在模型外完成，Demo Expert 接收已经定位的 Local Chunk。

推荐总体方案：

\[
\boxed{\text{重叠时间块检索}+\text{在线单调阶段定位}+\text{40/60 Local Window}}
\]

## 2. 为什么不能只匹配当前帧

只把当前 RGB/State 与 Demo 各帧分别计算距离，本质上是最近邻，不是 DTW。抓取前后、接近和撤离等阶段可能具有相似画面，因此需要：

- 当前执行最近约 1 秒的 RGB+State 历史；
- 上一时刻的匹配位置；
- 单调或近似单调的时间约束；
- 低置信度时的恢复机制。

记 Demo 局部特征序列为

\[
D=\{d_1,\ldots,d_N\},
\]

当前执行历史为

\[
Q_{1:t}=\{q_1,\ldots,q_t\}.
\]

当前 Query 应与 Demo 检索 Chunk 覆盖相近的物理时间。若统一重采样到 10 Hz，则当前最近 1 秒和每个 Demo Chunk 都包含约 10 个 observations。推理时每获得一次 observation 或每次重新规划，只更新在线 DTW 的最新状态。

## 3. 对齐特征

视觉特征应来自当前时刻附近的短片段。第一版可以直接复用 SmolVLA 已有的 SigLIP：

\[
f_{t,n}^V=\operatorname{SigLIP}(I_{t,n}),
\qquad
v_t=E_{\mathrm{temporal}}(f_{t,1}^V,\ldots,f_{t,L}^V).
\]

SigLIP 负责逐帧视觉语义，但本身没有跨帧时序建模。轻量 Temporal Encoder 可以使用一层 GRU、Temporal Conv，或使用：

\[
v_t=
\left[
\operatorname{Mean}(f_t^V),
f_{t,L}^V-f_{t,1}^V
\right].
\]

后续也可以比较视频骨干的中间 temporal features，但不能直接使用已经全局池化的 Global Embedding。

State 特征同时包含位置和运动趋势：

\[
r_t=
\left[
s_t^{\mathrm{start}},
s_t^{\mathrm{end}},
s_t^{\mathrm{end}}-s_t^{\mathrm{start}},
\operatorname{Mean}(\Delta s_t),
g_t
\right],
\]

其中 $g_t$ 表示夹爪状态。如果初始条件变化，应优先使用末端相对位姿、归一化关节变化和夹爪—物体关系，而不是绝对关节值。

融合距离可定义为：

\[
c(t,i)=
\alpha\left(1-\cos(v_t,v_i^D)\right)
+\beta\,\operatorname{Huber}(r_t-r_i^D)
+\gamma\|\Delta s_t-\Delta s_i^D\|_2^2.
\]

第一版可从 $\alpha=0.6,\beta=0.3,\gamma=0.1$ 开始，再通过消融调整。

## 4. 在线受限 DTW

离线标准 DTW 会看到完整当前轨迹，不适用于真实在线推理。对于第 $k$ 个 Demo 检索 Chunk，推荐：

\[
C_t(k)=c(t,k)+
\min
\begin{cases}
C_{t-1}(k)+\lambda_{\mathrm{stay}},\\
C_{t-1}(k-1),\\
C_{t-1}(k-2)+\lambda_{\mathrm{skip}}.
\end{cases}
\]

- 第一项：允许连续多个当前 Query 匹配同一个 Active Chunk；
- 第二项：正常前进一个 Chunk；
- 第三项：当前执行较快时允许跳过一个 Chunk；
- 更大的前跳默认禁止，或给予更高惩罚。

搜索范围限制为：

\[
k\in[k_{t-1}-R_{\mathrm{back}},\;k_{t-1}+R_{\mathrm{forward}}].
\]

关键约束是：

\[
k_t\ge k_{t-1},
\]

而不是严格的 $k_t>k_{t-1}$。如果匹配过的 Chunk 立刻禁止复用，执行速度稍慢、短暂停顿或较高重规划频率都会迫使系统过早进入未来阶段，也失去了 DTW 处理速度差异的能力。

每个 Chunk 维护三种状态：

- **Future**：尚未到达；
- **Active**：当前可能仍处于该阶段，允许重复匹配；
- **Completed**：已经可靠通过，正常搜索中不再使用。

只有当新 Chunk 连续多次胜出、代价明显更低且 confidence 足够高时，才把旧 Chunk 从 Active 标为 Completed。失败恢复时可以允许回退一个 Chunk，但需施加较大惩罚。

匹配位置和概率为：

\[
\hat k_t=\arg\min_k C_t(k),
\qquad
p_t(k)=\operatorname{softmax}(-C_t(k)/\tau).
\]

概率分布熵可作为 confidence。普通全局检索只用于初始化和丢失后的恢复，DTW 用于连续跟踪。

## 5. 一秒重叠 Chunk 与粗到细定位

可以预先按约 1 秒建立 Demo 检索 Chunk，但不建议采用互不重叠的硬切分，因为时间边界通常不等于动作语义边界。第一版建议：

- Chunk duration：1.0 秒；
- stride：0.5 秒；
- 相邻窗口：50% overlap；
- 当前 Query：最近 1.0 秒 RGB+State；
- 保存每个窗口中心到原始 Demo 时间戳和 observation 索引的映射。

~~~text
Chunk 0: 0.0s ───── 1.0s
Chunk 1:       0.5s ───── 1.5s
Chunk 2:             1.0s ───── 2.0s
~~~

粗到细定位流程：

1. 离线为每个 1 秒 Chunk 计算并缓存 SigLIP+State 时序特征；
2. 在线 DTW 在这些 Chunk 特征上定位候选区域；
3. 在候选及相邻区域内进行 observation 级精匹配；
4. 将匹配 Chunk 中心映射为连续 Demo 时间锚点 $\hat t_D$ 和索引 $\hat j_t$；
5. 围绕该锚点从原始 Demo 轨迹动态提取 Local Chunk。

如果一秒窗口无法分辨持续 0.2–0.5 秒的夹爪闭合、接触等事件，可以保留一秒窗口做粗检索，再在候选区域中使用 0.25–0.5 秒窗口精定位。

预分块只是编码和索引结构，最终 Local Chunk 可以跨越任意预分块边界。

### 检索特征与模型输入必须分开

DTW 使用的是压缩检索特征：

\[
z_k^{\mathrm{retrieval}}\in\mathbb{R}^{D_r}.
\]

它只负责寻找阶段。送入 Demo Expert 的 Local Chunk 仍应包含较完整的：

- RGB 或逐时刻视觉 tokens；
- State、Delta-State 和夹爪事件；
- relative time 与 global phase；
- valid mask 与 alignment confidence。

不能只把匹配到的单个 pooled SigLIP vector 交给模型，否则模型只能知道粗略阶段，无法获得后续运动轨迹。

## 6. 40/60 Local Window

如果 Action Chunk 长度为 $H_a=50$，首版令：

\[
L_{\mathrm{local}}=2H_a=100.
\]

如果检索到 Chunk $k$，两个历史 Chunk 与“当前 Chunk + 两个未来 Chunk”在数量上近似 40/60。但由于检索 Chunk 存在重叠，不能直接拼接这五个 Chunk，否则重复帧会被多次送入模型。

正确做法是取匹配 Chunk 的中心时间 $\hat t_D$，从原始连续 Demo 轨迹截取：

\[
\left[
\hat t_D-0.4T_{\mathrm{local}},
\hat t_D+0.6T_{\mathrm{local}}
\right],
\]

然后重新采样为固定的 $L_{\mathrm{local}}=100$ 个 tokens。用索引表示为：

\[
D_{\mathrm{local}}=D[\hat j_t-40:\hat j_t+60].
\]

固定定义为：

- 前 40 个 token 是匹配点之前的历史；
- 后 60 个 token 包含匹配点和未来 59 个 token；
- 匹配点始终位于 Local Chunk 的第 40 位。

历史用于阶段消歧，更多未来信息用于指导下一段动作。在 Demo 起点或终点附近使用 Padding + Mask，不要移动锚点位置。

Local Token 应加入：

\[
e_i=
e_{\mathrm{relative}}\left(\frac{i-\hat j_t}{L_{\mathrm{local}}}\right)
+e_{\mathrm{phase}}\left(\frac{i}{N-1}\right),
\]

分别表示相对匹配点的位置和完整 Demo 中的全局进度。

## 7. 时间尺度

不能机械地按原始视频帧数切片。Demo 与当前执行可能有不同 FPS 和速度，因此应：

1. 根据时间戳同步 RGB 和 State；
2. 重采样到统一的 policy/alignment 频率；
3. 按物理时间确定窗口跨度；
4. 将窗口重采样为固定 token 数。

例如，若 action chunk 对应约 1.5 秒，Local Chunk 可以覆盖约 3 秒，而不一定对应原视频的 100 帧。

## 8. 与模型和控制循环的连接

~~~text
加载 Demo
   ├─ 完整 RGB+State ─> Global Encoder ─> Global Task Tokens ─> 缓存
   └─ 1 秒重叠 Chunk ─> SigLIP + State + Temporal Head ───> 检索特征缓存

每次重新规划
   最近 1 秒 RGB + State
          │
     在线 DTW 更新
          │
 Chunk index + phase + confidence
          │
 候选区域内精定位并映射时间锚点
          │
  动态提取 40/60 Local Chunk
          │
       Demo Expert
          │
 Action Expert Cross-Attention
 Q=Action Hidden, K/V=Local Demo Hidden
          │
    预测 Action Chunk
~~~

SmolVLA 默认预测并执行 50 步动作。如果 50 步全部执行完才重新对齐，Local Chunk 会逐渐过时。建议保持 **chunk_size=50**，先测试只执行前 $M\in\{10,20,50\}$ 步后重新观测、对齐和规划。

## 9. 第一版实现顺序与实验

第一版不必让 DTW 可微：

1. 将 Demo RGB 和 State 按时间戳同步并重采样；
2. 建立 1.0 秒、stride 0.5 秒的重叠检索 Chunk；
3. 离线预计算 SigLIP+State+Temporal Head 特征；
4. 当前端维护最近 1 秒历史；
5. 在线 DTW 允许 stay、前进 1 或前进 2 个 Chunk；
6. 输出 **chunk_index**、**demo_index**、**phase**、**confidence**；
7. 围绕连续时间锚点按 40/60 截取原始 Demo，并重采样为 100 tokens；
8. Demo Expert 编码，Action Expert 通过 Cross-Attention 读取；
9. 每执行 10–20 个动作重新观测和对齐。

首版建议参数：

| 项目 | 初始设置 |
|---|---|
| Demo 检索窗口 | 1.0 秒 |
| Demo stride | 0.5 秒 |
| Current Query | 最近 1.0 秒 |
| RGB 特征 | 复用 SmolVLA SigLIP |
| Temporal Head | 一层 GRU，或 mean + 首尾差分 |
| State 特征 | State + Delta-State + gripper events |
| DTW 单步转移 | stay / +1 / +2 |
| 回退 | 第一版关闭，低置信度恢复时可开放 -1 |
| Local Window | 40% 历史 + 60% 当前/未来 |
| Local 输出 | 固定 100 tokens + valid mask |
| 重新对齐 | 每次 Action Expert 重新规划前 |

应比较：

- Oracle Alignment；
- 单帧最近邻；
- State-only、RGB-only、RGB+State DTW；
- 50/50、40/60、20/80 窗口；
- 正确 Demo 与错误 Demo；
- 每执行 10、20、50 步后的重规划；
- 非重叠 1 秒 Chunk 与 50% overlap Chunk；
- 严格禁止复用与 Active Chunk 可停留；
- 打乱 Demo 顺序、冻结错误阶段等反事实测试。

最重要的判断是：40/60 窗口必须建立在因果、单调、持续更新的阶段定位上。已经确认 Completed 的阶段正常情况下不再使用，但 Active Chunk 必须允许短时间重复匹配。只根据当前单帧全局检索，或首次匹配后立即删除 Chunk，都会造成阶段跳跃。
