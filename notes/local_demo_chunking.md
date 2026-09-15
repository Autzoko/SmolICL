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
\boxed{\text{在线单调阶段定位}+\text{以匹配点为锚点的 40/60 Local Window}}
\]

## 2. 为什么不能只匹配当前帧

只把当前 RGB/State 与 Demo 各帧分别计算距离，本质上是最近邻，不是 DTW。抓取前后、接近和撤离等阶段可能具有相似画面，因此需要：

- 当前执行的短历史；
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

推理时每获得一次 observation，只更新在线 DTW 的最新状态。

## 3. 对齐特征

视觉特征应来自当前时刻附近的短片段：

\[
v_t=E_{\mathrm{align}}(I_{t-h:t}).
\]

可使用视频骨干的中间 temporal features，不能直接使用已经全局池化的 Global Embedding。

State 特征同时包含位置和运动趋势：

\[
r_t=[s_t,\Delta s_t,\Delta^2s_t,g_t],
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

离线标准 DTW 会看到完整当前轨迹，不适用于真实在线推理。推荐使用因果、带搜索带宽的递推：

\[
C_t(i)=c(t,i)+
\min_{\delta\in\{0,\ldots,J\}}
\left[
C_{t-1}(i-\delta)
+\lambda_{\mathrm{jump}}(\delta-\bar{\delta})^2
\right].
\]

- $\delta=0$：停留在同一 Demo 阶段；
- $\delta=1$：正常向前推进；
- $\delta>1$：允许当前执行比 Demo 更快；
- 大跨度跳跃应受到额外惩罚。

搜索范围限制为：

\[
i\in[j_{t-1}-R_{\mathrm{back}},\;j_{t-1}+R_{\mathrm{forward}}].
\]

正常时只允许很小的回退；匹配置信度过低时扩大搜索范围。匹配位置和概率为：

\[
\hat j_t=\arg\min_i C_t(i),
\qquad
p_t(i)=\operatorname{softmax}(-C_t(i)/\tau).
\]

概率分布熵可作为 confidence。普通全局检索只用于初始化和丢失后的恢复，DTW 用于连续跟踪。

## 5. 是否预先均分 Demo

不建议将 Demo 硬性分成几个互不重叠的大 Chunk，然后直接检索其中一个，因为时间边界通常不等于语义阶段边界。

推荐使用有重叠的 Micro-Chunks：

- 每个 Micro-Chunk：4–16 个 observations；
- stride：窗口长度的一半；
- 相邻窗口约 50% overlap；
- 保存每个窗口中心到原始 Demo 时间索引的映射。

~~~text
Micro 1: [0────────7]
Micro 2:     [4────────11]
Micro 3:         [8────────15]
~~~

两级定位流程：

1. 在线 DTW 在缓存的 Micro-Chunk 特征上定位候选区域；
2. 在候选及相邻区域内进行 observation 级精匹配；
3. 得到连续时间锚点 $\hat j_t$；
4. 围绕锚点动态提取 Local Chunk。

预分块只是编码和索引结构，最终 Local Chunk 可以跨越任意预分块边界。

## 6. 40/60 Local Window

如果 Action Chunk 长度为 $H_a=50$，首版令：

\[
L_{\mathrm{local}}=2H_a=100.
\]

围绕匹配点提取：

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
   └─ 局部 RGB+State ─> Alignment Features ────────────────> 缓存

每次重新规划
   最新 RGB + State
          │
     在线 DTW 更新
          │
  index + phase + confidence
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

1. 离线预计算 Demo Micro-Chunk 的 RGB+State 特征；
2. 在线维护带窗口约束的单调 DTW；
3. 输出 **demo_index**、**phase**、**confidence**；
4. 按 40/60 规则提取固定长度 Local Chunk；
5. Demo Expert 编码，Action Expert 通过 Cross-Attention 读取；
6. 每执行 10–20 个动作重新观测和对齐。

应比较：

- Oracle Alignment；
- 单帧最近邻；
- State-only、RGB-only、RGB+State DTW；
- 50/50、40/60、20/80 窗口；
- 正确 Demo 与错误 Demo；
- 每执行 10、20、50 步后的重规划；
- 打乱 Demo 顺序、冻结错误阶段等反事实测试。

最重要的判断是：40/60 窗口必须建立在因果、单调、持续更新的阶段定位上。只根据当前单帧全局检索，再截取窗口，很容易在相似阶段间跳跃。
