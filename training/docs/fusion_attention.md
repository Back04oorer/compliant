# Stereo-ACT 中的 DINO–DeFM Attention 机制

本文解释 Stereo-ACT 的 DINO–DeFM Fusion block 中 Query、Key、Value 和
2D relative-position bias 的具体作用，并将通用 Attention 概念对应到当前实现。

实现文件：

- [`modeling_stereo_act.py`](./modeling_stereo_act.py)
- [`configuration_stereo_act.py`](./configuration_stereo_act.py)

## 1. 核心直觉

可以把 Query、Key、Value 理解为：

```text
Query：我正在寻找什么？
Key：我这里有什么，是否与你的需求匹配？
Value：如果匹配，我应该把什么信息传给你？
```

在当前 Stereo-ACT 的 cross-attention 中：

```text
Query = DINO 语义外观 token
Key   = DeFM 几何 token
Value = DeFM 几何 token
```

因此 cross-attention 的方向是：

```text
DINO 主动查询 DeFM
语义外观区域主动寻找与之相关的几何信息
```

注意，Key 和 Value 虽然都来源于 DeFM token，但会经过不同的可训练线性投影，
作用并不相同。

## 2. Stereo-ACT 的输入 token

Top RGB 经过冻结的 DINO encoder 后得到：

\[
D\in\mathbb{R}^{B\times800\times768}
\]

FastFS metric depth 经过 DeFM preprocessing 和冻结的 DeFM encoder 后得到：

\[
G\in\mathbb{R}^{B\times800\times384}
\]

其中：

```text
800 = 20 × 40 spatial patches
B   = batch size
```

两个 encoder 的输出维度不同，因此先用可训练线性层投影到共同的 512 维 policy
空间：

\[
D'=D W_D,\qquad G'=G W_G
\]

得到：

```text
DINO projected tokens：[B, 800, 512]
DeFM projected tokens：[B, 800, 512]
```

对应模块是：

```python
self.dino_input_proj = nn.Linear(768, 512)
self.defm_input_proj = nn.Linear(384, 512)
```

## 3. 为什么需要 Q、K、V 三套向量

Attention 不直接对 DINO token 和 DeFM token 做原始点积，而是学习三个不同投影：

\[
Q=D'W_Q
\]

\[
K=G'W_K
\]

\[
V=G'W_V
\]

它们分别负责：

- **Query**：把 DINO token 转换成适合提出查询的表达。
- **Key**：把 DeFM token 转换成适合判断匹配程度的表达。
- **Value**：把 DeFM token 转换成匹配后真正传递给 DINO 的几何内容。

例如，一个 DeFM token 可能同时编码：

```text
区域距离
局部表面结构
相对深度
深度边界
周围几何上下文
```

`W_K` 可以学习从这些信息中提取“用于匹配的特征”；`W_V` 可以学习提取“应该
注入 Fusion token 的内容”。因此 Key 和 Value 即使来自同一 DeFM token，也不等价。

## 4. 一个 DINO patch 如何查询全部 DeFM patch

假设 DINO 第 \(i\) 个 token 对应物体上的一个图像区域，它产生 Query：

\[
q_i
\]

全部 800 个 DeFM patch 分别产生 Key：

\[
k_1,k_2,\ldots,k_{800}
\]

对于每个 attention head，模型计算：

\[
s_{ij}=\frac{q_i^\top k_j}{\sqrt{d}}
\]

其中：

- \(i\) 是一个 DINO query patch。
- \(j\) 是一个候选 DeFM patch。
- 当前配置有 8 个 heads，因此每个 head 的维度 \(d=512/8=64\)。

分数越大，表示模型认为：

> 第 \(j\) 个 DeFM 几何区域对第 \(i\) 个 DINO 语义区域越有帮助。

这个匹配发生在学习后的 Q/K 空间中，不等于原始 DINO 与 DeFM feature 的 cosine
similarity。

## 5. 2D Relative-Position Bias

`cross_relbias` 模式还会向匹配分数加入可训练的二维相对位置偏置：

\[
s_{ij}=\frac{q_i^\top k_j}{\sqrt{d}}+b(\Delta row,\Delta col)
\]

其中：

```text
Δrow = DeFM patch 与 DINO query patch 的相对行距离
Δcol = DeFM patch 与 DINO query patch 的相对列距离
```

当前实现使用 factorized bias：

\[
b(\Delta row,\Delta col)
=b_{row}(\Delta row)+b_{col}(\Delta col)
\]

它让模型显式区分：

```text
完全对齐的位置
局部邻域
远距离区域
不同方向的空间偏移
```

例如，一个位于物体中心的 DINO query 可以：

- 从同位置 DeFM patch 读取表面距离。
- 从相邻 DeFM patch 读取深度边界和局部形状。
- 从附近桌面 patch 读取物体相对桌面的几何参照。

Relative bias 不是强制 attention 只关注同一 patch，而是提供一个可学习的二维空间
关系先验。

## 6. Softmax：从匹配分数变成权重

对一个 DINO query 的全部 800 个 DeFM 匹配分数进行 Softmax：

\[
\alpha_{ij}
=\frac{\exp(s_{ij})}{\sum_{m=1}^{800}\exp(s_{im})}
\]

因此对每一个 query patch：

\[
\sum_{j=1}^{800}\alpha_{ij}=1
\]

一行 \(\alpha_{i,:}\) 可以重新排列成 `20×40` heatmap，表示：

> 一个 DINO patch 在 DeFM 几何图上主要读取哪些区域。

概念示例：

```text
DINO query：物体中心

DeFM attention：
物体中心区域    35%
物体边界区域    25%
附近桌面区域    20%
其他区域        20%
```

以上百分比只是解释机制的例子，不是当前 checkpoint 的实测结果。

## 7. Value 如何把几何信息传给 DINO

得到权重后，对所有 DeFM Value 加权求和：

\[
o_i=\sum_{j=1}^{800}\alpha_{ij}v_j
\]

于是每个 DINO patch 都得到一个从全部 DeFM 区域中选择性读取的几何向量。

经过 multi-head concat 和 output projection 后，通过 residual connection 加回 DINO
token：

\[
f_i=D'_i+W_Oo_i
\]

因此 Fusion token 保留：

```text
原来的 DINO 语义外观信息
+
从 DeFM 中选择性读取的几何信息
```

随后还有 FFN residual：

\[
f_i\leftarrow f_i+
\operatorname{FFN}(\operatorname{LN}(f_i))
\]

最终仍然输出：

```text
Fusion tokens：[B, 800, 512]
```

它仍然是与图像平面对齐的 `20×40` token grid，但 token 内容已经包含由 DeFM 注入
的几何信息。它不是一个显式输出 XYZ 的 3D point cloud。

## 8. Multi-Head Attention

当前配置：

```text
dim_model   = 512
fusion_heads = 8
head_dim     = 64
```

每个 head 都有自己的 Q/K/V 投影，因此可以学习不同的匹配子空间。理论上，不同 head
可能偏向：

```text
同位置绝对距离
物体与桌面的相对高度
深度边界
局部表面形状
更大范围的空间关系
```

但不能仅根据 head 编号给它预先命名；这些含义必须通过 attention、intervention 或
action attribution 实验验证。

每个 Fusion block 内部的 cross-attention score 形状是：

```text
[B, 8 heads, 800 DINO queries, 800 DeFM keys]
```

也就是每个 head 有一个 `800×800` 的关系矩阵。

## 9. Cross-Attention 前的两条 Self-Attention

每个 `StereoFusionBlock` 在 cross-attention 前还会执行：

```text
DINO → DINO self-attention
DeFM → DeFM self-attention
```

DINO self-attention：

```text
Query = DINO
Key   = DINO
Value = DINO
```

它使一个 RGB patch 在查询 DeFM 之前先结合其他 RGB patch 的语义上下文。

DeFM self-attention：

```text
Query = DeFM
Key   = DeFM
Value = DeFM
```

它使一个 depth patch 在被 DINO 查询之前先结合局部和全局几何上下文。

因此 cross-attention 实际匹配的不是两个完全孤立的 patch，而是已经包含上下文的
语义 token 和几何 token。

对应实现的核心逻辑：

```python
rgb_q = self.rgb_self_norm(rgb + pos)
depth_q = self.depth_self_norm(depth + pos)

rgb = rgb + self.rgb_self(rgb_q, rgb_q, rgb)[0]
depth = depth + self.depth_self(depth_q, depth_q, depth)[0]

q = self.cross_norm(rgb + pos)
k = depth + pos
rgb = rgb + self.cross(q, k, depth, attn_mask=relative_bias)[0]
rgb = rgb + self.ffn(self.ffn_norm(rgb))
```

这里 cross-attention 的输入可以更精确地写成：

```text
Q input = normalized contextualized DINO + position
K input = contextualized DeFM + position
V input = contextualized DeFM
```

`nn.MultiheadAttention` 内部还会再执行可训练的 Q/K/V projection。

## 10. 两层 Fusion block

当前配置使用两层 Fusion block：

```text
fusion_layers = 2
```

第一层：

```text
DINO 语义 token
→ 第一次读取 DeFM 几何
→ 初步语义–几何 token
```

第二层：

```text
已经包含几何的 token
→ 再次查询更新后的 DeFM 上下文
→ 最终 Fusion token
```

因此第二层的 Query 已经不再是纯粹的 DINO feature，而是第一层融合后的表示。

## 11. 哪些部分冻结，哪些部分训练

当前训练配置中冻结：

```text
DINO foundation encoder
DeFM foundation encoder
```

但以下部分会训练：

```text
DINO input projection
DeFM input projection
DINO self-attention
DeFM self-attention
Cross-attention WQ / WK / WV / WO
2D relative-position bias
Fusion FFN
ACT encoder / decoder
腕部 ResNet18 branch
```

模型没有直接收到：

```text
“这里是物体”
“这里高 10 cm”
```

Fusion attention 的训练信号来自最终 action loss。如果读取某个深度区域有助于降低
action error，反向传播就会调整：

- `WQ`：DINO 应该提出怎样的查询。
- `WK`：哪些几何特征应该更容易匹配。
- `WV`：匹配后应该传递什么几何内容。
- `relative bias`：应该偏好哪些空间关系。
- `WO/FFN`：如何把读取的几何信息写回 Fusion token。

因此它学习的是面向机器人控制任务的语义–几何融合，而不是通过 segmentation 或
height regression supervision 直接训练出来的显式分割器/高度估计器。

## 12. 对 Throw Into Tank 的直观解释

当物体外观几乎相同、但实际高度发生变化时：

```text
DINO Query：
两个高度下可能仍然非常相似

DeFM Key / Value：
metric depth、相对高度和表面几何发生变化

Cross-Attention：
DINO query 读取到不同的 DeFM value

Fusion：
产生不同的语义–几何 token

ACT：
可能输出不同的抓取高度和轨迹
```

这也是可以通过高度 sweep 实验验证的核心假设。

## 13. Fusion Attention 与 ACT Attention 的区别

不要把两种 attention 混为一谈：

### Fusion attention

```text
DINO Query → DeFM Key / Value
```

作用是为 Top camera 构建语义–几何 fused representation。

### ACT encoder/decoder attention

Fusion 产生的 800 个 Top tokens 会与腕部视觉 token、robot state 等一起进入 ACT。
ACT 后续还有自己的 transformer attention，用于生成 action chunk。

因此：

```text
Fusion attention：决定怎样融合 Top RGB 与 Top depth
ACT attention：决定动作预测怎样读取所有 observation memory
```

## 14. 重要解释边界

1. FastFS metric depth 是 DeFM 的输入来源，但 Fusion cross-attention 不直接对标量
   depth 或 XYZ point cloud 做 attention。
2. Cross-attention 实际读取的是高维 DeFM token。
3. DINO 不是显式 segmentation model；DINO PCA 只是 soft semantic feature
   visualization。
4. DeFM PCA 颜色不是 metric height，也不是物体类别。
5. Attention weight 表示模型内部的匹配/路由关系，但不能单独作为最终因果贡献。
6. 若要回答 Top、Wrist、State 对最终执行 action 的贡献，应该使用 intervention、
   Shapley 或 action sensitivity，并正确经过 ACT temporal ensemble 聚合。

## 15. 一句话总结

```text
Query：DINO 语义区域想寻找什么几何信息？
Key：每个 DeFM 区域与这个语义需求有多匹配？
Value：匹配后应该把哪些几何内容传给 DINO？
Attention：对全部 DeFM 候选区域分配读取权重。
Fusion：原 DINO token + 加权读取的 DeFM 几何信息。
```

最准确的描述是：

> Stereo-ACT 使用 DINO token 作为 Query、DeFM token 作为 Key/Value，通过带二维
> 相对位置偏置的 cross-attention，把语义外观区域与相应的距离、表面形状和相对几何
> 关联起来，形成供 ACT 使用的 semantic-conditioned geometric representation。
