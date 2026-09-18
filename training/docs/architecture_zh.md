# Stereo Semantic–Geometric Fusion

本文档完整说明 Stereo Policy 中 DINO–DeFM 融合策略的研究动机、结构设计、
数学形式和创新点边界。这里仅讨论 Top RGB 与 metric depth 的融合，不展开
ACT、Diffusion Policy 或 Flow-Matching action head。

对应实现：

- [`modeling_stereo_act.py`](./modeling_stereo_act.py)
- [`configuration_stereo_act.py`](./configuration_stereo_act.py)
- [`defm_dino_fusion_explanation.md`](./defm_dino_fusion_explanation.md)

## 1. 研究问题与动机

机器人操作策略不仅需要识别场景中“有什么”，还需要判断目标物体“在哪里、
处于什么高度，以及哪些区域可以被抓取或操作”。纯 RGB 视觉主要描述外观和
语义，但在以下场景中存在明显歧义。

### 1.1 外观相似但几何状态不同

在固定俯视视角下，物体处于不同高度时，其二维 RGB 外观可能高度相似，但
对应的抓取高度、碰撞关系和可操作区域完全不同。仅依赖 RGB 特征，策略难以
区分这些几何状态。

### 1.2 位置变化与遮挡导致注意力漂移

当目标物体出现在桌面不同位置，或受到机械臂、容器和其他物体遮挡时，RGB
encoder 可能把注意力分散到纹理、背景和外观相似但与任务无关的区域。本文将
这种现象称为 **distracted attention**：策略对任务无关区域产生了分散关注，
而没有稳定聚焦到决定操作结果的目标和几何关系。

### 1.3 小物体操作要求保留细粒度空间信息

多位置泛化要求策略准确区分目标在二维图像中的位置。过度缩小图像、全局池化
或过早压缩视觉 token，可能丢失小物体边缘、容器开口和精细抓取区域。因而，
高分辨率、区域级 token 表示对于小物体识别和定位十分重要。

### 1.4 Depth 本身不等于策略能力

Metric depth 能够提供距离、高度差和表面结构，但它不包含明确的任务语义，
无法独立判断“哪个几何区域属于目标物体”或“哪个高度与当前操作相关”。因此，
depth 只有与 RGB 语义建立区域级关联后，才能有效转化为策略能力。

## 2. 核心研究问题

本文关注的核心问题是：

> 能否在 patch 层面对齐 RGB 语义特征与 metric-depth 几何特征，使策略能够
> 区分 RGB 外观相似但空间几何不同的场景，并减少对任务无关区域的分散注意？

进一步地：

> 如何在保留高分辨率空间信息的同时，将 DINO 的语义组织能力与 DeFM 的
> 深度几何表征结合，形成同时包含物体语义、空间位置和可操作几何信息的视觉
> token？

我们的核心假设是：

> DINO 能够提供稳定的区域语义和外观表征，DeFM 能够提供由 metric depth
> 导出的距离、表面和空间结构。如果在统一二维 patch 网格中，让 DINO 语义
> token 主动检索对应的 DeFM 几何信息，就可以缓解纯 RGB 在高度歧义、遮挡
> 和多位置场景中的注意力漂移。

可以将其简化为：

```text
DINO：识别“这个区域可能是什么”
DeFM：描述“这个区域的空间结构如何”
Fusion：建立“哪个语义对象具有怎样的几何状态”
```

## 3. 融合策略概览

我们的主要设计不是把 RGB 和 depth 直接拼接，而是：

> 将 DINO 提取的高维语义外观 token 作为主体，通过区域对齐的
> cross-attention，从 DeFM 几何 token 中读取与每个语义区域相关的距离、
> 表面形状和空间结构，形成 geometry-conditioned semantic tokens。

整体流程如下：

```text
Top RGB [B,3,320,640]
        │
        ▼
Frozen DINOv3 ViT-B/16
        │
        ▼
DINO tokens [B,800,768]
        │ trainable Linear 768→512
        ▼
Semantic tokens [B,800,512] ───────────────┐
                                            │
FastFS metric depth [B,320,640]             │
        │                                   │
        ▼                                   │
Frozen DeFM ViT-S/14                        │
        │                                   │
        ▼                                   │
DeFM tokens [B,800,384]                     │
        │ trainable Linear 384→512          │
        ▼                                   │
Geometric tokens [B,800,512] ───────────────┘
                         │
                         ▼
              2 × StereoFusionBlock
              ├─ DINO self-attention
              ├─ DeFM self-attention
              ├─ DINO Q ← DeFM K/V
              ├─ factorized 2D relative bias
              └─ fused-stream FFN
                         │
                         ▼
       Geometry-conditioned semantic tokens
                    [B,800,512]
```

## 4. 空间对齐的双分支 Tokenization

Top RGB 和 FastFS depth 来自同一个经过标定、去畸变、裁剪和空间对齐的 Top
视角。两个 foundation encoder 的 patch size 不同，但都被组织成同一个
`20 × 40` 归一化空间网格。

### 4.1 DINO 语义分支

RGB 输入调整为 `320 × 640`。DINO 使用 `16 × 16` patch，因此输出：

\[
20\times40=800\ \text{tokens}.
\]

DINO 最后一层 patch token 为：

\[
R_{\mathrm{DINO}}
\in
\mathbb{R}^{B\times800\times768}.
\]

经过可训练线性投影：

\[
R^0=R_{\mathrm{DINO}}W_R
\in
\mathbb{R}^{B\times800\times512}.
\]

### 4.2 DeFM 几何分支

FastFS 输出的 metric depth 以米为单位。Depth 经过 DeFM preprocessing，
调整到与 `20 × 40` token 网格对应的空间尺寸。DeFM 使用 `14 × 14` patch，
得到：

\[
G_{\mathrm{DeFM}}
\in
\mathbb{R}^{B\times800\times384}.
\]

经过可训练线性投影：

\[
G^0=G_{\mathrm{DeFM}}W_G
\in
\mathbb{R}^{B\times800\times512}.
\]

两个分支最终满足：

```text
DINO projected tokens：[B,800,512]
DeFM projected tokens：[B,800,512]
```

第 `(r,c)` 个 DINO token 和第 `(r,c)` 个 DeFM token 对应输入画面中的同一个
归一化空间区域。这里是区域级空间对齐，并不要求两个 foundation encoder 使用
相同 patch size。

## 5. 模态内 Self-Attention

融合前，两个分支先独立建立各自的上下文，而不是立即对孤立 patch 做
cross-attention。

令：

\[
R^l,G^l\in\mathbb{R}^{B\times800\times512}
\]

分别表示第 \(l\) 个 Fusion block 的 DINO 和 DeFM 输入，二维位置编码为：

\[
P\in\mathbb{R}^{B\times800\times512}.
\]

DINO self-attention：

\[
\widetilde R^l
=
R^l+
\operatorname{MSA}_R
\left(
\operatorname{LN}(R^l+P),
\operatorname{LN}(R^l+P),
R^l
\right).
\]

DeFM self-attention：

\[
\widetilde G^l
=
G^l+
\operatorname{MSA}_G
\left(
\operatorname{LN}(G^l+P),
\operatorname{LN}(G^l+P),
G^l
\right).
\]

两个 self-attention 分支分别承担：

```text
DINO self-attention：
建立物体、容器、机械臂和背景区域之间的语义上下文。

DeFM self-attention：
建立相邻表面、深度边界、高度变化和场景几何之间的上下文。
```

因此，后续 cross-attention 匹配的是已经包含模态内上下文的高维区域特征，
而不是独立的 RGB patch 或原始 depth 标量。

## 6. DINO Query、DeFM Key/Value Cross-Attention

融合方向是明确且非对称的：

```text
Query = DINO
Key   = DeFM
Value = DeFM
```

对于 DINO 区域 \(i\) 和 DeFM 区域 \(j\)：

\[
Q_i
=
W_Q\operatorname{LN}
\left(
\widetilde R_i^l+P_i
\right),
\]

\[
K_j
=
W_K
\left(
\widetilde G_j^l+P_j
\right),
\]

\[
V_j=W_V\widetilde G_j^l.
\]

注意力分数为：

\[
S_{ij}
=
\frac{Q_iK_j^\top}{\sqrt{d_h}}
+B_{ij}^{\mathrm{rel}}.
\]

经过 softmax：

\[
A_{ij}
=
\frac{\exp S_{ij}}
{\sum_k\exp S_{ik}}.
\]

DINO 区域 \(i\) 读取到的几何内容为：

\[
C_i=\sum_j A_{ij}V_j.
\]

几何内容通过残差写回 DINO 主体：

\[
R_{\mathrm{cross}}^l
=
\widetilde R^l+W_OC.
\]

随后执行 FFN：

\[
R^{l+1}
=
R_{\mathrm{cross}}^l+
\operatorname{FFN}
\left(
\operatorname{LN}(R_{\mathrm{cross}}^l)
\right).
\]

DeFM 分支保留 self-attention 后的结果：

\[
G^{l+1}=\widetilde G^l.
\]

当前配置使用两个 Fusion block：

```text
fusion_layers  = 2
fusion_dim     = 512
fusion_heads   = 8
fusion_ffn_dim = 2048
```

第二层在第一层已经注入几何信息的语义 token 上继续执行更高层的语义—几何
匹配。

## 7. 为什么使用 DINO 作为 Query

该方向体现了融合策略的核心设计：

```text
DINO 提供语义锚点
DeFM 提供几何证据
```

例如，DINO token 可能对杯子边缘、容器开口或夹爪产生较强语义响应。该 token
作为 Query 后，会从 DeFM token 中寻找：

- 对应区域的相对距离与高度变化；
- 表面方向和几何边界；
- 前后遮挡关系；
- 相邻区域的空间结构。

输出保持 DINO 语义组织作为主体，但每个语义区域都被几何信息重新调制。它不是：

```text
RGB feature + depth scalar
```

而是：

```text
semantic query × high-dimensional geometric evidence
```

## 8. Factorized 2D Relative-Position Bias

普通 cross-attention 只根据 token 内容计算匹配关系，并不知道两个 token 在原图中
相隔多远。我们的 Fusion block 加入二维相对位置偏置：

\[
B_{ij}^{\mathrm{rel}}
=
b_{\mathrm{row}}(\Delta r_{ij})
+b_{\mathrm{col}}(\Delta c_{ij}),
\]

其中：

\[
\Delta r_{ij}=r_j-r_i,
\qquad
\Delta c_{ij}=c_j-c_i.
\]

对于 `20 × 40` 网格：

```text
row bias参数：   2×20−1 = 39
column bias参数：2×40−1 = 79
每个Fusion block总计：118个可训练标量
```

完整注意力为：

\[
A_{ij}
=
\operatorname{softmax}_j
\left(
\frac{Q_iK_j^\top}{\sqrt{d_h}}
+b_{\mathrm{row}}(\Delta r_{ij})
+b_{\mathrm{col}}(\Delta c_{ij})
\right).
\]

这样，融合不仅依赖 DINO 与 DeFM 内容是否匹配，也能学习二维图像中的相对位置
关系。一个目标物体区域的 DINO Query 可以优先读取相同位置或附近的 DeFM 几何
特征，而减少对场景远端无关深度区域的关注。

Factorized 设计避免直接学习完整的 `800 × 800` 位置偏置矩阵，以 118 个参数
表达行列相对位移，提供了低参数量的空间归纳偏置。

## 9. 为什么不直接拼接 RGB 与 Depth Token

直接拼接会产生：

```text
800 DINO tokens + 800 DeFM tokens = 1600 Top tokens
```

这要求下游 policy 自行学习：

- 哪个 RGB token 对应哪个 depth token；
- 两种特征统计分布之间的差异；
- 何时使用语义、何时使用几何；
- 如何保持跨模态空间对应。

我们的 Fusion 在进入下游 policy 前完成：

```text
模态内上下文建模
+ 区域级语义—几何匹配
+ 二维相对空间关系建模
```

最终仍只输出 800 个 fused token，避免将 Top token 数量增加到 1600，同时保持
原始 `20 × 40` 空间网格。

## 10. 最终 Fused Representation

经过两个 Fusion block 后：

\[
F=R^2
\in
\mathbb{R}^{B\times800\times512}.
\]

输出进一步带有二维位置编码和独立的 fused-modality embedding：

\[
F_{\mathrm{out}}=F+P+E_{\mathrm{fused}}.
\]

每个 fused token 可以理解为：

```text
该区域的语义外观
+ 对应的距离与表面结构
+ 与周围区域的空间关系
```

最终表示称为：

```text
geometry-conditioned semantic tokens
或
geometry-aware semantic tokens
```

## 11. 训练边界

融合训练中：

```text
FastFS：policy外部，固定
DINO：冻结
DeFM：冻结
```

以下部分参与训练：

```text
DINO 768→512 projection
DeFM 384→512 projection
DINO self-attention
DeFM self-attention
DINO←DeFM cross-attention
factorized 2D relative-position bias
Fusion FFN
```

因此，训练目标不是修改 DINO 和 DeFM 的基础表示，而是学习如何把固定的语义
特征和固定的几何特征组织成对机器人动作最有效的联合表示。这降低了小规模机器人
数据对 foundation model 造成灾难性遗忘的风险。

## 12. 研究动机与方法设计的对应关系

| 研究问题 | 对应设计 |
|---|---|
| RGB 相似但高度不同 | 引入 FastFS metric depth 与 DeFM 几何 token |
| RGB 注意力分散到无关区域 | DINO Query 从 DeFM Key/Value 检索几何证据 |
| 多位置与小物体识别 | 保留 `20 × 40`、共 800 个空间 token |
| RGB 与 depth 空间对应不明确 | 两个分支对齐到统一二维 patch 网格 |
| Cross-attention 可能读取远处无关区域 | 加入 factorized 2D relative-position bias |
| Depth 缺少任务语义 | 以 DINO 为语义主体，将 DeFM 信息写回 DINO token |
| 简单拼接关系隐式且计算量大 | 输出 800 个 fused token，而不是拼成 1600 个 token |

## 13. 主要创新点

该融合策略可以概括为以下几个贡献：

1. **Foundation feature-level fusion**：融合 DINO 和 DeFM 的高维 patch feature，
   而不是拼接 RGB 像素与原始 depth scalar。
2. **Region-aligned dual-stream representation**：将不同 patch size、不同特征维度
   的 foundation encoder 输出统一到 `20 × 40 × 512` 的空间 token 网格。
3. **Independent intra-modal contextualization**：cross-attention 前分别对语义和
   几何分支做 self-attention。
4. **Asymmetric semantic-to-geometric retrieval**：以 DINO 为 Query、DeFM 为
   Key/Value，将几何证据注入语义主体。
5. **Factorized spatial prior**：利用行列分解的二维相对位置偏置显式建模空间
   对应关系，仅增加线性规模的位置参数。
6. **Compact fused output**：在保持 800 个高分辨率空间 token 的同时完成融合，
   不将下游 memory 扩展为 1600 个 token。
7. **Frozen-foundation task adaptation**：冻结 DINO 和 DeFM，仅通过投影与融合
   模块学习适用于机器人操作的语义—几何关系。

## 14. 准确的能力边界

为了避免对方法能力作不准确表述，需要明确：

- DINO token 不是显式的语义分割 mask；它是具有语义组织能力的高维 patch
  embedding。
- DeFM token 不是直接的三维点云；它是从 metric depth 中提取的高维几何特征。
- Fusion 输出不是显式 3D reconstruction，而是由下游任务损失学习得到的
  semantic–geometric latent representation。
- 当前 cross-attention 是单向的 `DINO Q ← DeFM K/V`，不是双向交换。
- 在 `cross_relbias` 模式下，最终只保留 geometry-conditioned DINO stream；
  DeFM stream 不作为第二组独立 memory 继续传递。

因此，更准确的表述是：

> 该融合模块生成由深度几何条件化的语义 patch token，而不是显式生成语义
> 分割图或三维场景模型。

## 15. 方法总结

完整逻辑可以总结为：

```text
纯RGB存在高度歧义和distracted attention
                    ↓
Metric depth提供几何信息，但缺少任务语义
                    ↓
在统一20×40 patch网格上对齐DINO与DeFM
                    ↓
独立self-attention建立各自模态上下文
                    ↓
DINO Query ← DeFM Key/Value cross-attention
                    ↓
2D relative bias保持跨模态空间对应
                    ↓
得到同时包含语义、位置和几何结构的fused token
```

## 16. 论文式中文描述

纯 RGB 策略在固定俯视、多位置操作及局部遮挡场景中容易产生几何歧义：具有
相似二维外观的状态可能对应不同的目标高度、碰撞关系和可操作区域。同时，RGB
encoder 的注意力可能受到背景纹理和外观相似区域干扰，形成与任务无关的分散
关注。虽然 metric depth 能够提供距离和表面结构，但其本身缺少任务语义，难以
直接转化为操作决策。为此，我们提出一种区域对齐的语义—几何融合模块，将冻结
DINO 提取的外观语义特征与冻结 DeFM 提取的深度几何特征投影到统一二维 token
网格。两个模态首先通过独立 self-attention 建立各自的上下文表示，随后以 DINO
token 作为语义 Query、以 DeFM token 作为几何 Key 和 Value 执行
cross-attention，使每个语义区域能够主动检索与其相关的几何证据。融合过程中
进一步加入因子化二维相对位置偏置，以较低参数开销显式保持跨模态空间对应关系。
最终表示保留 DINO 的语义组织，同时注入由 depth 导出的距离、表面和空间结构
信息，形成用于机器人控制的 geometry-aware semantic tokens。

## 17. Paper-Style English Description

Pure RGB visuomotor policies can be geometrically ambiguous under fixed
overhead views, multi-position manipulation, and partial occlusion: states
with similar two-dimensional appearance may require different grasp heights,
collision-aware motions, and actionable regions. RGB encoders may also exhibit
*distracted attention*, responding to background texture or visually similar
but task-irrelevant regions. Although metric depth provides distance and
surface structure, it lacks the task semantics required to identify which
geometric regions are relevant for manipulation.

We therefore propose a region-aligned semantic–geometric fusion module that
combines frozen DINO appearance features with frozen DeFM geometric features.
The two modalities are projected onto a shared two-dimensional patch grid and
contextualized through independent self-attention. DINO tokens then act as
semantic queries over DeFM keys and values, allowing each semantic region to
retrieve task-relevant geometric evidence. A factorized two-dimensional
relative-position bias explicitly preserves spatial correspondence with only
linear parameter overhead. The resulting representation retains the semantic
organization of DINO while being conditioned on depth-derived distance,
surface, and spatial structure, producing compact geometry-aware semantic
tokens for downstream visuomotor control.
