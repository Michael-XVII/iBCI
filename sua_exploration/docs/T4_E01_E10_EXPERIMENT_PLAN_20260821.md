# T4 原始路线的三项方法改版：独立验证任务单

> **任务定位**：本任务由独立负责人推进，目标是在**不改变当前 TF-SR / 循环动力学主线**的前提下，基于现有 T4 Carrier 系统验证三项方法改版是否成立。
> **组长侧并行工作**：TF-SR、逐时间步集合读入、状态条件查询与 recurrent dynamics 由组长继续负责；本任务原则上不改 TF-SR，以免架构变化与载体变化相互混淆。
> **基本原则**：三项均属于**待验证假设**，当前不预设一定有效。优先做最小实现、严格匹配对照、跨 session / 多 seed 验证，并完整保留负结果。

---

## 0. 当前基线与任务边界

### 0.1 当前系统主线

现有系统核心仍为：

\[
\text{source multi-session pretraining}
\rightarrow
\text{target-session calibration prefix}
\rightarrow
\text{closed-form T4 estimation}
\rightarrow
\text{frozen decoder inference}.
\]

目标约束：

- target session 不做 backpropagation；
- 不训练新的 unit embedding；
- 不启用目标端 optimizer；
- 仅允许使用 session 开始阶段规定的 calibration prefix；
- 评测阶段 decoder 参数严格冻结；
- 所有 target-side 统计量必须只来自允许使用的 calibration 数据。

当前 T4：

\[
r_i(\theta)
\approx
b_i+a_i\cos\theta+c_i\sin\theta,
\]

\[
\beta_i=
\begin{bmatrix}
a_i\\c_i
\end{bmatrix},
\qquad
m_i=\|\beta_i\|_2,
\]

\[
T4_i=[a_i,c_i,m_i,b_i].
\]

---

### 0.2 本任务不负责的内容

以下内容**暂不纳入本实验线**：

- TF-SR 架构本身；
- GRU / S4 / Mamba 等 recurrent backbone 替换；
- state-conditioned query 的进一步优化；
- CEBRA source-domain contrastive pretraining；
- 新的大模型 tokenizer / slot attention；
- 重新打开已经明确失败的 bilinear / multiplicative gate 路线。

如果三项 T4 改版中某一项明确成立，之后再由组长决定是否与 TF-SR 合并。

---

### 0.3 统一实验纪律

三条路线都遵守以下规则。

#### Matched baseline

默认基线应尽量固定为当前已经稳定的 T4 主干，例如：

- 同样的数据划分；
- 同样的 source sessions；
- 同样的 calibration prefix；
- 同样的 Activity encoder；
- 同样的 tokenizer / decoder；
- 同样的 optimizer、epoch、seed；
- 只替换待测试的 T4 相关模块。

除非实验本身要求，不同时改变多个组件。

#### Seed

建议至少：

\[
\text{seed}\in\{42,43,44\}.
\]

开发阶段可以先 seed 42 做 sanity check，但进入结论前至少补足 3 seeds。

#### 指标

除 mean \(R^2\) 外，至少记录：

- session-wise \(R^2\)；
- paired \(\Delta R^2\)；
- positive-session ratio；
- worst-session；
- across-seed variance；
- calibration cost；
- 参数量与额外 MAC / FLOP；
- 是否引入任何 target-side iterative optimization。

如果某个方法主要目标是提高鲁棒性，应特别看：

\[
\text{worst-session},
\qquad
\operatorname{Var}_{session},
\qquad
\operatorname{Var}_{seed},
\]

而不只看平均值。

---

# 1. Variant A：Permutation Invariance + SO(2) Task-Frame Equivariance

## 1.1 核心假设

当前 T4 中：

\[
\beta_i=[a_i,c_i]^\top
\]

并不是任意学习出来的 embedding，而可以视为 2D behavioral coordinate 中的 functional vector。

如果仅仅对行为坐标系做旋转重参数化：

\[
v' = Rv,
\qquad
R\in SO(2),
\]

则相应的 tuning vector 应满足：

\[
\beta_i'=R\beta_i.
\]

因此可以要求 decoder 满足：

\[
f(\{x_i,R\beta_i,s_i\}_{i=1}^{N})
=
R f(\{x_i,\beta_i,s_i\}_{i=1}^{N}),
\]

其中 \(s_i\) 表示 modulation、baseline、uncertainty 等 scalar / invariant features。

同时，对于任意 neuron permutation \(P\)，应保持：

\[
f(PX,P\beta,Ps)=f(X,\beta,s).
\]

因此目标不是仅有 permutation invariance，而是探索：

\[
S_N\times SO(2)
\]

下的结构约束。

**注意：这目前只是方法假设，需要实验验证其是否真的改善跨 session 泛化。**

---

## 1.2 第一阶段只做 SO(2)，不要直接推广到 O(7)

本任务第一阶段只在具有明确 2D behavioral frame 的任务上验证。

不建议当前直接声称：

\[
SO(2)\rightarrow O(7)
\]

可以自然解决 H1。

对于多自由度任务，不同 behavioral dimensions 未必同质，任意 \(O(d)\) mixing 不一定具有物理意义。

H1 的推广可以在后续考虑：

- 先学习 source-domain 低维 task subspace；
- 再讨论该 latent basis 中的 \(O(K)\) gauge / basis equivariance。

但这不是 Variant A 第一阶段的必要内容。

---

## 1.3 最小实现建议

### A0：Rotation consistency test

在不改模型的情况下，先检查当前 T4 decoder 对 task-frame rotation 有多不一致。

对 source / validation sample：

1. 随机采样
   \[
   R(\phi),\quad \phi\sim U(0,2\pi);
   \]

2. 将
   \[
   \beta_i\rightarrow R\beta_i;
   \]

3. 将 behavioral target
   \[
   v\rightarrow Rv;
   \]

4. neural activity 保持不变；

5. 测量
   \[
   \epsilon_{eq}
   =
   \|f(X,R\beta)-Rf(X,\beta)\|_2.
   \]

这是**坐标系重参数化测试**，不是生成新的真实 neural trial。

先回答一个问题：

> 当前模型究竟多大程度违反 task-frame equivariance？

如果当前模型本身已经近似 equivariant，则硬编码这一性质未必会带来很大收益。

---

### A1：SO(2)-aware readout 的最小版本

不要第一步就实现复杂 geometric Transformer。

建议先把 carrier 拆成：

#### Equivariant vector

\[
u_i=\frac{\beta_i}{\|\beta_i\|+\epsilon}.
\]

#### Invariant scalars

例如：

\[
m_i=\|\beta_i\|,
\qquad
b_i,
\qquad
q_i.
\]

其中 \(q_i\) 可以先留空，之后由 Variant C 提供 uncertainty / reliability。

attention score / weight 尽量只依赖 invariant quantities：

\[
\alpha_i
=
\operatorname{softmax}
\left(
\phi(x_i,m_i,b_i,\ldots)
\right).
\]

vector output 用：

\[
z
=
\sum_i \alpha_i
\left(
A_i u_i+B_iJu_i
\right),
\]

其中：

\[
J=
\begin{bmatrix}
0&-1\\
1&0
\end{bmatrix},
\]

\(A_i,B_i\) 均为 scalar functions。

在 \(SO(2)\) 下，这种形式可以保持 vector equivariance。

---

### A2：复杂 attention 仅在 A1 有结果后推进

如果 A1 已经出现稳定收益，再考虑：

- invariant pairwise angular relation；
- multi-head steerable attention；
- scalar / vector hidden channels；
- 更完整的 equivariant block。

不要一开始就投入较大工程量。

---

## 1.4 需要特别避免的错误

### 错误 1：把 coordinate rotation 当作真实 neural augmentation

允许：

\[
(v,\beta)\rightarrow(Rv,R\beta)
\]

作为 coordinate-frame reparameterization / consistency augmentation。

不应声称 neural activity 对真实物理 movement rotation 保持不变。

---

### 错误 2：只让 attention 等变，但后续普通网络破坏等变

如果 vector feature 后面进入普通 unrestricted MLP / GRU，则整体网络通常不再严格满足 SO(2) equivariance。

本任务若只验证静态 / 原始 T4 backbone，可先避免 recurrent block。

如果之后与 TF-SR 合并，再单独解决 equivariant recurrent state。

---

### 错误 3：SO(2) 和 O(2) 混用

第一阶段建议只声称：

\[
SO(2).
\]

reflection / \(O(2)\) 需要额外处理 parity，不作为当前必要目标。

---

## 1.5 最低对照组

| ID | 方法 |
|---|---|
| A-Base | 当前原始 T4 decoder |
| A-Aug | 原模型 + task-frame rotation consistency / augmentation |
| A-Eq | 最小 SO(2)-equivariant T4 consumer |
| A-Eq-NoPhase | 去掉 vector direction，仅保留 invariant scalars |
| A-Eq-Shuffle | \(\beta_i\) 在 neuron rows 间错配，用于确认 row binding |

如果 Variant C 已完成，再加：

| ID | 方法 |
|---|---|
| A-Eq-UQ | SO(2)-equivariant + uncertainty scalar |

---

## 1.6 主要观察指标

除了 \(R^2\)，必须增加：

\[
\epsilon_{eq}
=
\mathbb E_R
\|f(X,R\beta)-Rf(X,\beta)\|_2.
\]

同时看：

- unseen session mean；
- worst session；
- source / target generalization gap；
- 对 calibration direction coverage 的敏感性。

---

## 1.7 怎样才算“值得继续”

不要求一开始就显著涨很多。

比较合理的继续条件是至少满足其中两项：

1. equivariance error 明显下降；
2. unseen-session mean \(R^2\) 稳定不退；
3. worst-session 有改善；
4. seed variance 降低；
5. calibration 较少时优势更明显。

如果只在 source / within-session 提升，而 external session 不提升甚至下降，则该路线暂时不作为核心架构。

---

# 2. Variant B：Analytic Anchor + Learned Residual

## 2.1 核心假设

当前 T4 已经估计：

\[
B=
\begin{bmatrix}
\beta_1^\top\\
\vdots\\
\beta_N^\top
\end{bmatrix}.
\]

如果采用近似线性编码：

\[
r(t)-b
\approx
Bv(t)+\epsilon,
\]

则可以构造一个 target-session-specific analytic decoder：

\[
\hat v_{\mathrm{ana}}(t)
=
D(B)\,[r(t)-b],
\]

例如：

\[
D(B)
=
(B^\top WB+\lambda I)^{-1}B^\top W.
\]

于是考虑：

\[
\hat v(t)
=
\hat v_{\mathrm{ana}}(t)
+
\Delta v_\theta(t).
\]

其中：

- analytic branch 随 target session 的 T4 自动重构；
- residual branch 在 source domain 训练；
- target session 不更新 \(\theta\)。

待验证假设：

> session-specific functional geometry 主要由 analytic branch 负责，而 frozen network 更适合学习跨 session 相对稳定的 residual / dynamics correction。

这只是一个 decomposition hypothesis，目前需要通过 matched experiment 判断是否成立。

---

## 2.2 不要预设“性能下界有保证”

简单 additive residual：

\[
\hat v=v_{\mathrm{ana}}+\Delta v_\theta
\]

**并不能严格保证** target performance 一定不低于 analytic decoder。

如果 residual 在 OOD session 上错误很大，最终输出仍然可能恶化。

因此实验和文档中只使用以下较弱表述：

- analytic decoder 提供 explicit anchor；
- model family 中存在
  \[
  \Delta v_\theta=0
  \]
  时退化为 analytic solution 的路径；
- 后续可以研究 uncertainty-gated fallback。

不要使用“理论保证不会崩”之类结论，除非额外做了严格安全门控并给出证明。

---

## 2.3 第一阶段先建立 analytic-only baseline

在任何 residual 实验之前，先单独回答：

> 现有 T4 到底能否直接构成一个可用的 analytic decoder？

至少实现：

### B0-1：Population-vector-like

最简单的方向加权：

\[
\hat v
\propto
\sum_i (r_i-b_i)\beta_i.
\]

### B0-2：Ridge / weighted OLE

\[
\hat v
=
(B^\top WB+\lambda I)^{-1}
B^\top W(r-b).
\]

需要特别记录：

- 方向是否正确；
- speed scale 是否存在系统偏差；
- RT endpoint-only T4 是否只能提供 direction anchor；
- 不同 dataset 上 analytic-only 的可用程度。

如果 analytic-only 本身非常差，则直接 additive residual 的解释力会较弱，需要重新定义 analytic branch 的角色。

---

## 2.4 Residual 的最小实现

### B1：Direct additive residual

\[
\hat v
=
v_{\mathrm{ana}}
+
g_\theta(X,T4).
\]

要求：

- \(g_\theta\) 与 baseline 网络尽量同容量；
- source training target 改成：
  \[
  v-v_{\mathrm{ana}};
  \]
- target session 只重新计算 \(v_{\mathrm{ana}}\)，不更新 \(g_\theta\)。

这是最直接的假设检验。

---

### B2：Residual in analytic local frame

若 B1 有价值，进一步尝试：

\[
u_t
=
\frac{v_{\mathrm{ana},t}}
{\|v_{\mathrm{ana},t}\|+\epsilon},
\]

\[
\Delta v_t
=
\delta_{\parallel,t}u_t
+
\delta_{\perp,t}Ju_t.
\]

网络只输出 scalar correction：

\[
\delta_{\parallel,t},\delta_{\perp,t}.
\]

这一版本与 Variant A 的 SO(2) 结构更容易兼容。

---

### B3：Reliability-gated residual

与 Variant C 结合后：

\[
\hat v
=
v_{\mathrm{ana}}
+
\alpha_t\Delta v_\theta,
\qquad
0\le\alpha_t\le1.
\]

\(\alpha_t\) 只能依赖 target-available、无梯度的 reliability / conditioning 信息。

例如：

- population carrier precision；
- \(B^\top B\) condition number；
- angular coverage；
- analytic prediction magnitude / stability。

B3 不是第一阶段必要项。

---

## 2.5 与已有负结果的关系

可以把以下内容作为**待验证解释**，不要预设为已证明：

- bilinear / multiplicative gate 可能把 session-specific carrier error 直接乘到 activity representation 中；
- high-capacity model 可能更容易依赖 source-session-specific shortcuts；
- analytic + residual decomposition 可能减少这类纠缠。

如果 Variant B 最终没有优于 matched T4 baseline，则不能用旧的 BL / Large 负结果反向“证明” decomposition 一定正确。

---

## 2.6 最低对照组

| ID | 方法 |
|---|---|
| B-Base | 当前 T4 网络直接预测行为 |
| B-Ana | analytic-only |
| B-Res | analytic + direct residual |
| B-Res-Zero | residual 强制为 0，确认退化路径 |
| B-Res-Shuffle | analytic carrier row shuffle / mismatch |
| B-Res-LF | local-frame scalar residual（若 B-Res 有结果后再做） |

如果 Variant C 已完成：

| ID | 方法 |
|---|---|
| B-UQ | uncertainty-weighted analytic anchor |
| B-UQ-Gate | uncertainty-gated residual |

---

## 2.7 怎样才算“值得继续”

Variant B 的目标不只是 mean \(R^2\)。

比较理想的信号包括：

- analytic-only 已经提供非平凡性能；
- residual 明显小于 full target signal；
- external sessions 比直接 full-network prediction 更稳；
- worst-session 改善；
- residual 的跨 session 分布比原始输出更稳定；
- carrier 质量下降时，analytic branch / gate 能使性能退化更平缓。

如果 analytic-only 极差、residual 最后重新承担了几乎全部行为预测，则该方法很可能只是换了一种参数化，不宜作为核心方法。

---

# 3. Variant C：Uncertainty-Aware Closed-Form Carrier

## 3.1 核心假设

H1 等极稀疏 calibration 场景下，当前问题可能不仅是 carrier 表达形式，而是：

\[
\hat\beta_i
\]

本身具有较大的估计方差，并且不同 neuron 的可靠性差异很大。

当前 decoder 基本把所有 carrier 当作同等可信输入，因此可能无法区分：

- 稳定、方向明确的 neuron；
- tuning 很弱或 calibration coverage 不足的 neuron；
- residual noise 很大的 neuron。

待验证假设：

> 将 point estimate T4 升级为 closed-form posterior / uncertainty-aware carrier，可能在 sparse-prefix 场景下降低方差，并允许 decoder 对低可信 neuron 自动降权。

这是本任务中短期最值得优先测试的统计改版，但仍不预设一定在所有数据集上涨点。

---

## 3.2 不再把 OLS sampling covariance 称为 posterior

经典 OLS 下：

\[
\hat\beta_i\mid\beta_i
\sim
\mathcal N
\left(
\beta_i,
\sigma_i^2(X^\top X)^{-1}
\right)
\]

描述的是 estimator sampling distribution。

如果要使用 Bayesian posterior，应显式引入 prior。

建议最小版本采用 isotropic Gaussian prior：

\[
\beta_i
\sim
\mathcal N(0,\tau^2I).
\]

观测模型：

\[
r_i
\mid
\beta_i
\sim
\mathcal N(X\beta_i,\sigma_i^2I).
\]

则：

\[
\Sigma_{i,\mathrm{post}}
=
\left(
\frac{X^\top X}{\sigma_i^2}
+
\tau^{-2}I
\right)^{-1},
\]

\[
\mu_{i,\mathrm{post}}
=
\Sigma_{i,\mathrm{post}}
\frac{X^\top r_i}{\sigma_i^2}.
\]

目标 carrier 可以由：

\[
C_i=
(\mu_i,\Sigma_i,b_i)
\]

或其低成本摘要构成。

---

## 3.3 为什么优先 isotropic shrinkage

暂时不建议把每个 neuron 的方向向固定“群体平均方向”收缩。

原因：

1. motor tuning direction 本身可能覆盖较大角度范围；
2. 固定非零平均方向会引入 arbitrary orientation；
3. 与 Variant A 的 SO(2) symmetry 不容易兼容。

更稳妥的第一版：

\[
\beta_i\sim N(0,\tau^2I),
\]

使低置信度 neuron：

\[
\|\mu_i\|\rightarrow0,
\]

即“对其方向信息少信任”，而不是把方向强行拉向某个群体 PD。

---

## 3.4 第一阶段先验证 posterior mean 是否优于 OLS

不要一开始同时加入很多 uncertainty mechanisms。

### C0：OLS

当前：

\[
\hat\beta_i^{OLS}.
\]

### C1：Ridge / posterior mean

替换为：

\[
\mu_i.
\]

其余 decoder 不变。

先回答：

> 单纯 shrinkage 是否改善 H1 Date2 / sparse calibration？

---

## 3.5 第二阶段再加入 uncertainty-aware attention

如果 C1 有正向结果，再加入 reliability。

建议比较几个简单的 invariant reliability：

### Reliability 1：trace covariance

\[
q_i^{(1)}
=
-\log(\operatorname{tr}\Sigma_i+\epsilon).
\]

### Reliability 2：posterior functional SNR

\[
q_i^{(2)}
=
\mu_i^\top\Sigma_i^{-1}\mu_i.
\]

这个量在旋转：

\[
\mu_i\rightarrow R\mu_i,
\qquad
\Sigma_i\rightarrow R\Sigma_iR^\top
\]

下保持不变。

### Reliability 3：angular uncertainty

令：

\[
u_i=\frac{\mu_i}{\|\mu_i\|+\epsilon},
\qquad
u_i^\perp=Ju_i,
\]

可近似定义：

\[
\sigma_{\theta,i}^2
\approx
\frac{
(u_i^\perp)^\top
\Sigma_i
u_i^\perp
}{
\|\mu_i\|^2+\epsilon
}.
\]

然后：

\[
q_i^{(3)}
=
-\log(\sigma_{\theta,i}^2+\epsilon).
\]

对于 T4 这种 direction-centric carrier，这一指标值得优先关注。

---

## 3.6 uncertainty 如何进入 decoder

第一版推荐最保守的做法：

\[
\text{logit}_i
\leftarrow
\text{logit}_i+\gamma q_i.
\]

也可以把：

\[
q_i
\]

作为 scalar token feature。

暂时不建议：

\[
x_i\leftarrow q_i x_i
\]

这种直接 activity multiplicative gating，因为此前 multiplicative interaction 已经存在明显跨域失败风险。

---

## 3.7 \(\tau,\lambda\) 等超参数怎么处理

为了保持 target-side “optimizer-free”的叙事，优先方案：

- 在 source sessions 上估计 / 选择 \(\tau,\lambda,\gamma\)；
- 固定后带到所有 target sessions；
- target session 仅计算 posterior mean / covariance。

如果后续测试 GCV / marginal likelihood：

- 可以做为扩展实验；
- 但不要把其 hyperparameter selection 简化描述为严格 closed-form；
- 需要单独统计 target-side search / compute cost。

---

## 3.8 最低对照组

| ID | 方法 |
|---|---|
| C-OLS | 当前 T4 OLS |
| C-Ridge | 仅 ridge / posterior mean |
| C-UQ-Feat | posterior mean + reliability scalar feature |
| C-UQ-Logit | posterior mean + reliability attention bias |
| C-UQ-Random | 随机打乱 reliability 与 neuron 对应关系 |
| C-UQ-Const | 所有 neuron 使用相同 reliability |

如果 Variant A 已完成：

| ID | 方法 |
|---|---|
| C-Eq | SO(2) carrier + uncertainty invariant |
| C-Eq-NoUQ | 等变结构但不使用 uncertainty |

---

## 3.9 重点数据集

### 第一优先级：H1

原因：

- calibration 极少；
- 目前存在跨日期不稳定；
- carrier slope / intercept scale 问题明显；
- 最适合观察 shrinkage 与 uncertainty 是否真正降低方差。

重点看：

- Date2；
- 其他 dates 是否被牺牲；
- 4-trial 下的方向估计方差；
- neuron reliability distribution。

### 第二优先级：RT / M2 / sub-M

目的不是强求所有数据集明显涨点，而是检查：

- 是否至少不破坏已有 T4 优势；
- sparse-prefix 时是否收益更明显；
- worst-session / seed variance 是否下降。

---

## 3.10 怎样才算“值得继续”

Variant C 的成功判据可以比单纯 mean improvement 更宽：

1. H1 Date2 明显修复，且其他日期不显著退化；
2. low-calibration regime 下优于 OLS；
3. worst-session 提升；
4. seed / session variance 下降；
5. reliability 与真实 carrier estimation error 有相关性；
6. 打乱 reliability-neuron binding 后收益消失。

如果只是在训练集或高 calibration regime 涨点，而 H1 sparse regime 无改善，则说明 uncertainty 机制未解决真正瓶颈。

---

# 4. 三条路线的推荐执行顺序

建议不要三条同时大规模开发。

## Phase 1：最小统计升级

先做 Variant C：

\[
\text{OLS}
\rightarrow
\text{posterior mean / ridge}
\rightarrow
+\text{reliability}.
\]

原因：

- 实现成本最低；
- 最直接对应 H1 当前失败；
- 其 uncertainty scalar 后续可直接供 Variant A/B 使用。

---

## Phase 2：SO(2) 最小等变版本

基于原始 T4 主干，不引入 recurrent dynamics：

\[
S_N
\rightarrow
S_N\times SO(2).
\]

先做：

- rotation consistency；
- minimal equivariant readout；
- matched external-session comparison。

只有出现正向信号，再升级复杂 geometric attention。

---

## Phase 3：Analytic anchor / residual

先建立 analytic-only baseline，再做：

\[
v_{\mathrm{ana}}
+
\Delta v_\theta.
\]

如果 direct residual 成立，再尝试：

- local-frame residual；
- uncertainty gate；
- 与 SO(2) 版本结合。

---

# 5. 建议的最小实验矩阵

如果时间有限，先完成下列 10 个实验即可判断方向：

| 编号 | 实验 | 目的 |
|---|---|---|
| E01 | Original T4 baseline | 锚点 |
| E02 | Ridge / posterior-mean T4 | 判断 shrinkage 本身 |
| E03 | Posterior mean + reliability feature | 判断 uncertainty 是否可被 consumer 使用 |
| E04 | Posterior mean + reliability logit | 判断显式可信度降权 |
| E05 | Current model rotation consistency | 测量现有 SO(2) violation |
| E06 | Minimal SO(2)-equivariant consumer | 核心等变假设 |
| E07 | SO(2) model 去掉 direction | 确认 vector component 必要性 |
| E08 | Analytic-only decoder | 判断 analytic anchor 的真实下限 |
| E09 | Analytic + direct residual | 判断 decomposition 是否有价值 |
| E10 | Analytic + local-frame residual | 判断 residual 是否适合用相对方向表示 |

E01–E10 全部先 seed 42 开发。

值得继续的版本再补：

\[
43,44.
\]

---

# 6. 结果汇报模板

每个实验不要只报一个平均分。

建议固定以下格式：

```text
Experiment:
Code / commit:
Seed:
Dataset:
Source sessions:
Target sessions:
Calibration regime:
Target labels used:
Target gradient:
Target optimizer:
Carrier estimator:
Decoder changes:
Parameter count:
Target-side compute:

Mean R2:
Median R2:
Worst-session R2:
Positive sessions:
Mean paired delta vs baseline:
Seed variance:

Equivariance error (if applicable):
Carrier estimation error (if available):
Reliability/error correlation (if applicable):

Conclusion:
- Supported / Partially supported / Not supported
- Main positive evidence:
- Main negative evidence:
- Potential confound:
- Next minimal experiment:
```

---

# 7. Fail-Closed 原则

本任务不要求三条路线都成功。

出现以下情况可以直接关闭某条路线：

### Variant A

- equivariance error 虽下降，但 external \(R^2\) 稳定恶化；
- 需要大幅增加模型容量才勉强持平；
- 优势只存在于人为 rotation augmentation，真实 external sessions 不成立。

### Variant B

- analytic-only 接近无效；
- residual 最终重新承担几乎全部预测；
- external session 比 direct T4 baseline 更差；
- gate / residual 对 carrier noise 极度敏感。

### Variant C

- shrinkage 在 H1 sparse regime 无改善；
- reliability 与 carrier error 没有关系；
- uncertainty 加权只提升 source / internal，external 不成立；
- 多 seed 后收益消失。

负结果照样保留完整实验记录，不为了方法叙事反复调参到不可比较。

---

# 8. 如果三条都部分成立，后续可能的统一方向

这一部分当前只作为远期假设，不作为本轮实验必须目标。

可能统一为：

\[
\boxed{
\text{Bayesian analytic functional carrier}
\rightarrow
\text{permutation + task-frame equivariant consumer}
\rightarrow
\text{analytic anchor + learned residual}
}
\]

其中：

- posterior mean \(\mu_i\)：vector / equivariant functional coordinate；
- posterior covariance \(\Sigma_i\)：carrier uncertainty；
- modulation、baseline、reliability：invariant scalars；
- attention weights：主要由 invariant quantities 决定；
- vector value / output：由 equivariant carrier 张成；
- analytic branch：target-session closed-form 重构；
- learned residual：source-trained、target-frozen。

只有在独立消融已经证明 A/B/C 各自有价值后，再考虑这一联合版本。

不要一开始把三条全部合起来，否则一旦涨点无法判断来源。

---

# 9. 本任务最终需要交付的内容

负责人完成后需要给组长：

1. **代码分支 / commit 列表**；
2. **E01–E10 的实验状态表**；
3. **每个 target session 的 raw metrics**；
4. **seed 42/43/44 汇总**；
5. **H1 Date-wise 详细结果**；
6. **所有 carrier estimator 的公式与实现说明**；
7. **target-side compute / memory 开销**；
8. **所有失败实验的配置与结果**；
9. **是否存在数据泄露 / target adaptation 的自查说明**；
10. **负责人和 AI agent 对三条路线的独立判断**：
   - 是否值得继续；
   - 最大风险；
   - 下一步最小实验；
   - 是否建议之后与 TF-SR 合并。

---

## 最后说明

这轮任务的目标不是证明某个预先设定的论文故事，而是回答三个相对独立的问题：

1. **T4 的方向分量是否值得被当作具有明确 transformation law 的 geometric object，而不是普通 side information？**
2. **T4 是否不仅能提供 identity，还能构成一个有实际价值的 target-session analytic anchor？**
3. **当前 sparse calibration 的主要瓶颈是否来自 carrier estimation uncertainty，并能否通过 closed-form shrinkage / reliability 建模缓解？**

只要其中一条得到稳定、跨 session、跨 seed 的正证据，就值得进入下一轮；如果没有，也应以负结果收口，不影响组长侧 TF-SR / recurrent dynamics 主线继续推进。
