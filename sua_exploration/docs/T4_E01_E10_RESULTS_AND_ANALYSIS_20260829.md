# T4 E01–E10 实验结果汇总与独立分析

> 汇总日期：2026-08-29（Asia/Shanghai）
> 汇总分支：`exp/t4-e01-e10-summary`
> 实验基点：`exp/e10-analytic-local-frame-residual`，commit `7d65cf3`
> 原始任务书：[T4_E01_E10_EXPERIMENT_PLAN_20260821.md](T4_E01_E10_EXPERIMENT_PLAN_20260821.md)

## 1. 结论摘要

本轮 E01–E10 形成了三条相对清晰但强度不同的开发证据：

1. **Variant C（posterior mean / uncertainty）是最值得保留的方向，但机制尚未被严格证明。**
   E02 相对 E01 的 mean R² 提高 `+0.0150`，worst-session 从 `0.4280` 提高到
   `0.4462`，session 标准差从 `0.1136` 降到 `0.1067`。E03 在 E02 上再提高
   `+0.0080`，并把 worst-session 提到 `0.4725`。不过，本轮没有运行任务书优先要求的
   H1 sparse/Date-wise 实验，也没有完成 seeds 43/44、reliability shuffle/constant control
   或 reliability–carrier-error correlation。因此现有结果只能说明“posterior mean 和
   reliability feature 在 CO seed 42 上有弱正信号”，不能证明 uncertainty 是 sparse
   calibration 的主瓶颈。

2. **Variant A 的几何前提成立，但当前最小硬等变 consumer 明确失败。**
   E05 证明原始 T4 consumer 存在显著 SO(2) consistency violation；E06 的实现通过了严格
   SO(2) equivariance 和 neuron permutation invariance 测试，但 test mean R² 从 E01 的
   `0.6137` 降到 `0.3191`，8/8 session 均退化，worst-session 为 `−0.4055`。按任务书的
   fail-closed 原则，当前 E06 架构不应继续晋级。由于 E06 的 active trainable parameters
   仅 `67,208`，而匹配的 coupled 网络约为 `4,613,178`，该结果同时混入了约 68.6 倍的
   容量差；它否定当前最小实现，不足以否定所有 SO(2) 方法。E07 未实施，因此“去掉
   direction 后是否进一步退化”仍未回答。

3. **Variant B 证明 analytic T4 可以提供非零 anchor，但 direct/local-frame residual
   都没有带来稳定的尾部改善。**
   E08 ridge OLE 的 mean R² 为 `0.2056`，8/8 session 为正，但远低于 E01。E09 direct
   residual 达到 `0.6277`，相对 E01 为 `+0.0141`；E10 local-frame 达到本轮最高 mean
   R² `0.6379`，相对 E01 为 `+0.0243`，相对 E09 为 `+0.0102`。然而 E09/E10 的
   worst-session 分别降到 `0.3512/0.2816`，session 标准差升到 `0.1438/0.1690`；E10
   residual-energy fraction 也从 E09 的 `0.3711` 升到 `0.4108`。因此 local frame 带来的
   是小幅、异质的均值变化，而不是任务书期待的跨 session 稳定性改善。

4. **没有任何路线达到“稳定、跨 session、跨 seed”的正式结论标准。**
   除 E08 closed-form 计算外，所有开发比较只有 seed 42；E07、H1 sparse、seeds 43/44
   均缺失。learned runs 使用了 noisy validation argmax/early stopping，而不是仓库冻结的
   V4 fixed-epoch tail-average 估计量；后续实验又根据同一组 target-test 结果依次开启，故
   这些 test 指标属于 held-out-selected development evidence，不是 blind formal held-out
   证据。

综合判断：**当前不建议把 E06、E09 或 E10 合并进 TF-SR 主线。若只保留一个下一步，
优先复验 E02 posterior mean；E03 reliability feature 只有在补齐绑定对照后再决定是否叠加。**

## 2. 实验范围与统一协议

### 2.1 实际完成的共同范围

- Dataset：DANDI 000688，`sub-C`，centre-out（CO），SUA。
- Chronological split：37 source-train / 8 validation / 8 target-test sessions。
- Seed：42。
- T4/activity calibration：每个 session 前 50 个 rewarded trials。
- Evaluation：calibration prefix 后的 trial/window；E08 明确使用 trials `[50:]`。
- Learned baseline：B3S/T4 coupled consumer，task-only training，learning rate `1e-4`，
  batch size 32，最多 40 epochs，patience 10。
- Target side：不使用 optimizer、backward 或新 embedding；T4 所需的 endpoint direction
  label 只来自允许的 calibration prefix。
- Learned-run checkpoint：以 validation `val_heldout/r2_mean` 的最大值选择，再对 8 个
  target-test sessions 做 forward evaluation。

### 2.2 本轮没有覆盖的原任务范围

- seeds 43/44 与 across-seed variance；
- H1 Date-wise、Date2 和 4-trial sparse calibration；
- RT、M2、sub-M 跨数据集复验；
- calibration-budget curve；
- E07 direction ablation；
- E03/E04 所需的 shuffled/constant reliability controls；
- reliability 与真实 carrier estimation error 的相关性；
- E09/E10 的 residual-only 对照和 uncertainty gate；
- V4 固定 12 epochs、epochs 5–12 tail average 的确定性比较。

因此，本文件将 `supported / partially supported / not supported` 限定为 **CO seed-42
开发结论**，不把它外推为多 seed、sparse calibration 或跨数据集结论。

## 3. 分支、提交与状态

| ID | 实验 | 分支 | 代表提交 | 状态 |
|---|---|---|---|---|
| E01 | Original T4 baseline | `exp/template-ridge-db-heldout-spint` | `eda5962` | 完成 |
| E02 | Posterior-mean T4 | `exp/e02-posterior-mean-t4` | `f9f05da` | 完成 |
| E03 | Posterior mean + angular reliability feature | `exp/e03-posterior-angular-reliability` | `603e11b` | 完成 |
| E04 | Posterior mean + reliability logit bias | `exp/e04-posterior-reliability-logit` | `f0aeb61` | 完成 |
| E05 | Rotation consistency diagnostic | `exp/e05-rotation-consistency` | `f2bd586` | 完成并通过审计 |
| E06 | Minimal SO(2)-equivariant consumer | `exp/e06-minimal-so2` | `635bb73` | 完成；性能失败 |
| E07 | SO(2) no-direction ablation | — | — | **未实施/未运行** |
| E08 | Analytic-only decoder | `exp/e08-analytic-only` | `b33bfe9` | 完成 |
| E09 | Analytic + direct residual | `exp/e09-analytic-direct-residual` | `7fa5bd7` | 完成；记录文档状态未回填 |
| E10 | Analytic + local-frame residual | `exp/e10-analytic-local-frame-residual` | `7d65cf3` | 完成；记录文档状态未回填 |

E09 和 E10 的运行文档仍写着 `formal training running`，但对应 receipt 的权威状态均为
`completed`：E09 完成于 2026-08-28 17:25，E10 完成于 2026-08-29 12:55。本汇总使用
完成后的 JSON receipt，而不是启动时的状态文字。

## 4. 总体性能

### 4.1 跨 session 汇总

`SD_session` 是 8 个 target sessions 的样本标准差；`正向数` 是相对 E01 的 paired
delta 大于 0 的 session 数。E08-PV/Ridge 是两个 analytic-only decoder。

| 实验 | Mean R² | Median R² | Worst R² | SD_session | Mean Δ vs E01 | 正向数/8 |
|---|---:|---:|---:|---:|---:|---:|
| E01 T4 baseline | 0.6137 | 0.6235 | 0.4280 | 0.1136 | 0.0000 | — |
| E02 posterior mean | 0.6287 | 0.6535 | 0.4462 | 0.1067 | +0.0150 | 5 |
| E03 + reliability feature | 0.6367 | 0.6774 | **0.4725** | 0.1062 | +0.0231 | 5 |
| E04 + reliability logit | 0.6126 | 0.6343 | 0.4184 | **0.1024** | −0.0011 | 4 |
| E06 strict SO(2) | 0.3191 | 0.3820 | −0.4055 | 0.3157 | −0.2945 | 0 |
| E08 population vector | −0.0216 | 0.1818 | −1.3680 | 0.5476 | −0.6353 | 0 |
| E08 ridge OLE | 0.2056 | 0.2302 | 0.1093 | 0.0479 | −0.4080 | 0 |
| E09 direct residual | 0.6277 | 0.6840 | 0.3512 | 0.1438 | +0.0141 | 7 |
| E10 local-frame residual | **0.6379** | **0.6889** | 0.2816 | 0.1690 | **+0.0243** | 7 |

这里有两个不同的“最佳”：

- E10 的单次 mean/median 最高；
- E03 的 worst-session 最高，且 session spread 明显小于 E09/E10。

E10 只比 E03 高 `0.00118` mean R²，但 worst-session 低 `0.19093`，所以不能只根据
mean 排名把 E10 判为综合最优。

### 4.2 逐 target-session raw R²

| Session | E01 | E02 | E03 | E04 | E06 | E08-Ridge | E09 | E10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20160923 | 0.4786 | 0.4939 | 0.4761 | 0.5054 | 0.3902 | 0.1093 | 0.3512 | 0.2816 |
| 20160929 | 0.7049 | 0.6957 | 0.6942 | 0.6885 | 0.3737 | 0.2363 | 0.7242 | 0.7177 |
| 20161005 | 0.4280 | 0.4462 | 0.4725 | 0.4184 | 0.2515 | 0.1529 | 0.4640 | 0.4999 |
| 20161006 | 0.5973 | 0.6260 | 0.6410 | 0.6263 | 0.2992 | 0.2335 | 0.6840 | 0.6849 |
| 20161007 | 0.6882 | 0.6775 | 0.6851 | 0.6423 | 0.5636 | 0.2362 | 0.7054 | 0.6930 |
| 20161011 | 0.6120 | 0.6295 | 0.6697 | 0.6206 | 0.5724 | 0.2270 | 0.6412 | 0.6639 |
| 20161013 | 0.7653 | 0.7478 | 0.7660 | 0.7327 | 0.5081 | 0.2335 | 0.7679 | 0.7975 |
| 20161021 | 0.6349 | 0.7130 | 0.6893 | 0.6663 | −0.4055 | 0.2164 | 0.6839 | 0.7650 |

### 4.3 描述性不确定度

以 8 个固定 target sessions 为配对单位，mean delta vs E01 的 t-based 95% interval 为：

| 实验 | Mean Δ vs E01 | 描述性 95% interval | Paired t p-value |
|---|---:|---:|---:|
| E02 | +0.0150 | [−0.0104, +0.0405] | 0.205 |
| E03 | +0.0231 | [−0.0015, +0.0476] | 0.062 |
| E04 | −0.0011 | [−0.0259, +0.0237] | 0.920 |
| E06 | −0.2945 | [−0.5607, −0.0284] | 0.035 |
| E09 | +0.0141 | [−0.0382, +0.0664] | 0.545 |
| E10 | +0.0243 | [−0.0580, +0.1065] | 0.508 |

这些 interval/p-value 只描述这 8 个 session 的异质性，**不能替代跨 seed 方差**，也没有
校正 checkpoint argmax、顺序开启实验或多重比较。它们主要说明：除 E06 的大幅负效应外，
本轮所有小幅正负差异都处在单 seed 无法稳定分辨的范围内。

## 5. E01–E10 逐项分析

### 5.1 E01：Original T4 baseline

E01 使用 raw `[a,c,m,b]` T4、B3S calibrated identity 和 coupled decoder，是所有学习型
实验的锚点。其 mean/median/worst R² 分别为 `0.6137/0.6235/0.4280`，8 个 target
sessions 均为正，session SD 为 `0.1136`。

部署侧参考 receipt（batch 1、64 units、50-bin window）给出 coupled decoder 约
`57,970,688` MAC；activity/calibration encoder 在 50-trial reference 下约
`21,225,472` MAC/session、`18,290` parameters。E09/E10 与 B-Base 参数匹配 receipt
说明完整 student 为 `4,613,178` trainable parameters。

结论：E01 是有效且相对稳定的 matched baseline；后续小于约 `0.02–0.03` 的单 seed
变化不应脱离方差、worst-session 和机制对照单独解读。

### 5.2 E02：Posterior-mean T4

E02 把 OLS point estimate 替换为 isotropic Gaussian prior 下的 posterior mean：

\[
\mu_{i,\mathrm{post}}=
\left(X^TX+\operatorname{diag}(0,\sigma_i^2/\tau^2,\sigma_i^2/\tau^2)\right)^{-1}X^Ty_i.
\]

`tau²=1.4556518254` 仅由 37 个 source-train sessions 估计。目标端执行 per-unit 3×3
closed-form solve，不增加 decoder query MAC，不增加模型参数，也不做 target optimizer/backward。

结果：

- mean delta vs E01：`+0.0150`；
- median paired delta：`+0.0164`；
- 5/8 sessions 提升；
- worst-session：`0.4280 → 0.4462`；
- SD_session：`0.1136 → 0.1067`，约下降 6.1%。

最大正变化出现在 20161021（`+0.0781`），最大负变化出现在 20161013（`−0.0175`）。
这是 Variant C 中最干净的低成本正信号，但效应小、只有一个 seed，而且 50-trial 不是任务书
最关心的 sparse regime。

判定：**Partially supported（值得复验）**。

### 5.3 E03：Posterior mean + angular reliability feature

E03 在 posterior T4 后增加 invariant scalar：

\[
q_\theta=-\log\left(
\frac{(u^\perp)^T\Sigma_{ac}u^\perp}{\|\mu\|^2+\epsilon}+\epsilon
\right).
\]

该值作为普通 scalar feature 进入 B3S consumer；近零 modulation unit 使用 `q=-20` 的
fail-closed 值。相对 E02，encoder 增加 64 parameters，reference encoder MAC/session
从 `21,225,472` 增到 `21,229,568`；目标端增加 2×2 quadratic form，无 target optimizer。

结果：

- mean delta vs E02：`+0.0080`，5/8 sessions 提升；
- mean delta vs E01：`+0.0231`；
- worst-session：`0.4462 → 0.4725`；
- SD_session：`0.1062`，与 E02 基本相同；
- 相对 E02 的 95% interval：`[−0.0101,+0.0261]`。

E03 是本轮 tail behavior 最好的 learned model，但没有 `C-UQ-Random`、`C-UQ-Const`、
reliability binding shuffle，也没有验证 reliability 与真实 carrier error 的相关性。因此
`+0.0080` 不能唯一归因于 uncertainty 被 consumer 正确利用，也可能是训练噪声或额外输入
自由度。

判定：**Partially supported / mechanism unconfirmed**。

### 5.4 E04：Posterior mean + reliability logit bias

E04 不把 q 拼到普通 feature，而在 coupled cross-attention softmax 前加入：

\[
\operatorname{logit}_{l,h,c,i}\leftarrow
\operatorname{logit}_{l,h,c,i}+\gamma_lq_i,
\qquad \gamma_l=\operatorname{softplus}(\gamma_{l,\mathrm{raw}}).
\]

本实现只有一个额外 trainable gamma，最终 `gamma=0.0038454`。结果相对 E02：

- mean delta：`−0.0161`；
- median paired delta：`−0.0121`；
- 2/8 sessions 提升；
- worst-session：`0.4462 → 0.4184`；
- 20161021 最大退化 `−0.0467`。

E04 mean 与 E01 几乎相同（`−0.0011`），但它丢掉了 E02 的 mean/worst 收益。当前显式
monotone reliability logit bias 不受支持；正 gamma 的存在也不能抵消 external 指标的负证据。

判定：**Not supported（当前 logit 注入方式关闭）**。

### 5.5 E05：Current-model rotation consistency

E05 冻结 E01，在 37 train + 8 validation sessions、每 session 64 个 deterministic windows、
32 个随机 SO(2) angles（另有 zero-angle identity）上测量：

\[
\epsilon_{eq}=\|f(X,R\beta)-Rf(X,\beta)\|_2.
\]

排除 zero angle 后：

| Split | Rotation path | Mean epsilon | Mean relative RMS | Worst epsilon | Worst relative RMS |
|---|---|---:|---:|---:|---:|
| train | physical pipeline | 2.5399 | 0.6317 | 7.4778 | 1.2199 |
| train | normalized internal | 2.5088 | 0.6268 | 7.1053 | 1.1584 |
| validation | physical pipeline | 1.2834 | 0.4109 | 2.5285 | 0.6618 |
| validation | normalized internal | 1.3183 | 0.4377 | 2.5772 | 0.7169 |

审计 `pass=true`，test sessions 被排除，评估前后 student state SHA 相同。结论仅是：原 consumer
并不近似 SO(2)-equivariant，因而硬编码 symmetry 的问题具有实际诊断意义。E05 本身不证明
等变架构能提高泛化。

判定：**Diagnostic hypothesis supported**。

### 5.6 E06：Minimal SO(2)-equivariant consumer

E06 的 active path 严格使用：

\[
z_t=\sum_i\alpha_{i,t}(A_{i,t}u_i+B_{i,t}Ju_i),
\]

其中 scalar network 只看 invariant activity/calibration identity、modulation 和 baseline。
结构测试验证了 physical SO(2) equivariance、aligned neuron permutation invariance 和梯度流。

但性能为：

- mean R²：`0.3191`，相对 E01 `−0.2945`；
- 0/8 sessions 提升；
- worst-session 20161021：`−0.4055`；
- best validation R² 仅 `−1.9857`；
- SD_session：`0.3157`，约为 E01 的 2.78 倍。

硬件/容量侧，E06 active trainable parameters 为 `67,208`，scalar consumer reference MAC
为 `3,112,960`，明显低于 E01 coupled decoder 的约 `57,970,688` MAC。也就是说 E06 同时
改变了 symmetry 和容量/decoder family。它的负结果足以关闭当前 A1 实现，但不能区分是
严格 symmetry 的限制、容量骤减、优化困难还是 activity-to-vector bottleneck 导致失败。

判定：**Not supported；按 fail-closed 停止当前 E06 路线**。

### 5.7 E07：SO(2) model 去掉 direction

仓库中没有找到 E07 的实现、分支、运行 receipt 或结果文档。由于 E06 已经大幅低于 E01，
继续运行 no-direction ablation 对“是否值得部署 E06”的边际价值有限；但任务书原问题
“vector direction 是否在 strict SO(2) consumer 中必要”仍然没有被实验回答。

判定：**Not run；不得把缺失实验写成负结果**。

### 5.8 E08：Analytic-only decoder

E08 比较：

\[
\hat v_{PV}\propto\sum_i(r_i-b_i)\beta_i,
\]

以及 source-only 选择超参数的 ridge OLE：

\[
\hat v_{ana}=g(B^TB+\lambda I)^{-1}B^T(r-b),
\]

其中 `lambda=100`、`gain=0.3090719545`，均由 source sessions 选择/拟合。

- Population-vector-like：mean R² `−0.0216`，worst `−1.3680`，失败。
- Ridge OLE：mean/median/worst `0.2056/0.2302/0.1093`，8/8 sessions 为正。
- Ridge OLE 相对 E01 mean delta `−0.4080`，8/8 均更差。
- Median angular error：`55.0°`；median speed ratio：`0.859`；mean vector-scale slope：
  `0.183`。

Ridge OLE 说明 T4 direction/rate 中存在可用信号，但 endpoint-only T4 不包含 instantaneous
speed 和 within-trial dynamics；其绝对表现不足以作为独立 decoder，也不是强 anchor。

判定：**Partially supported：非平凡但较弱的 analytic anchor**。

### 5.9 E09：Analytic + direct residual

E09 使用：

\[
\hat v=v_{ana}+\Delta v_\theta,
\]

与 E01 保持相同的 coupled network 容量；output head 零初始化使初始 prediction 精确等于
E08 ridge OLE。target side 没有新参数、optimizer 或 backward。

结果：

- mean R²：`0.6277`，相对 E01 `+0.0141`；
- 7/8 sessions 提升，但 20160923 退化 `−0.1274`；
- worst-session：`0.4280 → 0.3512`；
- SD_session：`0.1136 → 0.1438`，约增加 26.6%；
- mean residual-energy fraction：`0.3711`，范围 `[0.1515,0.6622]`；
- aligned full model `0.6277`，analytic-only `0.2057`，analytic-row-shuffle full model
  `0.3927`。

Residual-energy fraction 的实现是 `mean(Δv²)/mean(v_target²)`，不是“解释 R² 的比例”，
也没有消除 analytic/residual cross term。它表明 residual 的 raw power 小于 target power，
但不能单独证明网络没有通过 cancellation/reparameterization 承担主要预测。

跨 8 sessions，residual-energy 与 E09-vs-E01 delta 的 Pearson correlation 为 `−0.862`
（描述性 p=`0.0059`）：越依赖 residual 的 session，E09 越可能失去相对基线的优势。
这一模式与任务书所期待的“更稳定 residual”相反。

判定：**Partially supported / unstable**。

### 5.10 E10：Analytic + local-frame residual

E10 保持与 E09 相同的参数量和 analytic branch，仅把两个 Cartesian outputs 改解释为：

\[
u=\frac{v_{ana}}{\|v_{ana}\|+10^{-6}},\qquad
\Delta v=\delta_\parallel u+\delta_\perp Ju.
\]

结果：

- mean R²：`0.6379`；相对 E01 `+0.0243`，7/8 sessions 提升；
- 相对 E09 mean delta：`+0.0102`，5/8 sessions 提升，95% interval
  `[−0.0265,+0.0469]`；
- 相对 E09 最大改善为 20161021 `+0.0810`，最大退化为 20160923 `−0.0696`；
- worst-session：`0.2816`，比 E09 再低 `0.0696`，比 E01 低 `0.1464`；
- SD_session：`0.1690`，比 E01 高约 48.8%；
- mean residual-energy fraction：`0.4108`，高于 E09 的 `0.3711`；
- analytic-row-shuffle full model mean R²：`−0.4585`。

E10 shuffle 同时改变 analytic anchor **和 residual reconstruction frame**，而 E09 shuffle
只改变 anchor、保留已形成的 Cartesian residual。因此 E10/E09 的 shuffle 分数不能直接用于
比较“谁更依赖 carrier”；E10 的更大崩溃至少部分来自更强的干预定义。

Local frame 在 `v_ana≠0` 时仍能用两个 scalar 表示任意二维 residual，并没有减少输出维度。
它引入的是 frame dependence/inductive bias，而不是额外表达能力。当前结果显示该 bias 带来
小幅 mean 改变，却使弱-anchor session 20160923 和整体尾部更差；这不满足任务书要求的
worst-session/稳定性信号。

判定：**Partially supported on mean；not supported as a robustness improvement**。

## 6. 三条 Variant 路线的独立判断

### 6.1 Variant C：Uncertainty-aware carrier

支持证据：

- E02 同时改善 mean、worst 和 session spread；
- E03 在 E02 上继续改善 mean/worst；
- target-side 仍是小矩阵 closed-form 计算，无 iterative optimization；
- E02 不增加网络参数或 decoder query MAC，部署代价最低。

反证与缺口：

- E04 显式 logit 降权失败；
- E03 的增益只有 `+0.0080`，无 binding/random/constant controls；
- 没有 reliability–error correlation；
- 没有 H1/low-calibration 结果，因此没有直接验证原始 sparse-uncertainty 假设；
- 无 seeds 43/44。

结论：**三条路线中最值得继续，但应先复验 E02，再决定 E03。** 当前不能宣传为已经证明
Bayesian uncertainty mechanism。

### 6.2 Variant A：SO(2) task-frame equivariance

支持证据：

- E05 明确发现现有 consumer 的 rotation inconsistency；
- E06 在实现层面通过严格 SO(2) 和 permutation invariance 测试。

反证与缺口：

- E06 external mean/worst 均大幅退化，0/8 session 提升；
- active network 容量和计算量远低于 E01，symmetry 与 capacity 未解耦；
- E07 未运行；
- 无 matched-capacity equivariant consumer、无 low-calibration 或多 seed 证据。

结论：**几何诊断成立，当前 consumer 失败。** 不建议继续复杂 geometric attention，更不建议
并入 TF-SR；只有在愿意做 matched-capacity rescue 时才值得重新打开。

### 6.3 Variant B：Analytic anchor + residual

支持证据：

- E08 ridge OLE 在所有 sessions 输出非零正 R²；
- E09/E10 residual power 小于 full target raw power；
- E09/E10 mean 均略高于 E01；
- analytic shuffle 会显著破坏最终输出，说明组合模型确实依赖 analytic alignment/frame。

反证与缺口：

- analytic-only 与 E01 差距为 `−0.4080`；
- E09/E10 worst-session 和 session variance 都比 E01 差；
- E10 只比 E09 高 `+0.0102`，且 residual power 更大；
- residual-energy 越高，relative gain 越差；
- 缺少 residual-only、quality-stratified 和 reliability-gated controls；
- 无多 seed 或确定性 checkpoint-window 复验。

结论：**decomposition 有弱 mean 信号，但没有表现出原任务要求的鲁棒性价值。** 当前不建议
把 direct/local-frame residual 作为核心方法或与 TF-SR 合并。

## 7. 对原任务三个核心问题的回答

### Q1：T4 direction 是否应被当作 geometric object？

**表示论层面：是。效用层面：尚未成立。** E05 说明 transformation consistency 不是空问题；
E06 则说明仅凭硬编码 SO(2) 并不会自动改善 external decoding。

### Q2：T4 能否构成有实际价值的 target-session analytic anchor？

**能构成非平凡但较弱的 anchor。** Ridge OLE 比 population vector 稳定得多，但 R² 只有
`0.2056`。与 residual 结合后的 mean 有小幅增长，尾部却明显恶化，所以“anchor 有信号”与
“anchor decomposition 有稳定泛化价值”必须分开陈述。

### Q3：sparse calibration 的主要瓶颈是否是 carrier uncertainty？

**本轮没有直接回答。** E02/E03 提供间接正信号，但实验只覆盖 50-trial CO，没有 H1
4-trial/Date2、budget curve、reliability-error correlation 或 shuffled controls。

## 8. 测量、选择与数据隔离审计

### 8.1 已满足的边界

- E02/E03/E04 audit 均 `pass=true`；prior 和 normalization 来自 source sessions；
- E05 排除 8 个 test sessions，评估前后 student state SHA 相同；
- E08 leakage audit `passed=true`，continuous target velocity 仅在 prediction 固定后评分；
- E09/E10 metadata 明确 `target_optimizer=false`、`target_backward=false`、
  `target_parameter_count=0`；
- E09/E10 analytic lambda/gain 锁定到 E08 source-only receipt；
- calibration trials 与 evaluation trials 分离。

没有发现 target-side gradient adaptation 或 optimizer 泄露。

### 8.2 仍需明确的证据边界

- learned runs 使用 40-epoch cap + patience 10 + noisy validation argmax，而冻结的
  [MEASUREMENT_PROTOCOL_V4.md](MEASUREMENT_PROTOCOL_V4.md) 要求固定 12 epochs 并平均
  epochs 5–12；
- E09/E10 receipt 的 `epoch_checkpoints=[]`，无法事后重建 V4 tail-average；
- 8 个 target-test sessions 被反复用于 E02→E03/E04→E09→E10 的继续/停止决策，故最终
  E10 test 已参与实验路线选择，不再是独立 blind test；
- 只有 seed 42，无法估计任务书要求的 across-seed variance；
- E09 经历断电/恢复和多 checkpoint 目录复用，虽完成了状态恢复，但不符合 V4 独占、
  clean-run 目录的严格形式；E10 是 fresh seed-42 fit。

因此最恰当的标签是：**held-out-selected seed-42 development screen**。

## 9. Target-side compute 与参数开销

| 实验 | 新增/主要 target-side 计算 | 参数/模型变化 | Target optimizer/backward |
|---|---|---|---|
| E01 | T4 OLS + B3S/coupled forward | 约 4.613M trainable | 无 |
| E02 | per-unit 3×3 posterior solve | 相对 E01 +0 | 无 |
| E03 | E02 + 2×2 angular quadratic form | encoder +64 params | 无 |
| E04 | E03 statistics + static pre-softmax bias | +1 gamma | 无 |
| E05 | 仅离线 rotation diagnostic | 无训练参数变化 | 无 |
| E06 | invariant encoder + scalar SO(2) consumer | active 67,208 params；3.113M consumer MAC | 无 |
| E07 | 未运行 | — | — |
| E08 | per-session 2×2 ridge solve + O(N) per window | target params 0 | 无 |
| E09 | E08 analytic branch + matched coupled residual | 与 E01 同参数；target params 0 | 无 |
| E10 | E09 + local-frame normalize/rotate | 与 E09 同参数；target params 0 | 无 |

E02 是现阶段精度、实现风险和 target-side 成本之间最合理的候选。E06 计算显著更低，但现有
精度不具备可用性；E09/E10 没有增加 learned parameter count，却仍保留 E01 的大 coupled
decoder，因此不是硬件简化方案。

## 10. 下一步建议与停止规则

### 10.1 推荐的最小复验

优先级 1：**E01 vs E02 clean replication**。

- seeds 42/43/44；
- 固定 12 epochs、保存每 epoch checkpoint；
- 使用 epochs 5–12 tail average，不使用 argmax；
- 同时跑 50-trial CO 和至少一个 low-calibration budget；
- 在不重复查看 formal test 的 validation development sessions 上做选择。

优先级 2：只有 E02 在 mean/worst/spread 上复现后，做 **E02 vs E03 + controls**：

- real q；
- neuron-row shuffled q；
- constant q；
- reliability–carrier-error correlation；
- H1 Date2 / 4-trial 或当前可行的最稀疏 regime。

### 10.2 当前应停止或暂缓的工作

- 不扩展 E04 logit bias；
- 不继续复杂化 E06 geometric attention；
- 不补 E07，除非重新开启 matched-capacity E06 rescue；
- 不把 E09/E10 与 TF-SR 合并；
- 不根据当前最高 mean R² 继续叠加 posterior + SO(2) + analytic residual 联合模型。

### 10.3 若仍要复验 Variant B

应先复验 E09，而不是继续发明 E11：

- clean E01/E09 seeds 42/43/44；
- 增加 `B-Res-Only`，推理时移除 anchor 但保留训练好的 residual；
- 按 analytic quality / residual-energy 分层报告；
- 只有 mean delta 达到预设效应门槛且 worst-session 不退化，才比较 local frame；
- 如研究 fallback，应使用仅依赖 target-available reliability 的预注册 gate。

## 11. 原任务交付项完成度

| 交付项 | 当前状态 |
|---|---|
| 代码分支 / commit 列表 | 已在本文件汇总 |
| E01–E10 状态表 | 已汇总；E07 明确为未运行 |
| 每个 target session raw metrics | 已汇总 E01/E02/E03/E04/E06/E08-Ridge/E09/E10 |
| seeds 42/43/44 汇总 | **未完成，仅 seed 42** |
| H1 Date-wise 结果 | **未完成** |
| carrier estimator 公式 | E02/E03/E08–E10 已汇总，详细定义见原任务书/实现 |
| target-side compute/memory | 主要 compute/参数已汇总；非所有算子都有统一 cycle receipt |
| 失败实验 | E04、E06、E08-PV 及部分负结果已保留 |
| 数据泄露/target adaptation 自查 | 已汇总；无 target gradient，但存在 adaptive test reuse 边界 |
| 三条路线独立判断 | 已完成 |

## 12. 证据来源

任务与测量规则：

- [T4_E01_E10_EXPERIMENT_PLAN_20260821.md](T4_E01_E10_EXPERIMENT_PLAN_20260821.md)
- [MEASUREMENT_PROTOCOL_V4.md](MEASUREMENT_PROTOCOL_V4.md)

单项文档：

- [E02_POSTERIOR_MEAN_T4_20260824.md](E02_POSTERIOR_MEAN_T4_20260824.md)
- [E03_POSTERIOR_ANGULAR_RELIABILITY_20260825.md](E03_POSTERIOR_ANGULAR_RELIABILITY_20260825.md)
- [E04_POSTERIOR_RELIABILITY_LOGIT_20260825.md](E04_POSTERIOR_RELIABILITY_LOGIT_20260825.md)
- [E05_ROTATION_CONSISTENCY_20260825.md](E05_ROTATION_CONSISTENCY_20260825.md)
- [E06_MINIMAL_SO2_CONSUMER_20260827.md](E06_MINIMAL_SO2_CONSUMER_20260827.md)
- [E08 experiment record](../results/e08_analytic_only_experiment.md)
- [E09 experiment record](../results/e09_analytic_direct_residual_experiment.md)
- [E10 experiment record](../results/e10_analytic_local_frame_residual_experiment.md)

Tracked aggregate/receipt：

- [E02 aggregate](../results/e02_posterior_mean_t4_v1/E02_POSTERIOR_MEAN_T4_AGGREGATE.json)
- [E02 audit](../results/e02_posterior_mean_t4_v1/E02_POSTERIOR_MEAN_T4_AUDIT.json)
- [E03 aggregate](../results/e03_posterior_angular_reliability_t4_v1/E03_POSTERIOR_ANGULAR_RELIABILITY_AGGREGATE.json)
- [E03 audit](../results/e03_posterior_angular_reliability_t4_v1/E03_POSTERIOR_ANGULAR_RELIABILITY_AUDIT.json)
- [E04 aggregate](../results/e04_posterior_reliability_logit_t4_v1/E04_POSTERIOR_RELIABILITY_LOGIT_AGGREGATE.json)
- [E04 audit](../results/e04_posterior_reliability_logit_t4_v1/E04_POSTERIOR_RELIABILITY_LOGIT_AUDIT.json)
- [E05 audit](../results/e05_rotation_consistency_t4_v1/E05_ROTATION_CONSISTENCY_AUDIT.json)
- [E08 receipt](../results/e08_analytic_only_t4_seed42.json)

E01/E06/E09/E10 的 `p3_*.json`、运行日志、checkpoints 和 TensorBoard 文件受 `.gitignore`
管理，未作为本汇总提交内容；为保证远程文档自包含，本文件已经固化其关键 aggregate、
逐 session、控制指标、参数和协议边界。原始二进制 checkpoint 和日志不应进入 Git。

## 13. 最终结论

本轮最有价值的发现不是“某个模型已经获胜”，而是三条方法假设被分解出了不同层级的答案：

- **posterior mean：**有低成本弱正信号，值得按严格协议复验；
- **reliability feature：**有额外弱信号，但机制对照缺失；
- **reliability logit：**当前实现为负；
- **SO(2) consistency：**原模型确实违反，但最小硬等变 consumer 性能失败；
- **analytic anchor：**有信息但很弱；
- **direct/local residual：**mean 略升，鲁棒性和 tail behavior 变差；
- **E07/H1/multi-seed：**仍未回答。

所以当前最稳健的路线是：**保留 E02/E03 作为候选，关闭 E04/E06，暂缓 Variant B 与
TF-SR 合并，并用 clean multi-seed + sparse-regime 实验决定最终去留。**
