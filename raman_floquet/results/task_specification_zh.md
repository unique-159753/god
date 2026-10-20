# Codex 科研计算任务：验证双驱动准周期 Raman 晶格中的 Floquet 共振、相位干涉及准能量选择性退局域化

## 一、研究目标

基于此前已经完成的 exact Floquet 与一阶有效模型对照，进一步分析：

1. 为什么在 \(\omega/t_0\approx6.0\sim6.6\) 的频率范围，exact Floquet 外侧态平均分形维数明显高于一阶有效理论？
2. 这一差异是否来自静态本征态之间的一光子 Floquet 共振？
3. 两种周期驱动之间的相位干涉，能否解释相位 \(\varphi\approx\pi/4\sim\pi/2\) 附近的异常增强？
4. 若一光子过程不能解释，是否存在高阶 Floquet 耦合或准能量分支混合的证据？
5. 这些效应是否可以解释特定准能量区域的退局域化，而不只是全谱平均 IPR 的变化？

请务必执行实际数值计算并生成论文风格的图。不得使用模拟数据、手工构造曲线或图像生成模型代替真实计算。

如有之前保存的 `fig2_corrected_Douter_comparison` 对应原始数据，请直接读取；如果没有，请按以下参数重新计算，不要通过图片像素反推数据。

## 二、Hamiltonian

使用一维具有两个自旋分量的准周期 Raman 晶格：

\[
H(\tau)=t(\tau)T+\Delta(\tau)S+H_Z
\]

其中

\[
T=-\sum_j\left[
c^\dagger_{j\uparrow}c_{j+1,\uparrow}
-c^\dagger_{j\downarrow}c_{j+1,\downarrow}
+\mathrm{H.c.}\right]
\]

\[
S=\sum_j\left[
c^\dagger_{j\uparrow}c_{j+1,\downarrow}
-c^\dagger_{j\downarrow}c_{j+1,\uparrow}
+\mathrm{H.c.}\right]
\]

\[
H_Z=\sum_j
[V\cos(2\pi\alpha j+\theta)+W]
(n_{j\uparrow}-n_{j\downarrow})
\]

周期驱动：

\[
t(\tau)=t_0+A\cos(\omega\tau)
\]

\[
\Delta(\tau)=\Delta_0+B\cos(\omega\tau+\varphi)
\]

参数：

- \(L=89\)
- \(\alpha=55/89\)
- \(t_0=1\)
- \(\Delta_0=0.5\)
- \(V=2.5\)
- \(W=0.01\)
- \(A=B=0.8\)
- \(\theta=0\)
- 周期边界 PBC
- \(\hbar=1\)

单粒子基底顺序：

\[
|1,\uparrow\rangle,|1,\downarrow\rangle,
\ldots,|L,\uparrow\rangle,|L,\downarrow\rangle.
\]

总矩阵维数为 \(2L=178\)。

使用相同的索引和符号约定构造所有模型。检查矩阵 Hermiticity，避免最近邻 SOC 负号写错。

## 三、计算一光子共振矩阵元

先构造静态 Hamiltonian：

\[
H_0=t_0T+\Delta_0S+H_Z.
\]

计算全部静态本征态：

\[
H_0|n\rangle=E_n|n\rangle.
\]

在静态能量本征基底中计算：

\[
T_{mn}=\langle m|T|n\rangle,
\]

\[
S_{mn}=\langle m|S|n\rangle.
\]

时间 Fourier 分量定义为：

\[
H(\tau)=\sum_l H_l e^{il\omega\tau}.
\]

因此

\[
H_{+1}=
\frac A2T+\frac B2e^{i\varphi}S,
\]

\[
H_{-1}=
\frac A2T+\frac B2e^{-i\varphi}S.
\]

对满足 \(E_m-E_n\approx+\omega\) 的吸收过程，使用 \(H_{-1}\) 对应的矩阵元：

\[
M_{mn}(\varphi)=
\langle m|H_{-1}|n\rangle.
\]

直接计算：

\[
|M_{mn}(\varphi)|^2.
\]

同时将其分解为：

\[
|M_{mn}|^2=
\frac14\left[
A^2|T_{mn}|^2+B^2|S_{mn}|^2
+2AB\operatorname{Re}
(e^{-i\varphi}T_{mn}^*S_{mn})
\right].
\]

特别注意：由于当前静态矩阵为实对称矩阵，使用实本征基底时，相位干涉为 \(\cos\varphi\) 依赖。

请数值验证该解析关系，并与一阶 van Vleck 有效跃迁

\[
g_F=\frac{AB}{\omega}\sin\varphi
\]

明确区分。

## 四、构造一光子共振权重

定义共振诊断量：

\[
\mathcal R(\omega,\varphi;\eta)
=
\frac1{N}
\sum_{m\ne n}
\frac{
|M_{mn}(\varphi)|^2
}{
(E_m-E_n-\omega)^2+\eta^2
}
\]

其中 \(N=2L\)。

这里 \(\mathcal R\) 是带展宽的共振响应诊断量，不是严格的实际跃迁概率或吸收率。

分别使用：

\[
\eta=0.02,\quad0.05,\quad0.10
\]

检验结果对展宽的敏感性。

另外构造有界的归一化共振权重：

\[
\mathcal R_{\rm norm}
=
\frac1N\sum_{m\ne n}
|M_{mn}|^2
\frac{\eta^2}
{(E_m-E_n-\omega)^2+\eta^2}.
\]

两种定义至少选择一种作为主要热图，并明确颜色条表示的量。

扫描：

\[
5.8\le\omega/t_0\le8.0
\]

\[
0\le\varphi\le\pi.
\]

建议使用：

- 频率方向 111 个点
- 相位方向 81 个点

由于一光子矩阵元可预先计算，请向量化计算共振权重，避免在每个参数点重新对角化静态 Hamiltonian。

## 五、Figure 1：一光子共振响应二维热图

请生成三幅面板：

(a) 全谱一光子共振权重

\[
\mathcal R(\omega,\varphi).
\]

(b) 仅保留涉及静态外侧能量态的共振权重。

静态外侧态按照 \(|E_n|\) 的最高 20% 选取，明确采用“至少一个态属于外侧区域”的选取规则。

(c) Exact Floquet 与 kick-corrected 一阶有效模型的外侧态分形维数差值

\[
\Delta\overline D_{\rm outer}(\omega,\varphi)
=
\overline D_{\rm exact}
-
\overline D_{\rm effective}.
\]

(c) 必须使用之前一致的 Floquet 准能区折叠和本征态选择规则。

三幅图使用完全相同的横纵坐标范围。

请分别标记：

- 共振权重最大的位置；
- \(\Delta\overline D_{\rm outer}\) 最大的位置；
- 静态谱给出的折叠参考频率 \(2\max|E_n|\)。

保存：

`results/fig1_resonance_vs_localization.png`

`results/fig1_resonance_vs_localization.pdf`

## 六、Figure 2：共振响应与局域化异常的定量相关性

从二维数据中提取以下相位：

\[
\varphi=0,\quad
\pi/4,\quad
\pi/3,\quad
\pi/2,\quad
3\pi/4,\quad
\pi.
\]

对每个相位绘制：

\[
\mathcal R(\omega,\varphi)
\]

以及

\[
\Delta\overline D_{\rm outer}(\omega,\varphi).
\]

可以使用两个独立面板，但不要通过人为缩放让不同物理量看起来一致。

计算两个量在相同网格上的 Pearson 与 Spearman 相关系数。

同时进行一个更严格的检验：分别扣除各自在每个频率处的相位平均值，再计算剩余相位响应的相关性，避免共同的频率变化趋势造成虚假的高相关。

要求特别检查：

如果 \(\mathcal R(\varphi)\) 主要遵循 \(\cos\varphi\)，而 exact Floquet 的增强集中在中间相位，则必须明确指出简单一光子干涉无法解释全部现象。

保存：

`results/fig2_resonance_correlation.png`

`results/fig2_resonance_correlation.pdf`

## 七、Figure 3：主要共振态对及其空间性质

固定代表性参数：

\[
\omega=6.2,\qquad\varphi=\pi/3.
\]

此外计算：

\[
\omega=6.2,\quad\varphi=0,
\]

\[
\omega=6.2,\quad\varphi=\pi/2,
\]

\[
\omega=8.0,\quad\varphi=\pi/2.
\]

挑选满足以下条件的代表性静态本征态对：

\[
|E_m-E_n-\omega|
\]

较小，且

\[
|M_{mn}|^2
\]

较大。

对于最重要的 5–10 对态，输出：

- \(E_n,E_m\)；
- 能级差 \(E_m-E_n\)；
- 共振失谐 \(\delta=E_m-E_n-\omega\)；
- \(|T_{mn}|^2\)；
- \(|S_{mn}|^2\)；
- \(|M_{mn}|^2\)；
- 两个静态态的 IPR 与 \(D_n\)；
- 两个态的实空间概率密度分布。

绘制最强共振态对的：

(a) 能级关系及一光子共振示意图（能级数值必须来自真实计算）。

(b) 两个静态本征态的空间密度。

(c) 代表性 exact Floquet 本征态在相应静态本征基底上的投影权重。

如果没有明显孤立的共振态对，应如实说明，不能强行挑选一对态解释整个外侧谱。

保存：

`results/fig3_resonant_state_pairs.png`

`results/fig3_resonant_state_pairs.pdf`

## 八、Figure 4：判断新跃迁与一光子共振谁占主导

比较以下模型：

A. 静态模型

\[
H_0.
\]

B. 一阶高频有效模型

\[
H_{\mathrm{eff}}=H_0+g_FJ_2^x,
\]

其中

\[
J_2^x=
i\sum_j
(\Psi^\dagger_j\sigma_x\Psi_{j+2}-\mathrm{H.c.}).
\]

C. 完整 exact Floquet 双驱动模型。

D. Exact Floquet 双驱动模型加反向补偿项：

\[
H_D(\tau)=H(\tau)-g_FJ_2^x.
\]

严格检查 \(g_F\) 的 Fourier 相位约定和正负号。

分别在

\[
\omega=6.2,\quad8,\quad12,\quad16
\]

以及

\[
\varphi=0,\quad\pi/4,\quad\pi/2
\]

计算外侧态平均 \(D_n\)。

对于时间无关有效模型，先折叠本征能量，再进行相同的 Floquet-zone 选态。需要比较实验室参考系 IPR 时，使用与定义一致的 kick 修正。

绘制四组模型的 \(D_n\) 对照曲线。

重点回答：

- 在高频时是否主要由 \(g_F\) 控制？
- 在 \(\omega=6.2\) 时抵消 \(g_F\) 后异常是否仍然存在？
- 是否存在独立于一阶 NNN-SOC 的有限频率增强？

保存：

`results/fig4_NNN_vs_resonance.png`

`results/fig4_NNN_vs_resonance.pdf`

## 九、额外检查：Floquet-Sambe 共振

若 Figure 1–3 无法充分解释异常，请在少量代表性参数点构建截断 Floquet-Sambe Hamiltonian：

\[
\mathcal H_{mn}^{F}
=
H_{m-n}+m\omega\,\delta_{mn}.
\]

首先取光子扇区：

\[
m=-1,0,+1.
\]

必要时再扩展到

\[
m=-2,-1,0,1,2
\]

检查收敛。

通过比较 Sambe 本征态的不同光子扇区权重，寻找靠近 \(\omega\approx6.2\) 的混合或避免交叉。

注意 Floquet-Sambe 截断存在重复准能量副本，必须正确识别物理上等价的态；不要把重复副本误判为新能带。

该步骤可作为附加 Figure 5。

## 十、计算精度与可靠性

必须执行以下检查：

1. Hamiltonian Hermiticity；
2. exact Floquet unitarity；
3. 时间切片 \(M=32,64,128\) 的收敛测试；
4. 共振展宽 \(\eta\) 敏感性；
5. 相位 \(\varphi\rightarrow-\varphi\) 的对称性检查；
6. 一阶有效 Hamiltonian 的傅里叶对易子系数验证；
7. 准能量折叠和外侧态选择规则保持一致；
8. 近简并本征态及可能的本征态分支交换检查。

图片必须使用真实数据，至少输出 300 DPI PNG 和矢量 PDF。

导出所有数值到 CSV/NPZ，完整保留可复现 Python 代码。

## 十一、最终科研分析报告

计算完成后，请明确回答以下六个问题：

1. 在 \(\omega\approx6.2\) 附近，是否存在明显的一光子共振？
2. 共振响应热图与 exact Floquet 局域化异常热图是否吻合？
3. 共振矩阵元的相位依赖能否解释异常区域的相位偏移？
4. 哪些态参与混合？它们原来的局域化性质如何？
5. 一阶 NNN-SOC 和有限频率共振分别解释了多少现象？
6. 目前是否有证据表明这一现象不同于已有的常规 Floquet 共振和单驱动局域化响应？

请在结果不支持假设时明确否定，不要预设必须存在新共振或新相变。

所有计算结果、图片及报告统一保存在 `results/` 文件夹。

## 本轮已由用户确认的观测量规则

站点ρj=|ψj↑|²+|ψj↓|²、IPRsite=Σρj²、Dsite=−lnIPRsite/ln89；准能折叠至[−ω/2,ω/2)后选|ε|最大的36态。另保留178分量IPR/ln178、静态外侧投影及重叠匹配对照。用户答复：采用上述明确规则，保留两套对照。
