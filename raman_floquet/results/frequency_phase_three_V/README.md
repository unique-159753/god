# 三种静态基准相的双驱动频率—相位扫描

本目录对应用户最新修订，横轴只使用无量纲ω/t₀=0…10，共50点。φ=2πk/50，k=0…49，共50个不重复周期点，三组V分别为0.5、2.0、4.0。另真实计算φ=π/2的50点切线；φ=0复用主网格同一点，不插值。

模型为H(τ)=[1+0.8cos(ωτ)]T+[0.5+0.8cos(ωτ+φ)]S+HZ。L=89、α=55/89、W=θ=0、PBC、ℏ=1。用户补充t₀/h=500Hz仅作为可选的物理标定元数据，主图和求解器均不使用Hz；若需要换算，f=(t₀/h)ω。

统一主定义IPRn=Σj,σ|ψn(j,σ;τ=0)|⁴，Dn=−lnIPRn/ln89。注意分母是lnL，而不是此前辅助定义ln(2L)；因此Dn可能大于1，其严格上界为ln178/ln89。这是有限尺寸、自旋分辨基底的参与指数，不直接作为热力学分形维数或相变判据。站点合并IPR另作辅助控制。

无驱动基准H₀=T+0.5S+HZ对每个V独立计算。主差值为逐态Dn的全谱算术均值减去对应H₀全谱算术均值；不是对平均IPR取负对数。

ω=0没有Floquet周期。程序直接对角化冻结驱动Hfrozen=1.8T+[0.5+0.8cosφ]S+HZ，保存真实静态能量及本征态。此列的quasienergies、outer_Dn及Floquet unitarity为NaN，外侧indices为−1，steps=0。它与H₀和ω→0⁺的单周期本征态都不同，不作插值。

ω>0由真实时间有序传播求解，不使用有效Hamiltonian代替。每点从M=32起，逐次加倍，联合检查单周期算符、匹配后的逐态Dn、全谱均值与外侧均值。生产扫描采用保持反射对称性的精确bond/star局部门和六阶Yoshida时间推进（每个宏时间片有七个对称子阶段），另以完整178维FFT和独立ODE交叉验证。四组局域生成元的和严格等于原89条边；相反射边在同组，固定顶点的两相邻边作为可精确求指数的三站点门，避免原偶奇拆分破坏Q。生产新后端与先前FFT后端分开保存，从头独立计算全网格，没有混用两种时间离散结果。

本地需要C99编译器（cc/gcc/clang均可），C源码嵌入`frequency_fast_native.py`并在可写临时目录自动编译；没有编译器会明确报错。对应独立验证为`frequency_fast_native_validation.*`，先前FFT部分计算保留在相邻的`frequency_phase_three_V_fft_crosscheck/`作交叉验证。

利用精确反射×σz分块及粒子空穴共轭关系，保存一个89维对称子空间的系数，并以附带代码无损重构完整178个本征态；这不是在参数网格间插值。主网格所有50相位均独立演化，φ与2π−φ用于实际数值对称性检验。

运行入口位于本目录上两级：

```bash
cd /workspace/god/raman_floquet
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scan_frequency_phases.py --workers 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python validate_frequency_scan.py
MPLCONFIGDIR=/tmp/raman_matplotlib python plot_frequency_phases.py
python write_frequency_report.py
python package_frequency_results.py
```

或在包含这些脚本的目录中运行`bash run_frequency_three_V.sh`执行完整顺序。可用`RAMAN_PYTHON`指定Python解释器。新计算需要`numpy/scipy/matplotlib/threadpoolctl/Pillow`（版本见requirements.txt）及C99编译器；已完成数据可直接画图，无需再次运行网格。

`frequency_phase_scan.npz`主标量场shape=(3V,50φ,50ω)，模态场再加178维。`frequency_phase_slices.npz`为(3V,2φ,50ω,...)，φ=[0,π/2]。`points/`保留每点收敛记录和全部本征态的无损重构信息。`eigenstate_observables.csv`每个模态一行，含真实能量/准能量、IPR、Dn、辅助站点指标、外侧标签和对称标签。frozen静态与Floquet不混用。

一阶van Vleck+kick只作为单独对照，不参与exact时间演化。另保存未加kick的一阶均值和kick-only均值，防止把全部高频观测量变化都归给NNN-SOC。

计算完成后还会生成共色标PNG/PDF、独立验证、数据字典和比较报告。若任何点未达到设定收敛阈值，程序明确标记，绘图程序不会将它伪称为已收敛结果。
