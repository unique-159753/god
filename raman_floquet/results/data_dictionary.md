# 数据字典

所有NPZ均可用`numpy.load(path, allow_pickle=False)`读取。数值为真实计算，未按图像反推。主参数、精度、物理时间原点及源哈希在`localization_parameters.json`。实际每一字段的shape/dtype见`npz_inventory.csv`，这里给出其物理语义。

## 主扫描

`localization_grid.npz`：`omega`111点、`phi`81点。标量场shape=(81,111)，模态场=(81,111,178)，outerindices=(81,111,36)。第一维为相位、第二维频率、最后一维模态。`localization_slices.npz`相位为0、π/4、π/3、π/2、3π/4、π，其shape=(6,111,...)；π/3是额外真实exact计算。

| 字段 | 含义 |
|---|---|
| `exact_*` / `effective_*` | 数值时间有序Floquet / 一阶van Vleck并做首阶kick的实验室参考系态 |
| `*_quasienergies` | 中心区[−ω/2,ω/2)准能量；模态标签随参数独立排序，不能视作连续分支编号 |
| `effective_energies` | 折叠前的一阶有效能量 |
| `*_ipr_site`, `*_D_site` | ρj=Σspin|ψ|²，IPR=Σρj²，D=−lnIPR/ln89 |
| `*_ipr_component`, `*_D_component` | Σ178分量|ψ|⁴，D以ln178归一化 |
| `*_D_component_logL` | 同一分量IPR，但D以ln89归一化；不与主D混用 |
| `*_outer_indices` | 同一模型中最高36个|ε|的零基模态索引 |
| `*_outer_*` / `*_all_*` | 所选36态 / 全178态的算术平均；平均D不同于平均IPR再取对数 |
| `*_outer_selection_gap` | 被选集合与未选集合的边界|ε|间隙 |
| `*_static_outer_weights` | 每个Floquet态对未折叠静态外侧36态子空间的投影权重，Σmode weight=36 |
| `*_projected_*` | 按上述权重/36加权的平均；是平滑对照而非硬外侧观测量 |
| `*_dominant_static_*` | 最大静态本征态投影及其零基编号；不表示态只含该一个分量 |
| `exact_to_effective` / permutation相关字段 | 最大总平方重叠Hungarian匹配；最后索引含义以字段名/源码为准 |
| `*_matched_overlap` | 逐态匹配平方重叠及其均值/最小值；小值时单态分支标签不可靠 |
| `delta_D_site` | 独立硬选态的exact外侧均值减effective外侧均值 |
| `matched_delta_D_site` | 以exact外侧集合为基准，匹配后逐态D差的平均 |
| `selection_delta_D_site` | 匹配到的effective集合与其自身硬外侧集合的D均值差 |
| `projected_delta_D_site` | 两模型静态投影加权平均之差 |
| 同名component等后缀 | 按相应替代IPR定义作同样分解 |
| `exact_unitarity_error` | 传播子相对Frobenius幺正误差 |
| `exact_eigen_residual`等 | 计算所得U的本征、正交、Schur、模长残差；不是时间离散误差上界 |
| `exact_min_eigenphase_gap` | 最小环形本征相位间隙，单位弧度；不是直接的能量间隙 |
| `steps`, `seconds` | 每周期时间步与运行耗时；耗时可能受并行竞争影响 |

`localization_grid.csv`8991行，包含各参数点的标量汇总；额外π/3的111行另存`localization_extra_pi_over_3.csv`。所有模态观测量由NPZ完整保留。`localization_points/pPPP_wWWW.npz`保存相位索引PPP、频率索引WWW的逐点缓存，PPP=81为额外π/3。缓存不存全部本征向量，只存模态观测量和权重；代表点和验证NPZ额外保存完整波函数。

## 共振诊断

`resonance_grid.npz/.csv`：三展宽η=0.02、0.05、0.10，111频率×81相位。`R_full`、`R_outer`、`Rnorm_full`、`Rnorm_outer`等场通常shape=(3,81,111)，依次为η、φ、ω。`R_outer`只保留静态未折叠外侧掩码中至少一个端点的态对。分母(Em−En−ω)²+η²，排除m=n，N=178。

`Rnorm=η²R`，不是概率；对本模型其上界0.64t₀²。`term_sums`储存T²、S²、交叉项的φ无关系数，交叉项乘cosφ；其确切轴在inventory和resonance.py中给出。`resonance_slices.*`为六个相位的解析精确矩阵元切线。

`resonance_static.npz`含静态E、实本征矢、T/S矩阵元、密度、两种IPR/D和静态外侧掩码。静态态按能量升序、索引从0开始；物理站点j从1开始。`resonance_state_pairs.csv/.npz`包含四个代表参数各前10对，按|M|²/(δ²+0.05²)排序；`resonance_near_only_pairs.csv`另存窄失谐窗口。排序权重不是实际跃迁概率。

## 代表态、补偿和单驱

`representative_floquet_projections.*`、`representative_state_pair_summary.json`含选中Floquet态的准能量、D、静态投影和密度；`representative_cache/*.npz`保留四模型各代表点完整模式及观测量。A=H₀，B=Heff+kick，C=exact128，D=exact128−gFJ₂ˣ；主图12组合共ω=6.2、8、12、16与φ=0、π/4、π/2。注意字母D模型与参与指数D是不同概念。

`four_model_outer_comparison.*`和`four_model_differences.csv`给硬外侧与静态投影比较；`kick_only_baseline.*`是把同一个首阶kick作用于H₀的附加控制。

`single_drive_controls.*`含A-only(A=.8,B=0)、B-only(A=0,B=.8)，ω=6.2、8、16与φ=0、π/4、π/2，共18标签组合。全部模式的静态/kick/exact准能、两种IPR/D、外侧索引和静态投影权重见NPZ，CSV含双驱参照。A-only复用相同φ无关传播子；B-only改变φ等于改变时间原点，准能不变而τ0的IPR可变。`single_drive_summary.json`记录幺正和M128→256检查。

## Sambe

`sambe_pointI_MJ.npz`中I=0..3代表(6.2,π/3)、(6.2,0)、(6.2,π/2)、(8,π/2)，J=1、2、3是光子截断mmax，不是时间步。Sambe维数(2J+1)×178。保存本征能量、光子系数、相位零重构、规范对齐的光子权重、exact匹配、中心区计数及副本回填标签。副本不是新能带。

`sambe_summary.*`给截断收敛；`sambe_outer_static_photon_analysis.json`给多态结构；图5数据在`sambe_figure_*.csv/.npz`。零驱动折叠态也可主要在±1扇区，因此非零边带不能单独判定共振。

## 验证、相关性与原始出处

`grid_validation.json`为全9102点完整性及9个代表点收敛审查；`validation_diagnostics.csv/.npz`保存32/64/128、独立ODE、敏感点256/512与Q分块完整模态。`grid_sensitive_point_validation.json`、`grid_min_gap_Q_diagnostic.npz`记录敏感性；没有替换统一M64生产数据。

`solver_validation.json`为独立Hamiltonian/Fourier/kick/FFT/native/ODE检验；`sambe_validation.json`为零驱动及副本规范检验。`correlation_summary.json/.csv`给不同η、窗口、掩码和定义的Pearson/Spearman；phase_centered意为分别扣除各量在每个频率处的相位算术平均，再关联残差。结构化样本不报独立样本p值。`fig1_fig2_plot_data.npz`保存主图实际绘图数组。

主网格M64；实际误差、近简并的基底敏感性与单尺寸限制在报告中明确列出。`file_manifest.json`和`SHA256SUMS`列出文件长度、SHA256，压缩包本身不放入其自己的manifest。`code/`是源代码镜像；完整zip在外层也放置同样脚本，使默认results相对位置可直接运行。
