# 频率—相位扫描数据字典

本轮所有主频率均为无量纲ω/t₀，主IPR=Σj,σ|ψjσ(τ=0)|⁴，Dn=−lnIPR/ln89。每个态都是固定驱动初相时间τ=0的单周期本征态；不是周期平均密度。对数分母沿全部模型与V保持一致，另存站点IPR作为辅助。

## 轴和主观测量

`frequency_phase_scan.npz`：V=(0.5,2,4)，omega=linspace(0,10,50)，phi=2πarange(50)/50。标量数组shape=(3,50,50)，依次为V、φ、ω；模态数组shape=(3,50,50,178)。`frequency_phase_slices.npz`的φ=(0,π/2)，相同字段shape=(3,2,50,...)。π/2不在50点主相位网格中，因此切线由额外真实计算得到。

| 字段 | 含义 |
|---|---|
| `IPR`、`Dn` | 全部178态的自旋分辨原始指标，Dn=-lnIPR/ln89 |
| `IPR_site`、`D_site` | 先合并每站点两自旋概率的辅助指标，以ln89归一化 |
| `D_component_log2L` | 同一自旋分辨IPR，另以ln178归一化；不是主Dn |
| `quasienergies` | ω>0时按升序排列的中心区[−ω/2,ω/2)准能量；ω0为NaN |
| `energies` | ω0冻结静态H的真实能量；ω>0为NaN，不混作准能量 |
| `parity_labels` | 每态的Q=reflection(j→−j)⊗σz标签±1 |
| `outer_indices` | ω>0最高36个|ε|的零基态索引；ω0全部−1 |
| `all_IPR`、`all_Dn` | 逐态指标的全谱算术平均，二者不经对数互相换算 |
| `outer_IPR`、`outer_Dn` | 中心准能区最高20%（ceil 35.6=36态）均值，ω0为NaN |
| `static_all_IPR`、`static_all_Dn` | 同V无驱动H₀的全谱均值，在每个驱动点重复记录便于对照 |
| `delta_all_Dn` | all_Dn−static_all_Dn，包含ω0真正冻结驱动的差值 |
| `steps` | 每周期宏时间片数，初始32，自适应倍增；ω0=0 |
| `is_frozen` | ω0标记；并不表示无驱动H₀ |
| `converged` | 是否达到联合时间步收敛准则；静态列表示静态对角化完成 |
| `unitarity_error` | 数值传播子相对Frobenius幺正残差；ω0没有传播子，因此NaN |
| `eigen_residual`、`orthogonality_error` | 对已计算传播子/静态H的本征方程与正交残差，不等同于时间离散误差 |
| `hermiticity_error`、`Q_commutator_error` | 实际T、S、HZ的Hermiticity和Q对易残差（静态列为冻结H残差） |
| `particle_hole_error` | Hamiltonian生成元的C H* C=−H残差；动力学U关系另由独立验证检查 |
| `sector_projection_error`、`sector_unitarity_error` | 对称块的投影漏量及幺正性残差 |
| `min_eigenphase_gap` | 全部Floquet乘子的最小环形相位间隙，单位弧度 |
| `final_*` | 最后相邻两档时间步的比较：算符、匹配逐态Dn、全谱/外侧均值、重叠及外侧成员交换 |
| `effective_IPR`、`effective_Dn`、`effective_*quasienergies` | 独立的一阶van Vleck+首阶kick对照；不会替代exact；ω0为NaN |
| `effective_all_Dn`、`effective_outer_Dn` | 上述近似的全谱/外侧均值，与exact用相同指标和选态 |
| `effective_no_kick_all_Dn` | 只用一阶Heff本征态的参考系对照 |
| `kick_only_all_Dn` | 只对H₀态施加首阶kick的辅助基线 |

`frequency_phase_scan.csv`为7500个主网格标量行；`frequency_extra_pi_over_2.csv`为150个额外相位点。`eigenstate_observables.csv`包含全部主网格及额外切线的每个态，明确区分静态能量、准能量、IPR/Dn、外侧及对称标签。

## 原始波函数及重构

逐点文件为`points/V{iv}_p{ip:02}_w{iw:02}.npz`，iv=0、1、2；ip=0…49对应主网格，ip=50为额外π/2；iw=0…49对应频率。ω>0时保存Q+扇区的`sector_vectors`（89×89复数）、`sector_quasienergies`和全178态的`reconstruction_order`。另一个扇区由精确C U* C=U、C=I⊗σx关系产生，与Q+基一起无损重构全部本征矢。不是插值，也不是删掉一半物理态。ω0直接保存`frozen_modes`。

```python
from read_frequency_point import read_point
data = read_point('results/frequency_phase_three_V/points/V0_p00_w01.npz')
modes = data['modes']  # shape=(178,178)，每列一个完整复本征态
```

脚本会重新计算每态IPR并核对原始值，也可`python read_frequency_point.py PATH --output full_modes.npz`导出完整展开版本。`convergence_history_json`保存从32起的全部比较，不能仅凭幺正残差判定收敛。

## 静态、验证及报告

`baselines.npz`包含H₀及全部冻结相位的真实本征态、Q基、IPR和Dn；其`zero_energy_outer_*`是单独命名的静态能量外侧诊断，不能当成ω0 Floquet外侧量。主`zero_outer_Dn`和`zero_quasienergies`恒NaN。

`scan_parameters.json`记录全部参数、积分阶数、容差及模型/扫描源码SHA256；`scan_completion.json`记录真实点数及未收敛列表。`frequency_fast_native_validation.*`是生产反射对称bond/star六阶方法的精确生成元、ODE和相位检查；`frequency_order6_validation.*`为独立FFT六阶对照。`frequency_solver_validation.*`及`method_benchmark.*`保留四阶辅助诊断，其中旧ln178数值不能混入新主图。全网格验证、代表点32/64/128与独立ODE对照另见`frequency_scan_validation.*`，跨后端对照表为`frequency_method_crosscheck.csv`。

`response_metrics.csv`和`analysis_summary.json`在实际主网格完成后生成；只比较有限尺寸观测量，排除ω0后统计Floquet响应。图中的2π边界只复制真实φ0行以闭合周期，不增加“计算点”或插值数据。所有主图同一指标跨三V共用色标。
