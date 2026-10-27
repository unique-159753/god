# 100×100 扫描数据字典

主指标统一为 spin-resolved IPR_n=sum_(j,sigma)|psi_n(j,sigma;tau=0)|^4，
Dn_n=-ln(IPR_n)/ln89。所有图均由完整 178 态算术平均得到。

## 轴与形状

`frequency_phase_scan.npz` 保存 V=(.5,2,4)、omega=linspace(.2,10,100)、phi=linspace(0,pi,100)。
数组轴依次为 V、phi、omega；主标量 shape=(3,100,100)，逐态数组 shape=(3,100,100,178)。
`frequency_phase_slices.npz` 的 phi=(0,pi/2)，主标量 shape=(3,2,100)。phi0 是主网格原值，
phiπ/2 是额外实际时间演化，不是相邻主网格行的插值。所有频率严格为正。

| 字段 | 含义 |
|---|---|
| `IPR`、`Dn` | 所有178态的原始自旋分辨指标；Dn 以 ln89 归一化 |
| `all_IPR`、`all_Dn` | 全谱逐态算术平均；平均Dn不等于平均IPR取对数 |
| `quasienergies` | 按升序排列、折叠到[-omega/2,omega/2)的准能量 |
| `parity_labels` | Q=reflection⊗sigma_z 对称扇区标签±1，用于固定近简并态基底和逐态匹配 |
| `steps` | 每周期宏时间片数，从32开始倍增，最终值逐点独立确定 |
| `converged` | 是否达到联合算符、匹配逐态Dn及全谱平均Dn收敛容差 |
| `is_frozen` | 本轮所有点均为false，不含omega0 |
| `unitarity_error`、`sector_unitarity_error` | 全传播子及Q+扇区归一化Frobenius幺正残差 |
| `hermiticity_error`、`Q_commutator_error` | 实际生成元的Hermiticity及Q对称残差 |
| `particle_hole_error` | 生成元 C H* C=-H 的残差，C为每站点自旋交换 |
| `eigen_residual`、`orthogonality_error` | 数值传播子本征方程和模态正交残差 |
| `sector_projection_error` | Q扇区投影漏量 |
| `schur_offdiagonal_error`、`eigenvalue_modulus_error` | Schur分解非对角残差与乘子模长偏离1 |
| `min_eigenphase_gap` | Floquet乘子间最小环形相位间隙，单位弧度 |
| `final_operator_delta` | 最终相邻时间片传播子归一化差，仅为数值收敛诊断 |
| `final_max_matched_Dn_delta` | 同Q扇区重叠匹配后的最大逐态Dn数值差 |
| `final_mean_Dn_delta` | 最终相邻时间片全谱平均Dn数值差 |
| `final_minimum_mode_overlap`、`final_mean_mode_overlap` | 匹配态重叠的最小值和均值 |
| `final_max_matched_quasienergy_delta` | 匹配态准能量环形差的最大值 |
| `seconds` | 对应逐点计算耗时，不是物理时间 |

带 `delta` 的 final/收敛历史字段只比较不同时间片离散，不能当作物理驱动差值。
本轮没有无驱动基准差值、外侧20%态或有效模型比较字段。

## CSV

`frequency_phase_scan.csv`：30000条主网格标量记录；`frequency_extra_pi_over_2.csv`：300条额外正交切线记录。
`eigenstate_observables.csv`：5393400条逐态记录，列为 V,phi,omega,state,quasienergy,IPR,Dn,parity。
state为0..177的零基态序号，不能跨参数点直接视为同一跟踪能带。
IPR/Dn与同一NPZ中的模态顺序一致；原始复波函数可从逐点文件无损重构。

## 波函数与断点文件

`points/V{iv}_p{ip:02d}_w{iw:02d}.npz`：iv=0..2；ip=0..99是主网格，ip=100是pi/2；iw=0..99。
其中 `sector_vectors` 为Q+扇区89×89复系数，`sector_quasienergies` 是其准能量，
`reconstruction_order` 确定全178态排列。另一扇区由精确 C U* C=U 对称关系恢复。
这是同一点完整波函数的无损对称系数存储，不是参数插值或少计算一半物理谱。
`read_frequency_point.py` 构造 shape=(178,178) 的完整本征矢矩阵，重新计算每态IPR并核对存档。
`convergence_history_json` 保存从32片开始全部相邻倍增比较；幺正残差本身不能替代时间片收敛。

Git仓库中将上述30300个原始点无损归档为三个按V分组的 `raw_points_V0.zip`、`raw_points_V1.zip`、
`raw_points_V2.zip`，通过Git LFS保存。每卷ZIP_STORED包含10100个原始NPZ，成员路径为 `points/<文件名>`。
`raw_point_archives.json` 记录各卷字节数、成员数、SHA256与CRC结果。`unpack_frequency_refined_points.py`
先核对分卷SHA256与成员索引/安全路径，再恢复本地 `points/`。本地目录及完整交付ZIP直接保留所有原始点。

## 参数、验证与图

`scan_parameters.json`：参数、精确坐标、时间积分阶数和容差、生产源码SHA256。
`scan_completion.json`：30000主点和300额外点的完成计数、最终步数范围、未收敛列表。
`frequency_refined_validation.json`：全网格/切线完整性、源码校验、逐点历史、Hermiticity、
代表点32/64/128、独立ODE、FFT、phi与2pi-phi镜像检查。配套CSV和NPZ保留实际比较数据。
`analysis_summary.json`：每种V在已计算网格上的IPR/Dn范围与极值坐标，未推断连续网格外极值。
`frequency_phase_plot_data.npz`：画图使用的真实均值数组；`frequency_phase_figures_summary.json`：统一色标和图规格。
绘图使用原始色块，没有为增加计算点数而插值，不闭合0..pi为假周期。
