# 100×100 双驱动 Raman Floquet 全谱 IPR / Dn

保留原 Hamiltonian 和 t0=1、Delta0=.5、A=B=.8、W=theta=0、L=89、alpha=55/89、PBC、hbar=1。
V=(.5,2,4)；omega/t0=linspace(.2,10,100)，phi=linspace(0,pi,100)，均含端点。
主网格为 3×100×100=30000 个真实 Floquet 参数点，另计算 300 个 phi=pi/2 点供频率切线使用。

只输出全谱 spin-resolved IPR 和 Dn。IPR_n=sum(abs(psi_n)**4)，Dn_n=-ln(IPR_n)/ln89；
图中是对 178 个本征态分别求指标后取均值。全谱平均 Dn 不是平均 IPR 的对数。
不输出无驱动差值或外侧态指标。全部频率为无量纲，未使用 Hz 代替计算频率。

## 图和数据

- `three_V_refined_IPR_Dn_summary.png/pdf`：两指标、三种 V 的总览。
- `three_V_refined_mean_IPR.png/pdf`、`three_V_refined_mean_Dn.png/pdf`：同类图跨 V 统一色标。
- `three_V_refined_frequency_cuts.png/pdf`：phi=0、pi/2 的真实切线。
- `frequency_phase_scan.npz/csv`：30000 点主网格。
- `frequency_phase_slices.npz`、`frequency_extra_pi_over_2.csv`：两切线及300个额外点。
- `eigenstate_observables.csv`：5393400 条逐本征态记录。
- `points/`：每个实际参数点的原始模态系数、IPR、Dn、准能量与全部收敛历史。
- `frequency_refined_validation.json` 及配套 CSV/NPZ：独立数值验证。
- `research_report_zh.md`、`data_dictionary.md`：结果说明与字段定义。
- `file_manifest.json`、`SHA256SUMS`：原始交付文件校验清单。

## 复现

需要 Python 3.12、`requirements.txt` 中依赖，以及本地 C99 编译器。原始模型和生产求解源码的 SHA256
保存在 `scan_parameters.json`。生产程序从 M=32 开始逐点倍增，独立收敛后才导出完整矩阵。
原数据可断点续算，若源码或配置变化则拒绝混用。请在全新目录中计算改变后的参数。

将 `code/` 内文件复制到同一工作目录，或解压交付 ZIP 后在顶层源码目录运行：

```bash
python3 -m pip install -r requirements.txt
bash run_frequency_refined.sh
```

也可以分别执行：

```bash
python3 scan_frequency_phase_refined.py --workers 4
python3 validate_frequency_phase_refined.py
python3 plot_frequency_phase_refined.py --verify-modal-statistics
python3 write_frequency_refined_report.py
python3 package_frequency_refined.py
```

只读取结果不需要重新扫描。完整波函数重构示例：

```bash
python3 read_frequency_point.py results/frequency_phase_100x100_phi_0_pi/points/V0_p00_w00.npz --output full_modes.npz
```

`frequency_refined_figures_report_code.zip` 包含图、说明、主网格标量CSV、切线NPZ及完整运行源码；
`frequency_refined_complete_data.zip` 包含全部 CSV/NPZ、逐点原始系数及上述交付内容，使用 ZIP64。
完整 ZIP 是原始文件的副本。仓库中的大 CSV/NPZ 及原始点分卷由 Git LFS 管理，克隆后请运行 `git lfs pull`。
逐点NPZ在仓库中以三个无损分卷 `raw_points_V0.zip`、`raw_points_V1.zip`、`raw_points_V2.zip` 保存，
每卷10100个点；本地和完整交付ZIP中仍直接保留 `points/` 下全部30300个文件。
分卷SHA256与CRC检查记录在 `raw_point_archives.json`。克隆后可验证并恢复所有逐点数据：

```bash
git lfs pull
python3 unpack_frequency_refined_points.py
```

`run_frequency_refined.sh` 会在点文件缺失但分卷已存在时先调用解包工具，避免重复计算。
ZIP 内顶层源码与 results/ 路径保持可直接复现的相对布局。

本结果仅为 L=89 的有限尺寸参与指标，不据此宣称热力学局域化相变。
