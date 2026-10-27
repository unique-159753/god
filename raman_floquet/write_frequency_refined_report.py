"""Write the 100x100 IPR/Dn report from completed, validated exact data only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
DEFAULT = BASE / "results" / "frequency_phase_100x100_phi_0_pi"
LABELS = ("Extended", "Critical", "Localized")


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|")
    return ("| " + " | ".join(map(cell, headers)) + " |\n| "
            + " | ".join(["---"] * len(headers)) + " |\n"
            + "".join("| " + " | ".join(map(cell, row)) + " |\n" for row in rows))


def extremum(values, omega, phi, maximum=False):
    flat = np.argmax(values) if maximum else np.argmin(values)
    ip, iw = np.unravel_index(flat, values.shape)
    return {"value": float(values[ip, iw]), "omega": float(omega[iw]),
            "phi": float(phi[ip]), "phi_over_pi": float(phi[ip] / np.pi),
            "index_phi": int(ip), "index_omega": int(iw)}


def location(record):
    return (f"{record['value']:.8f} @ ω={record['omega']:.7g}, "
            f"φ/π={record['phi_over_pi']:.7g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT)
    args = parser.parse_args()
    out = args.results
    completion = json.loads((out / "scan_completion.json").read_text())
    validation = json.loads((out / "frequency_refined_validation.json").read_text())
    metadata = json.loads((out / "scan_parameters.json").read_text())
    if completion["grid_points"] != 30000 or completion["nonconverged"]:
        raise RuntimeError("The report requires all 30000 converged main-grid points")
    if (not validation["data_integrity_passed"] or not validation["numerical_validation_passed"]
            or validation["failures"]):
        raise RuntimeError("The report requires a passed independent validation")
    for name, expected in metadata["code_sha256"].items():
        if hashlib.sha256((BASE / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Production source changed after scan: {name}")
    with np.load(out / "frequency_phase_scan.npz", allow_pickle=False) as data:
        omega, phi, V = data["omega"], data["phi"], data["V"]
        ipr, dn = data["all_IPR"], data["all_Dn"]
        if ipr.shape != (3, 100, 100) or dn.shape != ipr.shape:
            raise RuntimeError("Unexpected completed-grid shape")
        if not np.isfinite(ipr).all() or not np.isfinite(dn).all():
            raise RuntimeError("All scanned positive-frequency observables must be finite")
        if any(k.startswith(("outer", "delta_all", "effective")) for k in data.files):
            raise RuntimeError("Removed physical observables leaked into this delivery")
        max_unitarity = float(np.max(data["unitarity_error"]))
        numerical = {name: float(np.max(data[name])) for name in
                     ["final_operator_delta", "final_max_matched_Dn_delta", "final_mean_Dn_delta"]}
        step_values, step_counts = np.unique(data["steps"], return_counts=True)
    with np.load(out / "frequency_phase_slices.npz", allow_pickle=False) as data:
        cuts = {key: data[key] for key in ["V", "omega", "phi", "all_IPR", "all_Dn"]}
    per_V = []
    for iv, v in enumerate(V):
        entry = {"V": float(v), "static_reference_label": LABELS[iv]}
        for name, values in [("mean_IPR", ipr[iv]), ("mean_Dn", dn[iv])]:
            entry[name] = {"minimum": extremum(values, omega, phi),
                           "maximum": extremum(values, omega, phi, maximum=True)}
        per_V.append(entry)
    analysis = {"grid_points": 30000, "additional_pi_over_2_points": 300,
                "eigenstate_rows": 5393400, "shape_order": ["V", "phi", "omega"],
                "omega_range": [float(omega[0]), float(omega[-1])],
                "phi_range": [float(phi[0]), float(phi[-1])],
                "per_V": per_V, "maximum_grid_unitarity_error": max_unitarity,
                "maximum_final_time_slice_diagnostics": numerical,
                "grid_steps_distribution": {str(int(k)): int(v) for k, v in zip(step_values, step_counts)},
                "finite_size_limit": "L=89 only; participation indicators do not by themselves establish a phase transition"}
    (out / "analysis_summary.json").write_text(json.dumps(analysis, indent=2) + "\n")
    sampled = [0, 19, 39, 59, 79, 99]
    cut_rows = []
    for iv, v in enumerate(V):
        for iw in sampled:
            cut_rows.append([f"{v:g}", f"{omega[iw]:.7g}",
                             f"{cuts['all_IPR'][iv, 0, iw]:.8f}",
                             f"{cuts['all_Dn'][iv, 0, iw]:.8f}",
                             f"{cuts['all_IPR'][iv, 1, iw]:.8f}",
                             f"{cuts['all_Dn'][iv, 1, iw]:.8f}"])
    warnings = validation.get("warnings", [])
    warning_text = ("\n".join("- " + str(item) for item in warnings)
                    if warnings else "验证文件没有记录额外警告。")
    steps = completion["positive_steps_range"]
    report = "\n".join([
        "# 双 Floquet 驱动 Raman 晶格：100×100 全谱 IPR 与 Dn\n",
        "本轮重新计算三个静态基准标签 Extended（V=0.5）、Critical（V=2）和 Localized（V=4）的完整频率—相位网格。"
        "主图仅包含全谱平均 IPR 和全谱平均 Dn；所有色块均来自实际时间有序 Floquet 计算。\n",
        "## 参数与采样\n",
        "H(τ)=[1+0.8cos(ωτ)]T+[0.5+0.8cos(ωτ+φ)]S+HZ。"
        "保持 t₀=1、Δ₀=0.5、A=B=0.8、W=θ=0、L=89、α=55/89、PBC、ℏ=1。"
        "T、S、HZ 的矩阵约定与此前程序完全相同，原 Hamiltonian 未作修改。\n",
        "横轴是无量纲 ω/t₀=linspace(0.2,10,100)，纵轴是 φ=linspace(0,π,100)，两个范围均含端点。"
        "Δω=9.8/99≈0.0989899，Δφ=π/99。φ=0 与 φ=π 是不同驱动，不存在重复周期采样点。"
        "本轮不含 ω=0，因此每个点都有良好定义的周期 2π/ω，无需静态极限列。"
        "每种 V 有 10000 个真实计算点，三种 V 共 30000 点。\n",
        "为沿用同相/正交驱动的频率曲线，φ=0 直接使用主网格，φ=π/2 在同一 100 个频率上另计算。"
        "由于 100 点含端点相位网格没有 π/2，这 300 个切线点不是插值。"
        "总计 30300 个实际驱动参数点，对应 5393400 条逐本征态记录。\n",
        "## 指标\n",
        "IPRₙ=Σⱼ,σ|ψₙ(j,σ;τ=0)|⁴，Dnₙ=−ln(IPRₙ)/ln89。"
        "全谱平均均对全部 178 个本征态作算术平均，平均 Dn 先逐态取对数再平均，"
        "因此不等于 −ln(平均IPR)/ln89。波函数采用固定驱动初相 τ=0 的 Floquet 本征态；这不是周期平均 IPR。\n",
        "自旋分辨 IPR 的理论下界是 1/178，Dn 的上界为 ln178/ln89≈1.1544。"
        "Dn 大于 1 并非归一化错误；分母始终为 ln89。不同 V 的同类图片共享同一色标范围。\n",
        "## 实际网格上的取值范围与极值位置\n",
        table(["静态基准标签", "V", "平均IPR最小值及位置", "平均IPR最大值及位置"],
              [[x["static_reference_label"], f"{x['V']:g}", location(x["mean_IPR"]["minimum"]),
                location(x["mean_IPR"]["maximum"])] for x in per_V]),
        table(["静态基准标签", "V", "平均Dn最小值及位置", "平均Dn最大值及位置"],
              [[x["static_reference_label"], f"{x['V']:g}", location(x["mean_Dn"]["minimum"]),
                location(x["mean_Dn"]["maximum"])] for x in per_V]),
        "\n这些坐标是已计算网格点的位置，不能据此给出比网格间距更精细的连续极值或共振宽度。"
        "IPR 较小、Dn 较大表示在选定自旋分辨基底中参与程度较高。"
        "自旋重分布也影响该指标，不能仅由这两张图直接推断实空间输运或热力学相变。\n",
        "## 同相与正交驱动：部分实际切线数据\n",
        table(["V", "ω/t₀", "φ0 平均IPR", "φ0 平均Dn", "φπ/2 平均IPR", "φπ/2 平均Dn"], cut_rows),
        "\n完整 100 点频率曲线见 frequency_phase_slices.npz 及对应 PNG/PDF；逐点值保留在 CSV。\n",
        "## 数值方法与验证\n",
        "每个正频率点从每周期 32 片开始，以 64、128、256…倍增，"
        "使用反射对称 bond/star 精确子门的六阶时间有序分解。"
        "这是一种受收敛检验控制的单周期数值演化，不以高频有效 Hamiltonian 代替主计算。"
        "对相邻时间片结果，在固定 Q 扇区内用波函数重叠匹配本征态，联合检查传播子及逐态/全谱均值。\n",
        f"实际收敛时间片范围为 **{steps[0]}–{steps[1]}**。主网格最大归一化幺正残差为 **{max_unitarity:.3e}**。"
        f"最后相邻两档传播子差的最大值为 {numerical['final_operator_delta']:.3e}；"
        f"匹配逐态 Dn 差最大值为 {numerical['final_max_matched_Dn_delta']:.3e}；"
        f"全谱平均 Dn 的离散误差估计最大值为 {numerical['final_mean_Dn_delta']:.3e}。"
        "这些是数值时间片收敛诊断，不是相对于无驱动模型的物理差值观测量。\n",
        "独立验证包括 Hamiltonian Hermiticity、传播子 unitarity、代表点 32/64/128 片检查、"
        "最低正频率点的独立 DOP853 时间演化、FFT 算法对照，以及 φ 与 2π−φ 的准能谱和逐态 IPR 对称性。"
        "详细容差、误差、警告和实际代表点列于 frequency_refined_validation.json 及配套 CSV/NPZ。\n",
        warning_text + "\n",
        "所有主网格均达到配置的收敛标准，原始逐态指标和可无损重构波函数的系数均已保存。"
        "没有使用参数插值补点，也没有计算无驱动静态基准的 ΔIPR 或 ΔDn，未保留外侧 20% 态统计。\n",
        "## 文件与适用范围\n",
        "three_V_refined_IPR_Dn_summary.png/pdf 给出两行三列总览；"
        "three_V_refined_mean_IPR.png/pdf 与 three_V_refined_mean_Dn.png/pdf 分别为单项三列图；"
        "three_V_refined_frequency_cuts.png/pdf 给出真实 φ0、φπ/2 频率切线。"
        "PNG 至少 350 DPI，PDF 保持矢量。数据字典、复现命令、源码镜像及 SHA256 清单与结果同目录。"
        "本地 points/ 保留全部 30300 个原始点文件；GitHub 通过 Git LFS 保存三个无损的 raw_points_V0.zip、"
        "raw_points_V1.zip、raw_points_V2.zip 分卷，每卷 10100 点。克隆后使用 git lfs pull，"
        "再运行 unpack_frequency_refined_points.py 校验 SHA256 和安全路径后解包。完整交付 ZIP 直接包含全部点文件，"
        "不重复嵌套这些分卷。\n",
        "本研究固定 L=89，未新增尺寸标度或长期波包输运计算。"
        "Extended/Critical/Localized 是用户指定静态参考标签；驱动后有限尺寸平均指标的变化不能单独确立相变。\n",
    ])
    (out / "research_report_zh.md").write_text(report, encoding="utf-8")
    write_docs(out)
    print(json.dumps(analysis, indent=2), flush=True)


def write_docs(out):
    readme = """# 100×100 双驱动 Raman Floquet 全谱 IPR / Dn

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
"""
    dictionary = """# 100×100 扫描数据字典

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
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    (out / "data_dictionary.md").write_text(dictionary, encoding="utf-8")


if __name__ == "__main__":
    main()
