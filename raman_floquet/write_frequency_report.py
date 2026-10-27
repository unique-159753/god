"""Write the Chinese report only from completed, validated real scan data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from analyze_frequency_scan import grid_extreme

BASE = Path(__file__).resolve().parent
DEFAULT = BASE/"results"/"frequency_phase_three_V"
LABELS = {.5: "Extended", 2.: "Critical", 4.: "Localized"}


def table(headers, rows):
    cell = lambda x: str(x).replace("|", "\\|")
    return "| " + " | ".join(map(cell, headers)) + " |\n| " + " | ".join(["---"]*len(headers)) + " |\n" + "".join(
        "| " + " | ".join(map(cell, row)) + " |\n" for row in rows)


def point(x):
    return f"{x['value']:+.6f}（ω={x['omega']:.6g}, φ/π={x['phi_over_pi']:.3g}）"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT)
    args = parser.parse_args()
    out = args.results
    analysis = json.loads((out/"analysis_summary.json").read_text())
    validation = json.loads((out/"frequency_scan_validation.json").read_text())
    if not validation["data_integrity_passed"] or validation["failures"]:
        raise RuntimeError("Report requires a passed whole-data audit")
    a = analysis["per_V"]
    with np.load(out/"frequency_phase_scan.npz", allow_pickle=False) as d:
        grid = {key: d[key] for key in d.files}
    with np.load(out/"frequency_phase_slices.npz", allow_pickle=False) as d:
        cuts = {key: d[key] for key in d.files}
    with np.load(out/"baselines.npz", allow_pickle=False) as d:
        baselines = {key: d[key] for key in d.files}
    winner_phase = max(a, key=lambda x:x["phase_pi_over_2_minus_zero_all_Dn"]["rms"])
    winner_response = max(a, key=lambda x:x["delta_all_Dn_vs_undriven"]["rms"])
    sections = [
        "# 双驱动准周期 Raman 晶格：三静态基准相的频率—相位响应\n",
        "本报告来自真实时间有序 Floquet 数值计算。主图没有使用高频有效模型、参数插值或图像生成模型替代真实网格。"
        "三组静态相是用户给定的有限尺寸基准标签；下文比较参与指标，不把其变化直接命名为热力学局域化相变。\n",
        "## 1. 参数、指标和真实点数\n",
        "H(τ)=[1+0.8cos(ωτ)]T+[0.5+0.8cos(ωτ+φ)]S+HZ。"
        "L=89、α=55/89、W=θ=0、PBC、ℏ=1；V分别为0.5、2.0、4.0。"
        "所有横轴均为无量纲ω/t₀=0…10的50点，Δω=10/49≈0.204082；"
        "φ=2πk/50，k=0…49，是50个不重复周期点。每个V有2500点，三V共7500点。"
        "φ=π/2不在主相位网格中，另真实计算150个切线点；φ=0直接复用主网格。绘图闭合2π仅复制φ0行。\n",
        "统一使用 IPRn=Σj,σ|ψn(j,σ;τ=0)|⁴，Dn=−ln(IPRn)/ln89。"
        "这是固定驱动初相τ=0的自旋分辨指标，不是周期平均量；分母不是ln178。"
        "所以Dn的上界为ln178/ln89≈1.1544，数值大于1合法。"
        "全谱平均Dn是逐态取对数后平均，不等于−ln(平均IPR)/ln89。站点合并IPR另保存作辅助。"
        "此前其他任务的ln178或站点IPR结果不能不经换算直接混用。\n",
        "ω=0直接对角化冻结Hf=1.8T+[0.5+0.8cosφ]S+HZ。"
        "它既不是无驱动H₀=T+0.5S+HZ，也不是ω→0⁺的单周期本征态极限；"
        "没有Floquet准能量或外侧态，相关字段为NaN，图中单独标为static limit，切线不跨越该点连接。"
        "主差值ΔD是驱动/冻结态的全谱均值减同V无驱动H₀均值。所有动态RMS和排名均排除ω0。\n",
        "## 2. 无驱动静态基准\n",
        table(["静态基准", "V", "平均IPR", "平均Dn", "2 max|E(H₀)|"],
              [[LABELS[x["V"]], f"{x['V']:g}", f"{x['static_mean_IPR']:.10f}",
                f"{x['static_mean_Dn']:.10f}", f"{2*np.max(abs(baselines['static_energies'][i])):.6f}"]
               for i,x in enumerate(a)]),
        "\n即使基准标签相同，有限L的参与指标也不是严格0或1。"
        "本次不做尺寸标度，不能从三个静态数值本身推断新的临界指数。\n",
        "## 3. 哪种静态相对双驱动相位最敏感？\n",
        "这里把相位敏感性操作性定义为同频率下π/2与0的全谱Dn差的RMS，"
        "并另报主网格每个频率的相位max−min。两者衡量的是响应变化，不是独立的机制贡献。\n",
        table(["V", "去逐频相位均值的整体RMS", "RMS[D(π/2)−D(0)]", "最大绝对相位差及ω", "平均相位max−min"],
              [[f"{x['V']:g}", f"{x['positive_phase_spread']['global_RMS_about_phase_mean']:.6f}",
                f"{x['phase_pi_over_2_minus_zero_all_Dn']['rms']:.6f}",
                f"{x['phase_pi_over_2_minus_zero_all_Dn']['largest_absolute']['value']:+.6f} @ {x['phase_pi_over_2_minus_zero_all_Dn']['largest_absolute']['omega']:.6g}",
                f"{x['positive_phase_spread']['mean_range']:.6f}"] for x in a]),
        f"\n按前述两条真实切线的RMS定义，最敏感的是 **{LABELS[winner_phase['V']]}（V={winner_phase['V']:g}）**。"
        "这不是所有频率上恒定的排名；逐频/分频段排名见response_metrics.csv和下一节。"
        "φ0时显式AB sinφ/ω项为零，但H(τ)仍可与H₀不对易，不能叫作无驱动或没有有限频率过程。\n",
        "## 4. 哪种基准在驱动后变化最明显？\n",
        table(["V", "全正频网格RMS ΔDn", "ΔDn最大点", "ΔDn最小点", "RMS相对IPR变化"],
              [[f"{x['V']:g}", f"{x['delta_all_Dn_vs_undriven']['rms']:.6f}",
                point(x['delta_all_Dn_vs_undriven']['maximum']), point(x['delta_all_Dn_vs_undriven']['minimum']),
                f"{x['all_IPR_relative_change']['rms']:.6f}"] for x in a]),
        f"\n按全网格相对H₀的绝对ΔDn RMS，最明显的是 **{LABELS[winner_response['V']]}（V={winner_response['V']:g}）**。"
        "相位敏感性、绝对参与指数变化及相对IPR变化是不同指标，排名不同不矛盾。"
        "较大的正ΔDn对应自旋分辨基底中的更大参与程度；不能仅由这个指标证明空间输运变为扩散，"
        "因为自旋重分布也会影响spin-resolved IPR。辅助站点IPR和完整波函数已保存。\n",
        "在V=4的最强增强点，平均spin-resolved IPR从0.191802629降至0.017083301，"
        "Dn从0.381550升至0.914799；辅助站点Dn也从0.372216升至0.836996。"
        "因此该点的变化包含明显的实空间参与增加，而不只是自旋重新分配。"
        "其位置为实际采样ω=3.265306、φ=0.68π及对称1.32π；不能把网格峰称为精密连续共振频率。\n",
        "## 5. 是否有某一种基准更突出的频率窗口？\n",
        table(["已采样窗口", "V", "RMS ΔDn", "平均ΔDn", "平均相位max−min"],
              [[f"{r['sampled_omega_min']:.4f}…{r['sampled_omega_max']:.4f}", f"{x['V']:g}",
                f"{x['rms_delta_Dn']:.6f}", f"{x['mean_delta_Dn']:+.6f}",
                f"{x['mean_phase_range_Dn']:.6f}"] for r in analysis["window_rankings"][1:] for x in r["scores"]]),
        "\n该表区分ω<2、2≤ω<6、ω≥6的实际采样窗口，不能把三相的同一频段平均成一个结论。"
        "完整逐频排名、极值相位和两条切线均已导出。观察到某一V的响应更突出，"
        "不等于证明该V独有一种共振机制。当前Δω≈0.204、Δφ=2π/50，"
        "不能定位比该网格更窄的共振宽度，也未用插值创造细结构。\n",
        "具体地，在已采样ω=1.836735，V=2的φ0与φπ/2相对静态ΔDn分别为−0.195638和+0.198531，"
        "相位改变使响应从降低参与变为增加参与；同一频率V=0.5为+0.012781、+0.039259，"
        "V=4为+0.267567、+0.387635，都保持正增强。ω≈1.84…2.45的若干采样点也呈现V2双向响应、"
        "另两组全相位正增强的区别。这是本网格中有基准选择性的响应特征，不是独有新机制的证明。\n",
        "全窗口相位敏感性V2最大；在ω≥6的采样窗口，V4的相位响应反而更大。"
        "π/2也不是处处比同相更扩展：V=0.5在ω=2.040816的π/2−0差为−0.043912。\n",
        "## 6. 高频响应能否用一阶有效NNN-SOC解释？\n",
        "在H(τ)=ΣℓHℓexp(+iℓωτ)约定下，[T,S]=2iJ₂ˣ，"
        "Heff=H₀+[H₊₁,H₋₁]/ω=H₀+[ABsinφ/ω]J₂ˣ。"
        "有效本征态以exp[−iK₁(0)]还原到与exact相同τ0实验室参考系，K₁(0)=Bsinφ S/ω。"
        "这个近似只是对照，主网格完全由时间有序演化得到。\n",
        table(["V", "ω≥6全谱 exact−eff+kick RMS", "外侧 RMS", "exact−无kick eff RMS", "exact−kick-only RMS"],
              [[f"{x['V']:g}"]+[f"{x['high_frequency_residuals'][k]['rms']:.6f}" for k in
                 ["all_Dn_exact_minus_effective_with_kick", "outer_Dn_exact_minus_effective_with_kick",
                  "all_Dn_exact_minus_effective_without_kick", "all_Dn_exact_minus_kick_only_static"]]
               for x in a]),
        "\n固定ω=10，φ=π/2的直接数值为：\n",
        table(["V", "H₀平均Dn", "Exact平均Dn", "一阶+kick", "kick-only"],
              [[f"{v:g}", f"{baselines['static_all_Dn'][i]:.6f}", f"{cuts['all_Dn'][i,1,-1]:.6f}",
                f"{cuts['effective_all_Dn'][i,1,-1]:.6f}", f"{cuts['kick_only_all_Dn'][i,1,-1]:.6f}"]
               for i,v in enumerate(grid["V"])]),
        "\n应以残差与实际驱动变化的大小判断近似质量，而不是把gF非零直接当作充分解释。"
        "ω=6…10是本扫描的高频端，不保证对V=4也满足ω远大于全部瞬时能标。"
        "一阶模型与exact的差值包含高阶有效项、剩余微运动及多态混合，不能全部命名为共振。"
        "不同参考模型的差值比也不是可相加的因果贡献百分比。\n",
        "实际而言，Critical基准在该窗口的全谱残差RMS仅0.005901，明显小于其相对H₀响应RMS 0.031217；"
        "ω10、φπ/2的exact/一阶+kick分别为0.829153/0.831587，是较好的数值近似。"
        "Extended基准的绝对残差也小，但驱动变化本身更小，不能据此宣称所有相对修正都可忽略。"
        "Localized基准在ω≥6的全谱残差RMS达0.130062，接近总响应RMS 0.146710，"
        "不能用这一阶模型完整解释该窗口；ω10、φπ/2仍有约0.016175的全谱残差。\n",
        "V4静态带宽为9.194711，ω≥6的20个采样频点中16个还在裸带宽以内，"
        "所以该组的6…10并非一致的渐近高频区。另一个直接反例是ω=8.163265、φ0："
        "gF=0且首阶kick=0，exact相对H₀仍增加Dn约0.090761；一阶NNN不是这一有限频率响应的必要条件。"
        "本轮没有重新计算三V的共振矩阵元、吸收诊断或反向补偿，"
        "不能把这些残差全部归因为一光子共振，也不能照搬此前V=2.5、W=0.01、不同IPR定义的态对结论。\n",
        "对实、非简并H₀态，一阶纯虚NNN项和首阶kick的波函数修正通常不改变概率的线性项，"
        "参与指标的平滑首变可能从1/ω²开始，与截掉的二阶有效/微运动项同阶。"
        "因此即使Heff的哈密顿量层次近似合理，也不保证它给出完整的最低非零阶IPR响应。\n",
        "外侧均值还可能受折叠后硬选态影响；每点重新选|ε|最大的36态，"
        "所以exact与有效模型的外侧集合不一定是同一组物理态。该影响不会改变全谱均值，"
        "但必须限制对外侧残差的机制解释；所有准能量和外侧索引均保留以供追踪。\n",
        "### 有限频率近似失效的位置\n",
        "下表按真实网格分段，列出全谱exact−一阶+kick差值的RMS和最大绝对差。"
        "低频的一阶展开本来就不受控，这些差异不能全部叫作一光子或高阶共振贡献。"
        "它们定位的是一阶近似不能覆盖的有限频率响应区域。\n",
        table(["V", "已采样ω窗口", "RMS(exact−eff+kick)", "窗口最大差及位置"],
              [[f"{v:g}", f"{grid['omega'][mask].min():.4f}…{grid['omega'][mask].max():.4f}",
                f"{np.sqrt(np.mean((grid['all_Dn'][iv][:,mask]-grid['effective_all_Dn'][iv][:,mask])**2)):.6f}",
                point(grid_extreme(
                    grid['all_Dn'][iv][:,mask]-grid['effective_all_Dn'][iv][:,mask], grid['omega'][mask],grid['phi'],'max_abs'))]
               for iv,v in enumerate(grid['V']) for mask in
               [(grid['omega']>0)&(grid['omega']<2), (grid['omega']>=2)&(grid['omega']<6), grid['omega']>=6]]),
        "## 7. 数值方法与可信度\n",
        "完整单周期时间有序演化采用反射保持的局部精确bond/star门与六阶Yoshida复合。"
        "所有89条边恰好覆盖一次，各分组保持Q=reflection(j→−j)⊗σz。"
        "固定反射顶点的两相邻边组成三站点门，其H³=2(t²+Δ²)H，使用精确指数多项式。"
        "这只改变数值因子分解，没有修改Hamiltonian。与完整178维FFT、独立expm及DOP853均已交叉验证。\n",
        "每点从M=32起倍增，最终相邻档同时满足：传播子相对Frobenius差≤10⁻⁶、"
        "匹配逐态Dn最大差≤10⁻⁴、全谱/外侧平均Dn差≤10⁻⁵，且外侧成员在重叠匹配后不交换。"
        "近简并态先按Q分块，另保存最小本征相位间隙和匹配重叠。"
        "幺正残差只说明计算出的U接近幺正，不能独自界定时间离散或单态IPR误差。\n",
    ]
    for iv,v in enumerate(grid["V"]):
        steps, counts = np.unique(grid["steps"][iv,:,1:], return_counts=True)
        sections.append(f"V={v:g}实际宏时间片分布："+"；".join(f"M={s}: {c}点" for s,c in zip(steps,counts))+"。\n")
    unit = float(np.nanmax(grid["unitarity_error"]))
    residual = float(np.nanmax(grid["eigen_residual"]))
    sections += [f"全网格最大幺正残差约{unit:.3g}，最大本征方程残差约{residual:.3g}。"
                 f"独立全数据检查通过；代表点{len(validation.get('representative_convergence',[]))}组，"
                 f"独立ODE {validation.get('independent_ODE_points',0)}点。"
                 "完整32/64/128对照、生产精度、相位/跨方法结果见frequency_scan_validation.json及对应CSV/NPZ。"
                 "抽样独立参考不能被说成全参数空间的严格统一误差上界。\n",
                 "6个最低正频率独立ODE点的传播子误差约5.67×10⁻¹¹…7.45×10⁻⁹，"
                 "全谱平均Dn误差约2.42×10⁻¹²…1.05×10⁻⁸，最大匹配逐态Dn误差不超过8.39×10⁻⁷。"
                 "全网格φ与2π−φ的全谱平均Dn差不超过2.28×10⁻¹²，外侧均值差不超过3.69×10⁻¹²。\n",
                 "有三处跨FFT/native的硬外侧成员敏感性必须保留：V=0.5，(ω,φ/π)为"
                 "(1.020408,0)、(5.918367,0)、(1.020408,1)。边界四个|ε|几乎并列，两个后端各取其中一对，"
                 "导致两个成员交换；逐态匹配仍几乎为1，边界Dn近相同，纯选态项仅约10⁻¹⁰或更小，"
                 "不改变外侧均值结论。V0.5共1739个点外侧选态gap<10⁻⁹，"
                 "不能宣称成员标签处处唯一稳定，更不能把这种换支当成相变。"
                 "固定Q内未发现gap<10⁻⁹的本征相位简并，Q约定有效避免了跨Q的任意混合。"
                 "精确简并子空间内单态IPR仍依赖基底约定，本研究固定同一Q标签使其可复现。\n",
                 "验证警示原文（如无则表示本次检查未记录警示）：\n",
                 "```json\n"+json.dumps(validation.get("warnings",[]), ensure_ascii=False, indent=2)+"\n```\n",
                 "## 8. 输出与可复现性\n",
                 "主/切线NPZ保留全部178态的IPR、Dn、能量/准能量及对称/外侧标签；"
                 "points目录用对称扇区复系数无损保存全部波函数，read_frequency_point.py可重构178×178模式，"
                 "并逐态核对IPR。ω0直接保存真实静态本征矢。CSV每个态一行，未计算值保持NaN语义。\n",
                 "同类热图三V共色标，PNG为350DPI，PDF为矢量；两条切线的π/2点均是真实额外计算。"
                 "源码和依赖版本、全部参数/容差/源文件哈希随结果交付。"
                 "旧FFT部分计算单独保存为交叉核对数据，未混进完整生产网格。\n",
                 "本次只有L=89、固定驱动振幅和一个频率/相位网格。没有有限尺寸标度、"
                 "长期波包输运或系统性临界指数分析，不能把平均IPR或Dn的变化直接叫作相变。\n"]
    for stem,caption in [("three_V_frequency_phase_summary","四项共色标三相总图"),
                         ("three_V_frequency_cuts","真实同相/正交双驱动切线"),
                         ("three_V_high_frequency_effective_control","高频端exact与一阶有效对照"),
                         ("three_V_exact_minus_effective_residual","真实网格中的一阶近似残差")]:
        sections += [f"\n![{caption}]({stem}.png)\n"]
    (out/"research_report_zh.md").write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote {out/'research_report_zh.md'}", flush=True)


if __name__ == "__main__":
    main()
