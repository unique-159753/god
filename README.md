# Floquet 数值结果与下载

## 最新计算：双驱动 Raman 晶格的一光子共振与相位干涉

- [中文科研报告：六个问题、真实数值和误差说明](raman_floquet/results/research_report_zh.md)
- [下载9张图、矢量PDF、报告与复现代码（约5.9 MB）](https://github.com/unique-159753/god/raw/refs/heads/main/raman_floquet/results/raman_floquet_figures_report_code.zip)
- [全部CSV/NPZ、逐点缓存、代表态及验证文件](raman_floquet/results)
- [计算代码及复现步骤](raman_floquet/README.md)

L=89，111频率×81相位共8991点，另补算111个π/3切线点。主结果采用站点概率ρj=|ψj↑|²+|ψj↓|²和D=−ln(Σρj²)/ln89，同时保留自旋分量定义与静态投影对照。本轮有一光子多态混合，但裸共振响应不能完整解释空间局域化异常；抵消一阶NNN项后低频增强仍保留。目前没有新相变的充分证据。

![一光子共振与局域化差值](raman_floquet/results/fig1_resonance_vs_localization.png)

主网格NPZ超过100 MB，使用Git LFS；克隆后运行`git lfs pull`获取真实大文件。其余原始数值、全部9102个逐点缓存、图片和代码均直接保存在库中。重复的495 MB全量ZIP保留于计算环境，未重复提交；可运行`raman_floquet/build_delivery.py`从仓库文件重建。

以下保留此前计算及其原有IPR定义，不能直接与本轮站点IPR混用。

## 新计算：准能量折叠、kick 修正与分支追踪

- [下载五张图及完整分析报告](https://raw.githubusercontent.com/unique-159753/god/main/results/folding_study/figures_and_report.zip)
- [下载完整代码、CSV/NPZ 数据及全部图片](https://raw.githubusercontent.com/unique-159753/god/main/results/folding_study/full_study.zip)
- [直接阅读中文分析](results/folding_study/REPORT.md)
- [计算方法、复现说明与文件索引](results/folding_study/README.md)

本次统一比较未折叠选态、已折叠选态、已折叠且经一阶 kick 修正的有效模型。低频差异没有在统一折叠后整体消失；kick 使全网格外侧 D 的 RMS 误差降低约 6.67%。频率 7–10 的相位响应对外侧态更明显；这些有限尺寸结果没有建立新的迁移边或相变。

![Exact Floquet 与修正有效模型比较](results/folding_study/fig2_corrected_Douter_comparison.png)

以下保留此前的二维热图下载。

## Exact Floquet 二维热图

一维双组分准周期 Raman 晶格的真实时间有序 Floquet 数值结果。
主图参数：L=89，alpha=55/89，41×41 网格，每周期128个中点时间片。

## 下载到手机

- [下载热图压缩包（约2.5 MB）](https://github.com/unique-159753/god/raw/refs/heads/main/results/download/heatmaps.zip)
- [打开压缩包文件页面](results/download/heatmaps.zip)
- [直接查看四张主热图 PNG](results/download/exact_summary.png)
- [打开四张主热图 PDF](results/download/exact_summary.pdf)

压缩包内有30个 PNG/PDF 文件，包括四项 exact Floquet 热图、对应有效模型和差值热图、四图汇总、外侧态对比图与五条相位截线。

如果直接下载链接没有触发保存，请进入文件页面，使用 GitHub 的原始文件下载功能。私有仓库需要先登录有访问权限的 GitHub 账号。

## 主图预览

![Exact Floquet 四项二维热图](results/download/exact_summary.png)

上排：全谱与外侧准能量态的平均 D_n，数值越大表示态分布越扩展。
下排：对应平均 IPR，数值越小表示态分布越扩展。

## 计算约定

- t0=1，Delta0=0.5，V=2.5，W=0.01，A=B=0.8，theta=0，PBC。
- 横轴 omega/t0 从6到10；纵轴 phi 从0到pi。
- Delta(tau)=Delta0+B*cos(omega*tau+phi)，因此 phi=pi/2 对应 Delta0-B*sin(omega*tau)。
- IPR 在原始自旋分辨基底计算，D_n=-ln(IPR_n)/ln(2L)。
- 每个参数点按绝对主值准能量重新选择外侧最高20%（向上取整，即36态）。
- 这里展示有限尺寸局域化诊断，未将窄结构直接认定为热力学相界。

校验和见 [SHA256SUMS.txt](results/download/SHA256SUMS.txt)。
