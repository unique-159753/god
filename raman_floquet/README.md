# 双驱动准周期 Raman 晶格计算

最新计算为三组 V=0.5/2/4 的 **100×100 全谱 IPR、Dn** 扫描：ω/t₀=0.2…10、φ=0…π，两端均包含，L=89、W=0。见[新结果与原始数据](results/frequency_phase_100x100_phi_0_pi)、[中文报告](results/frequency_phase_100x100_phi_0_pi/research_report_zh.md)和[复现说明](results/frequency_phase_100x100_phi_0_pi/README.md)。运行 `bash run_frequency_refined.sh` 可复现。此次只计算全谱参与指标，不选外侧准能量态，不计算静态差值。IPR 为自旋分辨定义，Dn 分母为 ln89。生产后端要求 C99 编译器，采用自适应六阶时间有序数值演化。

此前的无量纲 ω=0…10、50×50 扫描仍保留于[原报告](results/frequency_phase_three_V/research_report_zh.md)与[原复现说明](results/frequency_phase_three_V/README.md)，对应 `bash run_frequency_three_V.sh`。

以下保留更早任务的参数与定义，不能与最新结果混用。

计算与报告为本次最新 Raman 模型任务，L=89、矩阵维数178、PBC。与此前 SVQL 双链任务无关。所有科研结果见 [results/research_report_zh.md](results/research_report_zh.md)，数据字典见 [results/data_dictionary.md](results/data_dictionary.md)。

本目录已保存到`unique-159753/god`。使用`git lfs install`和`git lfs pull`获取超过100 MB的`results/localization_grid.npz`；其余科学文件直接保存在Git中。完整重复ZIP不纳入版本管理，可用`python build_delivery.py`重建。环境安装脚本和计算入口均相对其自身目录运行，支持任意检出路径。

## 已完成范围

- 111频率×81相位的8991点 exact/effective 网格，额外111个 φ=π/3 exact点；全部178模态观测量。
- 三种展宽的一光子响应及完整静态本征基底。
- 代表态对、空间密度、Floquet投影、四模型12参数组合、kick-only控制。
- Sambe光子截断1/2/3、重复副本及光子规范检查。
- 18个单驱动参数组合；A-only无φ依赖时传播子按物理等价复用。
- 全数据完整性审查，9点32/64/128与独立ODE对照，敏感点256/512。
- 5张主图及补图，300或350DPI PNG与矢量PDF。

主定义：ρj=|ψj↑|²+|ψj↓|²，IPR=Σρj²，D=−lnIPR/ln89。另保存178分量IPR与D。先折叠到[−ω/2,ω/2)，再选|ε|最大的36态；静态响应掩码取未折叠|E|最大的36态。有效态在τ=0用exp[−iK₁(0)]修正。原始任务及用户确认规则在results/task_specification_zh.md。

## 环境

实际使用Python3.12.14，版本固定在requirements.txt。CPU计算，不需要GPU、凭据、在线服务或运行中的server。C99编译器可加速bond rotations；源码嵌入model.py，首次使用自动在临时目录编译；没有编译器时自动退回SciPy FFT路径。该替代路径时间离散的误差系数不同，已单独交叉验证。

推荐从完整压缩包解压到任意可写目录，保持Python文件和results目录为同级。进入该目录：

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg
.venv/bin/python validate_model.py
```

当前云实例已经执行真正的全流程；无需仅为看图重新扫描。`results/code/`是交付代码镜像。若只下载该镜像，请把其中脚本放在实际results目录旁，或使用各脚本的`--results`选项；不要误将默认输出位置设为已有结果的嵌套子目录。

## 复现顺序

`bash run_all.sh`依次运行下面流程（会复用已有合法缓存）。完整扫描依硬件需数分钟至更久，通常不要打开BLAS多线程再叠加大量workers。

```bash
python validate_model.py
python resonance.py
python representatives.py
python kick_baseline.py
python sambe.py
python sambe_diagnostics.py
python sambe_figure.py
python scan.py --steps 64 --workers 4
python figures12.py
python validate_grid.py
python single_drive_controls.py
```

运行前请检查对应脚本`--help`或源文件；`representatives.py`和`single_drive_controls.py`的代表点默认128步。`scan.py`源文件和参数哈希会校验缓存，主网格统一64步。需要改变M或模型时，指定新的输出目录或先备份旧results，不能将不同精度混入旧缓存。验证脚本只读生产数据，另外保存高精度参考，不静默替换主网格。

时间推进为四阶方法；Floquet通过复Schur分解，不能以某个实部对角化快捷方法替代。Fourier约定exp(+iℓωτ)，吸收矩阵元H₋₁；Heff首阶为+[H₊₁,H₋₁]/ω=+ABsinφ J₂ˣ/ω。单个M64网格并非无误差的数学exact：最敏感抽样点(6.2,π/40)对ODE的D误差约4.77e−4，M512降至1.10e−7；最小隙处还存在约1.2e−5的基底敏感性。报告给出实际范围而非全网格统一精度保证。

## 交付

完整包保留所有结果、逐点缓存、代表态波函数、验证数据和源代码；轻量包保留图/PDF、报告、代码及文档。`results/SHA256SUMS`和`results/file_manifest.json`用于完整性核对，`results/npz_inventory.csv`列出每个非缓存NPZ的字段、shape和dtype。`.venv`和临时编译产物不进入科研归档。

没有做尺寸标度、长期输运或完整振幅扫描；当前结果不证明新相变，也不把差值比称为机制贡献百分比。
