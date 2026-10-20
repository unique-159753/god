# 云计算环境记录

本工作流按 [cloud-environment-onboarding:setup](skill://plugin_connector_1p_ed5feb9070a08191b08c81c47947bc16/setup/SKILL.md) 核对并准备。`/workspace/god`为现有空Git检出，tracked/untracked状态在本轮设置前后均无变化；科研文件在其外的`/workspace/raman_floquet`。没有创建新worktree。

已实际运行`bash /workspace/raman_floquet/install_environment.sh`，以Python3.12.14创建使用系统包的本地venv，校验固定版本依赖，并执行`validate_model.py`。独立ODE、native C/FFT、Hermiticity、Fourier/kick、时间精度及对称性验证通过。无需GPU、凭据或驻留server。

环境配置草稿已保存`install_script`和`start_skill`，工具返回status=saved、requires_publish=true。它保存复用说明，不会自动执行配置或发布快照。若需要把当前准备好的环境用于后续新任务，请在环境设置中审阅保存并发布。尚未在新任务验证快照恢复；没有据此宣称已完成发布或跨任务复原。

本次科研结果已经在当前机器完成；查看报告和下载图/数据无需重新发布或重算。

## 保存至Git仓库

用户随后要求将结果保存在god库。本轮结果另复制到`god/raman_floquet/`，保留原始科学数据字节和两份模型源文件的SHA256；此前仓库结果均保留。该目录的安装脚本已改为相对脚本目录运行，打包脚本不再依赖会话附件的绝对路径；已经保存的任务原文仍在`task_specification_zh.md`中。上面的`/workspace/raman_floquet`路径记录的是原始计算环境，不是新检出位置的要求。
