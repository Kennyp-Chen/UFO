# tests 局部指南

本项目测试使用标准库 `unittest`，不是 pytest。继承根目录验证命令和不下载/不长训约束。

## 覆盖边界

- motion data、RobotSpec、manifest 和 robot-aware precedence：优先单文件测试。
- Stage2 shared contract：`test_piplus_h0w_stage2_common.py`。
- 速度跟踪 task/metadata/reward：`test_speed_stage2.py`。
- AMP Profile A/B 和 feature：`test_amp_stage2_piplus_22dof.py`。
- TeCH kick 合同、curriculum、ball reward：`test_tech_kick_stage2.py`。

测试应锁定维度、joint order、metadata rejection、GAE termination/timeout 等边界；不要用删除断言或放宽合同来适配旧 checkpoint。需要 MuJoCo 时设置 `MUJOCO_GL=egl`，快速验证输出放 `/tmp`。
