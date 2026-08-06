# configs 局部指南

这里是面向用户的完整 robot config 与 motion data manifest。它们由 `humanoidverse.train` 和 Stage2 CLI 消费，继承根目录 GPU、数据和 Git 规则。

## robots

- `robots/g1_29dof.yaml`：主路线 G1 29DoF。
- `robots/piplus_bfm.yaml`：PiPlus BFM 23DoF。
- `robots/piplus_h0w.yaml`：PiPlus H0W 22DoF，显式 control joint order；正式训练前复核 XML draft metadata。
- 文件之间不继承；每个文件必须能独立解析为 RobotTrainingSpec。

## data

manifest 明确 source type、robot config、fps、hash、路径/环境变量、权重和 clip 构建策略。RobotState 来源必须通过 RobotSpec 适配；混合数据不能跨机器人或跨 joint order 直接拼接。

新增配置时同步更新 `docs/training_config_overview_zh.md`，并用已有检查工具/targeted unittest 验证路径、维度和机器人兼容性。不要提交大型数据、checkpoint 或生成目录。
