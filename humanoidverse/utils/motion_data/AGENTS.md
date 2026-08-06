# motion_data 局部指南

这里是 RobotState、UFO PKL 和混合 motion manifest 的适配层。上级 `utils` 规则和根目录数据/Git 约束同样适用。

## 支持的来源

- `ufo_pkl`：已有裁剪动作序列。
- `robot_state_csv`、`robot_state_npz`、`robot_state_pkl`：通过 RobotSpec 做列、关节和 fps 校验。
- manifest 可组合多个 source、权重和 priority sampling；不要把推理完整序列误当训练 clip。

## 修改规则

- 新字段先更新 manifest schema/解析，再更新文档和测试。
- 机器人维度、fps、hash、glob 和 clip stride 都是显式合同；失败要指出 source/path/robot。
- auto-build 只在用户显式请求时执行，不下载、不覆盖已有大文件。
- motion data 与 checkpoint 形态绑定，跨机器人复用必须先 retarget 并重新验证。
