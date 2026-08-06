# utils 局部指南

本目录提供 RobotSpec、motion data、配置 schema、路径解析和通用工具。公共 helper 必须保持 robot-aware，不依赖某个机器人名称或固定关节顺序。

## 子边界

- `motion_data/`：manifest、RobotState 适配、clip 构建和校验。
- `robot_spec/`：XML/URDF、joint/action/contact 语义和 robot config 解析。
- `tracking_inference.py`、`goal_inference.py`、`reward_inference.py` 与 `export/`：模型输入输出及 metadata 合同（入口在包根，不是 `utils/inference/` 子目录）。

## 约束

- 路径用 `pathlib.Path`，结构化数据使用现有 schema/dataclass。
- 维度、fps、hash、joint order 等边界错误要尽早报错并带上下文。
- 大型数据只读校验，不在测试或 import 时自动下载/生成。
- 修改 shared helper 后优先运行 motion-data、robot-spec 和 Stage2 合同测试。
