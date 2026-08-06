# robot_spec 局部指南

本目录定义 RobotSpec 及 XML/URDF 到训练语义的转换。它是 robot-aware 训练的单一事实来源。

## 必须保持一致

- actuator/joint order、action dim、observation 相关维度、接触集合和 control scale。
- robot config 路径、Hydra robot 名称和实际 MuJoCo XML。
- checkpoint、motion manifest、decoder/ONNX metadata 的 robot identity。

解析或校验失败时带出机器人、文件路径和期望/实际维度。禁止在公共训练流程中增加 `if robot_name == ...` 的隐式分支；新机器人必须提供完整 config、匹配 XML 和 retargeted motion data。
