# tools 局部指南

这里放机器人检查、motion data 检查、manifest 构建和导出辅助 CLI。工具应验证资产并报告问题，不改变训练主流程的隐式默认。

## 约束

- 大文件检查尽量流式或只读 metadata；不要把 checkpoint、motion data、视频写入源码目录。
- CLI 新参数要覆盖解析、默认值、兼容 alias、错误信息和中文文档。
- 输出路径应由用户指定；临时结果使用 `/tmp`。
- 修改工具后用最小 fixture/unittest 覆盖错误维度、缺失路径和 robot mismatch。
