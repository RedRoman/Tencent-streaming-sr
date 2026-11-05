# v5 motion：预检证据快照

正式流程状态：`running`（由主流程在打包时明确指定）。仅发布8步实际NPU预检和12项CPU测试证据；尚无正式两臂完成/完整dev gate/test结论。

代码、公式与局限见[实验说明](../../experiments/h3_v5_motion_20261106/README.md)。`evidence/`为路径脱敏副本，原始bytes/SHA与公开bytes/SHA见[来源映射](../../experiments/h3_v5_motion_20261106/source_fingerprints.json)。嵌入JSON的原SHA不回填；启动包装器失败证据保留。不含权重、缓存或视频。
