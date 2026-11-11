# 原 restoration prompt 的逐张量校验

以下指纹于2026-11-11从原训练绑定的 `restoration_prompt.pt` 直接CPU读取，没有重新运行文本编码器。源文件289,149字节，SHA256仍为 `253b1ebea968a41fee598952b162dac8a5900c3dc18c1a7cdc2fa51d60da440a`，与[原训练收据](../../../results/solar_refiner_tf2500_20261111/evidence/prompt.json)一致。

本次提供张量级哈希，不上传 `.pt`。这样不依赖PyTorch ZIP容器的文件名、存储编号或重新序列化细节，也可确认对方的缓存是否保留相同数值。

## 直接比对

| 字段 | 原始类型和形状 | 连续有效数据字节的 SHA256 |
|---|---|---|
| `prompt_embeds` | BF16 `[28,5120]`，143,360元素 | `2572dc002742eacd7e80bd271948bada5ef38e74edd284c2ac73416ec98fc5d0` |
| `text_token_tags` | int64 `[28]`，全部为1 | `f08c1a12407ba17e3e4e8899e0617198bdbf4b319bb8ee8d722f17dd75c64c95` |

这里的raw hash只覆盖逻辑C顺序、连续、CPU、小端的有效tensor字节：BF16每元素2字节，int64每元素8字节。不含tensor header、storage padding、stride、device或`.pt`包装。shape与dtype须另行一致；不要把BF16转为NumPy float32后计算这个raw hash。

机器可读[完整指纹](restoration_prompt.tensor_sha256.json)还包含：

- 每个tensor的 `bound_sha256`：将name/dtype/shape/字节序/顺序与raw字节共同绑定，避免同字节但解释不同。
- `prompt_embeds` 的 `float32_value_sha256`：BF16精确转FP32后的辅助数值指纹，值为 `5141d974f37287701e301fa9478acd9384d6804fc53796612a592547bbfe21e4`。它能识别“只是把原BF16无损转存为FP32”的情况，但不替代原dtype检查。
- prompt文本、layer50、`add_special_tokens=false`、`presentation=verbatim_text_only`及元数据绑定。
- 整个条件的 `bundle_sha256`：`f3df2ab0f0e4b55a4e8752bed4a5d8ccceee21f767e7604b069f2ba13930a7f8`。

所有hash的canonical JSON和NUL分隔定义均在该JSON中列出；不需要自行猜序列化格式。

## 在复现机器运行

使用[CPU校验脚本](prompt_tensor_fingerprint.py)，仅需已安装PyTorch及其NumPy依赖，不需要SolarRenderer、Diffusers、模型权重或GPU。进入本目录后运行：

```bash
python prompt_tensor_fingerprint.py /your/path/restoration_prompt.pt \
  --reference restoration_prompt.tensor_sha256.json
```

脚本以 `map_location="cpu", weights_only=True` 读取缓存，要求原六个顶层字段，拒绝非有限值，且不会修改输入。只对自己信任的缓存执行。它会输出本机完整指纹及 `comparison`：

- `exact_condition_match=true`、退出码0：两个张量的名字/shape/dtype/每个有效字节及全部conditioning元数据精确一致。
- `container_sha256_match=false` 但上一项为true：仅`.pt`容器不同，不阻止张量级确认。
- `exact_condition_match=false`、退出码1：存在实际条件或元数据差异；查看各tensor结果。若仅FP32辅助指纹一致，说明数值可一致但存储dtype不等价，仍不按原条件通过。

只想生成自己缓存的指纹时，去掉 `--reference`。如果自己的导出格式是 `[1,28,5120]`、safetensors或字段名不同，应先显式适配为原消费接口并记录转换，不能无记录squeeze/转dtype后声称原始缓存相同。

原prompt为：

> Restore this video with sharp, natural details. Preserve the original subjects, identity, colors, lighting, composition, camera motion, and timing.

它由原CPU BF16 Qwen3-VL、SDPA、layer50、28个无special-token文本token生成。CUDA/H200上重新编码即便文字和权重相同，也应实测输出，不能预设跨后端bit-exact。hash不一致只说明未精确复现，不能单凭hash判断误差幅度或认定灰猫漂移根因；maxabs/RMS/容差比较需要双方实际tensor。

## 校验脚本的实际检查

[CPU检查收据](prompt_fingerprint_cpu_checks.json)已确认：原缓存重新序列化到内存后，文件SHA改变但tensor比对通过；修改单个embedding值、单个tag或encoder layer均被拒绝；BF16无损转FP32会呈现“辅助数值相同、精确dtype不通过”。测试全部在CPU与内存中完成，无模型前向、训练或加速器操作。
