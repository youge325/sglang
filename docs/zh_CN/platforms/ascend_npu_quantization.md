# Ascend NPU 量化

Ascend 上的量化。

要加载已量化的模型，只需加载模型权重和配置即可。同样，如果模型已离线量化，启动引擎时无需添加 `--quantization` 参数。量化方法将从下载的 `quant_model_description.json` 或 `config.json` 配置中自动解析。

[ModelSlim Ascend 支持](https://github.com/sgl-project/sglang/pull/14504)：
- [x] W4A4 动态线性层
- [x] W8A8 静态线性层
- [x] W8A8 动态线性层
- [x] W4A8 动态 MOE
- [x] W8A8 动态 MOE

[AWQ Ascend 支持](https://github.com/sgl-project/sglang/pull/10158)：
- [x] W4A16 线性层
- [x] W8A16 线性层 # 需要测试
- [x] W4A16 MOE # 需要测试

Compressed-tensors (LLM Compressor) Ascend 支持：
- [x] [W4A8 动态 MOE（含/不含激活裁剪）](https://github.com/sgl-project/sglang/pull/14736) # 需要测试
- [x] [W4A16 MOE](https://github.com/sgl-project/sglang/pull/12759)
- [x] [W8A8 动态线性层](https://github.com/sgl-project/sglang/pull/14504)
- [x] [W8A8 动态 MOE](https://github.com/sgl-project/sglang/pull/14504)
