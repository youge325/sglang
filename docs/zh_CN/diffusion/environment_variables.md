# 缓存加速

这些变量用于配置 Diffusion Transformer (DiT) 模型的缓存加速。
SGLang 支持多种缓存策略 - 请参阅[缓存文档](performance/cache/index.md)了解概述。

### Cache-DiT 配置

详细配置请参阅 [Cache-DiT 文档](performance/cache/cache_dit.md)。

| 环境变量                | 默认值 | 描述                              |
|-------------------------------------|---------|------------------------------------------|
| `SGLANG_CACHE_DIT_ENABLED`          | false   | 启用 Cache-DiT 加速            |
| `SGLANG_CACHE_DIT_FN`               | 1       | 始终计算的前 N 个块         |
| `SGLANG_CACHE_DIT_BN`               | 0       | 始终计算的后 N 个块          |
| `SGLANG_CACHE_DIT_WARMUP`           | 4       | 缓存前的预热步数              |
| `SGLANG_CACHE_DIT_RDT`              | 0.24    | 残差差异阈值            |
| `SGLANG_CACHE_DIT_MC`               | 3       | 最大连续缓存步数              |
| `SGLANG_CACHE_DIT_TAYLORSEER`       | false   | 启用 TaylorSeer 校准器             |
| `SGLANG_CACHE_DIT_TS_ORDER`         | 1       | TaylorSeer 阶数（1 或 2）                |
| `SGLANG_CACHE_DIT_SCM_PRESET`       | none    | SCM 预设（none/slow/medium/fast/ultra） |
| `SGLANG_CACHE_DIT_SCM_POLICY`       | dynamic | SCM 缓存策略                       |
| `SGLANG_CACHE_DIT_SCM_COMPUTE_BINS` | 未设置 | 自定义 SCM 计算分箱                  |
| `SGLANG_CACHE_DIT_SCM_CACHE_BINS`   | 未设置 | 自定义 SCM 缓存分箱                    |

## 云存储

这些变量用于配置 S3 兼容的云存储，以自动上传生成的图像和视频。

| 环境变量            | 默认值 | 描述                                            |
|---------------------------------|---------|--------------------------------------------------------|
| `SGLANG_CLOUD_STORAGE_TYPE`     | 未设置 | 设置为 `s3` 以启用云存储                    |
| `SGLANG_S3_BUCKET_NAME`         | 未设置 | S3 存储桶名称                              |
| `SGLANG_S3_ENDPOINT_URL`        | 未设置 | 自定义端点 URL（用于 MinIO、OSS 等）             |
| `SGLANG_S3_REGION_NAME`         | us-east-1 | AWS 区域名称                                      |
| `SGLANG_S3_ACCESS_KEY_ID`       | 未设置 | AWS Access Key ID                                      |
| `SGLANG_S3_SECRET_ACCESS_KEY`   | 未设置 | AWS Secret Access Key                                  |
