# 获取和预处理权重

本文说明怎样把固定来源的部署 checkpoint 变成可校验的训练 bundle。

训练基座由仓库、不可变 revision 和三个分片的 SHA256 共同绑定。预处理只重排与校验原字节，不重新量化流式权重，见[格式设计](../design/format.md)。

## 哪些来源已经验证

| 模型 | 来源与不可变 revision |
|---|---|
| Qwen3.6-35B-A3B | `nvidia/Qwen3.6-35B-A3B-NVFP4@1355db6a052410cfd62085d94b58866fd0f2c3c5` |
| Qwen3.8-27B | `RadixArk/Qwen3.8-27B-NVFP4@319f741cce68d7914884900c138a1fbb70a42f30` |
| Qwen3.6-27B | `nvidia/Qwen3.6-27B-NVFP4@0893e1606ff3d5f97a441f405d5fc541a6bdf404` |

模型主页链接集中在 [README](../../README.md#支持哪些模型)。分片的可信 SHA256 从该 revision 的 Hugging Face LFS 元数据获取，实际下载字节必须与之相同；完整绑定写入收据和 bundle 的 `manifest.source`。

## 怎样下载并校验

以下示例下载 Qwen3.8 的元数据和全部权重，不能用浮动分支名代替 revision。

```powershell
& '<venv>\Scripts\python.exe' tools/fetch_checkpoint.py --repo RadixArk/Qwen3.8-27B-NVFP4 --revision 319f741cce68d7914884900c138a1fbb70a42f30 --model-dir '<source>' --output '<fetch>' --download-weights
```

完整下载的 `<fetch>/receipt.json` 应为完成状态，并列出三个分片、元数据、字节数与实际 SHA256。仅下载元数据或单个分片的收据不能用于正式预处理。

## 怎样生成 bundle

统一入口从来源 `config.json` 识别模型族；可选的 `--family` 只检查预期身份，不覆盖来源声明。

```powershell
& '<venv>\Scripts\python.exe' tools/preprocess_checkpoint.py --model-dir '<source>' --source-receipt '<fetch>\receipt.json' --output '<bundle>'
```

正常输出包含族名、几何和内存峰值，最终 manifest 为完成状态。输出目录必须是新目录；运行中的 manifest 不能当作完整 bundle 使用。实际耗时、峰值内存受磁盘和缓存影响，本版本未重新测量完整真实预处理。

## 怎样检查重排后的权重

MoE 使用原始来源与 bundle 逐字段比对。

```powershell
& '<venv>\Scripts\python.exe' tools/validation/moe/check_preprocessed.py --source '<source>' --processed '<bundle>' --output '<check>'
```

dense 使用对应的完整原字节检查器。

```powershell
& '<venv>\Scripts\python.exe' tools/validation/dense/check_dense_preprocessed.py --source '<source>' --bundle '<bundle>' --output '<check>'
```

检查应正常退出，并报告原字节一致。真实 checkpoint 不能改成 tiny 来源跳过几何约束。目录布局和张量形状见[格式设计](../design/format.md)。

## 常见失败先查什么

| 现象 | 先检查 |
|---|---|
| 来源 SHA256 不符 | revision、LFS 元数据、下载是否完整；禁止替换收据中的哈希 |
| 来源身份不完整 | 是否下载全部分片和必要模板、分词器元数据 |
| 几何或布局未知 | 是否为已支持结构；不能修改模型名称强行通过 |
| 输出目录已存在 | 使用新目录，保留失败的记录和 manifest |
| 磁盘不足或锁页失败 | 同时核算原始分片、bundle、训练输出和物理锁页分配 |
