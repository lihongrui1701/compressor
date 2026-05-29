# Compressor

`Compressor/` 是项目的可部署压缩层，负责把“流水线选择 -> 压缩封装 -> 解压恢复”串成可以直接运行的命令行工具。输出文件统一使用 `.cpss` 后缀。

直接运行模块也会转发到 CLI：

```powershell
python -m Compressor --help
python -m Compressor.cli --help
```

## 快速命令

```powershell
python -m Compressor.cli check-env
python -m Compressor.cli compress .\example.bin --selector auto
python -m Compressor.cli inspect .\example.bin.cpss
python -m Compressor.cli decompress .\example.bin.cpss
```

科学数组文件建议传入 dtype 和 shape：

```powershell
python -m Compressor.cli compress .\field.f32 --dtype float32 --shape 100,500,500 --selector auto
```

## Compress

命令格式：

```powershell
python -m Compressor.cli compress <输入文件> [-o 输出文件] [参数]
```

默认输出规则：

- 输入 `example.bin`
- 输出 `example.bin.cpss`

参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `input` | 必填 | 要压缩的原始文件 |
| `-o`, `--output` | 原文件名追加 `.cpss` | 指定输出路径 |
| `--selector` | `model` | 选线模式：`auto`、`exhaustive`、`model`、`hybrid` |
| `--top-k` | `3` | `hybrid` 模式下试压模型排序前 K 条流水线 |
| `--model-dir` | `Model/` | 覆盖默认模型目录 |
| `--dtype` | 自动推断 | 手动指定数据类型，如 `float32`、`float64`、`int32` |
| `--shape` | 自动推断 | 手动指定数组形状，如 `100,500,500` |
| `--endian` | `little` | 字节序：`little` 或 `big` |
| `--pipelines` | 全部可部署流水线 | 逗号分隔的流水线白名单 |

选择器模式：

| 模式 | 行为 |
| --- | --- |
| `model` | 直接使用 nnmax 模型预测一条流水线，不做额外试压缩 |
| `hybrid` | 先用模型排序，再试压 Top-K 条并选择实际 `.cpss` 最小者 |
| `exhaustive` | 穷举全部可部署流水线 |
| `auto` | 优先使用 `hybrid`，失败时回退到 `exhaustive` |

首次在新环境里使用时，建议选择 `--selector auto`。如果想获得确定的模型预测路径，可以显式使用 `--selector model`。

## Decompress

命令格式：

```powershell
python -m Compressor.cli decompress <输入.cpss> [-o 输出文件]
```

默认输出规则：

- 输入 `example.bin.cpss`
- 输出 `example.bin`

解压不需要再传 `dtype`、`shape` 或流水线名，因为 `.cpss` 文件已经保存了 `pipeline_id`、流水线名、上下文元数据和尾字节信息。

## Inspect

命令格式：

```powershell
python -m Compressor.cli inspect <输入.cpss>
```

`inspect` 会输出 JSON，包含固定头字段和元数据，例如：

- format version
- selector mode code
- pipeline id
- metadata size
- tail size
- payload size
- original size
- pipeline name
- dtype、shape、endian 等上下文信息

## `.cpss` 格式

`.cpss` 文件结构：

1. 固定 64 字节头部。
2. UTF-8 JSON 元数据。
3. 为保证严格无损而保留的尾字节。
4. 压缩后的 payload。

头部保存 `pipeline_id`，解压时根据 ID 找回 `Compressor/deploy_registry.py` 中对应的可部署流水线。

## 关键文件

| 文件 | 作用 |
| --- | --- |
| `cli.py` | 命令行入口 |
| `selector.py` | `model`、`hybrid`、`exhaustive`、`auto` 选线逻辑 |
| `deploy_registry.py` | 可部署流水线 ID 与名称 |
| `reversible.py` | 可逆过滤器、压缩 payload 和解压恢复 |
| `container.py` | `.cpss` 固定头与 JSON 元数据封装 |
| `runtime.py` | dtype、shape、路径和流水线上下文构建 |
| `feature_extraction.py` | nnmax 模型推理所需特征提取 |
