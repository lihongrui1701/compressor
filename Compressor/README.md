# Compressor

`Compressor/` 是项目的落地压缩层，负责把“流水线选择 -> 压缩封装 -> 解压恢复”串成可以直接运行的工具。

最常用的两个入口在 [cli.py](<c:/Users/Lenovo/Desktop/project/Compressor/cli.py:1>)：

- 压缩入口：`python -m Compressor.cli compress ...`
- 解压入口：`python -m Compressor.cli decompress ...`

如果你直接运行：

```powershell
python -m Compressor
```

它会转发到 `Compressor.cli:main`。

## 快速开始

压缩一个文件：

```powershell
python -m Compressor.cli compress .\example.bin
```

解压一个 `.cpss` 文件：

```powershell
python -m Compressor.cli decompress .\example.bin.cpss
```

查看封装信息：

```powershell
python -m Compressor.cli inspect .\example.bin.cpss
```

检查当前环境支持哪些流水线：

```powershell
python -m Compressor.cli check-env
```

## 压缩入口

命令格式：

```powershell
python -m Compressor.cli compress <输入文件> [-o 输出文件] [其他参数]
```

压缩命令对应代码入口：

- 命令行入口函数：`command_compress`
- 核心文件：[cli.py](<c:/Users/Lenovo/Desktop/project/Compressor/cli.py:1>)

默认输出规则：

- 输入：`example.bin`
- 输出：`example.bin.cpss`

### 参数说明

| 参数 | 是否必填 | 默认值 | 功能 |
| --- | --- | --- | --- |
| `input` | 是 | 无 | 要压缩的原始文件路径。 |
| `-o`, `--output` | 否 | 原文件名后追加 `.cpss` | 指定压缩输出文件路径。 |
| `--selector` | 否 | `model` | 选择流水线的模式。可选值：`auto`、`exhaustive`、`model`、`hybrid`。默认直接使用模型预测，不做额外试压缩。 |
| `--top-k` | 否 | `3` | 当选择器使用 `hybrid` 时，只实际试压模型排序最靠前的 `K` 条流水线。 |
| `--model-dir` | 否 | `Model/` | 指定模型目录，覆盖默认的 nnmax 模型位置。 |
| `--dtype` | 否 | 自动推断 | 手工指定数据类型，例如 `float32`、`float64`、`int32`。适合科学数组文件。 |
| `--shape` | 否 | 自动推断 | 手工指定数据形状，格式如 `100,500,500`。 |
| `--endian` | 否 | `little` | 指定字节序。可选值：`little`、`big`。 |
| `--pipelines` | 否 | 使用全部可部署流水线 | 用逗号分隔的流水线白名单，只在这些流水线里做选择。 |

### `--selector` 的四种模式

如果不显式传 `--selector`，默认使用 `model`。这意味着普通压缩命令默认不会对多条流水线做试压缩比较；如果你希望通过试压来选更优结果，请显式指定 `hybrid` 或 `exhaustive`。

| 模式 | 功能 |
| --- | --- |
| `auto` | 优先尝试模型辅助的 `hybrid`，如果模型或环境不可用，会自动回退。 |
| `exhaustive` | 穷举所有可部署流水线，选择最终 `.cpss` 最小的方案。 |
| `model` | 直接使用模型预测一条流水线，不做额外试压比较。 |
| `hybrid` | 先由模型排序，再只试压 Top-K 条流水线。 |

## 解压入口

命令格式：

```powershell
python -m Compressor.cli decompress <输入.cpss> [-o 输出文件]
```

解压命令对应代码入口：

- 命令行入口函数：`command_decompress`
- 核心文件：[cli.py](<c:/Users/Lenovo/Desktop/project/Compressor/cli.py:1>)

默认输出规则：

- 输入：`example.bin.cpss`
- 输出：`example.bin`

### 参数说明

| 参数 | 是否必填 | 默认值 | 功能 |
| --- | --- | --- | --- |
| `input` | 是 | 无 | 要解压的 `.cpss` 文件路径。 |
| `-o`, `--output` | 否 | 去掉 `.cpss` 后缀后的路径 | 指定解压后的输出文件路径。 |

### 解压时会自动完成的事

解压不需要你手动再传 `dtype`、`shape` 或流水线名，因为 `.cpss` 文件内部已经保存了：

- `pipeline_id`
- `pipeline_name`
- 上下文元数据
- 尾字节信息

## Inspect 入口

命令格式：

```powershell
python -m Compressor.cli inspect <输入.cpss>
```

对应代码入口：

- 命令行入口函数：`command_inspect`

### 参数说明

| 参数 | 是否必填 | 默认值 | 功能 |
| --- | --- | --- | --- |
| `input` | 是 | 无 | 要查看的 `.cpss` 文件路径。 |

### 输出内容

`inspect` 会输出 JSON，包含：

- 固定头字段
- `pipeline_id`
- `pipeline_name`
- 原始大小
- payload 大小
- 元数据内容

## `.cpss` 文件格式

`.cpss` 文件结构如下：

1. 固定 64 字节头部
2. UTF-8 JSON 元数据
3. 为保证严格无损而保留的尾字节
4. 压缩后的 payload

头部里保存了 `pipeline_id`，因此解压时可以只依赖文件本身恢复所需流水线。

## 相关文件

- [cli.py](<c:/Users/Lenovo/Desktop/project/Compressor/cli.py:1>)：命令行入口
- [selector.py](<c:/Users/Lenovo/Desktop/project/Compressor/selector.py:1>)：流水线选择
- [reversible.py](<c:/Users/Lenovo/Desktop/project/Compressor/reversible.py:1>)：可逆压缩和解压核心
- [container.py](<c:/Users/Lenovo/Desktop/project/Compressor/container.py:1>)：`.cpss` 头部和元数据封装

