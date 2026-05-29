# 面向科学数据的自适应压缩技术研究与实现

本仓库是论文《面向科学数据的自适应压缩技术研究与实现》的工程化快照，当前版本重点保留三条主线：

- 科学数据下载、切片、抽样、压缩打标和训练数据生成脚本。
- `nnmax_quantile_ensemble_v1` 流水线选择模型及其测试报告。
- `Compressor/` 可部署压缩层，可将模型选线、压缩封装和无损解压串成命令行工具。

当前仓库不包含 `RawDataset/`、`MediumDataset/`、`Dataset/` 等大体积数据目录。模型权重 `Model/selector_nnmax_ensemble.pt` 约 140 MB，已通过 Git LFS 管理。

## 快速开始

首次克隆后建议先拉取 LFS 模型文件并安装依赖：

```powershell
git lfs install
git lfs pull

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

检查当前环境支持哪些可部署流水线：

```powershell
python -m Compressor.cli check-env
```

压缩、查看和解压文件：

```powershell
python -m Compressor.cli compress .\example.bin --selector auto
python -m Compressor.cli inspect .\example.bin.cpss
python -m Compressor.cli decompress .\example.bin.cpss
```

如果压缩的是原始科学数组，建议显式提供类型和形状，模型选线会更稳定：

```powershell
python -m Compressor.cli compress .\field.f32 --dtype float32 --shape 100,500,500 --selector auto
```

## 当前模型

随仓库提供的模型文件位于 `Model/` 根目录：

| 文件 | 作用 |
| --- | --- |
| `selector_nnmax_ensemble.pt` | nnmax 集成选择器权重，使用 Git LFS 管理 |
| `feature_names_nnmax.json` | 特征名和分箱配置 |
| `feature_norm_nnmax.npz` | 特征归一化参数和阈值 |
| `test_report_nnmax.json` | 训练与测试摘要、候选模型、预测明细 |

`Model/test_report_nnmax.json` 中的当前摘要：

| 指标 | 值 |
| --- | --- |
| 模型 | `nnmax_quantile_ensemble_v1` |
| 候选流水线标签数 | 27 |
| 训练 / 测试样本数 | 2080 / 520 |
| 原始特征维度 | 322 |
| 分箱后特征维度 | 5152 |
| Test accuracy | 0.8442 |
| Test top-2 accuracy | 0.9692 |
| Test relative CR | 0.9978 |
| Test macro F1 | 0.5757 |

## Compressor 压缩层

`Compressor/` 是当前项目的部署入口，输出格式固定为 `.cpss`。

核心特性：

- `.cpss` 文件包含 64 字节固定头、JSON 元数据、尾字节和压缩 payload。
- 固定头保存 `pipeline_id`，解压时可从文件本身恢复所需流水线。
- 对 dtype 不对齐的尾字节单独保存，保证严格无损。
- 可部署流水线固定在 `Compressor/deploy_registry.py`，当前共 27 条。
- 默认 `--selector model` 直接用模型预测；建议首次使用 `--selector auto`，模型或环境不可用时会回退到穷举模式。

选择器模式：

| 模式 | 说明 |
| --- | --- |
| `model` | 使用模型预测一条流水线，不做额外试压缩 |
| `hybrid` | 模型先排序，再试压 Top-K 条流水线 |
| `exhaustive` | 穷举所有可部署流水线，选择 `.cpss` 总大小最小的方案 |
| `auto` | 优先走 `hybrid`，失败时自动回退到 `exhaustive` |

更多 CLI 参数见 [Compressor/README.md](Compressor/README.md)。

## 数据与训练流程

如果只想下载一个不超过 1 GiB 的精选原始数据子集：

```powershell
python Script/download_benchmark_1gb_subset.py --remove-archives
```

如果已经有完整 `RawDataset/`，也可以把精选文件复制成独立目录：

```powershell
python Script/build_benchmark_subset.py --source RawDataset --dest BenchmarkDataset1GB
```

完整数据准备、压缩打标和训练流程如下：

```powershell
python Script/download_raw_datasets.py
python Script/sample_dataset_preserve_format.py --input-dir ./RawDataset --output-root ./MediumDataset
python Script/export_medium_subset_to_dataset.py --source-dir ./MediumDataset --dest-dir ./Dataset --per-folder 200
python Script/flatten_dataset_to_dataset_folder.py --dataset-root ./Dataset
python Script/compress_and_label.py --preset benchmark_curated
python Script/mian_train_and_test.py --model-dir ./Model --epochs 150 --top-k 3
```

说明：

- `download_raw_datasets.py` 需要联网，支持 `--only` 只下载指定数据源。
- `compress_and_label.py` 默认 preset 是 `compressors_plus_single_filter_x_compressor`，会跑大量组合，耗时较长；快速复现实验可先用 `--preset benchmark_curated`。
- `mian_train_and_test.py` 文件名中的 `mian` 是当前仓库保留的历史命名，运行时请按实际文件名输入。
- 训练依赖 `Dataset/train.json`、`Dataset/test.json` 以及 `Dataset/` 中的切片文件。

## 脚本总览

| 路径 | 作用 |
| --- | --- |
| `Script/download_benchmark_1gb_subset.py` | 下载精选的 <=1 GiB 原始数据子集到 `RawDataset/` |
| `Script/build_benchmark_subset.py` | 从已有 `RawDataset/` 复制精选文件到 `BenchmarkDataset1GB/` |
| `Script/download_raw_datasets.py` | 下载完整公开原始数据到 `RawDataset/` |
| `Script/sample_dataset_preserve_format.py` | 从原始数据结构保持切片，生成 `MediumDataset/` |
| `Script/export_medium_subset_to_dataset.py` | 从 `MediumDataset/` 按目录抽样生成 `Dataset/` |
| `Script/flatten_dataset_to_dataset_folder.py` | 将 `Dataset/` 扁平化为 `Dataset/dataset/` 并改写元数据 |
| `Script/compress_and_label.py` | 对切片运行候选流水线，写回最佳标签并导出 `train.json` / `test.json` |
| `Script/pipeline_registry.py` | 过滤器、压缩器、流水线 preset 和执行入口注册中心 |
| `Script/mian_train_and_test.py` | 训练 nnmax 分位数分箱神经集成模型 |

## 项目结构

```text
Compressor/       可部署压缩、解压、选线和 .cpss 封装代码
Model/            当前 nnmax 模型权重、归一化参数和测试报告
Script/           数据下载、切片、打标、训练相关脚本
requirements.txt  Python 依赖
.gitattributes    Git LFS 配置，管理 .pt 模型权重
```

生成的数据目录通常很大，不建议直接提交到 Git：

```text
RawDataset/
MediumDataset/
BenchmarkDataset1GB/
Dataset/
NewDataset/
```
