# 面向科学数据的自适应压缩技术研究与实现

本仓库是论文《面向科学数据的自适应压缩技术研究与实现》的精简工程快照，重点保留了数据构建、压缩打标、候选流水线注册、最新模型产物，以及部分测试与绘图脚本。



## 推荐工作流

如果你想顺着论文主线理解整个工程，建议按这个顺序看和用：

1. `python Script/download_raw_datasets.py`
2. `python Script/sample_dataset_preserve_format.py --input-dir ./RawDataset --output-root ./MediumDataset`
3. `python Script/export_medium_subset_to_dataset.py --source-dir ./MediumDataset --dest-dir ./Dataset`
4. `python Script/flatten_dataset_to_dataset_folder.py --dataset-root ./Dataset`
5. `python Script/compress_and_label.py --preset benchmark_curated`
6. `python Script/mian_train_and_test.py ...`

其中前 1 到 5 步是当前快照里最完整、最直接的“数据准备和打标链路”；第 6 步和 `Script/test/` 下的大部分脚本在当前快照中还缺少部分训练模块依赖，下面会单独说明。

## 先看这几个限制

- 当前仓库没有附带 `RawDataset/`、`MediumDataset/`、`Dataset/`、`NewDataset/` 等大体积数据目录。
- 当前仓库没有附带 `train_selector_model_science_accboost.py`、`run_ablation.py` 等旧训练/实验模块。
- 因此，`Script/mian_train_and_test.py` 和 `Script/test/` 下多数脚本目前不能在这个快照里直接运行。
- 当前仓库随附的模型文件位于 `Model/` 根目录，而不是部分脚本默认写的 `Model/nnmax/`。

## Compressor 落地层

根目录新增了 `Compressor/`，用于把训练好的流水线选择思路真正落地为可执行的压缩/解压工具。

当前实现特点：

- 输出格式固定为 `.cpss`
- 文件头固定 64 字节，包含 `pipeline_id`
- 解压时可以直接从文件头读取要使用的流水线
- 对类型不对齐的尾字节会单独保留，保证严格无损
- 默认支持 `auto / exhaustive / model / hybrid` 四种选线模式
- 在缺少 `torch` 或部分编码器依赖时，会自动回退到可运行的穷举模式

常用命令：

```powershell
python -m Compressor.cli check-env
python -m Compressor.cli compress .\example.bin
python -m Compressor.cli compress .\field.f32 --dtype float32 --shape 100,500,500
python -m Compressor.cli inspect .\example.bin.cpss
python -m Compressor.cli decompress .\example.bin.cpss
```

环境初始化：

```powershell
python Compressor/setup_env.py
# 或
powershell -ExecutionPolicy Bypass -File .\Compressor\setup_env.ps1
```

说明：

- `check-env` 会逐条探测当前环境下哪些可部署流水线可用。
- 如果希望模型选线效果更稳定，建议在压缩时尽量显式提供 `--dtype` 和 `--shape`。
- 浮点型 `delta / higher_order_delta` 在未安装 `imagecodecs` 时会被自动禁用，以保证位级可逆。

## 脚本总览

| 脚本 | 作用 | 当前快照 |
| --- | --- | --- |
| `Script/download_raw_datasets.py` | 下载公开原始数据到 `RawDataset/` | 可直接运行，需联网 |
| `Script/sample_dataset_preserve_format.py` | 结构保持切片，生成 `MediumDataset/` | 可直接运行 |
| `Script/export_medium_subset_to_dataset.py` | 从 `MediumDataset/` 抽样构造 `Dataset/` | 可直接运行 |
| `Script/flatten_dataset_to_dataset_folder.py` | 将 `Dataset/` 扁平化到 `Dataset/dataset/` | 可直接运行 |
| `Script/pipeline_registry.py` | 压缩器/过滤器注册中心和执行入口 | 模块，不是命令行脚本 |
| `Script/compress_and_label.py` | 对样本跑候选流水线并生成 `train.json`/`test.json` | 可直接运行，但要求已有 `Dataset/` |
| `Script/mian_train_and_test.py` | 训练 nnmax 集成模型 | 可查看 `--help`，实际训练仍缺依赖 |
| `Script/test/common_accboost_test.py` | 测试脚本公用函数 | 模块，不是命令行脚本 |
| `Script/test/plot_nnmax_pipeline_bar.py` | 绘制 nnmax 与 27 条流水线对比柱状图 | 可直接运行，优先读取 `Model/test_report_nnmax.json` |
| `Script/test/test_file_type_bar.py` | 按文件类型抽样比较模型与固定流水线 | 当前快照缺依赖，不能直接运行 |
| `Script/test/test_newdataset_generalization_bar.py` | 新数据集泛化柱状图 | 当前快照缺依赖，不能直接运行 |
| `Script/test/test_rawdataset_trainset_bar.py` | 按原始来源目录抽样比较 | 当前快照缺依赖，不能直接运行 |
| `Script/test/test_hyperparameter_line.py` | 小规模超参数 sweep 和折线图 | 当前快照缺依赖，不能直接运行 |
| `Script/test/test.py` | 批量调度上面几类测试 | 可查看 `--help`，实际运行仍依赖旧测试链路 |

## 逐脚本说明

### `Script/download_raw_datasets.py`

功能：

- 从公开 URL 下载论文中使用的一批示例科学数据。
- 自动把归档文件放到 `RawDataset/_archives/`，并把可解压的内容展开到 `RawDataset/` 对应目录。

输入输出：

- 输入：网络数据源。
- 输出：`RawDataset/`、`RawDataset/_archives/`。

当前快照：

- 可直接运行，但必须联网。

常用命令：

```powershell
python Script/download_raw_datasets.py
```

只下载指定数据源：

```powershell
python Script/download_raw_datasets.py --only oisst_v2_1 era5_pressure_level viirs_surface_albedo
```

`--only` 可选值：

- `cesm_atm_dataset1`
- `cesm_atm_dataset2`
- `era5_pressure_level`
- `exafel_dataset2`
- `hurricane_isabel`
- `igsr_fastq`
- `landsat_qa_pixel`
- `miranda_small`
- `nyx_512`
- `oisst_v2_1`
- `rcsb_mmcif`
- `refseq_ecoli`
- `refseq_human`
- `s3d`
- `viirs_i1_sdr_swath`
- `viirs_surface_albedo`

### `Script/sample_dataset_preserve_format.py`

功能：

- 从 `RawDataset/` 中读取原始文件。
- 按文件类型做“结构保持”切片，生成中等大小样本。
- 输出到 `MediumDataset/`，并生成 `MediumDataset/data.json`。

输入输出：

- 输入：`RawDataset/`
- 输出：`MediumDataset/` 和 `MediumDataset/data.json`


常用命令：

```powershell
python Script/sample_dataset_preserve_format.py --input-dir ./RawDataset --output-root ./MediumDataset
```


常用参数：

- `--min-kb` / `--max-kb`：目标切片大小范围，默认约 `10000` 到 `50000` KB。
- `--include-archives`：把归档目录中的文件也纳入处理。
- `--shuffle`：打乱切片顺序后再编号。
- `--max-files`：限制读取的原始文件数。
- `--max-pieces-per-file`：限制每个文件最多切出多少块。

### `Script/export_medium_subset_to_dataset.py`

功能：

- 从 `MediumDataset/data.json` 中按顶级目录抽样。
- 将抽中的样本复制到 `Dataset/`。
- 生成新的 `Dataset/data.json`。

输入输出：

- 输入：`MediumDataset/` 和 `MediumDataset/data.json`
- 输出：`Dataset/` 和 `Dataset/data.json`


常用命令：

```powershell
python Script/export_medium_subset_to_dataset.py --source-dir ./MediumDataset --dest-dir ./Dataset --per-folder 200
```


### `Script/flatten_dataset_to_dataset_folder.py`

功能：

- 把 `Dataset/` 下各顶级目录中的样本重新打乱。
- 统一移动到 `Dataset/dataset/`。
- 按 `000001.xxx` 这种编号方式重命名，并改写 `Dataset/data.json`。

输入输出：

- 输入：`Dataset/`
- 输出：更新后的 `Dataset/dataset/` 和 `Dataset/data.json`

常用命令：

```powershell
python Script/flatten_dataset_to_dataset_folder.py --dataset-root ./Dataset
```


### `Script/pipeline_registry.py`

功能：

- 注册过滤器、压缩器和候选流水线。
- 提供 `execute_pipeline`、`resolve_pipeline_names`、`check_component_availability` 等统一接口。
- 做 `dtype` / `shape` 推断，并支持多种 preset。



### `Script/compress_and_label.py`

功能：

- 读取 `Dataset/data.json`。
- 对每个样本运行候选流水线。
- 记录压缩结果、最佳流水线标签、元信息。
- 最后生成：
  - 更新后的 `Dataset/data.json`
  - `Dataset/train.json`
  - `Dataset/test.json`

输入输出：

- 输入：`Dataset/data.json` 和 `Dataset/` 中的实际样本文件
- 输出：更新后的 `Dataset/data.json`、`Dataset/train.json`、`Dataset/test.json`

当前快照：

- 可直接运行，但前提是 `Dataset/` 已经准备好。


### `Script/mian_train_and_test.py`

功能：

- 训练论文里的 nnmax 分位数分箱集成模型。
- 从 `Dataset/train.json` 和 `Dataset/test.json` 读取样本。
- 输出模型权重、特征归一化参数、测试报告。

输入输出：

- 输入：`Dataset/`、旧训练模块、ablation 结果摘要
- 输出：模型目录中的 `selector_nnmax_ensemble.pt`、`feature_norm_nnmax.npz`、`feature_names_nnmax.json`、`test_report_nnmax.json`

当前快照：

- 不能直接运行。
- 直接执行会报错：
  - `ModuleNotFoundError: No module named 'train_selector_model_science_accboost'`
  - 同时还依赖缺失的 `run_ablation.py`

完整仓库中的典型用法：

```powershell
python Script/mian_train_and_test.py --model-dir ./Model/nnmax --epochs 150 --top-k 3
```

如果你后面把缺失模块补回来了，建议同时检查：

- 默认 `--model-dir` 指向 `Model/nnmax/`
- 当前快照随附的现成模型文件实际在 `Model/` 根目录


