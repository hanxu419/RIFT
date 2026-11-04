# RIFT

论文 **RIFT: A Framework for Temporal Knowledge Graph Completion under Irregular Time Intervals** 的研究代码。

作者：Jiateng Xu、Dejun Wang、Bo Meng、Jiayi Han。

包含模型实现、训练与评估流程、配置文件及数据准备工具。

## 准备数据并训练

输入目录放置 `train.txt`、`valid.txt`、`test.txt`，每行四列依次为主体、关系、客体和时间。工具也支持已有的 `.del` 文件及ID映射。

```bash
python tools/prepare_dataset.py --source path/to/GDELT --output data/gdelt
python -m rift start data/config/gdelt_rift_causal.yaml --folder runs/gdelt
python -m rift test runs/gdelt --checkpoint best
```

其他配置：

- ICEWS05-15：`data/config/icews05-15_rift_causal_chrono.yaml`，数据目录为 `data/icews05-15-chrono`。
- ICEWS18：`data/config/icews18_rift_causal.yaml`，数据目录为 `data/icews18`。

Windows上运行基准训练时，增加 `--train.num_workers 0`。

数据工具保留划分和重复事件，并按时间标签排列时间ID。比较既有实验时，应沿用该实验的划分、映射、配置和查询集合。具体格式见 [DATASETS.md](DATASETS.md)。

## 主要文件

- `kge/model/rift.py`：CTE、事件编码、RCAP、门控融合和HEPE。
- `kge/model/causal_context.py`：历史事件检索。
- `kge/job/`：训练和排名评估。
- `data/config/`：训练配置。
- `tools/prepare_dataset.py`：数据准备工具。

上游代码说明见 [docs/provenance.md](docs/provenance.md)，保留的版权及许可文字见 `ThirdPartyNotices.txt`。
