# 日前电价预测代码基线

版本：`0.1.0`。本仓库当前保存已有电价预测代码，供后续研究使用。

本版本保留原始 PefCodeBench 的点预测、分位数回归、Normal、JohnsonSU、StudentT 五种 DNN，以及 QRA、CP、CQR 和在线共形校准。已经弃用的 VICQR、DMACC、派生版本和相关论文实验工具已从活动代码中移除。

## 目录

- `tools/`：数据处理、模型、共形校准、预测区间和统计评估。
- `run_recalibration.py`：滚动训练与日前预测。
- `exec_qra_cp.py`：集合预测与 QRA、CP、CQR、在线校准后处理。
- `results_analysis.py`：覆盖率检验、评分、图表和结果表。
- `data/datasets/`：德国和意大利七个市场的原始基线数据。
- `experiments/tasks/`：原始配置、已调参数及保存的基线预测结果。
- `tests/`：基础方法回归测试和五种 DNN 的小规模训练测试。

## 环境与验证

本版本在 Python 3.8.10、TensorFlow 2.13.0、TensorFlow Probability 0.21.0、NumPy 1.23.5、pandas 1.5.3 下验证。

```powershell
conda create -n price_baseline python=3.8.10
conda activate price_baseline
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

请从仓库根目录运行上述命令。测试包含真实模型的单轮训练，通常需等待一段时间。

## 原有实验流程

1. 在 `run_recalibration.py` 的 `main()` 中设置市场、模型和运行编号，然后运行 `python run_recalibration.py`。
2. 在 `exec_qra_cp.py` 的 `main()` 中设置对应运行编号，然后运行 `python exec_qra_cp.py`。
3. 在 `results_analysis.py` 的 `main()` 中选择市场和运行编号，然后运行 `python results_analysis.py`。

重新训练前请复制实验运行目录并使用新运行编号，以保留原始预测结果。三个入口在直接运行时执行实验；导入它们不会触发训练或改写结果。

本次验证覆盖基本数值方法、配置与数据加载、集合后处理，以及五种 DNN 的单轮训练和预测。未重新执行七个市场的完整滚动训练；仓库中的基线预测结果为原始保存结果。

`.worktrees/` 是本地历史分支的独立工作目录，已排除在当前版本的提交范围之外。

## 来源与许可

保留原始作者署名及 `LICENSE`。基础代码用于复现 Brusaferri、Ballarino、Grossi 和 Laurini 的论文：[On-line conformalized neural networks ensembles for probabilistic forecasting of day-ahead electricity prices](https://arxiv.org/abs/2404.02722)。部分统计与在线校准工具的来源和许可见各文件头部。
