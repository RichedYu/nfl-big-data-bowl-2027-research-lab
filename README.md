# NFL Big Data Bowl 2027 Research Lab

信息论实践项目。研究传统 Combine 体测指标之外，10 Hz 运动轨迹能给 NFL 常规赛表现提供多少样本外信息。

[研究看板](https://richedyu.github.io/nfl-big-data-bowl-2027-research-lab/) · [Kaggle 竞赛](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027) · [审计状态](reports/experiment_status.md)

## 环境与数据审计

使用 Python 3.11 或更高版本。CI 固定使用 Python 3.12，DuckDB 版本由 `pyproject.toml` 固定。

```bash
bash scripts/setup.sh
```

在 [Kaggle 规则页](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/rules) 加入竞赛并接受规则。以下命令使用官方 Kaggle CLI 和本人的 Kaggle 凭证。也可以把从竞赛页面下载的 ZIP 解压到 `data/raw`，或在 Kaggle Notebook 中将 `--data-dir` 指向竞赛挂载目录。

```bash
.venv/bin/python -m pip install kaggle
mkdir -p data/raw
.venv/bin/kaggle competitions download nfl-big-data-bowl-2027 -p data/raw
unzip -n data/raw/nfl-big-data-bowl-2027.zip -d data/raw
.venv/bin/python scripts/audit.py --data-dir data/raw --run-id g1-2026-11-26
```

`audit.py` 会将 DuckDB、逐表画像、逐条关联和 cohort 检查点保存到 `data/audit_runs/<run-id>`。恢复时使用相同的 `--run-id`、数据和计算参数；脚本会校验数据 SHA-256、代码版本和运行配置，只跳过完全匹配的已完成阶段。输出报告按相同 Run ID 写入 `reports/runs/<run-id>`，该目录已忽略。可用 `--output` 更改报告位置。

审计程序需要九个官方 CSV。每次运行单独保存 `audit.json`、`data_audit.md` 和 `data_dictionary.md`。内存上限默认 2 GB，DuckDB 中间表保存在对应 Run ID 目录中，以支持恢复。可用 `--memory-limit`、`--threads` 和 `--scratch-dir` 调整资源。

- [审计方法](reports/data_audit.md)和[字段与关联口径](docs/data_dictionary.md)
- [Codex Cloud 配置](docs/cloud.md)和[当前研究问题](docs/research_question.md)

## 更新看板

GitHub Actions 只部署 `site/`。审阅后可更新 `site/audit.json` 和 `site/runs.json`，格式见 [结果发布约定](docs/dashboard.md)。竞赛原始数据、逐条预测、模型文件和凭证留在授权环境内。

```bash
python3 -m http.server 8000 --directory site
```

本地预览地址为 `http://localhost:8000`。
