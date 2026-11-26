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
.venv/bin/python scripts/audit.py --data-dir data/raw
```

审计程序需要九个官方 CSV。每次运行单独保存带时间戳的 `audit.json`、`data_audit.md` 和 `data_dictionary.md`。内存上限默认 2 GB，DuckDB 将中间表写入 `data/` 下临时目录。可用 `--memory-limit`、`--threads` 和 `--scratch-dir` 调整资源。

- [审计方法](reports/data_audit.md)和[字段与关联口径](docs/data_dictionary.md)
- [Codex Cloud 配置](docs/cloud.md)和[当前研究问题](docs/research_question.md)

## 更新看板

GitHub Actions 只部署 `site/`。审阅后可更新 `site/audit.json` 和 `site/runs.json`，格式见 [结果发布约定](docs/dashboard.md)。竞赛原始数据、逐条预测、模型文件和凭证留在授权环境内。

```bash
python3 -m http.server 8000 --directory site
```

本地预览地址为 `http://localhost:8000`。
