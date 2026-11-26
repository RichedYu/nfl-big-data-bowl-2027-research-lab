# 看板结果发布

看板由 GitHub Actions 从 `site/` 发布，使用静态 HTML、CSS、JavaScript 和两份 JSON。

## 审计摘要

完整审计结束后，可以从运行报告生成待审阅的摘要。命令输出文件，审阅通过后更新 `site/audit.json`。

```bash
.venv/bin/python scripts/dashboard_summary.py reports/runs/<run>/audit.json > /tmp/nfl-audit-summary.json
```

摘要含审计时间和各文件的行数、唯一球员数、字段名及缺失率。`games.csv` 等没有球员 ID 的文件使用 `null`。

## 模型结果

`site/runs.json` 顶层是 `{"runs": []}`。每条记录使用以下字段。

| 字段 | 内容 |
|---|---|
| `id` | 实验记录中的唯一标识 |
| `model` | 模型标签 |
| `comparison` | 可比较的目标、样本与验证方案标识 |
| `split` | 验证描述 |
| `n_players` | 验证样本中的唯一球员数 |
| `metrics` | 带口径和单位的指标名及有限数值，例如 `Log loss (nats)` |

看板按指标名和 `comparison` 筛选。相同对比标识应使用相同目标、样本和划分。表格显示精确值，非负指标柱的右端对应当前对比中的最大值。实验记录应同时保留 Git 提交、数据版本、随机种子和完整评估方案。
