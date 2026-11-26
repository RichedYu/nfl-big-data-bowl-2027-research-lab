# Codex Cloud

创建仅自己可使用的可复用环境，并选中 `RichedYu/nfl-big-data-bowl-2027-research-lab`。从仓库根目录执行安装命令。

```bash
bash scripts/setup.sh
```

这个安装流程会用合成输入跑九表测试。安装脚本只准备代码和依赖。正式审计可使用 Kaggle Notebook 挂载竞赛数据，或在独立 Codex Cloud 任务里通过官方 CLI 下载。

Cloud 下载使用个人保险库中的 `KAGGLE_API_TOKEN`，并把访问范围限定在该环境。受限出口至少需要允许 `api.kaggle.com`（竞赛 API）和 `storage.googleapis.com`（官方文件对象存储）；只允许 API 主机时，文件列表可以成功而文件传输会被出口策略拒绝。不要打印或保存签名下载地址。以九个 CSV 文件均严格读取为准。完整下载和运行命令见 [README](../README.md)。云任务的原始数据与运行报告存放于该任务工作区。

[Codex Cloud 环境文档](https://learn.chatgpt.com/docs/environments/cloud-environments) · [Kaggle CLI 认证](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/auth.md)
