# 数据字典

Kaggle 数据页列出九个 CSV，覆盖 2023–2025 年选秀或签约的 510 名球员。候选键在读取完整 CSV 后检查。实际字段名、推断类型与缺失比例由每次运行生成的 `data_dictionary.md` 记录。

| 文件 | 粒度与候选键 |
|---|---|
| `players.csv` | 一名球员，`nfl_id` |
| `combine_results.csv` | 一名球员的体测记录，`nfl_id` |
| `player_career_successes.csv` | 一名球员的职业结果，`nfl_id` |
| `games.csv` | 一场比赛，`game_id` |
| `player_play.csv` | 一名球员的一档进攻，`game_id` + `play_id` + `nfl_id` |
| `combine_tracking.csv` | 球员试次追踪，`event_id` + `nfl_id` + `time` |
| `game_tracking_2023.csv` | 2023 赛季比赛追踪，球员-回合键 + `time` |
| `game_tracking_2024.csv` | 2024 赛季比赛追踪，球员-回合键 + `time` |
| `game_tracking_2025.csv` | 2025 赛季比赛追踪，球员-回合键 + `time` |

比赛追踪分别核对球员、比赛及球员-回合关联。追踪中的空球员键单独计数，结合 `entity_type` 等字段判定其含义。位置和选秀年使用球员表的 `nfl_position` 与 `draft_year`。Combine 覆盖优先取追踪表的 `draft_year`，并在结果中记录年份来源。`combine_results.csv` 中的 `combine_position` 是试训分组，后续按训练分组分析时使用该字段。

[Kaggle 数据说明](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/data)
