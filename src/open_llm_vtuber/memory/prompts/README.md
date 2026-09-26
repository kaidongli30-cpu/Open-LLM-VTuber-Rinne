# 记忆生成提示词

这里存放日记背景和记忆整理使用的提示词。正常安装不需要修改。

## 第二层背景

| 文件 | 用途 |
| --- | --- |
| `layer2_background_cloud_neutral_v1.txt` | 根据当天日记更新事实记录 |
| `layer2_model_facing_blind_v1.txt` | 把完整事实记录整理成供对话使用的个人背景 |
| `layer2_model_facing_rolling_patch_v1.txt` | 保留上次背景，只修改当天需要更新的部分 |

## 每日子事件

“子事件”是从每天的记忆中整理出的具体事件，供之后检索。

- `child_event_system_v23.txt`：整理事件的总规则。
- `child_event_task_v23.txt`：每次整理时使用的任务模板。

对话时何时调用记忆检索工具，由 [cloud_memory_tool.py](../cloud_memory_tool.py) 中的工具说明控制，不在这里。

## 修改后怎样检查

修改提示词后，要用已有案例重新测试，确认生成结果仍符合要求。不要把测试案例的答案直接写进提示词。
