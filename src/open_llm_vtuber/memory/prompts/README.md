# Rinne 记忆提示词索引

此目录集中保存会影响凛祢记忆生成语义的正式提示词。

- `layer2_background_cloud_neutral_v1.txt`：第二层事实账本的每日中性差量更新。
- `layer2_model_facing_blind_v1.txt`：把完整第二层账本压缩为每轮直接送给对话模型的个人背景。
- `layer2_model_facing_rolling_patch_v1.txt`：以最后实际保存的个人背景为正文基础，只把当日事实差量局部写入下一版。
- `child_event_system_v23.txt`：第三层长期记忆的每日子事件生成系统提示词。
- `child_event_task_v23.txt`：第三层长期记忆的每日子事件任务模板。

第三层“是否调用检索工具”的工具说明仍位于
`src/open_llm_vtuber/memory/cloud_memory_tool.py`，因为它与工具定义及参数 Schema
共同组成运行时工具契约，不属于每日离线生成提示词。

修改任一正式提示词后，都应重新执行相应的盲测或回放；不要直接把人工审查中的
正确答案写回提示词。
