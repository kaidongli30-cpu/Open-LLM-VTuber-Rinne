import os
"""周记和月记生成器的本地API配置。

直接编辑这个文件即可，不需要在CMD中设置环境变量。
日记生成器继续使用它原来的配置和生成逻辑。
"""

import diary_generator


# ==================== 请在这里查看或修改API ====================

# 默认沿用 diary_generator.py 中现有的API Key。
# 如需让周记/月记单独使用另一把Key，可改成：API_KEY = "你的Key"
API_KEY = os.getenv("MEMORY_LLM_API_KEY", "")

# OpenAI-compatible 的完整聊天补全地址
BASE_URL = "https://xingjiabiapi.org/v1/chat/completions"

# 周记和月记使用的模型
MODEL = "claude-sonnet-4-6"

# 这是API允许返回的最大token数，不是要求模型必须写满的字数。
# 保留较高硬上限以避免正文中途截断；实际篇幅由生成提示词中的软范围控制。
WEEKLY_MAX_TOKENS = 10000
MONTHLY_MAX_TOKENS = 10000

# API等待时间，单位为秒
WEEKLY_TIMEOUT_SECONDS = 300
MONTHLY_TIMEOUT_SECONDS = 300

# ==============================================================
