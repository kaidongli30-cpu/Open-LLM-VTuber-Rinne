"""周记和月记生成器的本地API配置。

默认沿用日记生成器的 API 设置，也可用环境变量单独指定。
日记生成器继续使用它原来的生成逻辑。
"""

import os

import diary_generator


# ==================== API 设置 ====================

# 默认沿用 diary_generator.py 的 API Key。
# 如需让周记/月记单独使用另一把 Key，可设置 MEMORY_GENERATION_API_KEY。
API_KEY = os.getenv("MEMORY_GENERATION_API_KEY") or diary_generator.LLM_API_KEY

# OpenAI-compatible 的完整聊天补全地址
BASE_URL = "https://apinebula.ai/v1/chat/completions"

# 周记和月记使用的模型
MODEL = "claude-opus-4-6"

# 这是API允许返回的最大token数，不是要求模型必须写满的字数。
# 保留较高硬上限以避免正文中途截断；实际篇幅由生成提示词中的软范围控制。
WEEKLY_MAX_TOKENS = 10000
MONTHLY_MAX_TOKENS = 10000

# API等待时间，单位为秒
WEEKLY_TIMEOUT_SECONDS = 300
MONTHLY_TIMEOUT_SECONDS = 300

# ==============================================================
