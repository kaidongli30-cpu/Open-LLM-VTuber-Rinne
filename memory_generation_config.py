import os
"""周记和月记生成器的 API 配置。

默认沿用日记生成器的 API、地址和模型，也可以通过 MEMORY_LLM_* 环境变量
为周记和月记单独覆盖。这样公开仓库不需要保存任何私人密钥。
"""

import diary_generator


# ==================== 请在这里查看或修改API ====================

# 默认沿用日记生成器配置；MEMORY_LLM_* 的优先级更高。
API_KEY = os.getenv("MEMORY_LLM_API_KEY", diary_generator.LLM_API_KEY)

# OpenAI-compatible 的完整聊天补全地址
BASE_URL = os.getenv("MEMORY_LLM_API_URL", diary_generator.LLM_API_URL)

# 周记和月记使用的模型
MODEL = os.getenv("MEMORY_LLM_MODEL", diary_generator.LLM_MODEL)

# 这是API允许返回的最大token数，不是要求模型必须写满的字数。
# 保留较高硬上限以避免正文中途截断；实际篇幅由生成提示词中的软范围控制。
WEEKLY_MAX_TOKENS = 10000
MONTHLY_MAX_TOKENS = 10000

# API等待时间，单位为秒
WEEKLY_TIMEOUT_SECONDS = 300
MONTHLY_TIMEOUT_SECONDS = 300

# ==============================================================
