"""周记和月记生成器的 API 配置。"""

import diary_generator


# ==================== 请在这里查看或修改API ====================

# 默认沿用日记生成器配置，不需要重复填写。
API_KEY = diary_generator.LLM_API_KEY

# 完整聊天补全地址
BASE_URL = diary_generator.LLM_API_URL

# 周记和月记使用的模型
MODEL = diary_generator.LLM_MODEL

# 如果希望周记和月记单独使用另一套 API，把上面三项改成：
# API_KEY = "另一套 API Key"
# BASE_URL = "https://服务商地址/chat/completions"
# MODEL = "服务商提供的模型名"

# 这是API允许返回的最大token数，不是要求模型必须写满的字数。
# 保留较高硬上限以避免正文中途截断；实际篇幅由生成提示词中的软范围控制。
WEEKLY_MAX_TOKENS = 10000
MONTHLY_MAX_TOKENS = 10000

# API等待时间，单位为秒
WEEKLY_TIMEOUT_SECONDS = 300
MONTHLY_TIMEOUT_SECONDS = 300

# ==============================================================
