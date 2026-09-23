# Open-LLM-VTuber-Rinne

凛祢桌面客户端：对话、日记与第二层背景、游戏原画渲染及换装。项目基于 Open-LLM-VTuber；桌面客户端源码位于 [前端仓库](https://github.com/kaidongli30-cpu/Open-LLM-VTuber-Rinne-Frontend)，本仓库的 `frontend` 是指向它的 Git 子模块。

仓库不附带《凛祢乌托邦》与《凛绪轮回》的游戏原文件、转换后的凛祢游戏模型，也不包含 API Key、聊天记录、日记或 `rinne_library` 数据。GPT-SoVITS V2 参考 WAV 已包含在仓库中，两份语音权重由安装脚本从项目 Release 下载。请使用自己持有的游戏源文件，在本机运行肖像导入器。项目自制服装按 [CC BY-NC 4.0](assets/rinne-original-outfits/LICENSE.md) 授权，商业使用需另行获得许可。

本指南以 Windows PowerShell 和 CMD 为主。完成安装后，可在桌面客户端看到游戏原画凛祢、进行文字或语音对话，并听到 GPT-SoVITS V2 声线。语音翻译使用本地 Ollama 的 `qwen3.5:4b-q4_K_M`；SenseVoice 用于麦克风识别。近期记忆检索、每日子事件和已审核日记的第二层背景也已启用。没有日记时不会凭空生成背景。

## 必要术语

- **API Key**：你自己的模型服务凭据，像密码一样保管。
- **CMD / PowerShell**：Windows 的两种命令窗口。下面分别给出写法；不要把 PowerShell 的 `$env:` 命令粘到 CMD。
- **Git 子模块**：后端仓库记录前端仓库的一个确定版本。因此克隆和更新都要带 `--recurse-submodules`。
- **PCK / SDK / 运行包**：PCK 是你本机游戏的源文件；仓库附带的自制 SDK 负责只读转换；生成的凛祢游戏资源包只留在你电脑上，不进入公开仓库。
- **GPT-SoVITS / Ollama**：前者使用凛祢 V2 权重把日语文字合成语音；后者用指定的本地模型把中文回复翻成日语，并用另一个 24B 本地模型生成每日子事件。它们都要作为独立服务运行，`uv sync` 不会代替安装或启动它们。
- **第二层背景**：从你审核过的日记提炼出的长期背景，用于让凛祢了解你大致是怎样的人、目前处于什么状态。它不同于原始日记；没有日记或尚未审核时不会凭空生成。

## 1. 准备软件

需要 Git、[uv](https://docs.astral.sh/uv/getting-started/installation/) 和 Python 3.10–3.12（推荐 3.12）。还需要 [Ollama](https://ollama.com/download/windows)、[7-Zip](https://www.7-zip.org/) 与 [GPT-SoVITS 官方 Windows 整合包 `GPT-SoVITS-v2pro-20250604.7z`](https://huggingface.co/lj1995/GPT-SoVITS-windows-package/blob/8b081e1fa1b3ad121e0f310e525dc80fcf15becc/GPT-SoVITS-v2pro-20250604.7z)；下载并解压整合包，记住**直接包含 `api_v2.py` 和 `runtime\python.exe` 的目录**，后文称为“语音目录”。确认基础命令：

PowerShell 或 CMD：

```text
git --version
uv --version
uv python install 3.12
```

若命令不存在，请先从 [Git for Windows](https://git-scm.com/download/win) 安装 Git、从 uv 官方文档安装 uv，然后重新打开命令窗口。启动 Ollama 后执行 `ollama --version`；翻译模型约 3.4 GB、每日子事件模型约 15 GB，另需 GPT-SoVITS 整合包、约 1 GB 的 SenseVoice 模型及首次记忆检索所需的模型缓存，预留足够磁盘空间。前端桌面源码构建还需要 Node.js 和 npm；只运行后端不需要它们。

## 2. 全新部署

PowerShell：

```powershell
git clone --recurse-submodules https://github.com/kaidongli30-cpu/Open-LLM-VTuber-Rinne.git
Set-Location .\Open-LLM-VTuber-Rinne
uv sync
Copy-Item .\config_templates\conf.rinne.public.yaml .\conf.yaml
$env:RINNE_DEEPSEEK_API_KEY = '填入你自己的 DeepSeek API Key'
$env:RINNE_APINEBULA_API_KEY = '填入你自己的 APINebula API Key'
$env:RINNE_MEDIA_GEMINI_API_KEY = '填入可调用 Gemini 的 API Key'
$env:BOCHA_API_KEY = '填入你自己的博查 API Key'
```

CMD：

```bat
git clone --recurse-submodules https://github.com/kaidongli30-cpu/Open-LLM-VTuber-Rinne.git
cd Open-LLM-VTuber-Rinne
uv sync
copy config_templates\conf.rinne.public.yaml conf.yaml
set "RINNE_DEEPSEEK_API_KEY=填入你自己的 DeepSeek API Key"
set "RINNE_APINEBULA_API_KEY=填入你自己的 APINebula API Key"
set "RINNE_MEDIA_GEMINI_API_KEY=填入可调用 Gemini 的 API Key"
set "BOCHA_API_KEY=填入你自己的博查 API Key"
```

`conf.yaml` 被 Git 忽略。默认使用 APINebula 的 `claude-opus-4-6` 对话、DeepSeek 生成第二层背景、Gemini 观察视频和屏幕、博查搜索、GPT-SoVITS V2 合成语音，以及本地 Ollama 翻译。以上 Key 分别由对应功能读取；使用这些功能需持有相应服务的凭据。若 Gemini 也通过 APINebula 调用，可按服务提供的说明使用相应 Key。环境变量只在当前命令窗口及从它启动的程序中生效，不要把 Key 写进 Git 仓库。

### 安装 V2 语音

先确保 Ollama 已启动，再在 PowerShell 或 CMD 执行：

```text
ollama pull qwen3.5:4b-q4_K_M
ollama pull mistral-small3.2:24b
ollama list
```

最后一行应能看到两个模型。`mistral-small3.2:24b` 用于每日子事件。接着在**后端项目根目录**安装凛祢 V2 权重。把下面的语音目录示例换成你的实际解压路径。

PowerShell：

```powershell
uv run python .\setup_rinne_voice.py --gpt-root 'D:\Rinne-Voice\GPT-SoVITS-v2pro-20250604'
```

CMD：

```bat
uv run python setup_rinne_voice.py --gpt-root "D:\Rinne-Voice\GPT-SoVITS-v2pro-20250604"
```

脚本从 [V2 语音权重 Release](https://github.com/kaidongli30-cpu/Open-LLM-VTuber-Rinne/releases/tag/rinne-gpt-sovits-v2-20260923) 下载并校验两份模型，在语音目录生成 `rinne_public_v2_tts_infer.yaml`。文件校验失败时会报错并停止，请检查下载和整合包版本。

另开一个命令窗口，进入**语音目录**并启动 GPT-SoVITS。PowerShell：

```powershell
Set-Location 'D:\Rinne-Voice\GPT-SoVITS-v2pro-20250604'
.\runtime\python.exe .\api_v2.py -c .\rinne_public_v2_tts_infer.yaml
```

CMD：

```bat
cd /d "D:\Rinne-Voice\GPT-SoVITS-v2pro-20250604"
runtime\python.exe api_v2.py -c rinne_public_v2_tts_infer.yaml
```

让这个语音窗口保持运行，默认监听本机 `9880`，与 `conf.yaml` 中的语音地址一致。若端口已被占用，可启动语音服务时指定其他端口，并同步修改 `gpt_sovits_tts.api_url`。参考 WAV 已包含在仓库中，无需从游戏提取。

回到最初设置了各项 API Key 的项目窗口启动后端，并让它保持运行。下面的游戏资源导入和桌面客户端构建命令请**另开命令窗口**执行；在运行 `uv run` 前先进入刚克隆的后端项目根目录。

```text
uv run run_server.py
```

后端默认在本机 `127.0.0.1:12393` 提供接口。请使用后文的桌面客户端完成对话和换装。

首次启动时会下载 SenseVoice 识别模型以及记忆检索所需的 `BAAI/bge-base-zh-v1.5`、`BAAI/bge-reranker-base`。保持网络连接并等待下载完成。搜索、音乐与 Library 工具也会随程序启动；新安装的 Library 为空。

### 把自己的游戏源文件导入

先在游戏目录里找到包含 `MP060101.pck` 等文件的 `Data\Data\Mp\1st` 文件夹。项目不提供这些 PCK，也不自动下载。默认第一套的 PowerShell 示例：

```powershell
uv run python setup_rinne_game_assets.py build 'D:\Games\DATE A LIVE Rio Reincarnation\Data\Data\Mp\1st'
uv run python setup_rinne_game_assets.py status
```

CMD：

```bat
uv run python setup_rinne_game_assets.py build "D:\Games\DATE A LIVE Rio Reincarnation\Data\Data\Mp\1st"
uv run python setup_rinne_game_assets.py status
```

把示例路径换成你实际的游戏目录。也可以运行 `uv run python setup_rinne_game_assets.py build --gui` 打开文件夹选择窗口。导入器会把转换结果放入本机应用数据目录，并在被忽略的 `local_config` 建立指针；不会修改 PCK。第 2–4 套、手动配置、状态检查与安全移除见 [游戏资源导入说明](public_docs/GAME_ASSET_SETUP.md)。第一套转换要处理 15 个肖像，可能需要一些时间。导入完成后应完全退出并重新启动桌面前端。

### 从源码启动桌面前端

仓库中的 `frontend` 供服务器使用。要运行 Live Mode、Pet Mode 和本地游戏肖像，请构建桌面客户端：

```powershell
git clone https://github.com/kaidongli30-cpu/Open-LLM-VTuber-Rinne-Frontend.git
Set-Location .\Open-LLM-VTuber-Rinne-Frontend\source
npm ci
npm run build:unpack
```

CMD 中把 `Set-Location` 换成 `cd`，其余命令相同。构建完成后，在 `source\release\1.2.1\win-unpacked` 中启动 `open-llm-vtuber-electron.exe`。请保持后端窗口运行。如果输出目录随版本变化，以实际 `win-unpacked` 目录为准。

安装后依次检查：`status` 显示肖像资源完整；桌面客户端显示凛祢；Live Mode 可以选择服装；输入文字后能收到回复并听到语音；切换灵装后仍能正常对话。若有文字但没有声音，检查 Ollama、GPT-SoVITS 两个窗口和后端日志。若没有画面，检查导入器状态并完全重启桌面客户端。

## 3. 代理仅按需设置

`proxy_url: null` 表示直连 DeepSeek。只有直连失败且已有可用 HTTP 代理时，才在本地 `conf.yaml` 中设置 `agent_config.llm_configs.deepseek_llm.proxy_url`。第二层背景沿用该连接设置。不要把自己的代理地址或凭据提交到仓库。

## 4. 已有用户升级而不是重装

升级前先备份私人 `conf.yaml` 与整个 `chat_history`，并停掉旧后端。不要运行会覆盖私人文件的复制命令。

- **原目录内升级**：更新代码和子模块，保留未受 Git 跟踪的 `conf.yaml`、`chat_history`、`rinne_library` 和 `local_config`；检查 `conf_uid` 仍是 `rinne_01`。代码默认继续使用 `chat_history\rinne_01`，不会清空旧记忆。
- **换到新目录**：把私人数据复制到新目录的 `chat_history`，或在启动窗口设置 `RINNE_DATA_ROOT` 为一个独立、绝对的数据目录，程序会在其下读写 `chat_history\rinne_01`。不要把两个正在运行的后端同时指向同一份数据；先在副本上验证，再切换。
- **保留 Library**：`rinne_library\rinne_01` 是独立的私人文件库，`RINNE_DATA_ROOT` 不会替它改位置。换目录时把旧库复制到新目录的同名位置，或用绝对路径环境变量 `RINNE_LIBRARY_ROOT` 指向要继续使用的旧库；并行测试应使用副本，避免两个进程同时写同一库。不要把库里的数据提交到 GitHub。

如果要在同一台电脑上同时运行两套凛祢，还要为新实例分别设置 `RINNE_CLIENT_USER_DATA_DIR`（客户端数据）、`RINNE_RENDERER_SETTINGS_PATH`（换装设置）与 `RINNE_DATA_ROOT`（日记和记忆），并让客户端连接新实例的后端端口。各变量应指向独立的绝对路径；不要让两个运行中的实例写入同一份数据。

PowerShell 更新代码：

```powershell
git pull --ff-only
git submodule update --init --recursive
uv sync
```

CMD 中命令相同。接着在原项目目录预览配置更新：

```text
uv run python scripts/update_rinne_config.py
```

确认列出的设置后，再执行：

```text
uv run python scripts/update_rinne_config.py --apply
```

脚本会先把原有 `conf.yaml` 备份为同目录下带日期的文件，然后在原文件中更新模型、语音、翻译和记忆设置；保留原有 API Key、代理、个人称呼及提示词、角色 ID、本机路径和语音参考文件。聊天、日记、背景与 `rinne_library` 不会被移动或清空。请在启动窗口设置对话、第二层和媒体功能所需的 Key。日记、聊天和 `rinne_library` 数据不能提交到 GitHub。

已有用户也按“安装 V2 语音”一节配置语音服务，再使用上述脚本更新原 `conf.yaml`。若已有自己的翻译词表和参考音，脚本会保留其路径；确认 `ollama list` 有指定模型，重启后端和桌面客户端，检查日常与灵装语音。

### 用已审核日记建立或续写第二层背景

配置更新后，`character_config.layer2_memory_generation.enabled` 为 `True`。只有处理已审核日记时才会调用配置的 DeepSeek API 并产生请求费用。先查看哪些历史日记有资格处理：

```powershell
uv run python -m src.open_llm_vtuber.memory.layer2_backfill --dry-run
```

只有你明确批准的日记才会用于背景更新。若需要逐篇在终端确认：

```powershell
uv run python -m src.open_llm_vtuber.memory.layer2_backfill --approve-interactively
```

命令会按日期顺序处理；缺失内容不会被猜测或补写。成功后新对话会读取生成的第二层背景，原始日记仍留在本地。后台启动时也只会补齐**已批准**的日记；未审核的日记不会被自动批准。

## 5. 范围与常见问题

- **换装菜单没有某套**：先运行该套 `build --outfit-number N` 并用 `status` 验证；菜单只展示完整可用的资源。作者自制服装的本地游戏头部合成也依赖用户自己的原画运行包。
- **API 连接失败**：检查自己的 Key、额度与网络；需要代理时，先确认代理服务已启动，再设置本地代理地址。
- **运行时找不到前端**：执行 `git submodule update --init --recursive`，确认 `frontend\index.html` 存在。
- **文字正常但没声音**：确认 Ollama 的 `qwen3.5:4b-q4_K_M` 已安装并运行、GPT-SoVITS 使用 `rinne_public_v2_tts_infer.yaml` 启动、语音端口与本地 `conf.yaml` 匹配。
- **旧记忆没出现**：确认正在启动的是正确的后端目录、`conf_uid=rinne_01`、`RINNE_DATA_ROOT` 没指错；不要删除旧 `chat_history`。

反馈问题时，请勿附上 API Key、游戏 PCK、转换后的资源、聊天记录或日记。使用游戏资源和第三方服务时，请遵守其使用条款。
