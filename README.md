# Open-LLM-VTuber-Rinne

凛祢桌面伙伴的公开代码版：对话、日记与第二层背景、游戏原画渲染及换装。项目基于 Open-LLM-VTuber；前端源码和网页构建位于 [前端仓库](https://github.com/kaidongli30-cpu/Open-LLM-VTuber-Rinne-Frontend)，本仓库的 `frontend` 是指向它的 Git 子模块。

公开仓库**不附带**游戏原文件或转换后的凛祢肖像模型、作者的 API Key、个人聊天与日记、`rinne_library` 数据和本机代理设置；但**附带作者当前使用的 GPT-SoVITS V2 参考 WAV**，对应的两份 V2 权重通过本项目 Release 下载。你需要使用自己持有的游戏源文件，在**自己的电脑上**运行肖像导入器。作者自制服装与程序代码分开授权；服装为 [CC BY-NC 4.0](assets/rinne-original-outfits/LICENSE.md)，不能未经另行授权用于商业用途。

本指南以 Windows PowerShell 和 CMD 为主。初次安装的目标是“在桌面客户端看到游戏原画凛祢、输入文字并听到 GPT-SoVITS V2 声线回复”。语音翻译与作者当前配置一样，使用本地 Ollama 的 `qwen3.5:4b-q4_K_M`。公开模板也启用了 SenseVoice 麦克风识别、近期记忆检索、每日子事件与已审核日记的第二层背景；初次安装尚无日记时不会凭空产生背景。不要把私人运行目录或密钥上传到 Git。

## 必要术语

- **API Key**：你自己的模型服务凭据，像密码一样保管。DeepSeek 网页聊天和 DeepSeek API 不是同一套配置。
- **CMD / PowerShell**：Windows 的两种命令窗口。下面分别给出写法；不要把 PowerShell 的 `$env:` 命令粘到 CMD。
- **Git 子模块**：后端仓库记录前端仓库的一个确定版本。因此克隆和更新都要带 `--recurse-submodules`。
- **PCK / SDK / 运行包**：PCK 是你本机游戏的源文件；仓库附带的自制 SDK 负责只读转换；生成的肖像运行包只留在你电脑上，不进入公开仓库。语音的参考 WAV 则已公开，无需从 PCK 提取。
- **GPT-SoVITS / Ollama**：前者使用凛祢 V2 权重把日语文字合成语音；后者用指定的本地模型把中文回复翻成日语，并用另一个本地模型生成每日子事件。它们都要作为独立服务运行，`uv sync` 不会代替安装或启动它们。
- **第二层背景**：从你审核过的日记提炼出的长期背景。它不同于原始日记；没有日记或尚未审核时不会凭空生成。

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
```

CMD：

```bat
git clone --recurse-submodules https://github.com/kaidongli30-cpu/Open-LLM-VTuber-Rinne.git
cd Open-LLM-VTuber-Rinne
uv sync
copy config_templates\conf.rinne.public.yaml conf.yaml
set "RINNE_DEEPSEEK_API_KEY=填入你自己的 DeepSeek API Key"
```

`conf.yaml` 被 Git 忽略；公开模板通过 `RINNE_DEEPSEEK_API_KEY` 环境变量读取凭据。上述临时变量只在当前命令窗口及从它启动的程序中生效。长期保存凭据应使用自己掌控的私有配置方式，**不要把 Key 写进受 Git 跟踪的源码或提交记录**。公开模板默认用 DeepSeek 对话、文字输入、**凛祢 GPT-SoVITS V2 声线**和本地 Ollama 日语翻译，不使用 Edge TTS。

### 安装与作者相同的 V2 语音链路

先确保 Ollama 已启动，再在 PowerShell 或 CMD 执行：

```text
ollama pull qwen3.5:4b-q4_K_M
ollama pull mistral-small3.2:24b
ollama list
```

最后一行应能看到两个模型。`mistral-small3.2:24b` 专供每日子事件；如果机器内存或磁盘不足，先在私人 `conf.yaml` 把 `character_config.daily_child_event_generation.enabled` 改成 `False`，但这会与作者现用体验不同。在**后端项目根目录**安装凛祢 V2 权重并校验仓库自带的五段参考 WAV。把下面的语音目录示例换成你解压后的真实路径，绝不能指向别人的私人安装目录。

PowerShell：

```powershell
uv run python .\setup_rinne_voice.py --gpt-root 'D:\Rinne-Voice\GPT-SoVITS-v2pro-20250604'
```

CMD：

```bat
uv run python setup_rinne_voice.py --gpt-root "D:\Rinne-Voice\GPT-SoVITS-v2pro-20250604"
```

脚本仅下载和校验本项目 [V2 语音权重 Release](https://github.com/kaidongli30-cpu/Open-LLM-VTuber-Rinne/releases/tag/rinne-gpt-sovits-v2-20260923) 中的两份模型（GPT 权重与 SoVITS 权重），并在语音目录生成独立的 `rinne_public_v2_tts_infer.yaml`；不会修改官方整合包原配置。它还会核对关键运行文件及五段 WAV 的 SHA-256；文件不匹配会停止，不会悄悄退回 Edge TTS。公开仓库中的 WAV 与作者当前使用的默认、惊讶/慌乱、害羞辅助、愤怒、灵装参考音逐字节一致；只有仓库路径不同。翻译词表也与作者当前运行的词表一致。

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

让这个语音窗口保持运行，默认监听本机 `9880`。公开配置已指向该端口；如果你要与另一套凛祢**同时运行**，请给隔离实例改用独立端口，并只在该实例私有的 `conf.yaml` 改 `gpt_sovits_tts.api_url`。运行声音并非只靠两份权重：公开配置还固定了作者当前的日语翻译模型、分词和情绪/灵装参考音路由；参考 WAV 不需要再次从游戏提取。

回到最初设置了 `RINNE_DEEPSEEK_API_KEY` 的项目窗口启动后端，并让它保持运行。下面的游戏资源导入和桌面客户端构建命令请**另开命令窗口**执行；在运行 `uv run` 前先进入刚克隆的后端项目根目录。

```text
uv run run_server.py
```

默认网页地址通常是 `http://127.0.0.1:12393`。但游戏原画渲染与本地资源读取需要桌面前端；仅打开网页不代表桌宠的衣服或语音已可用。

首次启动时，麦克风识别会按模板路径下载 SenseVoice 的 ONNX 模型，记忆检索会准备 `BAAI/bge-base-zh-v1.5` 与 `BAAI/bge-reranker-base`；这些不是翻译或每日子事件模型。保持网络连接并等待相关下载完成。公开模板还启用工具服务：`uvx`（随 uv 安装）会按需启动搜索工具，本机音乐及 Library 工具使用仓库附带程序；Library 的私人数据不随公开代码分发，新用户最初为空。

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

仓库的 `frontend` 是服务器使用的页面资源，**不是桌面客户端**；Electron 桌面客户端源码在另一个公开仓库。要使用当前源码的 Live Mode、Pet Mode 与本机游戏资源接口，可自行构建：

```powershell
git clone https://github.com/kaidongli30-cpu/Open-LLM-VTuber-Rinne-Frontend.git
Set-Location .\Open-LLM-VTuber-Rinne-Frontend\source
npm ci
npm run build:unpack
```

CMD 中把 `Set-Location` 换成 `cd`，其余命令相同。构建完成后，在 `source\release\1.2.1\win-unpacked` 中启动 `open-llm-vtuber-electron.exe`。请保持上一步启动的后端窗口运行。如果构建输出目录随版本变化，以实际 `win-unpacked` 目录为准。不要把旧版安装包的 `app.asar` 当作当前源码；本项目不会自动替换用户已安装客户端。

验收顺序：`status` 显示本地肖像资源完整 → 桌面前端显示凛祢 → Live Mode 选择服装无闪退 → 输入文字能取得回复并听到 **V2 凛祢日语声线** → 切换灵装后听到对应灵装参考音 → 关闭重启后仍能继续对话。文字出现但无语音**不算安装成功**；检查 Ollama、GPT-SoVITS 两个窗口和后端日志，不要换用 Edge TTS 冒充通过。若首次无画面，先检查导入器状态与桌面客户端是否彻底重启。

如果同一台电脑上还运行着另一套凛祢，不能仅换后端端口：桌面客户端的聊天设置、窗口状态和换装设置也必须分开。启动独立测试客户端前，可给它设置绝对路径环境变量 `RINNE_CLIENT_USER_DATA_DIR`（客户端数据目录）与 `RINNE_RENDERER_SETTINGS_PATH`（换装设置文件）；后端也使用同一个 `RINNE_RENDERER_SETTINGS_PATH`，并用 `RINNE_DATA_ROOT` 隔离日记和记忆。测试客户端连接地址必须指向测试后端的端口。普通首次安装无需设置这些变量。

## 3. 代理仅按需设置

公开模板中 DeepSeek 的 `proxy_url: null` 表示**直接连接**，没有作者的 Clash Verge 地址或端口。大多数能直接访问 DeepSeek 的用户无需改动。确实需要代理时，只改私人 `conf.yaml` 中 `agent_config.llm_configs.deepseek_llm.proxy_url`，填写自己可用的 HTTP 代理地址；第二层背景会沿用该 DeepSeek 配置。代理软件必须自行运行，填了地址并不代表连接一定通过。不要把个人网络规则或凭据提交到仓库。

## 4. 已有用户升级而不是重装

升级前先备份私人 `conf.yaml` 与整个 `chat_history`，并停掉旧后端。不要运行会覆盖私人文件的复制命令。

- **原目录内升级**：更新代码和子模块，保留未受 Git 跟踪的 `conf.yaml`、`chat_history`、`rinne_library` 和 `local_config`；检查 `conf_uid` 仍是 `rinne_01`。代码默认继续使用 `chat_history\rinne_01`，不会清空旧记忆。
- **换到新目录**：把私人数据复制到新目录的 `chat_history`，或在启动窗口设置 `RINNE_DATA_ROOT` 为一个独立、绝对的数据目录，程序会在其下读写 `chat_history\rinne_01`。不要把两个正在运行的后端同时指向同一份数据；先在副本上验证，再切换。
- **保留 Library**：`rinne_library\rinne_01` 是独立的私人文件库，`RINNE_DATA_ROOT` 不会替它改位置。换目录时把旧库复制到新目录的同名位置，或用绝对路径环境变量 `RINNE_LIBRARY_ROOT` 指向要继续使用的旧库；并行测试应使用副本，避免两个进程同时写同一库。不要把库里的数据提交到 GitHub。

PowerShell 更新代码：

```powershell
git pull --ff-only
git submodule update --init --recursive
uv sync
```

CMD 中命令相同。私人 `conf.yaml` 不应被模板覆盖；需要新选项时，对照 `config_templates/conf.rinne.public.yaml` 在私人文件中增补。日记、聊天和 `rinne_library` 数据不属于代码升级包，也不能提交到 GitHub。

已有用户要升级到与作者相同的 V2 声线时，仍使用本节的原目录/新目录数据升级方式，**不要**用公开模板覆盖已积累记忆的私人 `conf.yaml`。先按“安装与作者相同的 V2 语音链路”在自己的独立语音目录运行安装脚本并启动服务；然后仅把公开模板里 `character_config.tts_config.tts_model`、`gpt_sovits_tts` 整段和 `tts_preprocessor_config.translator_config` 中的翻译开关、模型及参数合并到私人 `conf.yaml`。若已有自己的翻译词表，保留其路径和私人词条，不要被公开词表覆盖。原有 DeepSeek Key、角色 ID、日记与 Library 保持不变。确认 `ollama list` 有指定模型，重启后端和桌面客户端，再按上面的语音验收顺序检查日常与灵装。

### 用已审核日记建立或续写第二层背景

公开模板默认已经将 `character_config.layer2_memory_generation.enabled` 设为 `True`；已有用户的私人配置若尚未启用，需手动改为 `True`。只有处理已审核日记时才会调用配置的 DeepSeek API 并产生请求费用。先查看哪些历史日记有资格处理：

```powershell
uv run python -m src.open_llm_vtuber.memory.layer2_backfill --dry-run
```

只有已验收、具备匹配批准标记的日记才会自动补齐。若需要逐篇在终端确认：

```powershell
uv run python -m src.open_llm_vtuber.memory.layer2_backfill --approve-interactively
```

命令会按日期顺序处理；缺失内容不会被猜测或补写。成功后新对话读取已发布的第二层背景，原始私人日记仍留在本地。后台启动时也只会补齐**已批准**的日记；未审核的日记不会被自动批准。周记/月记沿原有独立生成流程，不要把它们误认为第二层背景由 DeepSeek 生成。

## 5. 范围与常见问题

- **换装菜单没有某套**：先运行该套 `build --outfit-number N` 并用 `status` 验证；菜单只展示完整可用的资源。作者自制服装的本地游戏头部合成也依赖用户自己的原画运行包。
- **API 连接失败**：先检查自己的 Key、额度与网络；只有确实受网络限制时才设置私人代理。公开项目不会替你启动 Clash Verge。
- **运行时找不到前端**：执行 `git submodule update --init --recursive`，确认 `frontend\index.html` 存在。
- **文字正常但没声音**：确认 Ollama 的 `qwen3.5:4b-q4_K_M` 已安装并运行、GPT-SoVITS 由 `rinne_public_v2_tts_infer.yaml` 启动、语音端口与私人 `conf.yaml` 匹配。不要把“GPT-SoVITS 程序版本 V2Pro”误认为已加载“凛祢 V2 权重”；安装脚本和启动配置会分别检查这两件事。
- **旧记忆没出现**：确认正在启动的是正确的后端目录、`conf_uid=rinne_01`、`RINNE_DATA_ROOT` 没指错；不要删除旧 `chat_history`。
- **TypeScript 检查报警**：当前前端附带的旧 WebSDK 类型定义尚有历史报错。以构建和实际交互验收为准，并把新发现的独立问题单独报告；不能据此宣称所有功能已通过。

若要开发或反馈问题，请只提供脱敏的日志和复现步骤，不上传 API Key、游戏 PCK、转换后的资源、私人聊天或日记。使用本地文件、游戏资源和第三方服务时，请自行遵守其使用条款。
