# Open-LLM-VTuber-Rinne

凛祢桌面客户端：对话、日记与第二层背景、游戏原画渲染及换装。项目基于 Open-LLM-VTuber；桌面客户端源码位于 [前端仓库](https://github.com/kaidongli30-cpu/Open-LLM-VTuber-Rinne-Frontend)，本仓库的 `frontend` 是指向它的 Git 子模块。

仓库已包含凛祢的五套游戏服装和四套作者自制服装，克隆后不必再导入游戏文件。GPT-SoVITS V2 参考 WAV 已包含在项目中，两份语音权重从项目 Release 手动下载。项目自制服装按 [CC BY-NC 4.0](assets/rinne-original-outfits/LICENSE.md) 授权，商业使用需另行获得许可。

本指南以 Windows PowerShell 和 CMD 为主。完成安装后，可在桌面客户端看到游戏原画凛祢、进行文字或语音对话，并听到 GPT-SoVITS V2 声线。语音翻译使用本地 Ollama 的 `qwen3.5:4b-q4_K_M`；SenseVoice 用于麦克风识别。近期记忆检索、每日子事件和已审核日记的第二层背景也已启用。没有日记时不会凭空生成背景。

## 必要术语

- **API Key**：你自己的模型服务凭据，像密码一样保管。
- **CMD / PowerShell**：Windows 的两种命令窗口。下面分别给出写法，请按自己打开的窗口选择。
- **Git 子模块**：后端仓库记录前端仓库的一个确定版本。因此克隆和更新都要带 `--recurse-submodules`。
- **GPT-SoVITS / Ollama**：前者使用凛祢 V2 权重把日语文字合成语音；后者用指定的本地模型把中文回复翻成日语，并用另一个 24B 本地模型生成每日子事件。它们都要作为独立服务运行，`uv sync` 不会代替安装或启动它们。
- **第二层背景**：从你审核过的日记提炼出的长期背景，用于让凛祢了解你大致是怎样的人、目前处于什么状态。它不同于原始日记；没有日记或尚未审核时不会凭空生成。

## 1. 准备软件

需要 Git、[uv](https://docs.astral.sh/uv/getting-started/installation/) 和 Python 3.10–3.12（推荐 3.12）。还需要 [Ollama](https://ollama.com/download/windows) 和 [7-Zip](https://www.7-zip.org/)。确认基础命令：

PowerShell 或 CMD：

```text
git --version
uv --version
uv python install 3.12
```

若命令不存在，请先从 [Git for Windows](https://git-scm.com/download/win) 安装 Git、从 uv 官方文档安装 uv，然后重新打开命令窗口。启动 Ollama 后执行 

```text
ollama --version
```

翻译模型约 3.4 GB、每日子事件模型约 15 GB，另需 GPT-SoVITS 整合包、约 1 GB 的 SenseVoice 模型及首次记忆检索所需的模型缓存，预留足够磁盘空间。

## 2. 全新部署

先决定把凛祢放在哪个盘。以下以 `D:\AI\Open-LLM-VTuber-Rinne` 为例；把 `D:\AI` 换成你想使用的文件夹。即使当前命令窗口显示 `C:\Users\...`，下面的命令也会把项目放在指定的 D 盘目录，而不是 C 盘。

PowerShell：

```powershell
New-Item -ItemType Directory -Force 'D:\AI' | Out-Null
git clone --recurse-submodules https://github.com/kaidongli30-cpu/Open-LLM-VTuber-Rinne.git 'D:\AI\Open-LLM-VTuber-Rinne'
Set-Location 'D:\AI\Open-LLM-VTuber-Rinne'
uv sync
```

CMD：

```bat
if not exist "D:\AI" mkdir "D:\AI"
git clone --recurse-submodules https://github.com/kaidongli30-cpu/Open-LLM-VTuber-Rinne.git "D:\AI\Open-LLM-VTuber-Rinne"
cd /d "D:\AI\Open-LLM-VTuber-Rinne"
uv sync
```

安装完成后，在项目根目录里找到'conf.yaml'。这是凛祢的运行设置文件，非常重要，可以理解为把桌宠凛祢的一切组合起来的一份文件。找到下列四处，把各自的密钥填在单引号中间，保存文件即可；只改引号里的内容，不要删除缩进或冒号：

| 用途 | 在 `conf.yaml` 中找到 | 填写位置 |
| --- | --- | --- |
| 对话 | `openai_compatible_llm:` | 其下的 `llm_api_key: ''`，填写 APINebula API Key |
| 第二层背景等 DeepSeek 调用 | `deepseek_llm:` | 其下的 `llm_api_key: ''`，填写 DeepSeek API Key |
| 视频与屏幕观察 | `media_analysis:` | 其下的 `api_key: ''`，填写可调用 Gemini 的 API Key |
| 联网搜索 | `basic_memory_agent:` | 其下的 `bocha_api_key: ''`，填写博查 API Key |

例如，拿到 APINebula 密钥后，把 `openai_compatible_llm` 下面的 `llm_api_key: ''` 改成 `llm_api_key: '你的实际密钥'`。其余三处做法相同。如果 Gemini 也通过 APINebula 调用，在视频观察那一处填写可用于该模型的密钥。保存后重新启动后端，修改才会生效。项目目录里的虚拟环境和前端构建文件会留在所选磁盘；其他软件自己的下载缓存可能仍按各自默认设置使用 C 盘。

### 填写日记、周记和月记的密钥

这三项的密钥不在 `conf.yaml` 中，启动后端前还需要完成下面的填写。

1. 在项目根目录找到 `diary_generator.py`，右键选择“打开方式 → 记事本”。按 `Ctrl+F` 搜索 `LLM_API_KEY =`，把这一整行替换成下面的形式，再将引号中的文字换成你的 APINebula 密钥：

   ```python
   LLM_API_KEY = '你的实际密钥'
   ```

2. 保存文件。日记默认使用 `claude-sonnet-4-6`；周记和月记默认使用 `claude-opus-4-6`，两者会自动沿用刚填好的这把密钥，不需要填写三次。
3. 如果周记和月记需要另一把密钥，打开同一目录的 `memory_generation_config.py`，搜索 `API_KEY =`，只把该行替换为 `API_KEY = '另一把实际密钥'` 并保存。不需要另一把密钥时不要改它。

不要删除引号，也不要改动文件其他部分。修改后重新启动后端即可生效；项目的更新工具会保留上述两处填写的密钥。

### 下载本地模型

先确保 Ollama 已启动，再在 PowerShell 或 CMD 执行：

```text
ollama pull qwen3.5:4b-q4_K_M
ollama pull mistral-small3.2:24b
ollama list
```

最后一行应能看到两个模型。`mistral-small3.2:24b` 用于每日子事件。

### 安装 V2 语音

先下载 [GPT-SoVITS 官方 Windows 整合包 `GPT-SoVITS-v2pro-20250604.7z`](https://huggingface.co/lj1995/GPT-SoVITS-windows-package/blob/8b081e1fa1b3ad121e0f310e525dc80fcf15becc/GPT-SoVITS-v2pro-20250604.7z)，用 7-Zip 解压到你想放语音程序的位置。打开解压出来的文件夹，找到**直接包含 `api_v2.py` 文件和 `runtime` 文件夹**的那一层，这一层也是语音模型的根目录；下文把这一层称为“语音目录”。如果解压后有两层同名文件夹，请进入里面那一层，以实际看到 `api_v2.py` 为准。

然后打开 [凛祢 V2 语音权重下载页](https://github.com/kaidongli30-cpu/Open-LLM-VTuber-Rinne/releases/tag/rinne-gpt-sovits-v2-20260923)，下载页面下方的两个文件，不要改文件名。用文件资源管理器把它们分别放到刚才找到的语音目录中：

| 下载的文件 | 放到语音目录中的位置 |
| --- | --- |
| `rinne_e15.ckpt` | `GPT_weights_v2\rinne_e15.ckpt` |
| `rinne_e8_s456.pth` | `SoVITS_weights_v2\rinne_e8_s456.pth` |

如果 `GPT_weights_v2` 或 `SoVITS_weights_v2` 文件夹不存在，就在语音目录中新建。然后用记事本打开语音目录里的 `GPT_SoVITS\configs\tts_infer.yaml`，把文件最上方的 `custom:` 部分改成下面这样；下面的 `v1:`、`v2:` 等部分保持原样：

```yaml
custom:
  bert_base_path: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: GPT_SoVITS/pretrained_models/chinese-hubert-base
  device: cpu
  is_half: false
  t2s_weights_path: GPT_weights_v2/rinne_e15.ckpt
  version: v2
  vits_weights_path: SoVITS_weights_v2/rinne_e8_s456.pth
```

保存文件。这里的 `custom:`、两个权重路径和 `version: v2` 决定 GPT-SoVITS 实际加载凛祢声线；仅把文件放进文件夹、但不修改这几行，不会使用凛祢的权重。

配置保存后，仍在文件资源管理器中打开**语音目录**，也就是能直接看到 `api_v2.py` 的那个文件夹。单击窗口上方的地址栏，输入 `powershell` 并按回车；新打开的 PowerShell 会自动位于这个文件夹。输入：

```powershell
.\runtime\python.exe .\api_v2.py
```

如果使用 CMD，就在同一个文件夹的地址栏输入 `cmd` 并按回车，然后输入：

```bat
runtime\python.exe api_v2.py
```

让这个语音窗口保持运行；语音服务默认使用本机 `9880` 端口，与 `conf.yaml` 中的语音地址一致。参考 WAV 已包含在项目中，无需从游戏提取。

回到项目目录的命令窗口启动后端，并让它保持运行。五套游戏服装和四套自制服装已随项目下载，不需要另找游戏文件或运行导入命令。

```text
uv run run_server.py
```

后端默认在本机 `127.0.0.1:12393` 提供接口。请使用后文的桌面客户端完成对话和换装。

首次启动时会下载 SenseVoice 识别模型以及记忆检索所需的 `BAAI/bge-base-zh-v1.5`、`BAAI/bge-reranker-base`。保持网络连接并等待下载完成。搜索、音乐与 Library 工具也会随程序启动；新安装的 Library 为空。

### 安装桌面客户端

在后端窗口保持运行的情况下，打开 [Windows 客户端下载页](https://github.com/kaidongli30-cpu/Open-LLM-VTuber-Rinne-Frontend/releases/tag/rinne-desktop-v1.2.4-20260924)，下载与本版后端配套的 64 位安装程序 `open-llm-vtuber-1.2.4-setup.exe`。双击运行，安装过程中可以选择 D 盘等位置；完成后双击桌面快捷方式打开凛祢。先启动后端，再点击客户端与凛祢对话。

安装后依次检查：桌面客户端显示凛祢；Live Mode 可以选择九套服装；输入文字后能收到回复并听到语音；切换灵装后仍能正常对话。若有文字但没有声音，检查 Ollama、GPT-SoVITS 两个窗口和后端日志。若没有画面，请完全退出客户端并重新启动；仍不正常时，在项目目录运行 `uv run python setup_rinne_game_assets.py status` 检查服装文件。

## 3. 代理仅按需设置

### 用 QQ 与凛祢对话（可选）

要接入 QQ，请先完成上述桌面端部署，再按 [QQ 通道安装说明](public_docs/QQ_SETUP.md) 设置自己的 QQ 账号。支持 QQ 官方机器人私聊，也支持通过 NapCat／AstrBot 接入个人 QQ 小号；两种方式都需要自行完成对应平台的安装和登录。桌面端用户不需要安装 QQ 组件。

这个本作者还处在实验阶段，感觉还不能放心地把 QQ 上的凛祢开源出来，总感觉时不时会出点小问题。所以对自己没信心的朋友可以先不让凛祢住进 QQ ，就先让她住在自己的电脑上，感觉想试一试的朋友可以尝试一下，然后把遇到的问题反馈给我。

## 4. 已有用户升级而不是重装

升级前先在文件资源管理器中把 `conf.yaml`、整个 `chat_history` 复制到项目目录之外保存。确认备份可打开后，继续在**原项目目录**升级，不需要删除旧版或把密钥重新填写一遍。

如果你用的是 **1.2.1 客户端**：它不会自己弹出新版提示。得知更新后，打开上方的 Windows 客户端下载页，下载新版 `.exe`，完全退出旧客户端，再运行安装程序；可以选择原来的安装位置。安装完成后，先打开新版客户端。首次检查时选择**原来安装的后端项目文件夹**，也就是你平时打开 `conf.yaml`、运行 `uv run run_server.py` 的文件夹。若客户端要求关闭旧后端，先关闭运行 `run_server.py` 的命令窗口，再点击“开始更新”。客户端会在原目录备份配置、更新后端代码和依赖；完成后按原顺序启动语音服务、后端和客户端。旧 `chat_history`、日记和记忆留在原处，不需要重新填写 API Key。

如果你用的是 **1.2.2 或之后的客户端**：以后有配套的正式版本时，客户端会显示更新提示。点击“更新”后按提示关闭后端、确认项目文件夹；客户端先更新后端，再打开新客户端的安装包页面。下载 `.exe`，完全退出旧客户端后安装。客户端会记住选过的后端文件夹，搬家后也可以重新选择。

更新工具会保留 `conf.yaml` 的个人配置，以及 `diary_generator.py`、`memory_generation_config.py` 中上述指定位置填写的密钥。其余配置只自动合并没有冲突的项目；如果你和新版改了同一项，它会停下来报告冲突，不会覆盖你的选择。如果你自己改动过其他代码，自动更新也会停下来。已有的聊天、日记和私人文件不会被清空。

如果客户端无法完成更新，可先关闭旧后端，在原项目目录的 PowerShell 或 CMD 中依次运行下面的命令。第一条下载更新工具，第二条只检查，确认通过后再运行第三、四条：

```text
curl.exe --fail --location --output rinne-update-now.py https://raw.githubusercontent.com/kaidongli30-cpu/Open-LLM-VTuber-Rinne/main/tools/rinne_safe_update.py
uv run python rinne-update-now.py
uv run python rinne-update-now.py --apply
uv sync
```

更新成功后可删除临时下载的 `rinne-update-now.py`；下次可直接运行项目自带的 `uv run python tools/rinne_safe_update.py --apply`。

已有用户也按“安装 V2 语音”一节检查语音服务及两个权重文件。若已有自己的翻译词表，更新会保留你填写的路径。确认 `ollama list` 有指定模型，重启后端和桌面客户端，检查日常与灵装语音。

### 用已审核日记建立或续写第二层背景

配置更新后，`character_config.layer2_memory_generation.enabled` 为 `True`。只有处理已审核日记时才会调用配置的 DeepSeek API 并产生请求费用。先查看哪些历史日记有资格处理：

```powershell
uv run python -m src.open_llm_vtuber.memory.layer2_backfill --dry-run
```

只有你明确批准的日记才会用于背景更新。若需要逐篇在终端确认：

```powershell
uv run python -m src.open_llm_vtuber.memory.layer2_backfill --approve-interactively
```

命令会按日期顺序处理。至少需要一篇非空、已确认的日记：还没有第二层背景时，先根据第一篇建立背景，再逐篇更新；没有日记或尚未确认时不生成。缺失日期和日记未提及的内容不会被猜测或补写。成功后新对话会读取生成的第二层背景，原始日记仍留在本地。未审核的日记不会被自动批准。

## 5. 常见问题

- **换装菜单没有某套**：在项目目录运行 `uv run python setup_rinne_game_assets.py status` 检查随仓库提供的五套游戏资源；确认启动的是这个项目目录里的后端，并完全重启桌面客户端。九套服装不需要另外导入游戏文件。
- **API 连接失败**：检查自己的 Key、额度与网络；需要代理时，先确认代理服务已启动，再设置本地代理地址。
- **运行时找不到前端**：执行 `git submodule update --init --recursive`，确认 `frontend\index.html` 存在。
- **文字正常但没声音**：确认 Ollama 的 `qwen3.5:4b-q4_K_M` 已安装并运行、两个语音权重已放到指定位置、`GPT_SoVITS\configs\tts_infer.yaml` 的 `custom:` 已按上文修改，以及语音端口与本地 `conf.yaml` 匹配。
- **旧记忆没出现**：确认正在启动的是正确的后端目录、`conf_uid=rinne_01`、`RINNE_DATA_ROOT` 没指错；不要删除旧 `chat_history`。
