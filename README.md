# Open-LLM-VTuber-Rinne

凛祢桌宠支持文字和语音对话、换装、日记与长期记忆。本指南介绍 Windows 桌面客户端的安装和使用。

## 从哪里开始

- 第一次安装：从[准备软件](#prepare)开始，按顺序操作。
- 已经装过：直接看[旧版升级](#upgrade)，不用重新安装后端。
- 已经装好，只想启动：看[每天怎样启动](#daily-start)。
- 遇到问题：看[常见问题](#faq)。

<a id="prepare"></a>
## 1. 准备软件

先安装下面的软件：

| 软件 | 用途 |
| --- | --- |
| [Git for Windows](https://git-scm.com/download/win) | 下载和更新项目 |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | 安装和运行后端所需的 Python 环境 |
| [Ollama](https://ollama.com/download/windows) | 运行翻译和记忆整理用的本地模型 |
| [7-Zip](https://www.7-zip.org/) | 解压语音程序 |

安装完成后，重新打开一个 PowerShell 或 CMD 窗口。它们都是 Windows 的命令窗口，用其中一种即可。

不知道怎样打开？在开始菜单搜索 `PowerShell`，点击打开。

下面的命令请**每次复制一行，按回车，等执行完再输入下一行**。遇到报错，先解决报错，不要继续往下执行。

```text
git --version
uv --version
uv python install 3.12
```

前两行会显示软件版本；第三行安装 Python 3.12。本项目支持 Python 3.10–3.12，推荐使用 3.12。

再启动 Ollama，检查命令是否可用：

```text
ollama --version
```

### 提前留出磁盘空间

翻译模型约 3.4 GB，记忆整理模型约 15 GB。此外还要存放语音程序、约 1 GB 的语音识别模型和记忆检索模型。

**Ollama 可以在设置中选择模型保存位置。** 不想占用 C 盘时，请先改好位置，再下载模型。

<a id="download"></a>
## 2. 下载项目，选择安装位置

下面以 `D:\AI\Open-LLM-VTuber-Rinne` 为例。

想放在其他位置，就把命令中的 `D:\AI` 换成你选的文件夹。即使命令窗口当前显示 C 盘，项目也会下载到你指定的位置。

### 如果用 PowerShell

```powershell
New-Item -ItemType Directory -Force 'D:\AI' | Out-Null
git clone --recurse-submodules https://github.com/kaidongli30-cpu/Open-LLM-VTuber-Rinne.git 'D:\AI\Open-LLM-VTuber-Rinne'
Set-Location 'D:\AI\Open-LLM-VTuber-Rinne'
uv sync
```

### 如果用 CMD

```bat
if not exist "D:\AI" mkdir "D:\AI"
git clone --recurse-submodules https://github.com/kaidongli30-cpu/Open-LLM-VTuber-Rinne.git "D:\AI\Open-LLM-VTuber-Rinne"
cd /d "D:\AI\Open-LLM-VTuber-Rinne"
uv sync
```

两组命令选一组执行，**不需要都执行**。

`--recurse-submodules` 会一起下载项目需要的前端文件，请保留。最后的 `uv sync` 用来安装后端依赖，等待它完成。

下文说的“项目目录”，就是刚下载的 `Open-LLM-VTuber-Rinne` 文件夹，里面能看到 `conf.yaml` 和 `run_server.py`。

项目的 Python 环境会留在这个目录里。其他软件的缓存位置由各自设置决定，仍可能使用 C 盘。

<a id="keys"></a>
## 3. 填写 API 密钥

API Key 就是模型服务提供给你的密钥。把自己的密钥填进对应位置，程序才能调用服务。

### 3.1 对话、背景、视频和搜索

在项目目录找到 `conf.yaml`，右键选择一种合适的打开方式，如果没有Notepad++，就用记事本也可以，推荐下载一个Notepad++并把带这类后缀名的文件用Notepad++打开，这样人眼看的更方便。

（快去下载一个吧，下载完你用Notepad++打开这类文件之后，你会回来感谢我的）

按 `Ctrl+F` 搜索下表中的配置名称，再修改它下面对应的密钥：

| 用途 | 搜索这个名称 | 修改其下的这一项 |
| --- | --- | --- |
| 对话 | `openai_compatible_llm:` | `llm_api_key: ''`，填 APINebula 密钥 |
| 第二层背景等 DeepSeek 调用 | `deepseek_llm:` | `llm_api_key: ''`，填 DeepSeek 密钥 |
| 视频与屏幕观察 | `media_analysis:` | `api_key: ''`，填可调用 Gemini 的密钥 |
| 联网搜索 | `basic_memory_agent:` | `bocha_api_key: ''`，填博查密钥 |

APINebula 是一个国内的海外模型API中转站（应该可以信赖……吧，这个是赞助了国内著名开源项目CCSwitch的），直接搜就能搜到，创建账号后，在“令牌管理”处创建新令牌，也就是API-Key。

对话所用的模型推荐使用 claude-opus-4-6 ,当然你也可以尝试用用 claude-sonnet-4-6 , claude-opus-4-8 ,或者新出的 claude-opus-5-5 （这个更贵，我还没试过，说是更好）,我个人是一直在用 claude-opus-4-6  一次对话大概需要两三分钱吧。分组可以选择 CC-kiro 就足够，基本不会掉线，选择“永不过期”，然后在模型限制列表那里选择令牌能够调用的模型名称。

第二层背景我试过了许多20~30B的本地模型，但是不仅速度慢，质量也不过关。所以最终迫不得已选择了云端方案，继续使用 Deepseek 去生成记忆系统的第二层。

第二层背景就是凛祢会通过每日的日记逐渐明白你当前是什么状态，你是一个怎样的人，你当前与凛祢是什么状态，你正在做什么，未来计划做什么……用于模拟人类之间对对方形成的一个整体印象而不用回忆与对方的交互。

我调用 Gemini 模型的方式依旧是使用 APINebula ，然后因为 Gemini 的多模态做的很好，所以用它来让凛祢能看见视频，看见真正动态的世界。我选用的是 gemini-3.5-flash，视频时长较长或空间较大时，后台会进行分段发送，即使如此，也还是建议不要发送连续几分钟的视频，因为凛祢可能中间有一段视频分析失败。

把对话设置中的：

```yaml
llm_api_key: ''
```

改成：

```yaml
llm_api_key: '你的实际密钥'
```

只替换引号中间的内容，保留原来的缩进、冒号和引号（一定要注意不要改变这些，这种低级错误不要来问我了啊啊啊，还有人填写了密钥后没有保存文件然后来问我出了什么问题的，我真是服了）。

如果 Gemini 也通过 APINebula 调用，视频那一项就填可用于该模型的 APINebula 密钥。

全部填好后，按 `Ctrl+S` 保存。以后修改设置，也要保存并重启后端才会生效。

### 3.2 日记、周记和月记

这三项还需要填写一处 Python 文件。请在启动后端前完成。

1. 在项目目录找到 `diary_generator.py`，用记事本打开。
2. 按 `Ctrl+F` 搜索 `LLM_API_KEY =`。
3. 把这一行改成下面的形式，填入自己的 APINebula 密钥。
4. 按 `Ctrl+S` 保存。

```python
LLM_API_KEY = '你的实际密钥'
```

日记我一直使用的是 `claude-sonnet-4-6`；周记和月记默认使用 `claude-opus-4-6`。三者默认共用刚填的密钥，不必重复填写。

只有想给周记、月记使用另一把密钥时，才需要打开 `memory_generation_config.py`，搜索 `API_KEY =`，把那一行改为：

```python
API_KEY = '另一把实际密钥'
```

只改指定的密钥行，文件其他部分保持不变。项目的更新工具会保留这两处密钥。

## 4. 下载本地模型

确认 Ollama 已启动，并已选好模型保存位置。

先下载翻译模型：

```text
ollama pull qwen3.5:4b-q4_K_M
```

下载完成后，再下载每日记忆整理用的模型（这个模型会用于整理每日你和凛祢之间发生的事情，用于长期记忆检索）：

```text
ollama pull mistral-small3.2:24b
```

最后检查：

```text
ollama list
```

列表里应能看到这两个模型。第二个模型用于把每天的记忆整理成“子事件”，供之后检索。

<a id="voice"></a>
## 5. 安装凛祢 V2 语音

### 5.1 下载并解压语音程序

下载 [GPT-SoVITS 官方 Windows 整合包](https://huggingface.co/lj1995/GPT-SoVITS-windows-package/blob/8b081e1fa1b3ad121e0f310e525dc80fcf15becc/GPT-SoVITS-v2pro-20250604.7z)，用 7-Zip 解压到你想放的位置。

打开解压后的文件夹，找到**同时能看到 `api_v2.py` 和 `runtime` 文件夹**的那一层。下文把它叫作“语音目录”。

如果有两层同名文件夹，就继续打开里面那层，以看到这两个项目为准。语音目录和前面的项目目录不是同一个目录。

### 5.2 下载两份语音权重

打开[凛祢 V2 语音下载页](https://github.com/kaidongli30-cpu/Open-LLM-VTuber-Rinne/releases/tag/rinne-gpt-sovits-v2-20260923)，在页面下方的 Assets 中下载这两个文件。

不要改名，分别放到语音目录里的指定位置：

| 下载文件 | 放进这个文件夹 |
| --- | --- |
| `rinne_e15.ckpt` | `GPT_weights_v2` |
| `rinne_e8_s456.pth` | `SoVITS_weights_v2` |

这两个文件夹应直接位于语音目录下；不存在就新建。

放好后，文件的相对路径应是：

```text
GPT_weights_v2\rinne_e15.ckpt
SoVITS_weights_v2\rinne_e8_s456.pth
```

### 5.3 修改语音配置

在语音目录里，用记事本打开：

```text
GPT_SoVITS\configs\tts_infer.yaml
```

把最上方的 `custom:` 部分改成下面这样。后面的 `v1:`、`v2:` 等部分不要改。

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

按 `Ctrl+S` 保存。**只放入权重文件还不够，必须完成这一步，程序才会加载凛祢的声线。**

参考 WAV 已包含在项目里，不需要另外提取。

<a id="start"></a>
## 6. 启动语音和后端

### 6.1 启动语音服务

在文件资源管理器中打开**语音目录**。

点击上方地址栏，输入 `powershell`，按回车。新窗口会直接打开在这个文件夹里。

执行：

```powershell
.\runtime\python.exe .\api_v2.py
```

如果习惯用 CMD，就在地址栏输入 `cmd`，然后执行：

```bat
runtime\python.exe api_v2.py
```

启动后让这个窗口保持打开。语音服务默认使用本机 `9880` 端口，已经与项目配置对应。

### 6.2 启动后端

另外打开**项目目录**，也就是能看到 `conf.yaml` 和 `run_server.py` 的文件夹。

同样在地址栏输入 `powershell` 或 `cmd`，按回车，然后执行：

```text
uv run run_server.py
```

这个窗口也要保持打开。后端默认使用本机 `127.0.0.1:12393`。

首次启动会下载 SenseVoice 语音识别模型，以及 `BAAI/bge-base-zh-v1.5`、`BAAI/bge-reranker-base` 记忆检索模型。请保持网络连接，等下载和初始化完成。

如果窗口要求审核日记，按提示完成后再继续。搜索、音乐和 Library 工具也会随程序启动；新安装的 Library 还没有文件。

<a id="client"></a>
## 7. 安装桌面客户端，开始对话

1. 打开 [Windows 客户端下载页](https://github.com/kaidongli30-cpu/Open-LLM-VTuber-Rinne-Frontend/releases/tag/rinne-desktop-v2.0.0-20260925)。
2. 下载 `open-llm-vtuber-2.0.0-setup.exe`。
3. 双击安装，可以选择 D 盘等位置。
4. 保持语音和后端运行，双击桌面上的客户端快捷方式。

不需要自己编译客户端，也不需要另外导入服装。

打开后，依次检查：

- 能看到凛祢。
- 在 Live Mode 的换装菜单中能选择九套服装。
- 输入一句话后，能收到文字回复并听到语音。
- 切换灵装后，也能继续对话。

有问题时，先看文末的[常见问题](#faq)。

<a id="daily-start"></a>
## 8. 每天怎样启动

安装步骤只做一次。以后每次使用，按这个顺序：

1. 确认 Ollama 已运行。
2. 在语音目录启动 `api_v2.py`。
3. 在项目目录执行 `uv run run_server.py`，等后端启动完成。
4. 双击桌面客户端，与凛祢对话。

前两个命令窗口都要保持打开。具体命令见[启动语音和后端](#start)。

<a id="upgrade"></a>
## 9. 旧版升级

**在原来的项目目录升级，不需要删除旧版后端，也不需要搬走原有记忆。**

### 9.1 先备份

把项目目录中的 `conf.yaml` 和整个 `chat_history` 文件夹，复制到项目目录之外保存。

确认备份能打开后，再开始更新。要复制，不要剪切。

### 9.2 如果你使用 1.2.1 客户端

1.2.1 不会自动弹出更新提示，需要先手动安装新版客户端。

1. 从[客户端下载页](https://github.com/kaidongli30-cpu/Open-LLM-VTuber-Rinne-Frontend/releases/tag/rinne-desktop-v2.0.0-20260925)下载新版 `.exe`。
2. 完全退出旧客户端，再运行安装程序。可以选择原来的安装位置。
3. 打开新版客户端，按更新提示选择**原来的后端项目目录**，里面应有 `conf.yaml` 和 `run_server.py`。
4. 按提示关闭旧后端窗口，再点击“开始更新”。
5. 等待后端代码和依赖更新完成，再按[日常启动顺序](#daily-start)启动。

这次升级要先打开新版客户端来更新后端，和日常启动顺序不同。

### 9.3 如果你使用 1.2.2 或之后的客户端

有配套的正式新版时，客户端会显示更新提示。

1. 点击“更新”，按提示关闭后端。
2. 确认后端项目目录。第一次选择后，客户端会记住位置；搬家后可以重新选择。
3. 等后端更新完成，再从打开的下载页下载新版 `.exe`。
4. 完全退出旧客户端，运行安装程序。
5. 按[日常启动顺序](#daily-start)重新启动。

### 9.4 更新会保留什么

更新工具会保留：

- `conf.yaml` 中的个人设置和密钥。
- `diary_generator.py`、`memory_generation_config.py` 中前文指定位置填写的密钥。
- 原有聊天、日记、记忆和私人文件。

如果你和新版修改了同一个配置项，工具会停止并提示冲突，不会直接覆盖。

如果你修改过其他程序代码，也会停止，请先处理提示的问题。

更新后，检查 `ollama list` 中有前文指定的两个模型，并按[语音安装说明](#voice)检查 V2 权重和配置。已有翻译词表的路径会保留。

### 9.5 客户端更新失败时，手动更新

先关闭后端，在**原项目目录**打开 PowerShell 或 CMD。

第一步：下载更新工具。

```text
curl.exe --fail --location --output rinne-update-now.py https://raw.githubusercontent.com/kaidongli30-cpu/Open-LLM-VTuber-Rinne/main/tools/rinne_safe_update.py
```

第二步：只检查，暂不更新。

```text
uv run python rinne-update-now.py
```

**检查通过后**，再执行更新。有冲突或报错时，先停在这里处理，不要继续。

```text
uv run python rinne-update-now.py --apply
```

更新成功后，安装依赖：

```text
uv sync
```

完成后按[日常启动顺序](#daily-start)启动，并检查文字回复、日常服装和灵装语音。

临时下载的 `rinne-update-now.py` 可以删除。下次手动更新可使用项目自带的工具：

```text
uv run python tools/rinne_safe_update.py --apply
```

<a id="optional"></a>
## 10. 可选功能

### 从已有日记建立第二层背景

“第二层背景”是从你审核过的日记中整理出的长期背景，帮助凛祢了解你的经历和近况。它不会替换原始日记。

至少需要一篇非空、已审核的日记。没有日记时，不会凭空生成背景。

确认 `conf.yaml` 中 `character_config.layer2_memory_generation.enabled` 为 `True`，并已填写 DeepSeek 密钥。生成会调用 API，产生请求费用。

先在项目目录检查哪些日记可以处理：

```text
uv run python -m src.open_llm_vtuber.memory.layer2_backfill --dry-run
```

需要逐篇确认并生成时，再执行：

```text
uv run python -m src.open_llm_vtuber.memory.layer2_backfill --approve-interactively
```

程序按日期处理：先用第一篇建立背景，再用后续日记更新。

未审核的日记不会被自动批准，缺失的日期不会补写。完成后，新对话会读取生成的背景。

### 用 QQ 与凛祢对话

先完成桌面端安装，再按 [QQ 通道说明](public_docs/QQ_SETUP.md)操作。

可以使用 QQ 官方机器人私聊，或通过 NapCat／AstrBot 接入个人 QQ 小号。需要完成对应平台的安装和登录。

QQ 通道仍在试用阶段，可能遇到问题。只用桌面端时，可以跳过，不必安装 QQ 组件。

### 让凛祢读取你的资料

文件放在哪里、视频密钥怎样填写，见 [Library 使用说明](rinne_library/README.md)。

<a id="faq"></a>
## 11. 常见问题

### 没有凛祢画面，或缺少服装

先确认启动的是这个项目目录的后端，再完全退出并重开客户端。

仍有问题时，在项目目录检查服装文件：

```text
uv run python setup_rinne_game_assets.py status
```

九套服装已包含在项目中，不需要另外导入游戏文件。

### 有文字，但没有声音

依次检查：

1. Ollama 已启动，`ollama list` 中有 `qwen3.5:4b-q4_K_M`。
2. 两份 V2 权重已放在指定位置。
3. `tts_infer.yaml` 的 `custom:` 已按[语音安装说明](#voice)修改。
4. 语音服务窗口没有关闭，端口与 `conf.yaml` 的设置一致，默认是 `9880`。

仍无声音时，查看语音窗口和后端窗口的报错。

### API 连接失败

检查密钥是否填对、账户是否有额度、网络是否能访问对应服务。

默认不需要代理。只有自己的网络确实需要代理时，才配置本地代理地址，并保持代理软件运行。

### 提示找不到前端

在项目目录执行：

```text
git submodule update --init --recursive
```

完成后检查 `frontend\index.html` 是否存在，再启动后端。

### 旧记忆没出现

先确认启动的是原来的后端目录，旧 `chat_history` 仍在里面，不要删除它。

再检查 `conf.yaml` 中的 `conf_uid` 是否仍为 `rinne_01`。如果设置过 `RINNE_DATA_ROOT`（另行指定的数据目录），也要确认它指向原有数据。

## 项目与资源

项目基于 [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)。桌面客户端源码在[前端仓库](https://github.com/kaidongli30-cpu/Open-LLM-VTuber-Rinne-Frontend)。

自制服装按 [CC BY-NC 4.0](assets/rinne-original-outfits/LICENSE.md) 授权，商业使用需另行获得作者许可。
