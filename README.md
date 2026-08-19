# Open-LLM-VTuber-Rinne

这是一个以 **Open-LLM-VTuber** 为基础整理出的凛祢桌宠版本。它保留了当前可运行状态中的 Live2D、云端 LLM 对话、语音识别、GPT-SoVITS 日语语音、表情参考音频，以及“聊天记录 + 日记 + 周记 + 月记”长期记忆流程。

本仓库不包含任何作者的 API Key、私人聊天记录、私人日记/周记/月记或个人电脑绝对路径。第一次安装时，你需要填入自己的 API，并下载 GPT-SoVITS 与凛祢语音权重。

> 本教程目前以 **Windows 10/11 64 位** 为主要验收环境。macOS/Linux 用户请参考上游项目文档。

## 你会得到什么

- 凛祢 Live2D 模型、头像与表情映射
- 凛祢桌面客户端与透明桌宠模式
- DeepSeek、OpenAI、Claude、Gemini、智谱、Ollama、LM Studio 等多种 LLM 接口
- 本地 SenseVoice 语音识别（首次使用时可能自动下载模型）
- GPT-SoVITS V2 日语语音
- 中文回复先经本地 Ollama 翻译成日语，再交给 GPT-SoVITS
- 聊天记录、日记、周记、月记全部作为云端 LLM 上下文的当前稳定记忆方案
- 局域网或 HTTPS 隧道访问，方便手机聊天

## 仓库内容说明

本仓库直接公开后端代码、记忆系统、配置模板、Live2D 运行资源、参考音频和语音管理脚本。

`Open-LLM-VTuber-Web` 的前端源码不在本仓库重复发布。`frontend` 是一个 Git submodule（子模块，可以理解为“指向另一个 Git 仓库特定版本的引用”），指向上游公开的前端 **build 分支**。后端需要这些构建好的网页文件，因此克隆时必须带 `--recursive`。

桌面 `.exe` 不在本仓库重复打包。请从上游 [Open-LLM-VTuber Releases](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/releases) 下载 Windows 安装程序。

## 一、安装前确认

### 1. 建议硬件

- Windows 10/11 64 位
- 至少 16 GB 内存更稳妥
- 至少预留 15 GB 磁盘空间（GPT-SoVITS 整合包、依赖与模型会占用较多空间）
- 推荐 NVIDIA 显卡运行 GPT-SoVITS；无 CUDA 时可以使用 CPU，但语音生成会明显变慢
- 麦克风和扬声器/耳机

### 2. 检查前置软件

打开 PowerShell，逐条执行：

```powershell
git --version
uv --version
ffmpeg -version
```

如果命令能正常显示版本，说明程序已进入 PATH。

#### Git

从 [Git for Windows](https://git-scm.com/download/win) 安装，或执行：

```powershell
winget install --id Git.Git -e
```

安装后关闭并重新打开 PowerShell。

#### uv 与 Python

本项目要求 Python `>=3.10,<3.13`，推荐用 uv 安装 Python 3.12，不需要自己维护虚拟环境。

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv python install 3.12
```

uv 的官方安装说明见 [Astral uv 文档](https://docs.astral.sh/uv/getting-started/installation/)。

#### FFmpeg

可以先查找可用的软件包，再安装：

```powershell
winget search ffmpeg
winget install --id Gyan.FFmpeg -e
```

也可以从 [FFmpeg 官方下载页](https://ffmpeg.org/download.html) 选择 Windows 构建。安装后再次运行 `ffmpeg -version`。

#### Ollama（完整日语语音需要）

从 [Ollama Windows 下载页](https://ollama.com/download/windows) 安装。安装完成后打开新的 PowerShell：

```powershell
ollama --version
ollama pull qwen3.5:4b-q4_K_M
ollama list
```

当前 `conf.yaml` 默认启用 Ollama 翻译。如果只想先验证文字聊天，可以临时把 `translator_config` 下的 `translate_audio: True` 改为 `False`；这时中文回复不会先翻译成日语，不建议把它当作最终语音效果。

## 二、下载项目并安装 Python 依赖

打开 PowerShell，依次执行：

```powershell
git clone --recursive https://github.com/kaidongli30-cpu/Open-LLM-VTuber-Rinne.git
cd Open-LLM-VTuber-Rinne
uv sync
```

`--recursive` 会同时下载运行需要的 `frontend`。不要使用 GitHub 的“Download ZIP”。

然后打开 [Open-LLM-VTuber Releases](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/releases)，下载并安装 Windows 的 `open-llm-vtuber-1.2.1-setup.exe`。

## 三、配置云端 LLM API

API 是程序调用云端大模型的接口。凛祢需要通过 API 把用户输入和记忆发送给大模型，再取得回复。

本项目支持多种 LLM 接口。下面以 DeepSeek 为例；如果使用其他服务商，请在 `conf.yaml` 中选择对应的 `llm_provider`，并填写对应配置块。

### 1. 获取 DeepSeek API Key

打开 [DeepSeek API Key 管理页面](https://platform.deepseek.com/api_keys)，登录后创建一个新的 API Key。Key 只会完整显示一次，请复制并妥善保存。

### 2. 配置凛祢的主体大脑

用记事本打开项目根目录的 `conf.yaml`，找到 `basic_memory_agent`，把服务商改成：

```yaml
llm_provider: 'deepseek_llm'
```

继续向下找到 `deepseek_llm`，填入刚刚申请的 Key：

```yaml
deepseek_llm:
  llm_api_key: '把你的 API Key 粘贴在这里'
  model: 'deepseek-v4-flash'
```

DeepSeek 的接口地址已经由程序配置好，普通用户不需要另外填写。其他服务商的配置方法相同：选择对应的 `llm_provider`，再填写该服务商配置块中的 Key 和模型名。

### 3. 配置日记、周记和月记

打开项目根目录的 `diary_generator.py`，找到“LLM API 配置”，填写：

```python
LLM_API_KEY = "把你的 API Key 粘贴在这里"
LLM_API_URL = "https://api.deepseek.com/chat/completions"
LLM_MODEL = "deepseek-v4-flash"
```

区别是：主体大脑使用项目内置的服务商接口；日记生成器会直接发送网络请求，所以必须填写完整的 `/chat/completions` 地址。

周记和月记默认沿用上面的日记 API，不需要重复填写。如果希望周记和月记单独使用另一个 API，请打开 `memory_generation_config.py`，把 `API_KEY`、`BASE_URL` 和 `MODEL` 三项替换成另一套服务商信息；其中 `BASE_URL` 同样要填写完整的 `/chat/completions` 地址。

不要把含有真实 API Key 的 `conf.yaml`、`diary_generator.py` 或 `memory_generation_config.py` 上传到公开仓库。录制演示视频时使用临时 Key，并在录制完成后立即删除该 Key。

### 4. 配置联网搜索（博查 API）

联网搜索使用博查 API。获取博查 API Key 后，打开项目中的：

```text
src\open_llm_vtuber\mcpp\tool_executor.py
```

找到下面这一行：

```python
BOCHA_API_KEY = os.getenv("BOCHA_API_KEY", "")
```

把它改成下面这样，并将引号中的文字替换成你自己的 Key：

```python
BOCHA_API_KEY = "把你的博查 API Key 粘贴在这里"
```

保存文件即可。不要把这个 Key 填到 `conf.yaml`，也不要把填写了真实 Key 的文件再次上传到 GitHub。录制演示视频时可以临时使用一个新 Key，录制完成后立即在博查平台删除或禁用它。

## 四、安装 GPT-SoVITS 与凛祢语音

### 1. 下载 GPT-SoVITS 整合包

打开 [GPT-SoVITS 官方 Release](https://github.com/RVC-Boss/GPT-SoVITS/releases/tag/20250606v2pro)：

- 普通 NVIDIA 显卡下载第一个 `windows 7z package download`；
- RTX 50 系显卡下载第二个 `windows 7z package (for 50x0 Nvidia GPU) download`。

下载完成后解压即可。整合包已经包含 GPT-SoVITS 所需的 Python 运行环境，不需要再单独安装依赖。

把解压得到的 `GPT-SoVITS-v2pro-20250604` 文件夹放到本项目根目录。确认里面可以看到 `runtime` 文件夹和 `api_v2.py`：

```text
Open-LLM-VTuber-Rinne\
└─ GPT-SoVITS-v2pro-20250604\
   ├─ runtime\python.exe
   ├─ api_v2.py
   └─ GPT_SoVITS\
```

如果解压软件额外生成了一层同名文件夹，启动脚本也会自动识别。

### 2. 下载两个凛祢 V2 权重

分别下载：

- [下载 rinne_e15.ckpt](https://github.com/kaidongli30-cpu/Open-LLM-VTuber-Rinne-Assets/releases/download/rinne-voice-v1/rinne_e15.ckpt)
- [下载 rinne_e8_s456.pth](https://github.com/kaidongli30-cpu/Open-LLM-VTuber-Rinne-Assets/releases/download/rinne-voice-v1/rinne_e8_s456.pth)

把两个文件分别放到：

```text
GPT-SoVITS-v2pro-20250604\
├─ GPT_weights_v2\
│  └─ rinne_e15.ckpt
└─ SoVITS_weights_v2\
   └─ rinne_e8_s456.pth
```

运行需要的参考音频已经包含在本仓库中，不需要再下载原始游戏音频或训练数据。

### 3. 启动语音服务

双击：

```text
Rinne_model\rinne_voice_runtime_bundle\启动凛祢语音服务.bat
```

脚本显示“切换完成：v2”后，保持窗口运行。需要停止时，双击同一目录下的 `停止凛祢语音服务.bat`。

## 五、第一次启动并开始聊天

1. 打开 Ollama，确认 `ollama list` 中存在 `qwen3.5:4b-q4_K_M`。
2. 双击 `Rinne_model\rinne_voice_runtime_bundle\启动凛祢语音服务.bat`。
3. 在项目根目录打开 PowerShell，运行：

```powershell
uv run run_server.py
```

4. 后端启动完成后，双击桌面的 `open-llm-vtuber` 应用图标。
5. 第一次加载语音识别模型可能需要等待下载，请保持后端窗口运行。

### 验收清单

按顺序确认：

- 桌面客户端能显示凛祢 Live2D，而不是空白或默认角色
- 输入文字后，云端 LLM 能返回回复
- 回复字幕正常，能够听到日语语音
- 点击麦克风并授权后，语音能被识别为文字
- `chat_history` 下产生新的本地会话数据
- 达到日记/周记/月记生成条件后，文件只写入本机；API 请求成功
- 关闭并重新启动后，历史会话仍能继续，记忆上下文能够进入新对话

如果桌面客户端能够显示凛祢，并且输入文字后能收到回复，就已经完成最基本的安装。

## 六、Windows 桌面客户端说明

第二章已经要求安装上游桌面客户端。它只是桌面显示和交互界面；LLM、记忆、TTS 和凛祢角色配置仍由本仓库后端提供。

第一次聊天成功后，可以在桌面客户端中切换到透明背景的桌宠模式。

## 七、手机部署

### 方案 A：同一局域网，先验证页面与文字聊天

电脑和手机连接同一个路由器。电脑执行：

```powershell
ipconfig
```

找到电脑的 IPv4 地址，例如 `192.168.1.20`。允许 Windows 防火墙中 Python/本项目通过专用网络后，手机打开：

```text
http://192.168.1.20:12393
```

因为浏览器通常只允许在 `https` 或 `localhost` 安全环境使用麦克风，所以局域网 HTTP 更适合先验证页面和文字聊天；手机语音请用下面的 HTTPS 方案。

### 方案 B：Cloudflare Quick Tunnel，适合首次测试

安装 [cloudflared](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/)，保持后端运行，然后执行：

```powershell
cloudflared tunnel --url http://localhost:12393
```

终端会显示一个随机 `https://xxxx.trycloudflare.com` 地址。用手机打开该地址；若前端要求单独填写服务器地址：

- Base URL：`https://xxxx.trycloudflare.com`
- WebSocket：`wss://xxxx.trycloudflare.com/client-ws`

手机第一次进入页面后点击一次页面，并允许麦克风权限，这也能解除移动浏览器的自动播放限制。

Quick Tunnel 是临时测试通道：地址每次可能改变，终端关闭后失效，不适合长期公开服务。官方说明见 [Cloudflare Quick Tunnels](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)。

### 方案 C：固定域名的 Named Tunnel

长期使用需要 Cloudflare 账号及一个接入 Cloudflare 的域名：

```powershell
cloudflared tunnel login
cloudflared tunnel create rinne
cloudflared tunnel route dns rinne rinne.你的域名.com
```

在 `%USERPROFILE%\.cloudflared\config.yml` 写入创建命令返回的 Tunnel ID 和凭据路径：

```yaml
tunnel: 你的-Tunnel-ID
credentials-file: C:\Users\你的用户名\.cloudflared\你的-Tunnel-ID.json

ingress:
  - hostname: rinne.你的域名.com
    service: http://localhost:12393
  - service: http_status:404
```

启动：

```powershell
cloudflared tunnel run rinne
```

然后手机访问 `https://rinne.你的域名.com`。不要把 `.cloudflared` 中的凭据文件上传到 GitHub。

## 八、常见问题

### 页面空白或 frontend 目录为空

```powershell
git submodule sync --recursive
git submodule update --init --recursive
```

### API 报 401/403

检查 `conf.yaml` 中选择的 `llm_provider`、对应配置块里的 Key 和模型名是否正确。不要在 Issue 中粘贴完整 Key。

### 日记生成失败但主对话正常

主对话和日记使用的 URL 形式不同。确认 `diary_generator.py` 中的 `LLM_API_URL` 是完整 `/chat/completions` 地址，并检查 `LLM_API_KEY` 与 `LLM_MODEL`。

### 9880 端口被占用

```powershell
Get-NetTCPConnection -LocalPort 9880 -ErrorAction SilentlyContinue
```

不要直接结束不明进程。先确认它是否是你已启动的 GPT-SoVITS；若是本项目语音管理器启动的实例，使用 `停止凛祢语音服务.bat`。

### 没有日语语音

依次检查：

```powershell
ollama list
Get-NetTCPConnection -LocalPort 9880 -ErrorAction SilentlyContinue
```

同时查看语音服务和后端窗口是否出现翻译超时、权重缺失或参考音频路径错误。

### 手机能打开但麦克风不可用

不要使用普通局域网 HTTP 做最终语音测试。改用 Cloudflare Tunnel 等 HTTPS 入口，并在浏览器地址栏权限中允许麦克风。

## 九、更新、备份与回退

更新前先备份自己的本地数据：

- `chat_history`
- 本地日记/周记/月记文件
- 自己修改过的 `conf.yaml`
- 自己填写过 API 的 `diary_generator.py` 和 `memory_generation_config.py`

查看当前版本：

```powershell
git status
git log -1 --oneline
```

不要在有未提交私人改动时直接强制重置。若只想安全试用新版本，重新克隆到另一个目录，再把本地配置和记忆复制过去。

## 十、隐私与开源资源说明

- API Key 只应保存在自己的本机配置中，不要上传、截图或提交到公开仓库。
- 聊天记录和日记类文件默认属于用户私人数据，不应提交到公开仓库。
- 当前记忆方案会把聊天记录、日记、周记、月记发送给你配置的云端 LLM。使用前请自行确认服务商的隐私政策。
- Live2D 相关资源还受仓库内 `LICENSE-Live2D.txt` 约束。
- 项目代码沿用上游许可证，详见 `LICENSE.txt`。
- GPT-SoVITS、上游桌面客户端及其模型/资源各自遵循原项目许可；请在再分发前分别核对。

## 致谢与上游项目

- [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)
- [Open-LLM-VTuber-Web](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber-Web)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- 安装教程结构参考：[Open-LLM-VTuber-ATRI](https://github.com/ZL-Tian/Open-LLM-VTuber-ATRI)

这个仓库是可继续研究的开源基线，不代表上游项目的官方角色发行版。
