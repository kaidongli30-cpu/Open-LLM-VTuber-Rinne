# 凛祢 V4-B-R1 语音运行包

这个目录只保存桌宠运行所需的语音启动脚本、混合代理和参考音频，不包含训练集、私人对话或 API Key。

V4-B-R1 会同时启动：

- V2 后端：作为漏句时的回退；
- V4 后端：作为主要语音模型；
- `127.0.0.1:9880` 混合代理：桌宠只需要访问这个地址。

## 所需权重

将资源仓库 Release 中的四份权重放进 GPT-SoVITS：

```text
GPT_weights_v2/rinne_e15.ckpt
SoVITS_weights_v2/rinne_e8_s456.pth
GPT_weights_v4/rinne_v4-e10.ckpt
SoVITS_weights_v4/rinne_v4_e1_s1297_l32.pth
```

根目录 README 会给出完整下载和安装步骤。

## 指定 GPT-SoVITS 目录

默认目录是项目根目录下：

```text
GPT-SoVITS-v2pro-20250604/GPT-SoVITS-v2pro-20250604
```

如果放在别处，在 PowerShell 中设置：

```powershell
$env:RINNE_GPT_SOVITS_ROOT = 'D:\GPT-SoVITS-v2pro-20250604\GPT-SoVITS-v2pro-20250604'
```

## 启动与停止

- `切换到V4-B-R1.bat`：启动完整 V4-B-R1。
- `检查保留文件与当前模式.bat`：检查权重、参考音频、进程和端口。
- `停止凛祢语音服务.bat`：只停止本运行包启动的进程。
- `切换回稳定V2.bat`：仅启动 V2 后端。

如果显卡不支持 CUDA，可在启动前设置 CPU 模式；速度会明显下降：

```powershell
$env:RINNE_TTS_DEVICE = 'cpu'
$env:RINNE_TTS_HALF = 'false'
```

运行状态、日志、临时配置和配置备份均在本目录生成，已被 Git 忽略。
