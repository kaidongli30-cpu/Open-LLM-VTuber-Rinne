# Library 资料库

Library 是凛祢读取文档、图片和视频的地方。新安装时这里没有资料，需要你自己放入文件。

## 文件放在哪里

第一次启动后端后，会创建以下目录：

| 内容 | 项目中的位置 |
| --- | --- |
| 文档 | `rinne_library\rinne_01\documents\待整理` |
| 图片 | `rinne_library\rinne_01\images\待整理` |
| 视频 | `rinne_library\rinne_01\videos\待整理` |

可以把支持的文件复制到对应文件夹，或通过程序的上传功能添加。

文档支持 TXT、MD、PDF 和 DOCX。聊天时可以告诉凛祢文件名，请她查找或阅读。

程序通过 Library MCP 工具让凛祢列出文件、搜索文件名、读取文档和查看图片。MCP 在这里就是连接对话模型与资料库的工具接口。

## 使用视频观察

先完成[主安装指南](../README.md)，再检查下面三项：

1. 电脑能运行 `ffmpeg` 和 `ffprobe`。
2. `conf.yaml` 的 `media_analysis:` 下，`enabled` 为 `True`。
3. 同一段的 `api_key: ''` 已填入可调用 Gemini 的密钥。

密钥直接填在 `conf.yaml` 的引号中，保存后重启后端。具体填写方法见[密钥说明](../README.md#keys)。

可以在 PowerShell 或 CMD 检查视频工具：

```text
ffmpeg -version
ffprobe -version
```

如果提示找不到命令，需要先安装 FFmpeg，并把包含这两个程序的文件夹加入 Windows 的 Path 环境变量，然后重新打开命令窗口。Path 用来让系统找到这些程序，与 API 密钥无关。

### 视频怎样保存和分析

普通上传模式会保存一份较省空间的 720p H.264/AAC 视频；原始模式保留通过格式检查的源文件副本。

长视频会先分段，再交给视频模型观察。静态图片仍交给对话模型查看。

完成的观察会保存在资料库中，之后可以再次读取。视频文件变了，就不能再使用旧观察。

针对新问题重新分析，与读取此前的观察不同：旧观察只能说明当时看到了什么，不代表已经回答了新的问题。

## 可选：按含义搜索文件名

直接按文件名搜索不需要额外设置。

按含义搜索需要 `sentence-transformers` 和本地的 `BAAI/bge-base-zh-v1.5` 模型。缺少它们时，工具会自动改用普通文件名搜索，不会自行下载模型。

如需指定模型位置，可设置 `RINNE_LIBRARY_MODEL_CACHE`；指定模型使用 `RINNE_LIBRARY_EMBEDDING_MODEL`。正常安装可先使用默认设置。

## 上传接口（供自行接入时使用）

上传地址是 `POST /library/upload`。

视频参数 `video_mode=normal` 表示普通模式，也是默认值；`video_mode=original` 表示原始模式。
