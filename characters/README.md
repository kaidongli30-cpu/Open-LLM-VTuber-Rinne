# 自定义角色配置

这里只在你想添加其他角色时使用。正常使用凛祢，直接修改项目根目录的 `conf.yaml` 即可。

## 配置怎样生效

程序先读取 `conf.yaml`。

在客户端切换角色后，再用该角色文件中的设置覆盖同名配置。没有写的设置继续沿用 `conf.yaml`，不必把整份配置复制过来。

## 添加一个角色

1. 在本目录新建 `my_character.yaml`。
2. 填入下面的示例，再修改角色名称和设定。
3. 保存后，在客户端选择该角色。

```yaml
character_config:
  conf_name: '我的新角色'
  conf_uid: 'my_character_001'
  character_name: '小星'
  persona_prompt: |
    你是小星。你说话温和，认真倾听用户，并给予真诚的回应。
```

这个例子只改变角色设定，其余设置（包括语音）沿用主配置。

## 常用字段

| 字段 | 含义 |
| --- | --- |
| `conf_name` | 客户端角色列表中的名称 |
| `conf_uid` | 区分角色和聊天记录的唯一编号 |
| `character_name` | 对话中显示的角色名称 |
| `persona_prompt` | 角色的性格、背景和说话方式 |
| `avatar` | 头像文件名，图片放在项目的 `avatars` 目录 |
| `live2d_model_name` | Live2D 模型名称，需与 `model_dict.json` 中的名称一致 |

新角色要使用新的 `conf_uid`。**不要为了改显示名称而修改已有凛祢的 `rinne_01` 编号**，否则可能无法继续使用原来的聊天记录。

## 需要更多设置时

语音识别、语音合成和对话模型，分别在 `asr_config`、`tts_config`、`agent_config` 中设置；语音检测使用 `vad_config`。

从 `conf.yaml` 复制需要修改的那一小段，放在角色文件的 `character_config` 下，并保持原有缩进。

添加其他 Live2D 模型时，把模型放进 `live2d-models`，再在 `model_dict.json` 添加对应条目。凛祢已有的九套服装不需要这样操作。
