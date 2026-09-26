# Open-LLM-VTuber-Rinne

린네와 글이나 음성으로 대화하는 데스크톱 앱입니다. 의상 9종, 일기, 장기 기억을 지원합니다.

현재 설치 방법은 [README(중국어)](README.md)에 정리되어 있습니다. Windows에서는 아래 순서로 진행하세요.

## 처음 설치할 때

1. Git, uv, Ollama, 7-Zip을 설치합니다.
2. 원하는 폴더에 프로젝트를 내려받습니다. C 드라이브가 아니어도 됩니다.
3. `conf.yaml`에 API 키를 입력하고, 일기용 키도 설정합니다.
4. Ollama 모델을 내려받습니다.
5. GPT-SoVITS와 린네 V2 음성 모델 파일 2개를 내려받고 설정합니다.
6. 음성 서비스와 백엔드를 실행합니다.
7. 데스크톱 앱의 `.exe` 설치 파일을 실행하고 앱을 엽니다.

의상은 프로젝트에 포함되어 있습니다. 게임 파일을 따로 가져올 필요가 없습니다.

명령어, 파일을 넣을 위치, 설정 예시는 [설치 안내](README.md#prepare)를 참고하세요.

## 이미 설치했다면

평소에는 **Ollama → 음성 서비스 → 백엔드 → 데스크톱 앱** 순서로 실행합니다.

- [실행 방법](README.md#daily-start)
- [이전 버전에서 업데이트](README.md#upgrade)
- [문제 해결](README.md#faq)

업데이트는 기존 프로젝트 폴더에서 진행합니다. 먼저 `conf.yaml`과 `chat_history`를 백업하고, 기존 기억은 삭제하지 마세요.

이 프로젝트는 [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)를 기반으로 합니다. 린네를 설치하거나 업데이트할 때는 위의 린네용 안내를 따라 주세요.

[中文](README.md) | [日本語](README.JP.md)

## 📜 써드 파티 라이센스들 (Third-Party Licenses)

### Live2D 샘플 모델 고지 (Live2D Sample Models Notice)

이 프로젝트에는 **Live2D Inc.에서 제공한 Live2D 샘플 모델**이 포함되어 있습니다. 해당 자산은 **Live2D Free Material License Agreement** 및 **Live2D Cubism Sample Data 이용 약관**에 따라 별도로 라이선스가 부여되며, 이 프로젝트의 MIT 라이선스에는 포함되지 않습니다.

이 콘텐츠는 Live2D Inc.가 소유하고 저작권을 가진 샘플 데이터를 사용하며, Live2D Inc.에서 정한 **약관과 조건**에 따라 활용됩니다. (자세한 내용은 [Live2D Free Material License Agreement](https://www.live2d.jp/en/terms/live2d-free-material-license-agreement/) 및 [Terms of Use](https://www.live2d.com/eula/live2d-sample-model-terms_en.html) 참고)

참고: 특히 중견·대규모 기업에서 **상업적 사용** 시, 이 Live2D 샘플 모델의 사용은 추가 라이선스 요구 사항이 적용될 수 있습니다. 프로젝트를 상업적으로 활용할 계획이라면, 반드시 Live2D Inc.로부터 적절한 허가를 받거나, 해당 모델이 포함되지 않은 버전을 사용하시기 바랍니다.
