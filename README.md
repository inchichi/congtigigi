# 나의 예제 롤플레잉 게임

- 타일맵을 기반으로 2D 웹 게임을 만든다.
- Lua 스크립트 언어를 통해 웹 게임 엔진과 스크립팅을 분리하여 개발해본다.

## 시작하기

1. `npm install`
2. `npm run dev`

## 기본 검증

- `npm run test:run`
- `npm run build`

## 스타일 트랜스퍼 (AdaIN)

에디터(`/editor.html`)의 🎨 스타일 변환 기능은 로컬 Python 서비스가 필요하다.

1. `cd style-service && pip install -r requirements.txt` (최초 1회)
2. `python server.py`

자세한 구조·설정은 [docs/style-transfer.md](docs/style-transfer.md) 참고.
