# My Sample RPG — Qwen × FLUX 게임 자동화 에디터

자연어로 게임의 테마, 퀘스트, 아트 스타일을 설계하고 실제 게임에 적용하는 2D RPG 개발 자동화 에디터입니다.

현재는 `My Sample RPG`에 최적화되어 있으며, TypeScript·Vite·PixiJS 게임과 Python 기반 FLUX 스타일 서비스를 연결합니다.

## 핵심 기능

- 라이브 게임 미리보기와 맵·NPC·오브젝트 자산 트리
- 한국어 자연어 기반 테마 방향·퀘스트·스타일 대상 생성
- 로컬 Qwen(`Qwen3.6-27B`)을 이용한 구조화 JSON 생성
- FLUX.1-Kontext 기반 정적 오브젝트·NPC 초상화 스타일 변경
- 한국어/혼합 프롬프트를 FLUX 적용 직전에 자연스러운 영어로 변환
- 테마 작업실의 단계별 검토 및 재생성
- 최종 적용 전 JSON 검증과 dry-run
- 변경 오브젝트 비포/애프터 확인 및 테마 JSON 내보내기
- 기존 원본 자산 백업 및 적용 결과 복원

## 실행

```bash
npm install
npm run dev
```

- 게임: <http://localhost:5173/>
- 기본 에디터: <http://localhost:5173/editor.html>
- 테마 작업실: <http://localhost:5173/editor.html?workspace=theme>

에디터 설정에서 모델을 `Qwen`으로 선택하고 로컬 키 `qwen-local`을 사용합니다. Qwen과 FLUX 서비스가 별도 GPU 서버에서 실행되는 경우에는 Vite 개발 서버가 해당 서비스 포트로 연결된 SSH 포워딩을 사용합니다.

## 테마 작업 흐름

1. 기본 에디터에서 `테마 변경 시작`을 누릅니다.
2. 테마 작업실에서 한국어 프롬프트와 퀘스트 의뢰 NPC를 입력합니다.
3. Qwen이 테마 방향을 생성합니다.
4. 결과를 사용하거나 같은 단계에서 새로 생성합니다.
5. Qwen이 게임 구조에 맞는 퀘스트를 생성합니다.
6. Qwen이 FLUX에 적용할 정적 스타일 대상을 최대 2개 선택합니다.
7. 최종 JSON 검증 후 FLUX 적용과 퀘스트 저장을 실행합니다.

한국어 입력은 Qwen이 이해하고, FLUX 요청 직전에 비영어 스타일 문장만 Qwen이 영어로 변환합니다. 퀘스트·게임 데이터와 에디터 UI는 별도로 유지됩니다.

## 서비스 구성

| 구성 | 기본 주소 | 역할 |
|---|---:|---|
| Vite 에디터 | `5173` | 게임, 에디터, API 프록시 |
| Qwen OpenAI 호환 서버 | `8000` | 테마·퀘스트·번역 JSON 생성 |
| FLUX 스타일 서비스 | `8765` | 이미지 스타일 변경 및 오브젝트 추출 |

Vite 프록시 경로:

- `/api/qwen` → Qwen 서버
- `/api/style` → `style-service/server.py`
- `/api/openai`, `/api/anthropic` → 외부 LLM 선택 사용

FLUX 서비스만 별도로 실행하려면:

```bash
cd style-service
pip install -r requirements.txt
python server.py
```

## 안전한 적용 범위

- 현재 JSON 스키마는 `schema_version: 1`, `game_id: "my-sample-rpg"`로 고정되어 있습니다.
- FLUX 대상은 추출된 맵 오브젝트 또는 정적 PNG 자산만 허용합니다.
- 런타임 애니메이션 시트, motion/spritesheet 파일, 맵 타일셋은 덮어쓰지 않습니다.
- 기존 픽셀 해상도, 투명도, 캔버스 크기, 프레임 구조, 오브젝트 실루엣을 보존합니다.
- 적용 전 원본 백업을 만들며, 실패한 대상은 다른 대상의 적용을 막지 않고 결과에 표시합니다.

## 주요 명령

```bash
npm run build       # TypeScript 검사 + Vite 빌드
npm run test:run    # Vitest 단일 실행
npm run check       # TypeScript 검사 + 전체 테스트
npm run lua:build   # Lua WASM 빌드가 필요한 경우
```

## 주요 디렉터리

```text
src/games/my-sample-rpg/   PixiJS 게임 런타임
src/editor/                에디터, Qwen 파이프라인, 테마 작업실
style-service/             FastAPI FLUX 스타일 서비스
docs/                      파이프라인 및 설계 문서
public/                    게임 및 외부 런타임 자산
```

자세한 Qwen·FLUX 설계는 [`docs/qwen-flux-pipeline.md`](docs/qwen-flux-pipeline.md)를 참고하세요.

## 저장소

- GitHub: <https://github.com/inchichi/congtigigi>
- 작업 브랜치: `7.22`
