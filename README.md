# 마을 이야기 공방 · My Sample RPG

타일맵 기반 2D 웹 RPG와, 그 게임의 콘텐츠를 자연어로 만드는 **LLM 시나리오 에디터**, 그리고 스프라이트를 신경망으로 다시 칠하는 **스타일 변환 서비스**를 한 저장소에 담은 캡스톤 프로젝트입니다.

특징은 게임의 **순수 규칙/로직을 Lua(WebAssembly)로 전환**해, 엔진/렌더(TypeScript·PixiJS)와 게임 규칙(Lua)을 분리한 점입니다.

세 가지 축으로 구성됩니다.

| 축 | 위치 | 역할 |
|---|---|---|
| 🎮 **게임** | `src/games/my-sample-rpg` | PixiJS v8로 렌더하는 탑다운 액션 RPG (Tiled 맵) |
| 🛠 **에디터** | `src/editor` (`/editor.html`) | NPC 대사·퀘스트·배치·스타일을 자연어로 만드는 LLM 저작 도구 |
| 🎨 **스타일 서비스** | `style-service` (Python) | AdaIN 신경망 스타일 변환 + 오브젝트 누끼 추출 |

---

## 빠른 시작

> 요구사항: **Node.js 20.19+ / 22.12+** (Vite 7), npm. (스타일 변환을 쓰려면 Python 3 + PyTorch 추가 — 아래 참고)

```bash
cd chichi
npm install
npm run dev
```

- 게임: <http://localhost:5173/>
- 에디터: <http://localhost:5173/editor.html>

> 하나의 Vite dev 서버가 두 페이지를 함께 서빙합니다(별도 라우터 없이 파일 기반). 포트를 따로 지정하지 않아 Vite 7 기본값 **5173**을 씁니다.

---

## 🎮 게임 (`src/games/my-sample-rpg`)

PixiJS v8로 렌더하는 **탑다운 2D 액션 RPG**. 맵은 직교(orthogonal) **Tiled TMX**(50×50, 32px)이고 외부 `.tsx` 타일셋을 참조합니다.

- **3개 씬** — `town` / `hunting-ground` / `cave`, 각 씬 진입 시 인트로 타이틀 표시
- **NPC 대화/상호작용** — Tiled 이벤트 레이어에서 스폰, 일부 NPC는 Lua 컨트롤러 스크립트(`reply-with-message`, `wander-near-home`)로 동작
- **실시간 전투** — HP 기반 몬스터 전투(레벨 스케일), 처치 시 경험치·골드·스킬포인트·장비 드롭
- **성장/인벤토리** — 레벨업(XP 곡선), 5종 장비 슬롯, 슬롯형 인벤토리, 스킬(`smash`·`protect`, Q/W/E/R), 소비 퀵슬롯 6칸(1–6)
- **상점** — 물약 상점 / 대장간(구매·판매)
- **퀘스트** — 다단계 퀘스트 체인(q001–q008), 목표 타입(처치/사용/상점/씬진입/대화)과 퀘스트 진행 상태
- **조작/세이브** — 재바인딩 가능한 키 설정, 플레이어 상태(프로필·장비·인벤·퀵슬롯·스킬슬롯)를 `localStorage`에 버전드 포맷으로 저장/복원
- **회피/스킬 모션** — 방향 구르기(대시), 검 잔상 스매시

> 게임은 단독 실행되지만, 에디터가 `<iframe>`으로 임베드하면 이를 감지(`window.self !== window.top`)해 오디오를 강제 음소거하고, 에디터의 씬 전환·배치·이벤트 초안 메시지를 받습니다. (참고: 코드상 BGM은 기본 비활성 — SFX만 재생)

---

## 🛠 에디터 (`src/editor` · `/editor.html`)

비개발자가 **자연어로 게임 콘텐츠를 만드는** LLM 저작 도구("마을 이야기 공방"). 임베드된 라이브 게임을 보면서 작업합니다.

- **라이브 미리보기** — 게임을 iframe으로 띄우고 `postMessage`로 통신(에디터는 게임 런타임 코드를 직접 import하지 않음)
- **LLM 생성** — API 키 접두사로 **Claude/GPT 자동 감지**, 공급자별 모델 선택. 키 검증·호출은 로컬 프록시(`/api/anthropic`, `/api/openai`) 경유
- **생성 → 검증(dry-run) → 적용** 파이프라인, 퀘스트는 후보 생성 → 선택 → 이벤트 JSON 생성의 2단계
- **현재 맵 에셋 트리** — 현재 맵의 NPC·건물·오브젝트 목록(검색·전체 맵 토글)
- **배치 팔레트** — 타일/오브젝트/NPC를 마우스로 맵에 배치, 맵별 `localStorage` 영속, NPC 수기 추가(외형·이름·대사)
- **스타일 변환 UI** — AdaIN 스타일 변환 모달 + 원본 복원 + 외부 게임 스프라이트 변환
- **평가/지표** — 생성 결과 수용/거부 평가, 세션 생성·검증 통과율 집계
- **에디터 편의(이번 추가)** — 하단 입력창(컴포저) **드래그 높이 조절**, 트리·팔레트 항목 **마우스 호버 시 확대 미리보기 툴팁**

> 타일 스타일 변환·NPC 팔레트·타일셋 미리보기는 현재 맵이 my-sample-rpg 에셋일 때만 활성화됩니다. LLM 호출은 dev 서버 프록시가 필요하고, 스타일 변환은 아래 Python 서비스가 떠 있어야 합니다.

---

## 🎨 스타일 변환 서비스 (`style-service`)

게임 스프라이트/타일/타일셋에 **AdaIN 신경망 스타일 변환**을 적용하는 로컬 Python(FastAPI/uvicorn) 서비스. 형제 폴더의 `ADAIN` PyTorch 프로젝트를 감싸며, Vite 프록시 `/api/style → 127.0.0.1:8765`로 에디터와 연결됩니다.

```bash
cd chichi/style-service
pip install -r requirements.txt   # 최초 1회 (fastapi, uvicorn, python-multipart)
python server.py                  # 127.0.0.1:8765
```

- **전체/부분 스타일 변환** — 알파 보존, 변환 크기 제한, 경계 침식 옵션
- **오브젝트 누끼 추출** — 타일셋의 알파 타일을 합성해 투명 PNG로 추출(에디터가 맵 인식 시 자동 호출). *신경망 세그먼테이션이 아니라 타일 알파 합성 방식*
- **원본/백업/되돌리기** — 에셋별 원본 1회 시드 + 타임스탬프 백업, 항상 원본에서 다시 칠해 색 누적 방지
- **몬스터 시트** — 배경 보존 + 전경만 변환(프레임 슬라이싱 유지)
- **외부 게임** — `config.json`의 `lol`(Legend of Lua, Love2D) 에셋을 별도 네임스페이스로 변환

> ⚠️ **PyTorch(`torch`)가 필요**하지만 `requirements.txt`에는 없습니다(형제 `ADAIN` 쪽/기존 설치 가정). 이 환경은 GPU가 없어 CPU 추론으로 동작합니다. 가중치(`ADAIN/models/*.pth`)가 없으면 `/health`가 `degraded`로 보고합니다. 자세한 내용은 [docs/style-transfer.md](docs/style-transfer.md).

---

## ⚙️ Lua 아키텍처 (게임 로직의 WASM 전환)

게임의 **순수·결정적 로직**은 **Lua 5.3.6(WebAssembly)** 로 전환되어, `luaLogicHost.ts`가 만드는 **하나의 공유 Lua VM** 안에서 실행됩니다.

- **퍼사드** `lua/luaGameLogic.ts` 가 원본 TS와 **동일한 시그니처**로 함수를 내보내, 게임 호출부는 import 출처만 바꾸면 됩니다. `initLuaGameLogic()` 이전(또는 실패 시)에는 **TS 구현으로 폴백**합니다.
- **동등성 검증** — 각 모듈의 `*Lua.bridge.spec.ts`가 Node에서 실제 WASM VM을 띄워 **Lua 출력 == TS 출력**을 다양한 입력으로 확인합니다.
- **Lua인 것** — 몬스터(보상·전투·표시명·드롭 인덱스), 플레이어(스탯·성장·경험치·인벤·장비·소비·장착·스킬·스킬슬롯·퀵슬롯), 상점(물약·대장간), 퀘스트, 컨트롤/이동/구르기/스매시, 세이브 직렬화, 씬 인트로, 이벤트 검증/초안 등 (`assets/lua/*.lua`, 27개 래퍼).
- **TS(호스트)로 남는 것** — 렌더(PixiJS)·DOM·입력·오디오·네트워크는 본래 호스트 계층. 그 외 **동등성 검증이 불가능한** 것: 브라우저/번들러 API, 비결정적 LLM/네트워크, 그리고 **가변 횟수 난수**(`monsterPatrol.stepMonsterPatrol` — JS `Math.random` 시퀀스를 Lua에서 재현 불가). 정적 상수/데이터도 TS 소유로 직접 re-export.

```bash
# 게임 로직 Lua↔TS 동등성 스펙 실행 (전용 config)
npx vitest run --config vitest.lua-logic.config.ts

# Lua WASM 재빌드 (Emscripten)
npm run lua:build      # 또는 fetch+build: npm run lua:sync
```

> 주의: `npm run lua:bridge:test`는 `vitest.lua-bridge.config.ts`를 써서 **캐릭터 컨트롤러 런타임 스펙 하나만** 실행합니다. 게임 로직 동등성 스펙은 위 `vitest.lua-logic.config.ts`로 직접 실행하세요. 기본 `npm test`/`test:run`은 `*.test.ts`만 포함하고 `*.bridge.spec.ts`는 제외합니다.

---

## 📂 폴더 구조

```
chichi/
├─ index.html              # 게임 진입 (#app → src/games/my-sample-rpg/main.ts)
├─ editor.html             # 에디터 진입 (#editor-root → src/editor/editorPage.ts)
├─ vite.config.ts          # 멀티 페이지 빌드 + /api 프록시(openai·anthropic·style)
├─ vitest.lua-logic.config.ts   # 게임 로직 Lua 브리지 스펙용
├─ vitest.lua-bridge.config.ts  # 캐릭터 컨트롤러 런타임 브리지 스펙용
├─ src/
│  ├─ games/my-sample-rpg/
│  │  ├─ main.ts                 # 게임 엔트리(부팅·세이브·에디터 브리지·오디오)
│  │  ├─ rendering/              # PixiJS v8 렌더러·게임 루프·HUD/오버레이
│  │  ├─ tiled/                  # Tiled TMX/TSX 파서
│  │  ├─ interaction/ · events/  # 상호작용·이벤트
│  │  ├─ lua/                    # Lua 호스트·퍼사드·래퍼 + *.bridge.spec.ts
│  │  └─ assets/
│  │     ├─ maps/                # town.tmx · hunting-ground.tmx · cave.tmx
│  │     └─ lua/                 # 변환된 Lua 로직 스크립트(*.lua) + json-codec.lua
│  ├─ editor/                    # LLM 시나리오 에디터(생성·검증·배치·스타일)
│  └─ games/legend-of-lua/       # (빈 자리표시자 — 실제 빌드는 public/legend-of-lua)
├─ style-service/          # Python AdaIN 스타일 변환 서비스(server.py 등)
├─ public/
│  ├─ vendor/lua/          # 런타임 Lua WASM (lua-5.3.6.mjs / .wasm)
│  └─ legend-of-lua/       # Love2D love.js 사전 빌드(외부 게임)
├─ third_party/            # Lua 5.3.6 C 소스 + 공식 테스트 스위트(빌드 원본)
├─ scripts/                # Lua WASM fetch/build/test (.mjs)
├─ docs/                   # 설계·규칙·연동 문서(.md)
└─ licenses/ · notes/      # 라이선스 / (git 무시) 작업 노트
```

---

## 📜 스크립트

| 명령 | 설명 |
|---|---|
| `npm run dev` | Vite dev 서버(게임+에디터, `/api` 프록시) |
| `npm run build` | `tsc --noEmit` 후 멀티 페이지 빌드 |
| `npm run preview` | 프로덕션 빌드 미리보기 |
| `npm test` / `npm run test:run` | Vitest 감시 / 1회 실행(`*.test.ts`) |
| `npm run check` | `tsc --noEmit && vitest run` (통합 게이트) |
| `npm run lua:build` / `lua:sync` | Lua WASM 빌드 / 소스 fetch+빌드 |
| `npm run lua:test` | Node 기반 Lua WASM 테스트 하니스 |

프록시(개발 서버): `/api/openai → api.openai.com`, `/api/anthropic → api.anthropic.com`, `/api/style → 127.0.0.1:8765`.

---

## 🧰 기술 스택

TypeScript 5.9 · Vite 7 · PixiJS 8(`@pixi/tilemap`) · Tailwind CSS 4 · Vitest 3 · `@xmldom/xmldom`(TMX 파싱) · Lua 5.3.6(WASM, Emscripten) · Python(FastAPI/uvicorn) + PyTorch(AdaIN).

## 📚 문서 (`docs/`)

`architecture.md`(모듈 경계) · `tech-stack.md` · `coding-standards.md` · `git-rules.md` · `testing-strategy.md` · `ai-setup.md` · `style-transfer.md`(AdaIN 연동) · `lua-controller-api.md`(Lua 캐릭터 컨트롤러 계약) · `legend-of-lua-love-js.md` · `legend-of-lua-bridge-protocol.md`(외부 Love2D 게임 라이브 브리지).
