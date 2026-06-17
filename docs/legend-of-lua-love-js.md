# Legend of Lua를 에디터 패널에서 직접 플레이하기 (love.js)

Love2D 게임은 브라우저에서 그냥은 못 돈다. **love.js**(LÖVE를 WebAssembly로 컴파일한 런타임)로
게임을 웹 빌드하면, my-sample-rpg처럼 에디터 오른쪽 **패널 안에서 진짜로 플레이**할 수 있다
(캐릭터가 방향키로 움직인다). 에디터 쪽 배선은 이미 끝났고, **게임을 한 번 빌드해서 넣기만** 하면 된다.

> 참고: 이건 "맵 미리보기"(정적 그림)와 다르다. 웹 빌드 URL을 설정하면 패널이 미리보기 대신
> **실행되는 게임**으로 바뀐다.

---

## 1. 게임을 `.love`로 패키징

게임 폴더(= `main.lua`, `conf.lua`가 있는 곳)에서 그 안의 **내용물**을 zip으로 묶는다(폴더 자체가
아니라 `main.lua`가 zip 루트에 오게). 확장자를 `.love`로:

```bash
cd /경로/legend-of-lua
zip -9 -r ../legend-of-lua.love .   # main.lua가 zip 최상위에 있어야 함
```

## 2. love.js로 웹 빌드

[Davidobot/love.js](https://github.com/Davidobot/love.js)(LÖVE 11.x 호환)를 쓴다:

```bash
# 출력은 이 repo의 public/ 아래로 — Vite가 public/을 루트 경로로 서빙한다.
npx love.js ../legend-of-lua.love public/legend-of-lua -c -t "Legend of Lua"
#                                  └ 출력 폴더            └ -c: compat 빌드(아래 참고)
```

- `-c`(compat) 빌드를 권장한다. 일반(스레드) 빌드는 `SharedArrayBuffer` 때문에 COOP/COEP
  헤더가 필요해 설정이 번거롭다. compat 빌드는 그 헤더 없이 바로 뜬다.
- 빌드가 안 되면 온라인 빌더도 있다: [LoveWebBuilder](https://schellingb.github.io/LoveWebBuilder/)
  에 `.love`를 올려 받은 결과물을 `public/legend-of-lua/`에 넣으면 된다.

결과적으로 `public/legend-of-lua/index.html`(+ `game.js`, `*.wasm` 등)이 생긴다.

> Vite dev 서버는 원래 HTML/디렉터리 요청을 루트 index.html(my-sample-rpg)로 가로채서, public/에
> 둔 love.js 빌드의 index.html이 무시된다. 그래서 `vite.config.ts`에 dev 전용 플러그인을 넣어,
> `public/` 하위의 love.js 빌드 폴더(= `index.html` + `love.js`가 있는 곳)를 자동 탐지해 그 경로로
> 직접 서빙한다. **빌드 폴더를 새로 추가하면 dev 서버를 한 번 재시작**해야 탐지된다.

## 3. 에디터에 URL 넣기

`npm run dev` 상태에서 `http://localhost:5173/editor.html`을 열고:

1. legend-of-lua 폴더를 평소처럼 "게임 폴더 열기"로 연다.
2. ⚙ 설정 → **"love.js 웹 빌드 URL"** 에 `/legend-of-lua/` 입력.

그러면 오른쪽 패널이 **맵 미리보기 → 실제 게임 플레이**로 바뀐다. 패널을 클릭하고 방향키를
누르면 캐릭터가 움직인다. 헤더는 "게임 연결됨"으로 바뀐다.

> URL을 비우면 다시 맵 미리보기로 돌아간다(빌드 전이거나 빌드 없이 맵만 보고 싶을 때).

---

## 4. (선택) "게임에 적용"을 패널 게임에 연결

웹 빌드 모드에서 **게임에 적용**을 누르면, 에디터는 iframe에 아래 메시지를 보낸다:

```js
// 부모(에디터) → iframe(게임 빌드)
window.postMessage({ type: 'editor:apply', payload: <BridgeApplyMessage> }, '*')
```

`payload`는 [브리지 프로토콜](legend-of-lua-bridge-protocol.md)의 `BridgeApplyMessage`와 같다
(`kind:"entity_lines"`, `target`, `lines` …).

이걸 게임이 받아 반영하려면, love.js 빌드의 **호스트 페이지**(`public/legend-of-lua/index.html`)에
JS↔Lua 다리를 한 줄 놓는다. 가장 단순한 방법은 받은 payload를 전역에 쌓아두고 Lua가 폴링하는 것:

```html
<!-- index.html, love.js 로더 <script> 앞에 추가 -->
<script>
  window.__editorApplyQueue = []
  window.addEventListener('message', function (e) {
    if (e.data && e.data.type === 'editor:apply') {
      window.__editorApplyQueue.push(e.data.payload)
    }
  })
</script>
```

게임 쪽 Lua에서 love.js의 JS 호출(예: `love.window` 대신 Emscripten `js` FFI / `love.system`로
주입한 헬퍼)로 `window.__editorApplyQueue`를 비우며 적용한다. 적용 로직 자체는
[bridge.lua](legend-of-lua-bridge.lua) 하단 `applyToGame` 예시와 동일한 계약(`target`+`lines`)이다.

> 이 4번은 "플레이"와 독립이다. URL만 넣으면 **플레이는 바로** 되고, 적용 연동은 나중에 붙여도 된다.

---

## 5. (선택) 맵 버튼으로 게임 맵 전환

패널 위 맵 버튼(test / menu / testCave …)을 누르면, 에디터는 왼쪽 엔티티 트리를 그 맵으로 좁히고
**동시에 iframe 게임에 맵 전환 메시지**를 보낸다:

```js
// 부모(에디터) → iframe(게임 빌드)
window.postMessage({ type: 'editor:goto-map', mapId: 'test', mapName: 'test' }, '*')
```

`mapId`는 `.tmx` 파일명(확장자 제외)이다. 실행 중인 게임이 이 명령으로 실제 맵을 바꾸려면, 4번과
같은 자리(`index.html`)에 핸들러를 한 줄 더 놓고 Lua가 처리하면 된다:

```html
<script>
  window.__editorGotoMap = null
  window.addEventListener('message', function (e) {
    if (e.data && e.data.type === 'editor:goto-map') {
      window.__editorGotoMap = e.data.mapId   // Lua가 폴링해 그 맵으로 씬 전환
    }
  })
</script>
```

게임 Lua는 `window.__editorGotoMap`을 폴링해 값이 있으면 그 맵(`maps/<mapId>.tmx`)으로 전환하고
다시 `nil`로 비우면 된다.

> 게임-쪽 핸들러가 없으면 맵 버튼은 **왼쪽 트리만** 좁히고 게임 화면은 그대로다(에디터는 메시지를
> 보내지만 게임이 무시). 핸들러를 붙이면 게임 맵도 같이 바뀐다.

> **주의(현실적 난이도):** 위 "Lua가 `window.__editorGotoMap`을 폴링" 부분이 사실 제일 까다롭다.
> LÖVE에는 브라우저 `window`를 직접 읽는 기능이 없어서, love.js에서 JS↔Lua 다리를 따로 놓아야
> 한다(예: 호스트 페이지가 Emscripten 가상 FS에 명령 파일을 써두고 Lua가 `love.filesystem`으로
> 읽기, 또는 `love.thread` 채널). 이건 게임 내부 구조에 의존하므로 **게임을 소유한 팀원이 붙이는 게
> 맞다.** 에디터는 명령을 표준 형태로 보내주는 데까지 책임진다. 핸들러 전까지는, 게임 안에서 직접
> 이동(맵 경계의 포털 등)해 맵을 옮기면 된다.

---

## 6. 두 방식 비교 (요약)

| | love.js (이 문서) | HTTP 브리지([다른 문서](legend-of-lua-bridge-protocol.md)) |
|---|---|---|
| 게임이 도는 곳 | **에디터 패널 안**(iframe) | 별도 네이티브 LÖVE 창 |
| 패널에서 플레이 | ✅ (캐릭터 움직임) | ❌ (패널은 맵 미리보기) |
| 적용 채널 | iframe `postMessage` | 로컬 HTTP(`localhost:17320`) |
| 셋업 | 게임을 love.js로 1회 빌드 | 게임에 bridge.lua+json.lua 설치 |

패널 안에서 플레이하고 싶으면 **love.js**, 네이티브 창으로 돌리고 에디터로 내용만 쏘고 싶으면
**브리지**를 쓰면 된다. 에디터는 둘 다 지원한다(웹 빌드 URL이 있으면 love.js, 없으면 브리지/미리보기).

## Quest Export

- The editor can generate Legend of Lua quests from the opened game's entity catalog.
- The generated quest is validated against the opened maps/entities and rendered as a Lua module preview.
- Live apply is still not wired in the game runtime, so the current output path is preview/copy/export first.
