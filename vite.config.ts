import {
  createReadStream,
  existsSync,
  readdirSync,
  readFileSync,
  statSync
} from 'node:fs'
import { basename, extname, join, normalize } from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig, type Plugin } from 'vitest/config'
import tailwindcss from '@tailwindcss/vite'

const PUBLIC_DIR = fileURLToPath(new URL('./public', import.meta.url))

// love.js 빌드를 에디터 패널(iframe)에 임베드할 때, 게임 캔버스가 네이티브 해상도(예: 1920×1080)
// 그대로 떠서 좁은 패널 밖으로 잘린다. 빌드 파일은 그대로 두고, 서빙 시점에 이 스타일을 index.html에
// 주입해 캔버스를 박스에 맞춰(레터박스) 줄이고, 제목/푸터 같은 chrome는 숨긴다.
const LOVE_EMBED_STYLE = `
<style id="editor-embed-fit">
  html, body { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: #000; }
  body > center, body > center > div { margin: 0; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; }
  h1, footer { display: none !important; }
  #canvas, #loadingCanvas { max-width: 100%; max-height: 100%; width: auto; height: auto; }
</style>
`

// 에디터(부모 창) → love.js 게임 다리. 에디터가 보낸 postMessage를 받아 Emscripten 가상 FS에
// 명령 파일로 써둔다. 게임-쪽 Lua(pollEditorCommands)가 매 프레임 그 파일을 읽어 처리한다.
//
// 이 love.js 빌드는 완전한 Module.FS를 노출하지 않아서(writeFile 없음), 패키저가 노출하는
// Module.FS_createDataFile을 쓴다. 이건 덮어쓰기가 안 되므로 매번 고유 파일명(editorcmd-N.txt)을
// 쓰고, Lua가 순번대로 읽는다. FS가 늦게 준비될 수 있어 큐에 담고 짧게 재시도한다.
const LOVE_EMBED_BRIDGE_SCRIPT = `
<script id="editor-bridge">
(function () {
  console.log('[editor-bridge] loaded');
  var counter = 0;
  var queue = [];
  var reportedKeys = false;
  function reportKeys() {
    if (reportedKeys) return;
    try {
      var M = window.Module;
      if (!M) return;
      var keys = Object.keys(M).filter(function (k) { return /FS/.test(k); });
      console.log('[editor-bridge] Module FS keys:', keys.join(', ') || '(none)');
      reportedKeys = true;
    } catch (e) {}
  }
  // love.js에서 LÖVE가 읽는 위치가 빌드마다 달라, 여러 후보 디렉터리에 같은 파일을 써둔다.
  // (게임은 raw io(/tmp) 또는 love.filesystem(세이브 디렉터리) 중 되는 쪽으로 읽는다.)
  var DIRS = [
    '/tmp',
    '/home/web_user/.local/share/LOVE/legend-of-lua',
    '/home/web_user/.local/share/love/legend-of-lua',
    '/save'
  ];
  // 큐의 맨 앞 명령을 고유 파일명으로 후보 경로들에 써본다. 한 곳이라도 성공하면 true.
  function tryWrite(text) {
    var M = window.Module;
    if (!M || !M.FS_createDataFile) return false;
    var name = 'editorcmd-' + (counter + 1) + '.txt';
    var wroteAnywhere = false;
    var okDirs = [];
    for (var i = 0; i < DIRS.length; i++) {
      try {
        M.FS_createDataFile(DIRS[i], name, text, true, true);
        wroteAnywhere = true;
        okDirs.push(DIRS[i]);
      } catch (e) { /* 그 디렉터리가 없으면 무시 */ }
    }
    if (wroteAnywhere) {
      counter++;
      console.log('[editor-bridge] 명령 씀:', name, text, '→', okDirs.join(', '));
      return true;
    }
    return false;
  }
  function flush() {
    reportKeys();
    while (queue.length > 0) {
      if (!tryWrite(queue[0])) return; // FS 아직 없음 → 다음 틱에 재시도
      queue.shift();
    }
  }
  setInterval(flush, 200);
  window.addEventListener('message', function (event) {
    var data = event.data;
    if (!data || typeof data !== 'object') return;
    if (data.type === 'editor:goto-map' && data.mapId) {
      console.log('[editor-bridge] goto-map 큐:', data.mapId);
      queue.push('goto-map:' + data.mapId);
      flush();
    } else if (
      data.type === 'editor:apply' &&
      data.payload &&
      Array.isArray(data.payload.lines)
    ) {
      // 생성된 대사를 화면 오버레이로 라이브 반영. 대상이 있으면 이름을 앞에 붙인다.
      var target = data.payload.target;
      var prefix = target && target.name ? target.name + ' — ' : '';
      console.log('[editor-bridge] apply 큐:', data.payload.lines.length + '줄');
      queue.push('apply-lines:' + prefix + data.payload.lines.join('\\n'));
      flush();
    }
  });
})();
</script>
`

const MIME_TYPES: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.wasm': 'application/wasm',
  '.css': 'text/css',
  '.json': 'application/json',
  '.data': 'application/octet-stream',
  '.mem': 'application/octet-stream',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ttf': 'font/ttf'
}

// Vite dev 서버는 HTML/디렉터리 요청을 가로채 루트 index.html로 fallback한다. 그래서 public/ 안에
// 둔 love.js 빌드(자체 index.html을 가짐)가 무시되고 my-sample-rpg가 떠버린다. 이 플러그인은
// public/ 하위에서 love.js 빌드 폴더(index.html + love.js가 있는 곳)를 찾아, 그 경로 요청을
// Vite의 HTML 처리보다 먼저 가로채 정적 파일 그대로 내보낸다(.wasm MIME 포함). dev 전용이며
// 프로덕션(빌드 산출물 정적 서빙)에는 영향이 없다.
const serveLoveJsBuilds = (): Plugin => ({
  name: 'serve-lovejs-builds',
  apply: 'serve',
  configureServer(server) {
    const buildDirs = existsSync(PUBLIC_DIR)
      ? readdirSync(PUBLIC_DIR).filter((name) => {
          const dir = join(PUBLIC_DIR, name)
          return (
            statSync(dir).isDirectory() &&
            existsSync(join(dir, 'index.html')) &&
            existsSync(join(dir, 'love.js'))
          )
        })
      : []

    if (buildDirs.length > 0) {
      server.config.logger.info(
        `  ➜  love.js 빌드 서빙: ${buildDirs.map((name) => `/${name}/`).join(', ')}`
      )
    }

    // 직접 등록(반환 함수 아님) → Vite 내부 미들웨어보다 먼저 실행된다.
    server.middlewares.use((req, res, next) => {
      const rawUrl = (req.url ?? '').split('?')[0]
      const name = buildDirs.find(
        (dir) => rawUrl === `/${dir}` || rawUrl.startsWith(`/${dir}/`)
      )

      if (!name) {
        next()
        return
      }

      // 슬래시 없는 디렉터리 요청은 상대경로 자산이 깨지므로 트레일링 슬래시로 보낸다.
      if (rawUrl === `/${name}`) {
        res.statusCode = 301
        res.setHeader('Location', `/${name}/`)
        res.end()
        return
      }

      const relative = rawUrl.slice(name.length + 2) || 'index.html'
      const baseDir = join(PUBLIC_DIR, name)
      const filePath = normalize(join(baseDir, decodeURIComponent(relative)))

      // 디렉터리 탈출 방지 + 존재/파일 확인. 없으면 빌드의 index.html로(SPA처럼).
      const resolved =
        filePath.startsWith(baseDir) &&
        existsSync(filePath) &&
        statSync(filePath).isFile()
          ? filePath
          : join(baseDir, 'index.html')

      res.setHeader(
        'Content-Type',
        MIME_TYPES[extname(resolved)] ?? 'application/octet-stream'
      )
      // 재빌드 시 옛 game.data/스크립트가 캐시에서 떠 변경이 안 보이는 일을 줄인다(dev 한정).
      res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate')

      // index.html은 임베드용 스타일 + 에디터 다리 스크립트를 주입해 내보낸다(나머지는 스트리밍).
      if (basename(resolved) === 'index.html') {
        const html = readFileSync(resolved, 'utf8')
          .replace('</head>', `${LOVE_EMBED_STYLE}</head>`)
          .replace('</body>', `${LOVE_EMBED_BRIDGE_SCRIPT}</body>`)
        res.end(html)
        return
      }

      createReadStream(resolved).pipe(res)
    })
  }
})

export default defineConfig({
  plugins: [tailwindcss(), serveLoveJsBuilds()],
  build: {
    rollupOptions: {
      input: {
        main: fileURLToPath(new URL('./index.html', import.meta.url)),
        editor: fileURLToPath(new URL('./editor.html', import.meta.url))
      }
    }
  },
  server: {
    proxy: {
      '/api/openai': {
        target: 'https://api.openai.com',
        changeOrigin: true,
        secure: true,
        rewrite: (path) => path.replace(/^\/api\/openai/, '')
      },
      // 에디터의 Claude 호출을 서버사이드로 포워딩 → 브라우저 CORS 회피.
      '/api/anthropic': {
        target: 'https://api.anthropic.com',
        changeOrigin: true,
        secure: true,
        rewrite: (path) => path.replace(/^\/api\/anthropic/, '')
      }
    }
  },
  test: {
    include: ['src/**/*.test.ts']
  }
})
