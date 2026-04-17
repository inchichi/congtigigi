# Tech Stack

- 클라이언트/서버: Node.js
- 게임 기반 및 서버 개발 언어: TypeScript
- 플랫폼: 웹브라우저 (데스크탑 / 모바일)
- 렌더링: PixiJS
- 타일맵 렌더링: `@pixi/tilemap`
- 맵 포맷: Tiled TMX/TSX
- 게임 스크립팅: Lua 5.3 사용을 검토 중이다. 실행 방식은 추후 결정한다.
- 번들러: Vite
- 테스트: Vitest

## 리소스 패킹

- 정적 리소스는 `src/assets/` 아래에 둔다.
- 애플리케이션 코드에서는 Vite import 를 통해 필요한 리소스를 가져온다.
