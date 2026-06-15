// 에디터 전용 게임풍 아이콘 — 전부 손으로 그린 인라인 SVG.
// emoji·lucide·heroicons 같은 기성 아이콘은 쓰지 않는다(프로토타입처럼 보인다는 피드백).
// 공통 문법: 24×24 뷰박스에 바닥 그림자 → 본체(2톤 채색) → 진갈색 외곽선 → 상단 하이라이트 순서로
// 그려서, 게임 오브젝트를 축소한 미니어처처럼 보이게 한다.

const SHADOW = '<ellipse cx="12" cy="21" rx="6.5" ry="1.5" fill="rgba(0,0,0,0.3)"/>'

// 외곽선 기본값. 개별 도형에서 stroke-width를 줄여 겹침 부위만 얇게 조절한다.
const OUTLINE = 'stroke="#241a12" stroke-width="1" stroke-linejoin="round"'

const ICONS = {
  // RPG 주민 초상화(상반신) — NPC. 막대인간이 아니라 메이플 NPC식 흉상.
  npc:
    SHADOW +
    `<path d="M4.2 20.5c.6-3.6 3.7-5.4 7.8-5.4s7.2 1.8 7.8 5.4z" fill="#6e8a4f" ${OUTLINE}/>` +
    '<path d="M9.4 15.6 12 17.6l2.6-2" fill="none" stroke="#f3e3c8" stroke-width="1.1" stroke-linecap="round"/>' +
    `<circle cx="12" cy="9.8" r="5.8" fill="#f0c08e" ${OUTLINE}/>` +
    `<path d="M6.2 9.6C6.2 5.6 8.6 3.4 12 3.4s5.8 2.2 5.8 6.2c-1-2.2-2.3-3.1-5.8-3.1s-4.8.9-5.8 3.1z" fill="#8a5a32" ${OUTLINE}/>` +
    '<path d="M6.3 9.6c-.5 1-.5 2.2-.2 3.1l1-.4z" fill="#8a5a32" stroke="#241a12" stroke-width="0.7"/>' +
    '<path d="M17.7 9.6c.5 1 .5 2.2.2 3.1l-1-.4z" fill="#8a5a32" stroke="#241a12" stroke-width="0.7"/>' +
    '<circle cx="9.8" cy="10.6" r="0.95" fill="#241a12"/>' +
    '<circle cx="14.2" cy="10.6" r="0.95" fill="#241a12"/>' +
    '<path d="M10.6 13c.9.8 1.9.8 2.8 0" stroke="#a05c34" stroke-width="1.1" fill="none" stroke-linecap="round"/>' +
    '<circle cx="8.4" cy="12.2" r="1" fill="#ef8f78" opacity="0.55"/>' +
    '<circle cx="15.6" cy="12.2" r="1" fill="#ef8f78" opacity="0.55"/>' +
    '<path d="M7.6 6.6c1-1 2.4-1.6 4-1.7" stroke="rgba(255,255,255,0.45)" stroke-width="1.1" fill="none" stroke-linecap="round"/>',
  // 보라색 슬라임 — 몬스터
  monster:
    SHADOW +
    `<path d="M5 13.5C5 9 8 6 12 6s7 3 7 7.5c0 3.6-3 6-7 6s-7-2.4-7-6z" fill="#8a5fb0" ${OUTLINE}/>` +
    `<path d="M7.5 7.5 6 4.6l3 1.2z" fill="#6b4391" ${OUTLINE}/>` +
    `<path d="M16.5 7.5 18 4.6l-3 1.2z" fill="#6b4391" ${OUTLINE}/>` +
    '<circle cx="9.6" cy="12.4" r="1.7" fill="#fff"/>' +
    '<circle cx="14.4" cy="12.4" r="1.7" fill="#fff"/>' +
    '<circle cx="9.8" cy="12.6" r="0.8" fill="#241a12"/>' +
    '<circle cx="14.2" cy="12.6" r="0.8" fill="#241a12"/>' +
    '<path d="M10 16.2h4l-1 1.4-1-.8-1 .8z" fill="#fff" stroke="#241a12" stroke-width="0.7"/>' +
    '<path d="M7.6 8.6c1.1-1 2.6-1.6 4.4-1.6" stroke="rgba(255,255,255,0.4)" stroke-width="1.1" fill="none" stroke-linecap="round"/>',
  // 나무 표지판
  sign:
    SHADOW +
    `<rect x="10.9" y="10" width="2.2" height="10" fill="#8a5a32" ${OUTLINE}/>` +
    `<rect x="4.2" y="4" width="15.6" height="7.2" rx="1.4" fill="#b07c46" ${OUTLINE}/>` +
    '<path d="M6.2 6.4h11.6M6.2 8.8h8.4" stroke="#8a5a32" stroke-width="1" stroke-linecap="round"/>' +
    '<path d="M5.4 5.2h6" stroke="rgba(255,255,255,0.35)" stroke-width="1" stroke-linecap="round"/>',
  // 빛나는 파란 원형 게이트 — 포털
  portal:
    '<ellipse cx="12" cy="11.5" rx="8.6" ry="9.6" fill="#3b6fa8" opacity="0.22"/>' +
    '<path d="M5.5 20.6h13l-1.4-2.2H6.9z" fill="#5a6470" stroke="#241a12" stroke-width="0.9"/>' +
    `<ellipse cx="12" cy="11.5" rx="6.6" ry="7.8" fill="#1d3a5f" ${OUTLINE}/>` +
    '<ellipse cx="12" cy="11.5" rx="4.5" ry="5.7" fill="#3f8fd4" stroke="#241a12" stroke-width="0.8"/>' +
    '<ellipse cx="12" cy="11.5" rx="2.3" ry="3.1" fill="#9fd4f6"/>' +
    '<path d="M10.7 14c2.3-.6 3.3-2.9 1.9-5.2" stroke="#d6ecfb" stroke-width="1" fill="none" stroke-linecap="round"/>' +
    '<circle cx="17.6" cy="4.6" r="0.9" fill="#bfe0fa"/>' +
    '<circle cx="6.6" cy="6.8" r="0.7" fill="#bfe0fa" opacity="0.85"/>',
  // 보물상자
  chest:
    SHADOW +
    `<rect x="4.5" y="11" width="15" height="8.6" rx="1.2" fill="#9a6334" ${OUTLINE}/>` +
    `<path d="M4.5 11c0-3.4 2.4-5.4 7.5-5.4s7.5 2 7.5 5.4z" fill="#b07c46" ${OUTLINE}/>` +
    `<rect x="10.7" y="10" width="2.6" height="4.6" rx="0.6" fill="#e3b34d" ${OUTLINE}/>` +
    '<circle cx="12" cy="12.6" r="0.55" fill="#241a12"/>' +
    '<path d="M6 7.6c1.2-1 2.9-1.6 5-1.7" stroke="rgba(255,255,255,0.35)" stroke-width="1" fill="none" stroke-linecap="round"/>',
  // 금화 더미 — 전리품
  loot:
    SHADOW +
    `<ellipse cx="9" cy="16.5" rx="5.4" ry="3" fill="#e3b34d" ${OUTLINE}/>` +
    `<ellipse cx="9" cy="14.8" rx="5.4" ry="3" fill="#f2cd6e" ${OUTLINE}/>` +
    `<circle cx="16.2" cy="12.6" r="4.4" fill="#f2cd6e" ${OUTLINE}/>` +
    '<circle cx="16.2" cy="12.6" r="2.6" fill="none" stroke="#c79436" stroke-width="1"/>' +
    '<path d="M13.4 9.6c.8-.7 1.8-1.1 2.8-1.1" stroke="rgba(255,255,255,0.5)" stroke-width="1" fill="none" stroke-linecap="round"/>',
  // 집 — 지붕·창문·굴뚝·연기까지 있는 게임 건물 아이콘.
  building:
    SHADOW +
    '<circle cx="17.3" cy="3.8" r="1.1" fill="#cfc8bd" opacity="0.85"/>' +
    '<circle cx="18.8" cy="2.6" r="0.8" fill="#cfc8bd" opacity="0.6"/>' +
    '<rect x="15.4" y="4.8" width="2.6" height="4.6" fill="#8d6b4a" stroke="#241a12" stroke-width="0.9"/>' +
    `<path d="M3.4 12.5 12 5l8.6 7.5z" fill="#b5503c" ${OUTLINE}/>` +
    `<rect x="5" y="12.5" width="14" height="7.4" fill="#e2cda3" ${OUTLINE}/>` +
    `<path d="M10.3 19.9v-3.9c0-1 .7-1.7 1.7-1.7s1.7.7 1.7 1.7v3.9z" fill="#7a4526" ${OUTLINE}/>` +
    '<circle cx="7.7" cy="15.2" r="1.3" fill="#9ecbe8" stroke="#241a12" stroke-width="0.8"/>' +
    '<circle cx="16.3" cy="15.2" r="1.3" fill="#9ecbe8" stroke="#241a12" stroke-width="0.8"/>' +
    '<path d="M5.6 11 12 5.6l1.8 1.6" stroke="rgba(255,255,255,0.4)" stroke-width="1" fill="none" stroke-linecap="round"/>',
  // 모자 쓴 주민 — 캐릭터
  character:
    SHADOW +
    `<circle cx="12" cy="12.5" r="7" fill="#edbf92" ${OUTLINE}/>` +
    `<path d="M5.2 11.4C5.6 7 8.4 5 12 5s6.4 2 6.8 6.4z" fill="#3f6e9e" ${OUTLINE}/>` +
    '<rect x="4" y="10.2" width="16" height="2" rx="1" fill="#345b84" stroke="#241a12" stroke-width="0.8"/>' +
    '<circle cx="9.6" cy="14" r="0.9" fill="#241a12"/>' +
    '<circle cx="14.4" cy="14" r="0.9" fill="#241a12"/>' +
    '<path d="M10.4 16.6c1 .8 2.2 .8 3.2 0" stroke="#a05c34" stroke-width="1" fill="none" stroke-linecap="round"/>',
  // 천막
  tent:
    SHADOW +
    `<path d="M3.6 19.5 12 5l8.4 14.5z" fill="#c46a4a" ${OUTLINE}/>` +
    '<path d="M12 5v14.5" stroke="#241a12" stroke-width="0.8"/>' +
    '<path d="M12 19.5c-1.8 0-2.6-3-2.6-5.2L12 11.6l2.6 2.7c0 2.2-.8 5.2-2.6 5.2z" fill="#5e3a28" stroke="#241a12" stroke-width="0.8"/>' +
    '<path d="M6.4 16.5 11 8.6" stroke="rgba(255,255,255,0.3)" stroke-width="1" stroke-linecap="round" fill="none"/>',
  // 시계탑
  clocktower:
    SHADOW +
    `<rect x="8" y="7" width="8" height="13" fill="#cdb592" ${OUTLINE}/>` +
    `<path d="M6.8 7.4 12 3l5.2 4.4z" fill="#b5503c" ${OUTLINE}/>` +
    `<circle cx="12" cy="12" r="2.8" fill="#f6ead0" ${OUTLINE}/>` +
    '<path d="M12 10.6v1.6l1.1.8" stroke="#241a12" stroke-width="0.8" fill="none" stroke-linecap="round"/>' +
    '<rect x="10.6" y="16.4" width="2.8" height="3.6" fill="#7a4526" stroke="#241a12" stroke-width="0.8"/>',
  // 광장 분수 — 2단 수반 + 물줄기.
  fountain:
    SHADOW +
    `<ellipse cx="12" cy="17.6" rx="8.4" ry="3.1" fill="#9aa6ad" ${OUTLINE}/>` +
    '<ellipse cx="12" cy="16.9" rx="6.6" ry="2.1" fill="#74b8d8" stroke="#241a12" stroke-width="0.7"/>' +
    '<path d="M8.6 13.4c-.6 1-.9 2-.8 3M15.4 13.4c.6 1 .9 2 .8 3" stroke="#9fd4ee" stroke-width="0.9" fill="none" stroke-linecap="round"/>' +
    '<rect x="11" y="13" width="2" height="3.2" fill="#b9c2c8" stroke="#241a12" stroke-width="0.8"/>' +
    '<ellipse cx="12" cy="12.8" rx="3.9" ry="1.5" fill="#b9c2c8" stroke="#241a12" stroke-width="0.8"/>' +
    '<ellipse cx="12" cy="12.5" rx="2.7" ry="1" fill="#8ecbe8"/>' +
    '<rect x="11.2" y="8.2" width="1.6" height="3.6" fill="#b9c2c8" stroke="#241a12" stroke-width="0.7"/>' +
    '<path d="M12 7.4c0-1.7-1.4-2.2-2.5-1.6M12 7.4c0-1.7 1.4-2.2 2.5-1.6" stroke="#9fd4ee" stroke-width="1.1" fill="none" stroke-linecap="round"/>' +
    '<circle cx="12" cy="7" r="1" fill="#bfe4f6" stroke="#241a12" stroke-width="0.6"/>',
  // 가로등
  lamp:
    SHADOW +
    '<rect x="11.1" y="8" width="1.8" height="12" fill="#4d5258" stroke="#241a12" stroke-width="0.9"/>' +
    '<circle cx="12" cy="6" r="4.6" fill="#ffd98a" opacity="0.25"/>' +
    `<circle cx="12" cy="6" r="3" fill="#ffd98a" ${OUTLINE}/>` +
    '<path d="M9.6 3.4h4.8l-.8-1.4h-3.2z" fill="#4d5258" stroke="#241a12" stroke-width="0.8"/>' +
    '<rect x="9.6" y="19" width="4.8" height="1.6" rx="0.8" fill="#4d5258" stroke="#241a12" stroke-width="0.8"/>',
  // 길드 깃발 — 제비꼬리 깃발에 금색 문장.
  banner:
    SHADOW +
    '<rect x="6" y="3.4" width="1.8" height="16.6" rx="0.5" fill="#8a5a32" stroke="#241a12" stroke-width="0.9"/>' +
    `<path d="M7.8 4.6h11.6l-2.6 3.7 2.6 3.7H7.8z" fill="#a83838" ${OUTLINE}/>` +
    '<circle cx="12.3" cy="8.3" r="2" fill="#e3b34d" stroke="#241a12" stroke-width="0.8"/>' +
    '<path d="M12.3 7v2.6M11 8.3h2.6" stroke="#a83838" stroke-width="0.9" stroke-linecap="round"/>' +
    '<path d="M9 5.7h5.4" stroke="rgba(255,255,255,0.3)" stroke-width="0.9" stroke-linecap="round"/>' +
    '<circle cx="6.9" cy="3.2" r="1.2" fill="#e3b34d" stroke="#241a12" stroke-width="0.7"/>',
  // 나무
  tree:
    SHADOW +
    '<rect x="10.9" y="13" width="2.2" height="7" fill="#7a4526" stroke="#241a12" stroke-width="0.9"/>' +
    `<path d="M12 3.4c4 0 6.8 2.8 6.8 6.2 0 3.2-2.6 5.6-6.8 5.6S5.2 12.8 5.2 9.6C5.2 6.2 8 3.4 12 3.4z" fill="#5e8f46" ${OUTLINE}/>` +
    '<circle cx="9.4" cy="8.2" r="1" fill="#6da153"/>' +
    '<circle cx="14.2" cy="10.6" r="1.2" fill="#6da153"/>' +
    '<path d="M8.2 6.4c1-1.1 2.3-1.7 3.8-1.8" stroke="rgba(255,255,255,0.4)" stroke-width="1.1" fill="none" stroke-linecap="round"/>',
  // 생울타리
  hedge:
    SHADOW +
    `<path d="M4 18.5c-1.2 0-1.8-3 .4-3.6-.6-2.2 1.4-3.8 3-3 .4-2.4 3.2-3.2 4.6-1.7 1.4-1.5 4.2-.7 4.6 1.7 1.6-.8 3.6.8 3 3 2.2.6 1.6 3.6.4 3.6z" fill="#567f41" ${OUTLINE}/>` +
    '<path d="M6.6 13.4c.8-1.1 2-1.8 3.2-2" stroke="rgba(255,255,255,0.35)" stroke-width="1" fill="none" stroke-linecap="round"/>',
  // 꽃 여러 송이가 심긴 화단
  flower:
    SHADOW +
    `<ellipse cx="12" cy="17.4" rx="8.2" ry="2.9" fill="#6b4a2e" ${OUTLINE}/>` +
    '<path d="M7.6 15.4v-2.6M12 15.2v-3.6M16.4 15.4v-2.6" stroke="#4f7a3c" stroke-width="1.2" stroke-linecap="round"/>' +
    '<path d="M9.2 13.6c.8-.2 1.4 0 1.8.6" stroke="#5e8f46" stroke-width="1" fill="none" stroke-linecap="round"/>' +
    '<circle cx="7.6" cy="10.8" r="2" fill="#e87a9a" stroke="#241a12" stroke-width="0.8"/>' +
    '<circle cx="7.6" cy="10.8" r="0.8" fill="#f2cd6e"/>' +
    '<circle cx="12" cy="9.2" r="2.3" fill="#e85c5c" stroke="#241a12" stroke-width="0.8"/>' +
    '<circle cx="12" cy="9.2" r="0.95" fill="#f2cd6e"/>' +
    '<circle cx="16.4" cy="10.8" r="2" fill="#9a6fd0" stroke="#241a12" stroke-width="0.8"/>' +
    '<circle cx="16.4" cy="10.8" r="0.8" fill="#f2cd6e"/>' +
    '<path d="M5.4 16.2c1.4-.7 3-.9 4.4-.7" stroke="rgba(255,255,255,0.25)" stroke-width="0.9" fill="none" stroke-linecap="round"/>',
  // 상자 + 배럴 + 항아리 묶음 — 소품
  prop:
    SHADOW +
    '<rect x="3.6" y="9.6" width="8.2" height="8.4" rx="0.8" fill="#a9743f"/>' +
    '<path d="M4 10 11.4 17.6M11.4 10 4 17.6" stroke="#8a5a32" stroke-width="1.1"/>' +
    '<rect x="3.6" y="9.6" width="8.2" height="8.4" rx="0.8" fill="none" stroke="#241a12" stroke-width="0.9"/>' +
    `<path d="M13 10.6c0-1 1.5-1.8 3.5-1.8s3.5.8 3.5 1.8v5.8c0 1-1.5 1.8-3.5 1.8s-3.5-.8-3.5-1.8z" fill="#9a6334" ${OUTLINE}/>` +
    '<path d="M13 12.4c.9.5 2.1.8 3.5.8s2.6-.3 3.5-.8M13 15c.9.5 2.1.8 3.5.8s2.6-.3 3.5-.8" stroke="#5e3a28" stroke-width="0.9" fill="none"/>' +
    '<path d="M8.5 15.1c0-.8 1-1.4 2.1-1.4s2.1.6 2.1 1.4c.8.5 1.3 1.3 1.3 2.1 0 1.5-1.5 2.7-3.4 2.7s-3.4-1.2-3.4-2.7c0-.8.5-1.6 1.3-2.1z" fill="#b98a5a" stroke="#241a12" stroke-width="0.9"/>' +
    '<ellipse cx="10.6" cy="15" rx="1.6" ry="0.7" fill="#7a4526" stroke="#241a12" stroke-width="0.6"/>' +
    '<path d="M4.4 8.8h3.4" stroke="rgba(255,255,255,0.3)" stroke-width="0.9" stroke-linecap="round"/>',
  // 바위
  rock:
    SHADOW +
    `<path d="M5 18.5c-1.6-3 0-6.4 2.2-7.6.6-2.4 3-3.8 5.2-3.2 2.2-.8 4.8.4 5.4 2.6 2 1.4 2.6 4.6 1 8.2z" fill="#8d9298" ${OUTLINE}/>` +
    '<path d="M9 10.5l2.4 2.2-1 5.8" stroke="#6f747a" stroke-width="1" fill="none" stroke-linecap="round"/>' +
    '<path d="M7.6 9.6c.8-1 2-1.7 3.2-1.9" stroke="rgba(255,255,255,0.4)" stroke-width="1.1" fill="none" stroke-linecap="round"/>',
  // 계단
  stairs:
    SHADOW +
    `<path d="M4.5 19.5v-4h5v-4h5v-4h5v12z" fill="#a9a195" ${OUTLINE}/>` +
    '<path d="M9.5 19.5v-4M14.5 19.5v-8" stroke="#241a12" stroke-width="0.7"/>' +
    '<path d="M5.2 15.9h3.6M10.2 11.9h3.6M15.2 7.9h3.6" stroke="rgba(255,255,255,0.35)" stroke-width="1" stroke-linecap="round"/>',
  // 벽돌 벽
  wall:
    SHADOW +
    '<rect x="4" y="7" width="16" height="12.5" fill="#a8654a"/>' +
    '<path d="M4 11.2h16M4 15.4h16M9.4 7v4.2M14.8 7v4.2M6.8 11.2v4.2M12 11.2v4.2M17.2 11.2v4.2M9.4 15.4v4.1M14.8 15.4v4.1" stroke="#7d4a36" stroke-width="1"/>' +
    `<rect x="4" y="7" width="16" height="12.5" fill="none" ${OUTLINE}/>` +
    '<path d="M5.2 8.4h4" stroke="rgba(255,255,255,0.3)" stroke-width="1" stroke-linecap="round"/>',
  // 창문
  window:
    SHADOW +
    `<rect x="6" y="4.5" width="12" height="15" rx="1.6" fill="#8a5a32" ${OUTLINE}/>` +
    '<rect x="7.8" y="6.3" width="8.4" height="11.4" fill="#9ecbe8" stroke="#241a12" stroke-width="0.8"/>' +
    '<path d="M12 6.3v11.4M7.8 12h8.4" stroke="#8a5a32" stroke-width="1.2"/>' +
    '<path d="M9 9.4 12.6 7" stroke="rgba(255,255,255,0.55)" stroke-width="1.1" stroke-linecap="round"/>',
  // 접힌 지도
  map:
    SHADOW +
    `<path d="M4 6.5 9.3 4.5l5.4 2 5.3-2v13l-5.3 2-5.4-2-5.3 2z" fill="#e7d6ae" ${OUTLINE}/>` +
    '<path d="M9.3 4.5v13M14.7 6.5v13" stroke="#c9b489" stroke-width="0.9"/>' +
    '<path d="M6.5 13.5c2-2.4 5.4 1 8-2 1-1.2 2-1.6 3-1.4" stroke="#c45050" stroke-width="1" stroke-dasharray="1.6 1.4" fill="none" stroke-linecap="round"/>' +
    '<circle cx="17.5" cy="10" r="1" fill="#c45050" stroke="#241a12" stroke-width="0.6"/>',
  // 금색 조준 링 — 선택 대상
  target:
    `<circle cx="12" cy="12" r="7.6" fill="#2e2620" ${OUTLINE}/>` +
    '<circle cx="12" cy="12" r="7.6" fill="none" stroke="#c48a4a" stroke-width="1.4"/>' +
    '<circle cx="12" cy="12" r="4.4" fill="none" stroke="#c48a4a" stroke-width="1.2" opacity="0.7"/>' +
    '<circle cx="12" cy="12" r="1.6" fill="#e0b070" stroke="#241a12" stroke-width="0.6"/>' +
    '<path d="M12 2.6v3M12 18.4v3M2.6 12h3M18.4 12h3" stroke="#c48a4a" stroke-width="1.4" stroke-linecap="round"/>',
  // 겹친 타일 — 레이어
  layers:
    `<path d="M12 9.5 20 14l-8 4.5L4 14z" fill="#8d9298" ${OUTLINE}/>` +
    `<path d="M12 5.5 20 10l-8 4.5L4 10z" fill="#6da153" ${OUTLINE}/>` +
    '<path d="M8.4 9.2 12 7.2l3.6 2" stroke="rgba(255,255,255,0.4)" stroke-width="1" fill="none" stroke-linecap="round"/>',
  // 검 — 사냥터
  sword:
    SHADOW +
    `<path d="M9.4 14.6 18.2 4.2l1.6-.4-.4 1.6L9 15.8z" fill="#c6ccd2" ${OUTLINE}/>` +
    '<path d="M10.6 13.4 17.6 5.2" stroke="#9aa1a8" stroke-width="0.8"/>' +
    '<rect x="7.6" y="13.2" width="6" height="2" rx="1" transform="rotate(45 10.6 14.2)" fill="#e3b34d" stroke="#241a12" stroke-width="0.8"/>' +
    '<path d="M8.8 16.4l-2.6 2.6" stroke="#7a4526" stroke-width="2.4" stroke-linecap="round"/>' +
    '<circle cx="5.6" cy="19.6" r="1.2" fill="#e3b34d" stroke="#241a12" stroke-width="0.8"/>',
  // 수정 — 동굴
  crystal:
    SHADOW +
    `<path d="M8.6 19.5 5.8 13.5 9 8.5l2.2 4.5z" fill="#4f93c4" ${OUTLINE}/>` +
    `<path d="M11 19.5 9.4 11 13.6 4.5l3.6 8z" fill="#7ec3ec" ${OUTLINE}/>` +
    '<path d="M12.4 8.2 13.8 6" stroke="rgba(255,255,255,0.6)" stroke-width="1.1" stroke-linecap="round"/>',
  // 금속 톱니 — 설정
  gear:
    '<g fill="#8f969e" stroke="#241a12" stroke-width="0.9">' +
    '<rect x="10.6" y="3.2" width="2.8" height="17.6" rx="1.2"/>' +
    '<rect x="10.6" y="3.2" width="2.8" height="17.6" rx="1.2" transform="rotate(60 12 12)"/>' +
    '<rect x="10.6" y="3.2" width="2.8" height="17.6" rx="1.2" transform="rotate(120 12 12)"/>' +
    '</g>' +
    `<circle cx="12" cy="12" r="5.2" fill="#b9bfc7" ${OUTLINE}/>` +
    '<circle cx="12" cy="12" r="2.2" fill="#3a3530" stroke="#241a12" stroke-width="0.9"/>' +
    '<path d="M8.6 9.2c.7-.9 1.7-1.5 2.8-1.7" stroke="rgba(255,255,255,0.5)" stroke-width="1" fill="none" stroke-linecap="round"/>',
  // 폴더
  folder:
    SHADOW +
    `<path d="M4 6.8c0-.9.7-1.6 1.6-1.6h4.2l1.8 2h6.8c.9 0 1.6.7 1.6 1.6v9.4c0 .9-.7 1.6-1.6 1.6H5.6c-.9 0-1.6-.7-1.6-1.6z" fill="#b07c46" ${OUTLINE}/>` +
    `<path d="M4 10h16v8.2c0 .9-.7 1.6-1.6 1.6H5.6c-.9 0-1.6-.7-1.6-1.6z" fill="#c89254" ${OUTLINE}/>` +
    '<path d="M5.4 11.6h5" stroke="rgba(255,255,255,0.35)" stroke-width="1" stroke-linecap="round"/>',
  // 양피지 두루마리
  scroll:
    SHADOW +
    `<rect x="6.4" y="6" width="11.2" height="12.5" fill="#e7d6ae" ${OUTLINE}/>` +
    `<rect x="4.8" y="4" width="14.4" height="3.6" rx="1.8" fill="#cdb88c" ${OUTLINE}/>` +
    `<rect x="4.8" y="16.8" width="14.4" height="3.6" rx="1.8" fill="#cdb88c" ${OUTLINE}/>` +
    '<path d="M9 10.6h6M9 13.2h6" stroke="#9c8a64" stroke-width="1" stroke-linecap="round"/>',
  // 방패 — 검증 결과
  shield:
    `<path d="M12 3.4 18.8 5.8v5.4c0 4.7-2.9 7.8-6.8 9.4-3.9-1.6-6.8-4.7-6.8-9.4V5.8z" fill="#5f7fa8" ${OUTLINE}/>` +
    '<path d="M12 5.6 16.8 7.3v4c0 3.5-2.1 5.9-4.8 7.2-2.7-1.3-4.8-3.7-4.8-7.2v-4z" fill="#7ea3c9" stroke="#241a12" stroke-width="0.7"/>' +
    '<path d="M9.4 11.6l1.9 1.9 3.5-3.5" stroke="#f3f8fc" stroke-width="1.3" fill="none" stroke-linecap="round" stroke-linejoin="round"/>' +
    '<path d="M8.4 6.9c1-.6 2.2-1 3.6-1.2" stroke="rgba(255,255,255,0.45)" stroke-width="1" fill="none" stroke-linecap="round"/>',
  // 로켓 — 적용 상태
  rocket:
    SHADOW +
    `<path d="M19.9 4.1c.5 3.5-.7 6.8-3.4 9.5l-2.6 2.6-6.1-6.1 2.6-2.6c2.7-2.7 6-3.9 9.5-3.4z" fill="#c6ccd2" ${OUTLINE}/>` +
    '<circle cx="14.7" cy="9.3" r="1.8" fill="#7ea3c9" stroke="#241a12" stroke-width="0.8"/>' +
    `<path d="M8.6 9.3 5.6 12.3l3.4.5z" fill="#c45050" ${OUTLINE}/>` +
    `<path d="M14.7 15.4l-3 3 .5 3.4z" fill="#c45050" ${OUTLINE}/>` +
    '<path d="M8.2 15.8c-1.4.3-2.6 1.5-3.1 3.1 1.6-.5 2.8-1.7 3.1-3.1z" fill="#e3a23c" stroke="#241a12" stroke-width="0.7"/>' +
    '<path d="M16.4 5.9c.9.2 1.6.9 1.8 1.8" stroke="rgba(255,255,255,0.5)" stroke-width="1" fill="none" stroke-linecap="round"/>',
  // 수정구 — 라이브 미리보기
  orb:
    SHADOW +
    '<path d="M7.6 16.6h8.8l1 2.9H6.6z" fill="#8a5a32" stroke="#241a12" stroke-width="0.9"/>' +
    `<circle cx="12" cy="11" r="6.6" fill="#6fb6dc" ${OUTLINE}/>` +
    '<circle cx="12" cy="11" r="4" fill="#a8d9f0" opacity="0.7"/>' +
    '<path d="M8.6 8.4c.8-1.1 2-1.8 3.2-2" stroke="rgba(255,255,255,0.7)" stroke-width="1.2" fill="none" stroke-linecap="round"/>'
} as const

export type EditorIconName = keyof typeof ICONS

// 아이콘 span 생성. SVG 마크업은 위 정적 문자열뿐이라 innerHTML이 안전하다.
export const editorIcon = (name: EditorIconName, sizePx = 16): HTMLSpanElement => {
  const span = document.createElement('span')
  span.className = 'inline-block shrink-0 align-middle leading-none'
  span.setAttribute('aria-hidden', 'true')
  span.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="${sizePx}" height="${sizePx}">${ICONS[name]}</svg>`
  return span
}
