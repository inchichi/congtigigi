const UI_REFERENCE_WIDTH = 1600
const UI_REFERENCE_HEIGHT = 900
// 작은 창(에디터의 게임 프리뷰 iframe 등)에서는 UI가 창 크기에 비례해 계속 줄어야 한다.
// 예전 하한 0.72는 700px 남짓의 iframe에서도 HUD(설계 폭 ~770px)를 86% 크기로 고정해
// 오른쪽이 화면 밖으로 잘렸다. 0.3이면 320px 폭까지도 HUD가 잘리지 않는다(≈277px).
const UI_MIN_SCALE = 0.3
const UI_MAX_SCALE = 1
const UI_SCALE_MULTIPLIER = 1.2

export const getResponsiveUiScale = (): number => {
  const widthScale = window.innerWidth / UI_REFERENCE_WIDTH
  const heightScale = window.innerHeight / UI_REFERENCE_HEIGHT

  return (
    clamp(Math.min(widthScale, heightScale), UI_MIN_SCALE, UI_MAX_SCALE) *
    UI_SCALE_MULTIPLIER
  )
}

const clamp = (value: number, min: number, max: number): number =>
  Math.max(min, Math.min(value, max))
