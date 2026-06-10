const UI_REFERENCE_WIDTH = 1600
const UI_REFERENCE_HEIGHT = 900
const UI_MIN_SCALE = 0.72
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
