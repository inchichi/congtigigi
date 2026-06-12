import type { MonsterAnimationTextures } from './monsterAnimationTextures'
import {
  createFrameTexturesFromBand,
  createFrameTexturesFromRects,
  loadImage,
  type FrameBand,
  type FrameRect
} from './loadMonsterSheetTextures'

const MONSTER_SLIME_MOTION_SHEET_URL = new URL(
  '../assets/monsters/몬스터-말캉이.png',
  import.meta.url
).href
const MONSTER_SLIME_IDLE_BAND: FrameBand = {
  top: 45,
  bottom: 140
}
const MONSTER_SLIME_RUN_BAND: FrameBand = {
  top: 250,
  bottom: 355
}
const MONSTER_SLIME_HIT_FRAME: FrameRect = {
  left: 1140,
  top: 610,
  width: 260,
  height: 180
}
const MONSTER_SLIME_ATTACK_BAND: FrameBand = {
  top: 935,
  bottom: 1045
}
const MONSTER_SLIME_IDLE_FRAME_COUNT = 4
const MONSTER_SLIME_RUN_FRAME_COUNT = 6
const MONSTER_SLIME_ATTACK_FRAME_COUNT = 7

const isSlimeForegroundPixel = (
  red: number,
  green: number,
  blue: number,
  alpha: number
): boolean => {
  if (alpha === 0) {
    return false
  }

  const maxChannel = Math.max(red, green, blue)
  const minChannel = Math.min(red, green, blue)
  const brightness = (red + green + blue) / 3
  const saturation = maxChannel - minChannel

  if (brightness > 220 && saturation < 28) {
    return false
  }

  if (saturation < 24 && brightness < 165) {
    return false
  }

  if (brightness < 70 && saturation < 18) {
    return false
  }

  return true
}

export const loadMonsterSlimeAnimationTextures =
  async (): Promise<MonsterAnimationTextures> => {
    const image = await loadImage(MONSTER_SLIME_MOTION_SHEET_URL)
    const sourceCanvas = document.createElement('canvas')
    const sourceContext = sourceCanvas.getContext('2d')

    if (!sourceContext) {
      throw new Error('Could not create canvas context for monster slime motion')
    }

    sourceCanvas.width = image.naturalWidth
    sourceCanvas.height = image.naturalHeight
    sourceContext.drawImage(image, 0, 0)
    const sourceImageData = sourceContext.getImageData(
      0,
      0,
      sourceCanvas.width,
      sourceCanvas.height
    )

    return {
      idleLeft: createFrameTexturesFromBand({
        sourceImageData,
        band: MONSTER_SLIME_IDLE_BAND,
        expectedFrameCount: MONSTER_SLIME_IDLE_FRAME_COUNT,
        framePaddingY: 2,
        pixelPredicate: isSlimeForegroundPixel
      }),
      idleRight: createFrameTexturesFromBand({
        sourceImageData,
        band: MONSTER_SLIME_IDLE_BAND,
        expectedFrameCount: MONSTER_SLIME_IDLE_FRAME_COUNT,
        mirrorX: true,
        framePaddingY: 2,
        pixelPredicate: isSlimeForegroundPixel
      }),
      runLeft: createFrameTexturesFromBand({
        sourceImageData,
        band: MONSTER_SLIME_RUN_BAND,
        expectedFrameCount: MONSTER_SLIME_RUN_FRAME_COUNT,
        framePaddingY: 2,
        pixelPredicate: isSlimeForegroundPixel
      }),
      runRight: createFrameTexturesFromBand({
        sourceImageData,
        band: MONSTER_SLIME_RUN_BAND,
        expectedFrameCount: MONSTER_SLIME_RUN_FRAME_COUNT,
        mirrorX: true,
        framePaddingY: 2,
        pixelPredicate: isSlimeForegroundPixel
      }),
      hitLeft: createFrameTexturesFromRects({
        sourceImageData,
        frames: [MONSTER_SLIME_HIT_FRAME],
        pixelPredicate: isSlimeForegroundPixel
      }),
      hitRight: createFrameTexturesFromRects({
        sourceImageData,
        frames: [MONSTER_SLIME_HIT_FRAME],
        mirrorX: true,
        pixelPredicate: isSlimeForegroundPixel
      }),
      attackLeft: createFrameTexturesFromBand({
        sourceImageData,
        band: MONSTER_SLIME_ATTACK_BAND,
        expectedFrameCount: MONSTER_SLIME_ATTACK_FRAME_COUNT,
        framePaddingY: 2,
        pixelPredicate: isSlimeForegroundPixel
      }),
      attackRight: createFrameTexturesFromBand({
        sourceImageData,
        band: MONSTER_SLIME_ATTACK_BAND,
        expectedFrameCount: MONSTER_SLIME_ATTACK_FRAME_COUNT,
        mirrorX: true,
        framePaddingY: 2,
        pixelPredicate: isSlimeForegroundPixel
      })
    }
  }
