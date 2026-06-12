import type { MonsterAnimationTextures } from './monsterAnimationTextures'
import {
  createFrameTexturesFromBand,
  createFrameTexturesFromRects,
  loadImage,
  type FrameBand,
  type FrameRect
} from './loadMonsterSheetTextures'

const MONSTER_PIG_MOTION_SHEET_URL = new URL(
  '../assets/monsters/monster-pig-sheet.png',
  import.meta.url
).href
const MONSTER_PIG_IDLE_BAND: FrameBand = {
  top: 40,
  bottom: 145
}
const MONSTER_PIG_RUN_BAND: FrameBand = {
  top: 235,
  bottom: 340
}
const MONSTER_PIG_HIT_FRAME: FrameRect = {
  left: 1180,
  top: 640,
  width: 228,
  height: 140
}
const MONSTER_PIG_ATTACK_FRAME_RECTS: FrameRect[] = [
  { left: 48, top: 844, width: 156, height: 124 },
  { left: 236, top: 844, width: 184, height: 124 },
  { left: 416, top: 842, width: 248, height: 126 },
  { left: 648, top: 836, width: 242, height: 134 },
  { left: 886, top: 841, width: 168, height: 124 },
  { left: 1088, top: 853, width: 174, height: 116 },
  { left: 1308, top: 850, width: 166, height: 116 }
]
const MONSTER_PIG_IDLE_FRAME_COUNT = 4
const MONSTER_PIG_RUN_FRAME_COUNT = 7

const isPigForegroundPixel = (
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

  if (brightness < 18 && saturation < 18) {
    return false
  }

  if (brightness > 76 && saturation < 28) {
    return false
  }

  return true
}

export const loadMonsterPigAnimationTextures =
  async (): Promise<MonsterAnimationTextures> => {
    const image = await loadImage(MONSTER_PIG_MOTION_SHEET_URL)
    const sourceCanvas = document.createElement('canvas')
    const sourceContext = sourceCanvas.getContext('2d')

    if (!sourceContext) {
      throw new Error('Could not create canvas context for monster pig motion')
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
        band: MONSTER_PIG_IDLE_BAND,
        expectedFrameCount: MONSTER_PIG_IDLE_FRAME_COUNT,
        framePaddingY: 2,
        pixelPredicate: isPigForegroundPixel
      }),
      idleRight: createFrameTexturesFromBand({
        sourceImageData,
        band: MONSTER_PIG_IDLE_BAND,
        expectedFrameCount: MONSTER_PIG_IDLE_FRAME_COUNT,
        mirrorX: true,
        framePaddingY: 2,
        pixelPredicate: isPigForegroundPixel
      }),
      runLeft: createFrameTexturesFromBand({
        sourceImageData,
        band: MONSTER_PIG_RUN_BAND,
        expectedFrameCount: MONSTER_PIG_RUN_FRAME_COUNT,
        framePaddingY: 2,
        pixelPredicate: isPigForegroundPixel
      }),
      runRight: createFrameTexturesFromBand({
        sourceImageData,
        band: MONSTER_PIG_RUN_BAND,
        expectedFrameCount: MONSTER_PIG_RUN_FRAME_COUNT,
        mirrorX: true,
        framePaddingY: 2,
        pixelPredicate: isPigForegroundPixel
      }),
      hitLeft: createFrameTexturesFromRects({
        sourceImageData,
        frames: [MONSTER_PIG_HIT_FRAME],
        pixelPredicate: isPigForegroundPixel
      }),
      hitRight: createFrameTexturesFromRects({
        sourceImageData,
        frames: [MONSTER_PIG_HIT_FRAME],
        mirrorX: true,
        pixelPredicate: isPigForegroundPixel
      }),
      attackLeft: createFrameTexturesFromRects({
        sourceImageData,
        frames: MONSTER_PIG_ATTACK_FRAME_RECTS,
        pixelPredicate: isPigForegroundPixel
      }),
      attackRight: createFrameTexturesFromRects({
        sourceImageData,
        frames: MONSTER_PIG_ATTACK_FRAME_RECTS,
        mirrorX: true,
        pixelPredicate: isPigForegroundPixel
      })
    }
  }
