import { Texture } from 'pixi.js'

export type FrameBand = {
  top: number
  bottom: number
}

export type FrameRect = {
  left: number
  top: number
  width: number
  height: number
}

type FrameBounds = {
  minX: number
  minY: number
  maxX: number
  maxY: number
}

export const loadImage = async (url: string): Promise<HTMLImageElement> => {
  const image = new Image()
  image.decoding = 'async'
  image.src = url
  await image.decode()
  return image
}

export const createFrameTexturesFromBand = ({
  sourceImageData,
  band,
  expectedFrameCount,
  framePaddingY = 10,
  pixelPredicate,
  mirrorX = false
}: {
  sourceImageData: ImageData
  band: FrameBand
  expectedFrameCount: number
  framePaddingY?: number
  pixelPredicate: (
    red: number,
    green: number,
    blue: number,
    alpha: number
  ) => boolean
  mirrorX?: boolean
}): Texture[] => {
  const bounds = findForegroundBounds({
    sourceImageData,
    band,
    pixelPredicate
  })

  if (!bounds) {
    throw new Error(
      `Could not find sprite frames in band ${band.top}-${band.bottom}`
    )
  }

  const totalWidth = bounds.maxX - bounds.minX + 1
  const frameWidth = Math.ceil(totalWidth / expectedFrameCount)
  const totalCoveredWidth = frameWidth * expectedFrameCount
  const startLeft = clamp(
    Math.round(bounds.minX - (totalCoveredWidth - totalWidth) / 2),
    0,
    sourceImageData.width - totalCoveredWidth
  )
  const frameTop = clamp(
    bounds.minY - framePaddingY,
    0,
    sourceImageData.height - 1
  )
  const frameBottom = clamp(
    bounds.maxY + framePaddingY,
    frameTop,
    sourceImageData.height - 1
  )
  const frameHeight = frameBottom - frameTop + 1

  return Array.from({ length: expectedFrameCount }, (_, index) =>
    createCroppedTexture({
      sourceImageData,
      frame: {
        left: startLeft + index * frameWidth,
        top: frameTop,
        width: frameWidth,
        height: frameHeight
      },
      mirrorX,
      pixelPredicate
    })
  )
}

export const createFrameTexturesFromRects = ({
  sourceImageData,
  frames,
  mirrorX = false,
  pixelPredicate
}: {
  sourceImageData: ImageData
  frames: FrameRect[]
  mirrorX?: boolean
  pixelPredicate: (
    red: number,
    green: number,
    blue: number,
    alpha: number
  ) => boolean
}): Texture[] =>
  frames.map((frame) =>
    createCroppedTexture({
      sourceImageData,
      frame,
      mirrorX,
      pixelPredicate
    })
  )

const createCroppedTexture = ({
  sourceImageData,
  frame,
  mirrorX = false,
  pixelPredicate
}: {
  sourceImageData: ImageData
  frame: FrameRect
  mirrorX?: boolean
  pixelPredicate: (
    red: number,
    green: number,
    blue: number,
    alpha: number
  ) => boolean
}): Texture => {
  const frameCanvas = document.createElement('canvas')
  const frameContext = frameCanvas.getContext('2d')

  if (!frameContext) {
    throw new Error('Could not create canvas context for monster frame')
  }

  frameCanvas.width = frame.width
  frameCanvas.height = frame.height
  frameContext.imageSmoothingEnabled = false

  let frameImageData = frameContext.createImageData(frame.width, frame.height)

  for (let sourceY = 0; sourceY < frame.height; sourceY += 1) {
    const rowOffset = sourceY * frame.width

    for (let sourceX = 0; sourceX < frame.width; sourceX += 1) {
      const sampleX = mirrorX ? frame.width - 1 - sourceX : sourceX
      const sampleIndex =
        ((frame.top + sourceY) * sourceImageData.width + (frame.left + sampleX)) *
        4
      const targetIndex = (rowOffset + sourceX) * 4
      const red = sourceImageData.data[sampleIndex] ?? 0
      const green = sourceImageData.data[sampleIndex + 1] ?? 0
      const blue = sourceImageData.data[sampleIndex + 2] ?? 0
      const alpha = sourceImageData.data[sampleIndex + 3] ?? 0

      if (!pixelPredicate(red, green, blue, alpha)) {
        continue
      }

      frameImageData.data[targetIndex] = red
      frameImageData.data[targetIndex + 1] = green
      frameImageData.data[targetIndex + 2] = blue
      frameImageData.data[targetIndex + 3] = 255
    }
  }

  frameContext.putImageData(frameImageData, 0, 0)
  const texture = Texture.from(frameCanvas)

  texture.source.scaleMode = 'nearest'
  texture.source.addressMode = 'clamp-to-edge'

  return texture
}

const findForegroundBounds = ({
  sourceImageData,
  band,
  pixelPredicate
}: {
  sourceImageData: ImageData
  band: FrameBand
  pixelPredicate: (
    red: number,
    green: number,
    blue: number,
    alpha: number
  ) => boolean
}): FrameBounds | undefined => {
  const { width, data } = sourceImageData
  let minX = Number.POSITIVE_INFINITY
  let minY = Number.POSITIVE_INFINITY
  let maxX = Number.NEGATIVE_INFINITY
  let maxY = Number.NEGATIVE_INFINITY

  for (let y = band.top; y < band.bottom; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (
        !isForegroundPixel(data, width, x, y, pixelPredicate)
      ) {
        continue
      }

      minX = Math.min(minX, x)
      minY = Math.min(minY, y)
      maxX = Math.max(maxX, x)
      maxY = Math.max(maxY, y)
    }
  }

  if (
    minX === Number.POSITIVE_INFINITY ||
    minY === Number.POSITIVE_INFINITY ||
    maxX === Number.NEGATIVE_INFINITY ||
    maxY === Number.NEGATIVE_INFINITY
  ) {
    return undefined
  }

  return {
    minX,
    minY,
    maxX,
    maxY
  }
}

const isForegroundPixel = (
  data: Uint8ClampedArray,
  width: number,
  x: number,
  y: number,
  pixelPredicate: (
    red: number,
    green: number,
    blue: number,
    alpha: number
  ) => boolean
): boolean => {
  const index = (y * width + x) * 4

  return pixelPredicate(
    data[index] ?? 0,
    data[index + 1] ?? 0,
    data[index + 2] ?? 0,
    data[index + 3] ?? 0
  )
}

const clamp = (value: number, min: number, max: number): number =>
  Math.min(Math.max(value, min), max)
