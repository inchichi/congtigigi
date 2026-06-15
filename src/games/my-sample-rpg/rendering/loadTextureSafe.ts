import { Assets, Texture } from 'pixi.js'

let placeholderTexture: Texture | undefined

// 다운로드되지 않은 Git LFS 포인터처럼 디코딩 불가한 이미지를 만나도 렌더러가 통째로 죽지
// 않도록, 1x1 투명 텍스처로 대체한다. (에셋 하나 누락에 게임 전체가 "Renderer Failed" 되는 것 방지)
const getPlaceholderTexture = (): Texture => {
  if (placeholderTexture) {
    return placeholderTexture
  }

  const canvas = document.createElement('canvas')
  canvas.width = 1
  canvas.height = 1
  placeholderTexture = Texture.from(canvas)
  return placeholderTexture
}

// Assets.load 의 안전 버전: 실패하면 던지지 않고 플레이스홀더 텍스처를 돌려준다.
export const loadTextureSafe = async (url: string): Promise<Texture> => {
  try {
    return await Assets.load<Texture>(url)
  } catch (error) {
    console.warn(`[asset] 텍스처 로드 실패 → 플레이스홀더 사용: ${url}`, error)
    return getPlaceholderTexture()
  }
}
