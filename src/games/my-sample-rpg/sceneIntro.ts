const SCENE_INTRO_MESSAGES: Record<string, string> = {
  town: '티르코네일 마을',
  'hunting-ground': '슬라임 숲',
  cave: '동굴'
}

export const getSceneIntroMessage = (sceneId: string): string =>
  SCENE_INTRO_MESSAGES[sceneId] ?? ''
