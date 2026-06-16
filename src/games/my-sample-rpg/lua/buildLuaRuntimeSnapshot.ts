import {
  QUEST_DEFINITIONS,
  getQuestProgress,
  isQuestUnlocked,
  type QuestLogState
} from '../questLog'
import type { PlayerInventory } from '../playerInventory'
import type { PlayerProfile } from '../playerProfile'
import type { LuaRuntimeSnapshot } from './createLuaCharacterControllerRuntime'

// Phase 2 읽기 채널: 게임의 권위 있는 상태를 Lua 가 읽을 평면 키 스냅샷으로 변환한다.
// 순수 함수라 단위 테스트가 쉽고, 렌더러는 매 변경 시 이 결과를 pushSnapshot 으로 밀어넣는다.
// 키 규약은 createLuaCharacterControllerRuntime 의 engine.quest/inventory/player/scene 읽기와 1:1.
export const buildLuaRuntimeSnapshot = ({
  questLog,
  inventory,
  profile,
  sceneId
}: {
  questLog: QuestLogState
  inventory: PlayerInventory
  profile: PlayerProfile
  sceneId: string
}): LuaRuntimeSnapshot => {
  const strings: Record<string, string> = {
    'scene:id': sceneId,
    'p:name': profile.name
  }
  const numbers: Record<string, number> = {
    'p:level': profile.level,
    'p:hp': profile.hp.current,
    'p:max_hp': profile.hp.max,
    'p:gold': inventory.gold
  }
  const booleans: Record<string, boolean> = {}

  for (const definition of QUEST_DEFINITIONS) {
    const progress = getQuestProgress(questLog, definition.id)
    strings[`q:status:${definition.id}`] = progress.status
    booleans[`q:unlocked:${definition.id}`] = isQuestUnlocked(questLog, definition.id)

    for (const [objectiveId, count] of Object.entries(progress.objectives)) {
      numbers[`q:obj:${definition.id}:${objectiveId}`] = count
    }
  }

  for (const slot of inventory.slots) {
    if (slot) {
      const key = `inv:${slot.id}`
      numbers[key] = (numbers[key] ?? 0) + slot.quantity
    }
  }

  return { strings, numbers, booleans }
}
