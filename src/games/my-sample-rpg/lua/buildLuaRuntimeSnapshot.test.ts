import { describe, expect, it } from 'vitest'
import { buildLuaRuntimeSnapshot } from './buildLuaRuntimeSnapshot'
import { createInitialQuestLog, QUEST_DEFINITIONS } from '../questLog'
import { createInitialPlayerProfile } from '../playerProfile'
import type { PlayerInventory } from '../playerInventory'

describe('buildLuaRuntimeSnapshot', () => {
  const inventory: PlayerInventory = {
    gold: 250,
    slots: [
      { id: 'potion_hp', label: 'HP 물약', quantity: 3 },
      undefined,
      { id: 'potion_hp', label: 'HP 물약', quantity: 2 }
    ]
  }

  it('flattens scene/profile/inventory/quest state into snapshot keys', () => {
    const profile = createInitialPlayerProfile()
    const snapshot = buildLuaRuntimeSnapshot({
      questLog: createInitialQuestLog(),
      inventory,
      profile,
      sceneId: 'town'
    })

    expect(snapshot.strings['scene:id']).toBe('town')
    expect(snapshot.strings['p:name']).toBe(profile.name)
    expect(snapshot.numbers['p:level']).toBe(profile.level)
    expect(snapshot.numbers['p:max_hp']).toBe(profile.hp.max)
    expect(snapshot.numbers['p:gold']).toBe(250)
    // 같은 item id 슬롯 수량이 합산된다.
    expect(snapshot.numbers['inv:potion_hp']).toBe(5)
    // 정의된 모든 퀘스트가 status/unlocked 키를 갖는다.
    const firstQuestId = QUEST_DEFINITIONS[0].id
    expect(typeof snapshot.strings[`q:status:${firstQuestId}`]).toBe('string')
    expect(typeof snapshot.booleans[`q:unlocked:${firstQuestId}`]).toBe('boolean')
  })
})
