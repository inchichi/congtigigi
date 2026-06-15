import { describe, expect, it } from 'vitest'

import { createInitialPlayerEquipment } from './playerEquipment'
import { createInitialPlayerInventory } from './playerInventory'
import { createInitialPlayerProfile } from './playerProfile'
import { createInitialPlayerQuickslots } from './playerQuickslots'
import { createInitialPlayerSkillSlots } from './playerSkillSlots'
import {
  PLAYER_SAVE_STATE_VERSION,
  normalizeStoredPlayerSaveState,
  parseStoredPlayerSaveState,
  serializePlayerSaveState,
  type PlayerSaveStateInput
} from './playerSaveState'

const createValidInput = (): PlayerSaveStateInput => ({
  profile: createInitialPlayerProfile(),
  equipment: createInitialPlayerEquipment(),
  inventory: createInitialPlayerInventory(),
  quickslots: createInitialPlayerQuickslots(),
  skillSlots: createInitialPlayerSkillSlots()
})

describe('serializePlayerSaveState / parseStoredPlayerSaveState', () => {
  it('round-trips a progressed player state unchanged', () => {
    const input = createValidInput()
    // 레벨업·스탯 소비·체력 손실·골드 변동·장비/퀵슬롯/스킬슬롯 배치를 모사한다.
    input.profile.level = 7
    input.profile.experience.current = 320
    input.profile.statPoints = 5
    input.profile.hp.current = 11
    input.profile.stats.strength = 12
    input.inventory.gold = 4250
    input.inventory.slots[0] = { id: 'iron-sword', label: '강철검', quantity: 1 }
    input.quickslots.slots[0] = { inventorySlotIndex: 3 }
    input.skillSlots.slots[1] = { skillId: 'smash' }

    const restored = parseStoredPlayerSaveState(serializePlayerSaveState(input))

    expect(restored).toEqual({
      version: PLAYER_SAVE_STATE_VERSION,
      ...input
    })
  })

  it('stamps the current schema version into the serialized payload', () => {
    const serialized = serializePlayerSaveState(createValidInput())

    expect(JSON.parse(serialized).version).toBe(PLAYER_SAVE_STATE_VERSION)
  })

  it('restores empty inventory slots (JSON null) back to undefined', () => {
    const input = createValidInput()
    input.inventory = createInitialPlayerInventory({ slotCount: 3 })
    input.inventory.slots[1] = { id: 'health-potion', label: '포션', quantity: 5 }

    const restored = parseStoredPlayerSaveState(serializePlayerSaveState(input))

    expect(restored?.inventory.slots).toEqual([
      undefined,
      { id: 'health-potion', label: '포션', quantity: 5 },
      undefined
    ])
  })
})

describe('parseStoredPlayerSaveState rejects unusable data', () => {
  it('returns undefined for empty or missing raw input', () => {
    expect(parseStoredPlayerSaveState(null)).toBeUndefined()
    expect(parseStoredPlayerSaveState(undefined)).toBeUndefined()
    expect(parseStoredPlayerSaveState('')).toBeUndefined()
  })

  it('returns undefined for malformed JSON', () => {
    expect(parseStoredPlayerSaveState('{ not valid json')).toBeUndefined()
  })

  it('discards a save from a different schema version', () => {
    const serialized = serializePlayerSaveState(createValidInput())
    const downgraded = JSON.parse(serialized)
    downgraded.version = PLAYER_SAVE_STATE_VERSION + 1

    expect(
      parseStoredPlayerSaveState(JSON.stringify(downgraded))
    ).toBeUndefined()
  })

  it('discards a save with a structurally broken profile', () => {
    const serialized = serializePlayerSaveState(createValidInput())
    const broken = JSON.parse(serialized)
    delete broken.profile.hp

    expect(parseStoredPlayerSaveState(JSON.stringify(broken))).toBeUndefined()
  })

  it('discards a save whose numeric fields are the wrong type', () => {
    const serialized = serializePlayerSaveState(createValidInput())
    const broken = JSON.parse(serialized)
    broken.profile.level = 'seven'

    expect(parseStoredPlayerSaveState(JSON.stringify(broken))).toBeUndefined()
  })
})

describe('normalizeStoredPlayerSaveState forgiving slot handling', () => {
  it('coerces a corrupted inventory item into an empty slot without dropping the save', () => {
    const input = createValidInput()
    const raw = JSON.parse(serializePlayerSaveState(input))
    raw.inventory.slots[0] = { id: 'broken' } // quantity/label 누락

    const restored = normalizeStoredPlayerSaveState(raw)

    expect(restored).toBeDefined()
    expect(restored?.inventory.slots[0]).toBeUndefined()
  })
})
