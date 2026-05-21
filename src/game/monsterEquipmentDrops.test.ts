import { describe, expect, it } from 'vitest'

import {
  MONSTER_EQUIPMENT_DROP_DEFINITIONS,
  rollMonsterEquipmentDrop
} from './monsterEquipmentDrops'

describe('rollMonsterEquipmentDrop', () => {
  it('returns one weapon or armor drop within the 90 percent drop chance', () => {
    const randomValues = [0.29, 0.99]

    expect(rollMonsterEquipmentDrop(() => randomValues.shift() ?? 0)).toEqual({
      dropId: 'Iron_Helmet_drop',
      itemId: 'Iron_Helmet',
      label: 'Iron_Helmet_drop'
    })
  })

  it('returns no equipment drop outside the 90 percent drop chance', () => {
    expect(rollMonsterEquipmentDrop(() => 0.9)).toBeUndefined()
  })

  it('exposes all armor folder equipment drops', () => {
    expect(MONSTER_EQUIPMENT_DROP_DEFINITIONS.map((drop) => drop.itemId)).toEqual([
      'iron-sword',
      'battle-axe',
      'long-spear',
      'quick-dagger',
      'spiked-mace',
      'magic-staff',
      'Leather_Armor',
      'Leather_Helmet',
      'Chain_Armor',
      'Chain_Helmet',
      'Iron_Armor',
      'Iron_Helmet'
    ])
  })
})
