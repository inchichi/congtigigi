import { describe, expect, it } from 'vitest'

import {
  createInitialPlayerEquipment,
  getPlayerEquipmentItemDefinitionBySlotId
} from './playerEquipment'

describe('createInitialPlayerEquipment', () => {
  it('creates a level 1 starter gear set', () => {
    expect(createInitialPlayerEquipment()).toEqual({
      setName: '기본 장비',
      level: 1,
      slots: [
        {
          id: 'weapon',
          label: '무기',
          item: {
            id: 'basic-sword',
            label: '기본 무기',
            level: 1,
            description: '초보용 근접 무기'
          }
        },
        {
          id: 'armor',
          label: '옷',
          item: {
            id: 'basic-armor',
            label: '기본 옷',
            level: 1,
            description: '초보용 옷'
          }
        },
        {
          id: 'boots',
          label: '신발',
          item: {
            id: 'basic-boots',
            label: '기본 신발',
            level: 1,
            description: '초보용 신발'
          }
        },
        {
          id: 'accessory',
          label: '장신구',
          item: {
            id: 'basic-charm',
            label: '기본 장신구',
            level: 1,
            description: '초보용 장신구'
          }
        }
      ]
    })
  })

  it('uses the tiny dungeon weapon tile for the starter weapon', () => {
    expect(getPlayerEquipmentItemDefinitionBySlotId('weapon')).toMatchObject({
      icon: {
        key: 'tiny-dungeon-weapon',
        scale: 1.6
      }
    })
  })
})
