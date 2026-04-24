import { describe, expect, it } from 'vitest'

import { createNpcCharacter } from '../characterState'
import { resolveCharacterInteractionTarget } from './resolveCharacterInteractionTarget'

describe('resolveCharacterInteractionTarget', () => {
  const sourceCharacter = createNpcCharacter({
    id: 'player',
    appearanceType: 'character_adventurer_brown_hair',
    position: {
      x: 5,
      y: 5
    },
    facing: 'right',
    collisionSize: {
      width: 1,
      height: 1
    }
  })

  it('returns the closest interactable target in front of the source character', () => {
    const nearestTarget = createNpcCharacter({
      id: 'blacksmith',
      appearanceType: 'character_bearded_apron_man',
      position: {
        x: 6,
        y: 5
      },
      collisionSize: {
        width: 1,
        height: 1
      }
    })
    const fartherTarget = createNpcCharacter({
      id: 'guard',
      appearanceType: 'character_knight_closed_helmet',
      position: {
        x: 6,
        y: 5.4
      },
      collisionSize: {
        width: 1,
        height: 1
      }
    })

    expect(
      resolveCharacterInteractionTarget({
        sourceCharacter,
        targetCharacters: [fartherTarget, nearestTarget],
        canReceiveInteraction: () => true
      })
    ).toMatchObject({
      id: 'blacksmith'
    })
  })

  it('ignores targets that are not in the facing probe', () => {
    const behindTarget = createNpcCharacter({
      id: 'wizard',
      appearanceType: 'character_wizard_purple',
      position: {
        x: 4,
        y: 5
      },
      collisionSize: {
        width: 1,
        height: 1
      }
    })

    expect(
      resolveCharacterInteractionTarget({
        sourceCharacter,
        targetCharacters: [behindTarget],
        canReceiveInteraction: () => true
      })
    ).toBeUndefined()
  })
})
