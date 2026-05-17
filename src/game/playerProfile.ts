export type PlayerStatBlock = {
  attack: number
  defense: number
  agility: number
}

export type PlayerResource = {
  current: number
  max: number
}

export type PlayerSkillSlot = {
  hotkey: string
  label: string
  description: string
}

export type PlayerProfile = {
  name: string
  job: string
  level: number
  hp: PlayerResource
  mp: PlayerResource
  stats: PlayerStatBlock
  skills: PlayerSkillSlot[]
}

export const createInitialPlayerProfile = (): PlayerProfile => ({
  name: 'Arin',
  job: 'Wanderer',
  level: 1,
  hp: {
    current: 24,
    max: 24
  },
  mp: {
    current: 12,
    max: 12
  },
  stats: {
    attack: 5,
    defense: 3,
    agility: 4
  },
  skills: [
    {
      hotkey: '1',
      label: 'Slash',
      description: 'Quick close-range strike'
    },
    {
      hotkey: '2',
      label: 'Guard',
      description: 'Brace against incoming damage'
    },
    {
      hotkey: '3',
      label: 'Dash',
      description: 'Short burst of movement'
    },
    {
      hotkey: '4',
      label: 'Focus',
      description: 'Recover a small amount of mana'
    }
  ]
})
