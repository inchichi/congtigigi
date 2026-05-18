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

export const PLAYER_JOB_PROMOTION_LEVEL = 10
export const PLAYER_STARTING_JOB = '초보자'

export const createInitialPlayerProfile = (): PlayerProfile => ({
  name: '준수',
  job: PLAYER_STARTING_JOB,
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
      label: '베기',
      description: '재빠른 근거리 공격'
    },
    {
      hotkey: '2',
      label: '방어 자세',
      description: '들어오는 피해를 막아냅니다'
    },
    {
      hotkey: '3',
      label: '돌진',
      description: '짧게 빠르게 이동합니다'
    },
    {
      hotkey: '4',
      label: '집중',
      description: '마나를 조금 회복합니다'
    }
  ]
})

export const getPlayerJobDisplayName = ({
  job,
  level
}: Pick<PlayerProfile, 'job' | 'level'>): string =>
  level >= PLAYER_JOB_PROMOTION_LEVEL ? `${job} · 전직 가능` : job

export const isPlayerJobPromotionAvailable = ({
  level
}: Pick<PlayerProfile, 'level'>): boolean =>
  level >= PLAYER_JOB_PROMOTION_LEVEL
