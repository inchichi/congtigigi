export type MonsterEquipmentDropDefinition = {
  dropId: string
  itemId: string
  label: string
}

export const MONSTER_EQUIPMENT_DROP_CHANCE = 0.9

export const MONSTER_EQUIPMENT_DROP_DEFINITIONS: MonsterEquipmentDropDefinition[] = [
  {
    dropId: 'iron-sword_drop',
    itemId: 'iron-sword',
    label: '강철 검'
  },
  {
    dropId: 'battle-axe_drop',
    itemId: 'battle-axe',
    label: '전투 도끼'
  },
  {
    dropId: 'long-spear_drop',
    itemId: 'long-spear',
    label: '장창'
  },
  {
    dropId: 'quick-dagger_drop',
    itemId: 'quick-dagger',
    label: '단검'
  },
  {
    dropId: 'spiked-mace_drop',
    itemId: 'spiked-mace',
    label: '철퇴'
  },
  {
    dropId: 'magic-staff_drop',
    itemId: 'magic-staff',
    label: '마법 지팡이'
  },
  {
    dropId: 'Leather_Armor_drop',
    itemId: 'Leather_Armor',
    label: '가죽 갑옷'
  },
  {
    dropId: 'Leather_Helmet_drop',
    itemId: 'Leather_Helmet',
    label: '가죽 투구'
  },
  {
    dropId: 'Chain_Armor_drop',
    itemId: 'Chain_Armor',
    label: '사슬 갑옷'
  },
  {
    dropId: 'Chain_Helmet_drop',
    itemId: 'Chain_Helmet',
    label: '사슬 투구'
  },
  {
    dropId: 'Iron_Armor_drop',
    itemId: 'Iron_Armor',
    label: '철 갑옷'
  },
  {
    dropId: 'Iron_Helmet_drop',
    itemId: 'Iron_Helmet',
    label: '철 투구'
  }
]

export const rollMonsterEquipmentDrop = (
  random: () => number = Math.random
): MonsterEquipmentDropDefinition | undefined => {
  if (random() >= MONSTER_EQUIPMENT_DROP_CHANCE) {
    return undefined
  }

  const dropIndex = Math.min(
    MONSTER_EQUIPMENT_DROP_DEFINITIONS.length - 1,
    Math.floor(random() * MONSTER_EQUIPMENT_DROP_DEFINITIONS.length)
  )

  return MONSTER_EQUIPMENT_DROP_DEFINITIONS[dropIndex]
}
