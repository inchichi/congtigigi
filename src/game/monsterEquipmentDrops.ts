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
    label: 'Leather_Armor_drop'
  },
  {
    dropId: 'Leather_Helmet_drop',
    itemId: 'Leather_Helmet',
    label: 'Leather_Helmet_drop'
  },
  {
    dropId: 'Chain_Armor_drop',
    itemId: 'Chain_Armor',
    label: 'Chain_Armor_drop'
  },
  {
    dropId: 'Chain_Helmet_drop',
    itemId: 'Chain_Helmet',
    label: 'Chain_Helmet_drop'
  },
  {
    dropId: 'Iron_Armor_drop',
    itemId: 'Iron_Armor',
    label: 'Iron_Armor_drop'
  },
  {
    dropId: 'Iron_Helmet_drop',
    itemId: 'Iron_Helmet',
    label: 'Iron_Helmet_drop'
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
