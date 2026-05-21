export const getMonsterDisplayName = ({
  id,
  displayText
}: {
  id: string
  displayText?: string
}): string => {
  if (displayText) {
    return displayText
  }

  const suffixSeparatorIndex = id.lastIndexOf('-')

  if (suffixSeparatorIndex > 0) {
    return id.slice(0, suffixSeparatorIndex)
  }

  return id
}
