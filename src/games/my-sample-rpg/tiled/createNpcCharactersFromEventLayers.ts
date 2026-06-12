import {
  createIdleNpcCharacterController,
  createLuaCharacterController,
  createNpcCharacter,
  type CharacterState,
  type LuaCharacterControllerConfig,
  type LuaCharacterControllerConfigValue
} from '../characterState'
import type {
  ParsedTiledEvent,
  ParsedTiledMap,
  ParsedTiledPropertyValue
} from './parseTiledMap'

type CreateNpcCharactersFromEventLayersInput = {
  map: ParsedTiledMap
  defaultPixelWidth: number
  defaultPixelHeight: number
}

export const createNpcCharactersFromEventLayers = ({
  map,
  defaultPixelWidth,
  defaultPixelHeight
}: CreateNpcCharactersFromEventLayersInput): CharacterState[] =>
  map.eventLayers.flatMap((eventLayer) =>
    eventLayer.events.flatMap((event) => {
      if (!event.visible || event.className !== 'character' || !event.appearanceType) {
        return []
      }

      return [
        createNpcCharacterFromEvent({
          event,
          tileWidth: map.tileWidth,
          tileHeight: map.tileHeight,
          defaultPixelWidth,
          defaultPixelHeight
        })
      ]
    })
  )

const createNpcCharacterFromEvent = ({
  event,
  tileWidth,
  tileHeight,
  defaultPixelWidth,
  defaultPixelHeight
}: {
  event: ParsedTiledEvent
  tileWidth: number
  tileHeight: number
  defaultPixelWidth: number
  defaultPixelHeight: number
}): CharacterState => {
  const characterPixelWidth = event.width || defaultPixelWidth
  const characterPixelHeight = event.height || defaultPixelHeight
  const blocksMovement =
    typeof event.properties.blocksMovement === 'boolean'
      ? event.properties.blocksMovement
      : true
  const level =
    getNumberProperty(event.properties, 'monster.level') ??
    getNumberProperty(event.properties, 'level')
  const displayText = getStringProperty(event.properties, 'displayText')
  const controller = createControllerFromEvent(event)

  return createNpcCharacter({
    id: event.name || `character-${event.id}`,
    appearanceType: event.appearanceType ?? '',
    level,
    displayText,
    position: {
      x: (event.x - characterPixelWidth / 2) / tileWidth,
      y: (event.y - characterPixelHeight) / tileHeight
    },
    collisionSize: {
      width: characterPixelWidth / tileWidth,
      height: characterPixelHeight / tileHeight
    },
    blocksMovement,
    controller
  })
}

const CONTROLLER_PROPERTY_PREFIX = 'controller.'
const RESERVED_CONTROLLER_CONFIG_KEYS = new Set([
  'kind',
  'scriptId',
  'radiusInTiles',
  'moveSpeedTilesPerSecond'
])

const createControllerFromEvent = (event: ParsedTiledEvent) => {
  const controllerKind = getStringProperty(event.properties, 'controller.kind')
  const controllerScriptId = getStringProperty(
    event.properties,
    'controller.scriptId'
  )
  const resolvedControllerKind =
    controllerKind ?? (controllerScriptId ? 'lua' : 'idle')
  const moveSpeedTilesPerSecond = getNumberProperty(
    event.properties,
    'controller.moveSpeedTilesPerSecond'
  )

  if (resolvedControllerKind === 'idle') {
    return createIdleNpcCharacterController({
      moveSpeedTilesPerSecond
    })
  }

  if (resolvedControllerKind !== 'lua') {
    throw new Error(
      `Unsupported controller kind "${resolvedControllerKind}" on character ${event.name || event.id}`
    )
  }

  if (!controllerScriptId) {
    throw new Error(
      `Lua controller on character ${event.name || event.id} requires controller.scriptId`
    )
  }

  return createLuaCharacterController({
    scriptId: controllerScriptId,
    radiusInTiles:
      getNumberProperty(event.properties, 'controller.radiusInTiles') ?? 0,
    moveSpeedTilesPerSecond,
    config: createLuaControllerConfig(event.properties)
  })
}

const createLuaControllerConfig = (
  properties: Record<string, ParsedTiledPropertyValue>
): LuaCharacterControllerConfig =>
  Object.fromEntries(
    Object.entries(properties).flatMap(([propertyName, propertyValue]) => {
      if (!propertyName.startsWith(CONTROLLER_PROPERTY_PREFIX)) {
        return []
      }

      const configKey = propertyName.slice(CONTROLLER_PROPERTY_PREFIX.length)

      if (RESERVED_CONTROLLER_CONFIG_KEYS.has(configKey)) {
        return []
      }

      return [[configKey, normalizeLuaControllerConfigValue(configKey, propertyValue)]]
    })
  )

const normalizeLuaControllerConfigValue = (
  configKey: string,
  propertyValue: ParsedTiledPropertyValue
): LuaCharacterControllerConfigValue => {
  if (Array.isArray(propertyValue)) {
    if (propertyValue.every((value) => typeof value === 'string')) {
      return propertyValue
        .map((value) => value.trim())
        .filter((value) => value.length > 0)
    }

    if (
      propertyValue.every(
        (value) => typeof value === 'number' || typeof value === 'boolean'
      )
    ) {
      return propertyValue
    }

    throw new Error(
      `Unsupported controller config property "controller.${configKey}": list items must all share one primitive type`
    )
  }

  if (typeof propertyValue !== 'string') {
    return propertyValue
  }

  if (configKey.endsWith('Lines')) {
    const multilineValue = propertyValue

    return multilineValue
      .split(/\r?\n/u)
      .map((line) => line.trim())
      .filter((line) => line.length > 0)
  }

  return propertyValue
}

const getStringProperty = (
  properties: Record<string, ParsedTiledPropertyValue>,
  propertyName: string
): string | undefined =>
  typeof properties[propertyName] === 'string'
    ? properties[propertyName]
    : undefined

const getNumberProperty = (
  properties: Record<string, ParsedTiledPropertyValue>,
  propertyName: string
): number | undefined =>
  typeof properties[propertyName] === 'number'
    ? properties[propertyName]
    : undefined
