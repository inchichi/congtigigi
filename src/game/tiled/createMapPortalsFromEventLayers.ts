import type { CharacterMoveDirection } from '../characterState'
import type {
  ParsedTiledEvent,
  ParsedTiledMap,
  ParsedTiledPropertyValue
} from './parseTiledMap'

type CreateMapPortalsFromEventLayersInput = {
  map: ParsedTiledMap
}

export type MapPortal = {
  id: string
  appearanceType: string
  position: {
    x: number
    y: number
  }
  collisionSize: {
    width: number
    height: number
  }
  targetSceneId: string
  targetSpawn: {
    x: number
    y: number
  }
  targetFacing?: CharacterMoveDirection
}

export const createMapPortalsFromEventLayers = ({
  map
}: CreateMapPortalsFromEventLayersInput): MapPortal[] =>
  map.eventLayers.flatMap((eventLayer) =>
    eventLayer.events.flatMap((event) => {
      if (!event.visible || event.className !== 'portal') {
        return []
      }

      return createMapPortalFromEvent({
        event,
        tileWidth: map.tileWidth,
        tileHeight: map.tileHeight
      })
    })
  )

const createMapPortalFromEvent = ({
  event,
  tileWidth,
  tileHeight
}: {
  event: ParsedTiledEvent
  tileWidth: number
  tileHeight: number
}): MapPortal[] => {
  const targetSceneId = getStringProperty(event.properties, 'targetSceneId')
  const targetSpawnTileX = getNumberProperty(
    event.properties,
    'targetSpawnTileX'
  )
  const targetSpawnTileY = getNumberProperty(
    event.properties,
    'targetSpawnTileY'
  )

  if (
    !targetSceneId ||
    targetSpawnTileX === undefined ||
    targetSpawnTileY === undefined
  ) {
    return []
  }

  return [
    {
      id: event.name || `portal-${event.id}`,
      appearanceType:
        getStringProperty(event.properties, 'appearanceType') ??
        'stairs_stone_step_base_00',
      position: {
        x: event.x / tileWidth,
        y: event.y / tileHeight
      },
      collisionSize: {
        width: (event.width || tileWidth) / tileWidth,
        height: (event.height || tileHeight) / tileHeight
      },
      targetSceneId,
      targetSpawn: {
        x: targetSpawnTileX,
        y: targetSpawnTileY
      },
      targetFacing: getTargetFacing(
        getStringProperty(event.properties, 'targetFacing')
      )
    }
  ]
}

const getTargetFacing = (
  facing: string | undefined
): CharacterMoveDirection | undefined => {
  switch (facing) {
    case 'up':
    case 'down':
    case 'left':
    case 'right':
      return facing
    default:
      return undefined
  }
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
