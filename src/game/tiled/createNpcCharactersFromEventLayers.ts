import {
  createIdleNpcCharacterController,
  createNpcCharacter,
  type CharacterState
} from '../characterState'
import type { ParsedTiledEvent, ParsedTiledMap } from './parseTiledMap'

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

  return createNpcCharacter({
    id: event.name || `character-${event.id}`,
    appearanceType: event.appearanceType ?? '',
    position: {
      x: (event.x - characterPixelWidth / 2) / tileWidth,
      y: (event.y - characterPixelHeight) / tileHeight
    },
    collisionSize: {
      width: characterPixelWidth / tileWidth,
      height: characterPixelHeight / tileHeight
    },
    blocksMovement,
    controller: createIdleNpcCharacterController()
  })
}
