import { DOMParser } from '@xmldom/xmldom'

const FLIPPED_HORIZONTALLY_FLAG = 0x80000000
const FLIPPED_VERTICALLY_FLAG = 0x40000000
const FLIPPED_DIAGONALLY_FLAG = 0x20000000
const ROTATED_HEXAGONAL_120_FLAG = 0x10000000
const ALL_TRANSFORM_FLAGS =
  FLIPPED_HORIZONTALLY_FLAG |
  FLIPPED_VERTICALLY_FLAG |
  FLIPPED_DIAGONALLY_FLAG |
  ROTATED_HEXAGONAL_120_FLAG

export type ParsedTiledTile = {
  x: number
  y: number
  gid: number
  localId: number
  flipHorizontally: boolean
  flipVertically: boolean
  flipDiagonally: boolean
}

export type ParsedTiledLayer = {
  id: number
  name: string
  width: number
  height: number
  opacity: number
  visible: boolean
  tiles: ParsedTiledTile[]
}

export type ParsedTiledPropertyValue =
  | boolean
  | number
  | string
  | boolean[]
  | number[]
  | string[]

export type ParsedTiledEvent = {
  id: number
  name: string
  className: string
  x: number
  y: number
  width: number
  height: number
  visible: boolean
  properties: Record<string, ParsedTiledPropertyValue>
  appearanceType?: string
}

export type ParsedTiledEventLayer = {
  id: number
  name: string
  opacity: number
  visible: boolean
  events: ParsedTiledEvent[]
}

export type ParsedTiledTileset = {
  firstGid: number
  source: string
  name: string
  tileWidth: number
  tileHeight: number
  spacing: number
  margin: number
  tileCount: number
  columns: number
  image: {
    source: string
    width: number
    height: number
  }
  tileTypes: Record<number, string>
  tileProperties: Record<number, Record<string, ParsedTiledPropertyValue>>
}

export type ParsedTiledMap = {
  width: number
  height: number
  tileWidth: number
  tileHeight: number
  pixelWidth: number
  pixelHeight: number
  layers: ParsedTiledLayer[]
  eventLayers: ParsedTiledEventLayer[]
  tilesets: ParsedTiledTileset[]
}

export type ParseTiledMapInput = {
  mapXml: string
  externalTilesets: Record<string, string>
}

type DecodedTiledGlobalId = {
  gid: number
  flipHorizontally: boolean
  flipVertically: boolean
  flipDiagonally: boolean
}

export const decodeTiledGlobalId = (rawGid: number): DecodedTiledGlobalId => {
  const unsignedRawGid = rawGid >>> 0

  return {
    gid: unsignedRawGid & ~ALL_TRANSFORM_FLAGS,
    flipHorizontally: (unsignedRawGid & FLIPPED_HORIZONTALLY_FLAG) !== 0,
    flipVertically: (unsignedRawGid & FLIPPED_VERTICALLY_FLAG) !== 0,
    flipDiagonally: (unsignedRawGid & FLIPPED_DIAGONALLY_FLAG) !== 0
  }
}

export const parseTiledMap = ({
  mapXml,
  externalTilesets
}: ParseTiledMapInput): ParsedTiledMap => {
  const mapDocument = parseXmlDocument(mapXml)
  const mapElement = mapDocument.documentElement

  if (mapElement.nodeName !== 'map') {
    throw new Error('Expected the TMX root element to be <map>')
  }

  const orientation = getRequiredAttribute(mapElement, 'orientation')

  if (orientation !== 'orthogonal') {
    throw new Error(`Unsupported TMX orientation: ${orientation}`)
  }

  const width = getRequiredNumberAttribute(mapElement, 'width')
  const height = getRequiredNumberAttribute(mapElement, 'height')
  const tileWidth = getRequiredNumberAttribute(mapElement, 'tilewidth')
  const tileHeight = getRequiredNumberAttribute(mapElement, 'tileheight')
  const tilesetReferences = getDirectChildElements(mapElement, 'tileset')
  const tilesets = tilesetReferences
    .map((tilesetReference) => {
      const firstGid = getRequiredNumberAttribute(tilesetReference, 'firstgid')
      const source = getRequiredAttribute(tilesetReference, 'source')
      const tilesetXml = externalTilesets[source]

      if (!tilesetXml) {
        throw new Error(`Missing external TSX content for ${source}`)
      }

      return parseTiledTileset({ firstGid, source, tilesetXml })
    })
    .sort((left, right) => left.firstGid - right.firstGid)

  const layers = getDirectChildElements(mapElement, 'layer').map((layerElement) =>
    parseLayer({
      layerElement,
      mapWidth: width,
      mapHeight: height,
      tilesets
    })
  )
  const eventLayers = getDirectChildElements(mapElement, 'objectgroup').map(
    (objectGroupElement) => parseEventLayer(objectGroupElement)
  )

  return {
    width,
    height,
    tileWidth,
    tileHeight,
    pixelWidth: width * tileWidth,
    pixelHeight: height * tileHeight,
    layers,
    eventLayers,
    tilesets
  }
}

export const parseTiledTileset = ({
  firstGid,
  source,
  tilesetXml
}: {
  firstGid: number
  source: string
  tilesetXml: string
}): ParsedTiledTileset => {
  const tilesetDocument = parseXmlDocument(tilesetXml)
  const tilesetElement = tilesetDocument.documentElement

  if (tilesetElement.nodeName !== 'tileset') {
    throw new Error(`Expected the TSX root element to be <tileset> for ${source}`)
  }

  const imageElement = getRequiredDirectChildElement(tilesetElement, 'image')
  const tileTypes = Object.fromEntries(
    getDirectChildElements(tilesetElement, 'tile')
      .map((tileElement) => {
        const type =
          getStringAttribute(tileElement, 'class') ??
          getStringAttribute(tileElement, 'type')

        if (!type) {
          return undefined
        }

        return [getRequiredNumberAttribute(tileElement, 'id'), type] as const
      })
      .filter((entry): entry is readonly [number, string] => entry !== undefined)
  )
  const tileProperties = Object.fromEntries(
    getDirectChildElements(tilesetElement, 'tile')
      .map((tileElement) => {
        const id = getRequiredNumberAttribute(tileElement, 'id')
        const propertiesElement = getDirectChildElements(tileElement, 'properties')[0]

        if (!propertiesElement) {
          return undefined
        }

        const properties = parseProperties(propertiesElement)

        if (Object.keys(properties).length === 0) {
          return undefined
        }

        return [id, properties] as const
      })
      .filter((entry): entry is readonly [number, Record<string, ParsedTiledPropertyValue>] =>
        entry !== undefined
      )
  )

  return {
    firstGid,
    source,
    name: getRequiredAttribute(tilesetElement, 'name'),
    tileWidth: getRequiredNumberAttribute(tilesetElement, 'tilewidth'),
    tileHeight: getRequiredNumberAttribute(tilesetElement, 'tileheight'),
    spacing: getNumberAttribute(tilesetElement, 'spacing', 0),
    margin: getNumberAttribute(tilesetElement, 'margin', 0),
    tileCount: getRequiredNumberAttribute(tilesetElement, 'tilecount'),
    columns: getRequiredNumberAttribute(tilesetElement, 'columns'),
    image: {
      source: getRequiredAttribute(imageElement, 'source'),
      width: getRequiredNumberAttribute(imageElement, 'width'),
      height: getRequiredNumberAttribute(imageElement, 'height')
    },
    tileTypes,
    tileProperties
  }
}

const parseLayer = ({
  layerElement,
  mapWidth,
  mapHeight,
  tilesets
}: {
  layerElement: Element
  mapWidth: number
  mapHeight: number
  tilesets: ParsedTiledTileset[]
}): ParsedTiledLayer => {
  const width = getRequiredNumberAttribute(layerElement, 'width')
  const height = getRequiredNumberAttribute(layerElement, 'height')

  if (width !== mapWidth || height !== mapHeight) {
    throw new Error(
      `Layer ${getRequiredAttribute(layerElement, 'name')} has unexpected dimensions`
    )
  }

  const dataElement = getRequiredDirectChildElement(layerElement, 'data')
  const encoding = getRequiredAttribute(dataElement, 'encoding')

  if (encoding !== 'csv') {
    throw new Error(
      `Only CSV-encoded TMX layers are supported, received ${encoding}`
    )
  }

  const rawGids = parseCsvData(getElementTextContent(dataElement), width, height)
  const tiles = rawGids.flatMap((rawGid, index) => {
    const decodedGlobalId = decodeTiledGlobalId(rawGid)

    if (decodedGlobalId.gid === 0) {
      return []
    }

    const tileset = resolveTileset(decodedGlobalId.gid, tilesets)
    const x = index % width
    const y = Math.floor(index / width)

    return [
      {
        x,
        y,
        gid: decodedGlobalId.gid,
        localId: decodedGlobalId.gid - tileset.firstGid,
        flipHorizontally: decodedGlobalId.flipHorizontally,
        flipVertically: decodedGlobalId.flipVertically,
        flipDiagonally: decodedGlobalId.flipDiagonally
      }
    ]
  })

  return {
    id: getRequiredNumberAttribute(layerElement, 'id'),
    name: getRequiredAttribute(layerElement, 'name'),
    width,
    height,
    opacity: getNumberAttribute(layerElement, 'opacity', 1),
    visible: getNumberAttribute(layerElement, 'visible', 1) !== 0,
    tiles
  }
}

const parseEventLayer = (objectGroupElement: Element): ParsedTiledEventLayer => ({
  id: getRequiredNumberAttribute(objectGroupElement, 'id'),
  name: getRequiredAttribute(objectGroupElement, 'name'),
  opacity: getNumberAttribute(objectGroupElement, 'opacity', 1),
  visible: getNumberAttribute(objectGroupElement, 'visible', 1) !== 0,
  events: getDirectChildElements(objectGroupElement, 'object').map((objectElement) =>
    parseEvent(objectElement)
  )
})

const parseEvent = (objectElement: Element): ParsedTiledEvent => {
  const propertiesElement = getDirectChildElements(objectElement, 'properties')[0]
  const properties = propertiesElement ? parseProperties(propertiesElement) : {}
  const className =
    getStringAttribute(objectElement, 'class') ??
    getStringAttribute(objectElement, 'type') ??
    ''
  const appearanceType =
    className === 'character' && typeof properties.type === 'string'
      ? properties.type
      : undefined

  return {
    id: getRequiredNumberAttribute(objectElement, 'id'),
    name: getStringAttribute(objectElement, 'name') ?? '',
    className,
    x: getNumberAttribute(objectElement, 'x', 0),
    y: getNumberAttribute(objectElement, 'y', 0),
    width: getNumberAttribute(objectElement, 'width', 0),
    height: getNumberAttribute(objectElement, 'height', 0),
    visible: getNumberAttribute(objectElement, 'visible', 1) !== 0,
    properties,
    appearanceType
  }
}

const parseCsvData = (
  csvText: string,
  width: number,
  height: number
): number[] => {
  const values = csvText
    .split(',')
    .map((value) => value.trim())
    .filter((value) => value.length > 0)
    .map((value) => Number.parseInt(value, 10) >>> 0)

  const expectedTileCount = width * height

  if (values.length !== expectedTileCount) {
    throw new Error(
      `Expected ${expectedTileCount} tile values but received ${values.length}`
    )
  }

  return values
}

const parseProperties = (
  propertiesElement: Element
): Record<string, ParsedTiledPropertyValue> =>
  Object.fromEntries(
    getDirectChildElements(propertiesElement, 'property').map((propertyElement) => [
      getRequiredAttribute(propertyElement, 'name'),
      parsePropertyValue(propertyElement)
    ])
  )

const parsePropertyValue = (propertyElement: Element): ParsedTiledPropertyValue => {
  const type = propertyElement.getAttribute('type') ?? 'string'
  const rawValue =
    getStringAttribute(propertyElement, 'value') ?? getElementTextContent(propertyElement)

  switch (type) {
    case 'bool':
      return rawValue === 'true'
    case 'int':
    case 'float':
      return Number(rawValue)
    case 'list':
      return parseListPropertyValue(propertyElement)
    default:
      return rawValue
  }
}

const parseListPropertyValue = (
  propertyElement: Element
): boolean[] | number[] | string[] => {
  const defaultItemType = getStringAttribute(propertyElement, 'propertytype') ?? 'string'
  const values = getDirectChildElements(propertyElement, 'item').map((itemElement) =>
    parseListItemValue(itemElement, defaultItemType)
  )

  if (values.every((value) => typeof value === 'boolean')) {
    return values
  }

  if (values.every((value) => typeof value === 'number')) {
    return values
  }

  if (values.every((value) => typeof value === 'string')) {
    return values
  }

  throw new Error(
    `Mixed list item types are not supported for property "${getRequiredAttribute(propertyElement, 'name')}"`
  )
}

const parseListItemValue = (
  itemElement: Element,
  defaultItemType: string
): boolean | number | string => {
  const itemType = getStringAttribute(itemElement, 'type') ?? defaultItemType
  const rawValue =
    getStringAttribute(itemElement, 'value') ?? getElementTextContent(itemElement)

  switch (itemType) {
    case 'bool':
      return rawValue === 'true'
    case 'int':
    case 'float':
      return Number(rawValue)
    default:
      return rawValue
  }
}

const resolveTileset = (
  gid: number,
  tilesets: ParsedTiledTileset[]
): ParsedTiledTileset => {
  for (let index = tilesets.length - 1; index >= 0; index -= 1) {
    const tileset = tilesets[index]

    if (tileset.firstGid <= gid) {
      return tileset
    }
  }

  throw new Error(`Could not resolve a tileset for gid ${gid}`)
}

const parseXmlDocument = (xmlText: string): Document => {
  return new DOMParser().parseFromString(xmlText, 'text/xml')
}

const getRequiredAttribute = (element: Element, attributeName: string): string => {
  const attributeValue = element.getAttribute(attributeName)

  if (!attributeValue) {
    throw new Error(
      `Missing required attribute "${attributeName}" on <${element.nodeName}>`
    )
  }

  return attributeValue
}

const getStringAttribute = (
  element: Element,
  attributeName: string
): string | undefined => {
  const attributeValue = element.getAttribute(attributeName)

  if (attributeValue === null || attributeValue.length === 0) {
    return undefined
  }

  return attributeValue
}

const getRequiredNumberAttribute = (
  element: Element,
  attributeName: string
): number => {
  const attributeValue = getRequiredAttribute(element, attributeName)
  const parsedValue = Number.parseInt(attributeValue, 10)

  if (Number.isNaN(parsedValue)) {
    throw new Error(
      `Expected "${attributeName}" on <${element.nodeName}> to be a number`
    )
  }

  return parsedValue
}

const getNumberAttribute = (
  element: Element,
  attributeName: string,
  fallbackValue: number
): number => {
  const attributeValue = element.getAttribute(attributeName)

  if (attributeValue === null || attributeValue.length === 0) {
    return fallbackValue
  }

  const parsedValue = Number.parseFloat(attributeValue)

  if (Number.isNaN(parsedValue)) {
    throw new Error(
      `Expected "${attributeName}" on <${element.nodeName}> to be numeric`
    )
  }

  return parsedValue
}

const getRequiredDirectChildElement = (
  element: Element,
  childName: string
): Element => {
  const childElement = getDirectChildElements(element, childName)[0]

  if (!childElement) {
    throw new Error(
      `Missing required <${childName}> child under <${element.nodeName}>`
    )
  }

  return childElement
}

const getDirectChildElements = (
  element: Element,
  childName: string
): Element[] => {
  const childElements: Element[] = []

  for (let index = 0; index < element.childNodes.length; index += 1) {
    const childNode = element.childNodes[index]

    if (childNode.nodeType !== 1) {
      continue
    }

    if (childNode.nodeName !== childName) {
      continue
    }

    childElements.push(childNode as Element)
  }

  return childElements
}

const getElementTextContent = (element: Element): string => {
  return element.textContent ?? ''
}
