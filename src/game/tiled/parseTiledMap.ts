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
}

export type ParsedTiledMap = {
  width: number
  height: number
  tileWidth: number
  tileHeight: number
  pixelWidth: number
  pixelHeight: number
  layers: ParsedTiledLayer[]
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

      return parseTileset({ firstGid, source, tilesetXml })
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

  return {
    width,
    height,
    tileWidth,
    tileHeight,
    pixelWidth: width * tileWidth,
    pixelHeight: height * tileHeight,
    layers,
    tilesets
  }
}

const parseTileset = ({
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
    }
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
