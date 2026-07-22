import townMapXml from '../games/my-sample-rpg/assets/maps/town.tmx?raw'
import huntingGroundMapXml from '../games/my-sample-rpg/assets/maps/hunting-ground.tmx?raw'
import caveMapXml from '../games/my-sample-rpg/assets/maps/cave.tmx?raw'
import townTilesetXml from '../games/my-sample-rpg/assets/tilesets/town-32.tsx?raw'
import tinyDungeonTilesetXml from '../games/my-sample-rpg/assets/tilesets/tiny-dungeon-16.tsx?raw'
import type { GameFile } from './loadGame'

// 에디터와 테마 작업실이 공유하는 My Sample RPG 기준 파일 묶음.
// 두 페이지가 서로 다른 프로필을 만들지 않도록 한 곳에서만 정의한다.
export const SAMPLE_GAME_FILES: GameFile[] = [
  { name: 'town.tmx', path: 'src/games/my-sample-rpg/assets/maps/town.tmx', text: townMapXml },
  {
    name: 'hunting-ground.tmx',
    path: 'src/games/my-sample-rpg/assets/maps/hunting-ground.tmx',
    text: huntingGroundMapXml
  },
  { name: 'cave.tmx', path: 'src/games/my-sample-rpg/assets/maps/cave.tmx', text: caveMapXml },
  {
    name: 'town-32.tsx',
    path: 'src/games/my-sample-rpg/assets/tilesets/town-32.tsx',
    text: townTilesetXml
  },
  {
    name: 'tiny-dungeon-16.tsx',
    path: 'src/games/my-sample-rpg/assets/tilesets/tiny-dungeon-16.tsx',
    text: tinyDungeonTilesetXml
  }
]
