import levelUpSoundUrl from '../assets/sounds/level-up.wav'
import playerDamageSoundUrl from '../assets/sounds/player-damage.wav'
import playerGameOverSoundUrl from '../assets/sounds/player-game-over.wav'
import playerSwordHitSoundUrl from '../assets/sounds/player-sword-hit.wav'
import slimeAttackSoundUrl from '../assets/sounds/slime-attack.wav'
import slimeDeathSoundUrl from '../assets/sounds/slime-death.wav'

export type GameSoundEffectId =
  | 'levelUp'
  | 'playerDamage'
  | 'playerGameOver'
  | 'playerSwordHit'
  | 'slimeAttack'
  | 'slimeDeath'

type GameSoundEffectConfig = {
  url: string
  volume: number
  poolSize: number
}

type GameSoundEffectPool = {
  audios: HTMLAudioElement[]
  nextAudioIndex: number
  volume: number
}

export type GameSoundEffects = {
  play: (soundId: GameSoundEffectId) => void
  destroy: () => void
}

const SOUND_EFFECT_CONFIGS: Record<GameSoundEffectId, GameSoundEffectConfig> = {
  levelUp: {
    url: levelUpSoundUrl,
    volume: 0.85,
    poolSize: 2
  },
  playerDamage: {
    url: playerDamageSoundUrl,
    volume: 0.75,
    poolSize: 3
  },
  playerGameOver: {
    url: playerGameOverSoundUrl,
    volume: 0.85,
    poolSize: 1
  },
  playerSwordHit: {
    url: playerSwordHitSoundUrl,
    volume: 0.85,
    poolSize: 4
  },
  slimeAttack: {
    url: slimeAttackSoundUrl,
    volume: 0.85,
    poolSize: 3
  },
  slimeDeath: {
    url: slimeDeathSoundUrl,
    volume: 0.9,
    poolSize: 3
  }
}

const createSoundEffectPool = (
  config: GameSoundEffectConfig
): GameSoundEffectPool => ({
  audios: Array.from({ length: config.poolSize }, () => {
    const audio = new Audio(config.url)

    audio.preload = 'auto'
    audio.volume = config.volume

    return audio
  }),
  nextAudioIndex: 0,
  volume: config.volume
})

export const createGameSoundEffects = (): GameSoundEffects => {
  const soundEffectPools = new Map<GameSoundEffectId, GameSoundEffectPool>(
    Object.entries(SOUND_EFFECT_CONFIGS).map(([soundId, config]) => [
      soundId as GameSoundEffectId,
      createSoundEffectPool(config)
    ])
  )

  const play = (soundId: GameSoundEffectId): void => {
    const soundEffectPool = soundEffectPools.get(soundId)

    if (!soundEffectPool) {
      return
    }

    const audio = soundEffectPool.audios[soundEffectPool.nextAudioIndex]

    soundEffectPool.nextAudioIndex =
      (soundEffectPool.nextAudioIndex + 1) % soundEffectPool.audios.length
    audio.pause()
    audio.volume = soundEffectPool.volume
    audio.currentTime = 0
    void audio.play().catch(() => undefined)
  }

  const destroy = (): void => {
    for (const soundEffectPool of soundEffectPools.values()) {
      for (const audio of soundEffectPool.audios) {
        audio.pause()
        audio.removeAttribute('src')
        audio.load()
      }
    }
    soundEffectPools.clear()
  }

  return {
    play,
    destroy
  }
}
