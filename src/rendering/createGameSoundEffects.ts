import grassFootstepSoundUrl from '../assets/sounds/grass-footstep.wav'
import levelUpSoundUrl from '../assets/sounds/level-up.wav'
import playerDamageSoundUrl from '../assets/sounds/player-damage.wav'
import playerGameOverSoundUrl from '../assets/sounds/player-game-over.wav'
import playerSkillSoundUrl from '../assets/sounds/player-skill.wav'
import playerSwordHitSoundUrl from '../assets/sounds/player-sword-hit.wav'
import slimeAttackSoundUrl from '../assets/sounds/slime-attack.wav'
import slimeDeathSoundUrl from '../assets/sounds/slime-death.wav'

export type GameSoundEffectId =
  | 'grassFootstep'
  | 'levelUp'
  | 'playerDamage'
  | 'playerGameOver'
  | 'playerSkill'
  | 'playerSwordHit'
  | 'slimeAttack'
  | 'slimeDeath'

type GameSoundEffectConfig = {
  url: string
  volume: number
  poolSize: number
  loop?: boolean
}

type GameSoundEffectPool = {
  audios: HTMLAudioElement[]
  nextAudioIndex: number
  volume: number
}

export type GameSoundEffects = {
  play: (soundId: GameSoundEffectId) => void
  startLoop: (soundId: GameSoundEffectId) => void
  stop: (soundId: GameSoundEffectId) => void
  stopAllLoops: () => void
  setMasterVolume: (volume: number) => void
  destroy: () => void
}

const SOUND_EFFECT_CONFIGS: Record<GameSoundEffectId, GameSoundEffectConfig> = {
  grassFootstep: {
    url: grassFootstepSoundUrl,
    volume: 0.75,
    poolSize: 1,
    loop: true
  },
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
  playerSkill: {
    url: playerSkillSoundUrl,
    volume: 0.85,
    poolSize: 3
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
    audio.loop = config.loop === true
    audio.volume = config.volume

    return audio
  }),
  nextAudioIndex: 0,
  volume: config.volume
})

export const createGameSoundEffects = ({
  masterVolume = 1
}: {
  masterVolume?: number
} = {}): GameSoundEffects => {
  const soundEffectPools = new Map<GameSoundEffectId, GameSoundEffectPool>(
    Object.entries(SOUND_EFFECT_CONFIGS).map(([soundId, config]) => [
      soundId as GameSoundEffectId,
      createSoundEffectPool(config)
    ])
  )
  let currentMasterVolume = clampVolume(masterVolume)

  const applyAudioVolume = (
    audio: HTMLAudioElement,
    soundEffectPool: GameSoundEffectPool
  ) => {
    audio.volume = soundEffectPool.volume * currentMasterVolume
  }

  const play = (soundId: GameSoundEffectId): void => {
    const soundEffectPool = soundEffectPools.get(soundId)

    if (!soundEffectPool) {
      return
    }

    const audio = soundEffectPool.audios[soundEffectPool.nextAudioIndex]

    soundEffectPool.nextAudioIndex =
      (soundEffectPool.nextAudioIndex + 1) % soundEffectPool.audios.length
    audio.pause()
    applyAudioVolume(audio, soundEffectPool)
    audio.currentTime = 0
    void audio.play().catch(() => undefined)
  }

  const startLoop = (soundId: GameSoundEffectId): void => {
    const soundEffectPool = soundEffectPools.get(soundId)

    if (!soundEffectPool) {
      return
    }

    const audio = soundEffectPool.audios[0]

    if (!audio.paused) {
      return
    }

    audio.currentTime = 0
    applyAudioVolume(audio, soundEffectPool)
    void audio.play().catch(() => undefined)
  }

  const stop = (soundId: GameSoundEffectId): void => {
    const soundEffectPool = soundEffectPools.get(soundId)

    if (!soundEffectPool) {
      return
    }

    for (const audio of soundEffectPool.audios) {
      audio.pause()
      audio.currentTime = 0
    }
  }

  const stopAllLoops = (): void => {
    for (const [soundId, config] of Object.entries(SOUND_EFFECT_CONFIGS)) {
      if (config.loop) {
        stop(soundId as GameSoundEffectId)
      }
    }
  }

  const setMasterVolume = (volume: number): void => {
    currentMasterVolume = clampVolume(volume)

    for (const soundEffectPool of soundEffectPools.values()) {
      for (const audio of soundEffectPool.audios) {
        applyAudioVolume(audio, soundEffectPool)
      }
    }
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
    startLoop,
    stop,
    stopAllLoops,
    setMasterVolume,
    destroy
  }
}

const clampVolume = (volume: number): number =>
  Number.isFinite(volume) ? Math.max(0, Math.min(volume, 1)) : 1
