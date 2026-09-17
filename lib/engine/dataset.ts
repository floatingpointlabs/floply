/**
 * Modality-specific token count and storage size estimation.
 *
 * Every division here floors. Using `/` instead would give plausible-but-wrong numbers:
 * (336 // 32) ** 2 is 100, not 110.25.
 */

/** Average BPE tokens per English word (GPT-2 / LLaMA tokeniser empirical average). */
export const WORDS_TO_BPE_TOKENS = 1.3;

/** Bytes per token when stored as raw uint32 token IDs. */
export const TEXT_BYTES_PER_TOKEN = 4;

/** Approximate JPEG compression ratio for natural images (10:1 lossiness). */
export const JPEG_COMPRESSION_RATIO = 10;

/** Raw audio bytes per second: 16kHz x 16-bit mono = 32,000 bytes/s. */
export const AUDIO_BYTES_PER_SECOND = 32_000;

/** Token rate for EnCodec 24kHz per codebook level. */
export const ENCODEC_TOKENS_PER_SECOND = 75;

/** Token rate for SoundStream per codebook level. */
export const SOUNDSTREAM_TOKENS_PER_SECOND = 50;

/** Token rate for Whisper mel-spectrogram tokenisation. */
export const WHISPER_TOKENS_PER_SECOND = 50;

export function tokensPerTextSample(avgSeqLenWords: number): number {
  return Math.trunc(avgSeqLenWords * WORDS_TO_BPE_TOKENS);
}

export function bytesPerTextSample(tokens: number): number {
  return tokens * TEXT_BYTES_PER_TOKEN;
}

export function tokensPerImageSample(resolution: number, patchSize: number): number {
  return Math.floor(resolution / patchSize) ** 2;
}

export function bytesPerImageSample(resolution: number): number {
  return Math.floor((resolution * resolution * 3) / JPEG_COMPRESSION_RATIO);
}

/**
 * Estimate token count for an audio clip.
 * @param numCodebooks Number of RVQ codebook levels (1 for Whisper).
 */
export function tokensPerAudioSample(
  clipDuration: number,
  baseRate: number,
  numCodebooks = 1
): number {
  return Math.trunc(clipDuration * baseRate * numCodebooks);
}

export function bytesPerAudioSample(clipDuration: number): number {
  return Math.trunc(clipDuration * AUDIO_BYTES_PER_SECOND);
}

export function tokensPerVideoSample(
  duration: number,
  fps: number,
  resolution: number,
  patchSize: number
): number {
  const tokensPerFrame = tokensPerImageSample(resolution, patchSize);
  const numFrames = Math.trunc(duration * fps);
  return tokensPerFrame * numFrames;
}

export function bytesPerVideoSample(duration: number, fps: number, resolution: number): number {
  const bytesPerFrame = bytesPerImageSample(resolution);
  const numFrames = Math.trunc(duration * fps);
  return bytesPerFrame * numFrames;
}
