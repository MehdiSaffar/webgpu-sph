export function range(start: number, end: number, step: number) {
  const length = Math.floor((end - start) / step) + 1
  return Array.from({ length }, (_, i) => start + i * step)
}

export function hexToRGB(hex: string) {
  return hex
    .slice(1)
    .match(/.{1,2}/g)!
    .map((e) => parseInt(e, 16) / 255) as [number, number, number]
}
export const roundTo = (num = 0, decimals = 2) => Math.round(num * 10 ** decimals) / 10 ** decimals
