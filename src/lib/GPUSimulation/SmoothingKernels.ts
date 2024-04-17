class SpikyDensityKernel {
  normalizationDensity: number
  normalizationNearDensity: number

  constructor(particleMass: number, smoothingRadius: number) {
    this.normalizationDensity = (6 * particleMass) / (Math.PI * smoothingRadius ** 4)
    this.normalizationNearDensity = (10 * particleMass) / (Math.PI * smoothingRadius ** 5)
  }

  evaluateDensityKernel(distance: number) {
    if (distance >= 0 && distance <= 1) {
      return this.normalizationDensity * (this.smoot - distance) ** 3
    } else {
      return 0
    }
  }
}
