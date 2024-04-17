import { GPUMin } from '$lib/GPUMin/GPUMin'
import { GPUSort } from '../GPUSort/GPUSort'
import shaderCode from './simulation.wgsl?raw'

export const KERNELS = {
  spiky: (PARTICLE_MASS: number, SMOOTHING_RADIUS: number) => ({
    NORMALIZATION_DENSITY: (6 * PARTICLE_MASS) / (Math.PI * SMOOTHING_RADIUS ** 4),
    NORMALIZATION_NEAR_DENSITY: (10 * PARTICLE_MASS) / (Math.PI * SMOOTHING_RADIUS ** 5)
  }),
  soft: (PARTICLE_MASS: number, SMOOTHING_RADIUS: number) => ({
    NORMALIZATION_DENSITY: (4 * PARTICLE_MASS) / (Math.PI * SMOOTHING_RADIUS ** 8),
    NORMALIZATION_NEAR_DENSITY: (5 * PARTICLE_MASS) / (Math.PI * SMOOTHING_RADIUS ** 10)
  })
}

export type GPUSimulationSettingsInput = {
  SMOOTHING_RADIUS: number
  BASE_DENSITY: number
  PARTICLE_MASS: number
  DYNAMIC_VISCOSITY: number
  SCENE_SIZE: [number, number]
  DAMPING_COEFF: number
  GRAVITY: number
  DENSITY_TO_PRESSURE_FACTOR: number
  NEAR_DENSITY_TO_NEAR_PRESSURE_FACTOR: number
  VIEWPORT_SIZE: [number, number]
  TIME_STEP: number
  RADIUS: number
  DENSITY_KERNEL: 'soft' | 'spiky'
  INTERACTION_STRENGTH: number
  SELECTED_PROPERTY: 'density' | 'near_density' | 'force' | 'velocity'
}

export type GPUSimulationSettings = {
  N: number
  SMOOTHING_RADIUS: number
  BASE_DENSITY: number
  PARTICLE_MASS: number
  DYNAMIC_VISCOSITY: number
  SCENE_SIZE: [number, number]
  DAMPING_COEFF: number
  GRAVITY: number
  DENSITY_TO_PRESSURE_FACTOR: number
  NEAR_DENSITY_TO_NEAR_PRESSURE_FACTOR: number
  VIEWPORT_SIZE: [number, number]
  TIME_STEP: number
  RADIUS: number

  INTERACTION_AMOUNT: number
  INTERACTION_POS: [number, number]
  INTERACTION_STRENGTH: number

  DENSITY_KERNEL: 'soft' | 'spiky'
  SELECTED_PROPERTY: 'density' | 'near_density' | 'force' | 'velocity'
}

export class GPUSimulation {
  device: GPUDevice
  gpuSort: GPUSort

  // GPU BUFFERS
  bufUBO: GPUBuffer
  bufPositions: GPUBuffer
  bufVelocities: GPUBuffer
  bufNearDensities: GPUBuffer
  bufDensities: GPUBuffer
  bufSpatialLookup: GPUBuffer
  bufStartIndices: GPUBuffer
  bufRanges: GPUBuffer

  // GPU PIPELINES
  computeSpatialLookupPassOnePipeline: GPUComputePipeline
  computeSpatialLookupPassTwoPipeline: GPUComputePipeline
  computeDensitiesPipeline: GPUComputePipeline
  computeForcesPipeline: GPUComputePipeline

  bindGroup: GPUBindGroup
  pipelineLayout: GPUPipelineLayout

  settings: GPUSimulationSettings
  settingsDirty = false

  UBOValues = new ArrayBuffer(80)
  UBOViews = {
    N: new Uint32Array(this.UBOValues, 0, 1),
    SMOOTHING_RADIUS: new Float32Array(this.UBOValues, 4, 1),
    BASE_DENSITY: new Float32Array(this.UBOValues, 8, 1),
    NORMALIZATION_DENSITY: new Float32Array(this.UBOValues, 12, 1),
    NORMALIZATION_NEAR_DENSITY: new Float32Array(this.UBOValues, 16, 1),
    NORMALIZATION_VISCOUS_FORCE: new Float32Array(this.UBOValues, 20, 1),
    SCENE_SIZE: new Float32Array(this.UBOValues, 24, 2),
    DAMPING_COEFF: new Float32Array(this.UBOValues, 32, 1),
    GRAVITY: new Float32Array(this.UBOValues, 36, 1),
    DENSITY_TO_PRESSURE_FACTOR: new Float32Array(this.UBOValues, 40, 1),
    NEAR_DENSITY_TO_NEAR_PRESSURE_FACTOR: new Float32Array(this.UBOValues, 44, 1),
    TIME_STEP: new Float32Array(this.UBOValues, 48, 1),
    RADIUS: new Float32Array(this.UBOValues, 52, 1),
    INTERACTION_STRENGTH: new Float32Array(this.UBOValues, 56, 1),
    INTERACTION_POS: new Float32Array(this.UBOValues, 64, 2),
    DENSITY_KERNEL: new Uint32Array(this.UBOValues, 72, 1)
  }

  gpuMin: GPUMin
  bufForcesMag: GPUBuffer
  bufVelocitiesMag: GPUBuffer
  predictPositionsPipeline: GPUComputePipeline
  bufPredictedPositions: GPUBuffer
  WK_SIZE: number
  tick: number

  constructor({
    device,
    settings,
    initialValues: { positions }
  }: {
    device: GPUDevice
    settings: GPUSimulationSettingsInput
    initialValues: {
      positions: Float32Array
    }
  }) {
    this.tick = 0
    this.device = device
    this.settings = { ...settings, N: positions.length / 2, INTERACTION_POS: [0, 0], INTERACTION_AMOUNT: 0 }
    this.WK_SIZE = 256

    const shaderModule = device.createShaderModule({ label: 'Shader Module', code: shaderCode })

    this.bufUBO = this.device.createBuffer({
      label: 'UBO',
      size: this.UBOValues.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    })

    this.bufPositions = this.device.createBuffer({
      label: 'Positions',
      size: positions.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    })

    this.bufPredictedPositions = this.device.createBuffer({
      label: 'Predicted Positions',
      size: positions.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    })

    this.bufVelocities = this.device.createBuffer({
      label: 'Velocities',
      size: this.settings.N * 2 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    })

    this.bufNearDensities = this.device.createBuffer({
      label: 'Near Densities',
      size: this.settings.N * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    })

    this.bufDensities = this.device.createBuffer({
      label: 'Densities',
      size: this.settings.N * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    })

    this.bufRanges = this.device.createBuffer({
      label: 'Ranges',
      size: 4 * 2 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    })

    this.bufSpatialLookup = this.device.createBuffer({
      label: 'Spatial Lookup',
      size: this.settings.N * 2 * Int32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE
    })

    this.bufStartIndices = this.device.createBuffer({
      label: 'Start Indices',
      size: this.settings.N * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE
    })

    this.bufForcesMag = this.device.createBuffer({
      label: 'Forces Magnitude',
      size: this.settings.N * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    })

    this.bufVelocitiesMag = this.device.createBuffer({
      label: 'Velocities Mag',
      size: this.settings.N * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    })

    const bindGroupLayout = this.device.createBindGroupLayout({
      label: 'Bind Group Layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {} },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
      ]
    })

    this.bindGroup = this.device.createBindGroup({
      label: 'Bind Group',
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.bufUBO } },
        { binding: 1, resource: { buffer: this.bufPositions } },
        { binding: 2, resource: { buffer: this.bufPredictedPositions } },
        { binding: 3, resource: { buffer: this.bufVelocities } },
        { binding: 4, resource: { buffer: this.bufNearDensities } },
        { binding: 5, resource: { buffer: this.bufDensities } },
        { binding: 6, resource: { buffer: this.bufSpatialLookup } },
        { binding: 7, resource: { buffer: this.bufStartIndices } },
        { binding: 8, resource: { buffer: this.bufRanges } },
        { binding: 9, resource: { buffer: this.bufForcesMag } },
        { binding: 10, resource: { buffer: this.bufVelocitiesMag } }
      ]
    })

    this.pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] })

    this.predictPositionsPipeline = this.device.createComputePipeline({
      label: 'predictPositionsPipeline',
      layout: this.pipelineLayout,
      compute: { module: shaderModule, entryPoint: 'predict_positions' }
    })

    this.computeSpatialLookupPassOnePipeline = this.device.createComputePipeline({
      label: 'computeSpatialLookupPassOnePipeline',
      layout: this.pipelineLayout,
      compute: { module: shaderModule, entryPoint: 'spatial_lookup_pass_one' }
    })

    this.computeSpatialLookupPassTwoPipeline = this.device.createComputePipeline({
      label: 'computeSpatialLookupPassTwoPipeline',
      layout: this.pipelineLayout,
      compute: { module: shaderModule, entryPoint: 'spatial_lookup_pass_two' }
    })

    this.computeDensitiesPipeline = this.device.createComputePipeline({
      label: 'computeDensities',
      layout: this.pipelineLayout,
      compute: { module: shaderModule, entryPoint: 'compute_densities' }
    })

    this.computeForcesPipeline = this.device.createComputePipeline({
      label: 'computeForcesPipeline',
      layout: this.pipelineLayout,
      compute: { module: shaderModule, entryPoint: 'compute_forces' }
    })

    this.device.queue.writeBuffer(this.bufPositions, 0, positions)
    this.writeUniforms()

    this.gpuMin = new GPUMin(device, positions.length)
    this.gpuSort = new GPUSort(device, this.bufSpatialLookup)
  }

  public updateSettings(settings: Partial<GPUSimulationSettings>) {
    this.settings = { ...this.settings, ...settings }
    this.settingsDirty = true
  }

  private writeUniforms() {
    this.UBOViews.N.set([this.settings.N])
    this.UBOViews.SMOOTHING_RADIUS.set([this.settings.SMOOTHING_RADIUS])
    this.UBOViews.BASE_DENSITY.set([this.settings.BASE_DENSITY])

    const { PARTICLE_MASS, DYNAMIC_VISCOSITY, SMOOTHING_RADIUS } = this.settings
    const { NORMALIZATION_DENSITY, NORMALIZATION_NEAR_DENSITY } = KERNELS[this.settings.DENSITY_KERNEL](
      PARTICLE_MASS,
      SMOOTHING_RADIUS
    )
    const NORMALIZATION_VISCOUS_FORCE = (4 * DYNAMIC_VISCOSITY * PARTICLE_MASS) / (Math.PI * SMOOTHING_RADIUS ** 8)

    this.UBOViews.NORMALIZATION_DENSITY.set([NORMALIZATION_DENSITY])
    this.UBOViews.NORMALIZATION_NEAR_DENSITY.set([NORMALIZATION_NEAR_DENSITY])

    this.UBOViews.NORMALIZATION_VISCOUS_FORCE.set([NORMALIZATION_VISCOUS_FORCE])

    this.UBOViews.SCENE_SIZE.set(this.settings.SCENE_SIZE)
    this.UBOViews.DAMPING_COEFF.set([this.settings.DAMPING_COEFF])
    this.UBOViews.GRAVITY.set([this.settings.GRAVITY])

    this.UBOViews.DENSITY_TO_PRESSURE_FACTOR.set([this.settings.DENSITY_TO_PRESSURE_FACTOR])
    this.UBOViews.NEAR_DENSITY_TO_NEAR_PRESSURE_FACTOR.set([this.settings.NEAR_DENSITY_TO_NEAR_PRESSURE_FACTOR])

    this.UBOViews.TIME_STEP.set([this.settings.TIME_STEP])
    this.UBOViews.RADIUS.set([this.settings.RADIUS])
    this.UBOViews.INTERACTION_POS.set(this.settings.INTERACTION_POS)

    this.UBOViews.INTERACTION_STRENGTH.set([this.settings.INTERACTION_AMOUNT])

    this.UBOViews.DENSITY_KERNEL.set([this.settings.DENSITY_KERNEL === 'spiky' ? 0 : 1])

    this.device.queue.writeBuffer(this.bufUBO, 0, this.UBOValues)
    this.settingsDirty = false
  }

  public setInteraction(x: number, y: number, type: 'push' | 'pull' | 'none') {
    this.updateSettings({
      INTERACTION_AMOUNT: type === 'none' ? 0 : (type === 'pull' ? -1 : 1) * this.settings.INTERACTION_STRENGTH,
      INTERACTION_POS: [x, y]
    })
  }

  public simulate() {
    if (this.settingsDirty) this.writeUniforms()

    let encoder = this.device.createCommandEncoder()
    let pass = encoder.beginComputePass()
    pass.setPipeline(this.predictPositionsPipeline)
    pass.setBindGroup(0, this.bindGroup)
    pass.dispatchWorkgroups(this.settings.N / this.WK_SIZE)
    pass.setPipeline(this.computeSpatialLookupPassOnePipeline)
    pass.setBindGroup(0, this.bindGroup)
    pass.dispatchWorkgroups(this.settings.N / this.WK_SIZE)
    pass.end()
    this.device.queue.submit([encoder.finish()])

    this.gpuSort.sort()

    encoder = this.device.createCommandEncoder()
    pass = encoder.beginComputePass()
    pass.setPipeline(this.computeSpatialLookupPassTwoPipeline)
    pass.setBindGroup(0, this.bindGroup)
    pass.dispatchWorkgroups(this.settings.N / this.WK_SIZE)

    pass.setPipeline(this.computeDensitiesPipeline)
    pass.setBindGroup(0, this.bindGroup)
    pass.dispatchWorkgroups(this.settings.N / this.WK_SIZE)

    pass.setPipeline(this.computeForcesPipeline)
    pass.setBindGroup(0, this.bindGroup)
    pass.dispatchWorkgroups(this.settings.N / this.WK_SIZE)
    pass.end()
    this.device.queue.submit([encoder.finish()])

    if (this.tick === 0) {
      this.device.queue.writeBuffer(
        this.bufRanges,
        0,
        new Float32Array([
          // density
          Number.POSITIVE_INFINITY,
          Number.NEGATIVE_INFINITY,
          // near_density
          Number.POSITIVE_INFINITY,
          Number.NEGATIVE_INFINITY,
          // force
          Number.POSITIVE_INFINITY,
          Number.NEGATIVE_INFINITY,
          // velocity
          Number.POSITIVE_INFINITY,
          Number.NEGATIVE_INFINITY
        ])
      )

      if (this.settings.SELECTED_PROPERTY === 'density') {
        this.gpuMin.compute('min', this.bufDensities, this.bufRanges, 0)
        this.gpuMin.compute('max', this.bufDensities, this.bufRanges, 1)
      } else if (this.settings.SELECTED_PROPERTY === 'near_density') {
        this.gpuMin.compute('min', this.bufNearDensities, this.bufRanges, 2)
        this.gpuMin.compute('max', this.bufNearDensities, this.bufRanges, 3)
      } else if (this.settings.SELECTED_PROPERTY === 'force') {
        this.gpuMin.compute('min', this.bufForcesMag, this.bufRanges, 4)
        this.gpuMin.compute('max', this.bufForcesMag, this.bufRanges, 5)
      } else if (this.settings.SELECTED_PROPERTY === 'velocity') {
        this.gpuMin.compute('min', this.bufVelocitiesMag, this.bufRanges, 6)
        this.gpuMin.compute('max', this.bufVelocitiesMag, this.bufRanges, 7)
      }
    }

    this.tick += 1
    this.tick %= 2
  }
}
