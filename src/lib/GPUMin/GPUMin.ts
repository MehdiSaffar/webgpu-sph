import shaderCode from './min.wgsl?raw'

export class GPUMin {
  bufUBO: GPUBuffer
  computeMinPipeline: GPUComputePipeline
  bindGroupLayout: GPUBindGroupLayout
  computeMaxPipeline: GPUComputePipeline
  bufCopy: GPUBuffer
  bindGroup: GPUBindGroup

  constructor(private device: GPUDevice, N: number) {
    this.bufUBO = this.device.createBuffer({
      label: 'GPUMin UBO',
      size: Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    })

    const shaderModule = this.device.createShaderModule({ label: 'Shader Module', code: shaderCode })

    this.bindGroupLayout = this.device.createBindGroupLayout({
      label: 'GPUMin Bind Group Layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
      ]
    })

    this.computeMinPipeline = this.device.createComputePipeline({
      label: 'Compute Min Pipeline',
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
      compute: { module: shaderModule, entryPoint: 'compute_min' }
    })

    this.computeMaxPipeline = this.device.createComputePipeline({
      label: 'Compute Max Pipeline',
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
      compute: { module: shaderModule, entryPoint: 'compute_max' }
    })

    this.bufCopy = this.device.createBuffer({
      label: 'Copy Buffer',
      size: N * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    })

    this.bindGroup = this.device.createBindGroup({
      label: 'Bind Group',
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.bufUBO } },
        { binding: 1, resource: { buffer: this.bufCopy } }
      ]
    })
  }

  compute(op: 'min' | 'max', inputBuffer: GPUBuffer, outputBuffer: GPUBuffer, writeAt: number = 0) {
    const pipeline = op === 'min' ? this.computeMinPipeline : this.computeMaxPipeline
    const bytesPerUnit = Float32Array.BYTES_PER_ELEMENT
    const bufferByteLength = inputBuffer.size
    const bufferUnitLength = bufferByteLength / bytesPerUnit

    let encoder = this.device.createCommandEncoder({ label: 'Copy Min Encoder - Copy input' })
    encoder.copyBufferToBuffer(inputBuffer, 0, this.bufCopy, 0, inputBuffer.size)
    this.device.queue.submit([encoder.finish()])
    for (let size = 2; size <= bufferUnitLength; size *= 2) {
      encoder = this.device.createCommandEncoder({ label: `Compute Min Encoder size=${size}` })
      this.device.queue.writeBuffer(this.bufUBO, 0, new Uint32Array([size]))
      let pass = encoder.beginComputePass()
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, this.bindGroup)
      pass.dispatchWorkgroups(bufferUnitLength / size / 256)
      pass.end()
      this.device.queue.submit([encoder.finish()])
    }

    encoder = this.device.createCommandEncoder({ label: 'Copy Min Encoder - Write value' })
    encoder.copyBufferToBuffer(this.bufCopy, 0, outputBuffer, writeAt * bytesPerUnit, bytesPerUnit)
    this.device.queue.submit([encoder.finish()])
  }
}
