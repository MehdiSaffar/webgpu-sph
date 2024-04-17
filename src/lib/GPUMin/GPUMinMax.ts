import shaderCode from './min.wgsl?raw'
import { varEx } from 'varex'

export class GPUMinMax {
  bufUBO: GPUBuffer
  computeMinPipeline: GPUComputePipeline
  bindGroupLayout: GPUBindGroupLayout
  computeMaxPipeline: GPUComputePipeline
  bufCopy: GPUBuffer
  bindGroup: GPUBindGroup
  initMinPipeline: GPUComputePipeline
  initMaxPipeline: GPUComputePipeline

  UBOValues = new ArrayBuffer(12)
  UBOViews = {
    size: new Uint32Array(this.UBOValues, 0, 1),
    start: new Uint32Array(this.UBOValues, 4, 1),
    end: new Uint32Array(this.UBOValues, 8, 1)
  }

  WK_SIZE: number

  constructor(private device: GPUDevice, N: number) {
    this.bufUBO = this.device.createBuffer({
      label: 'GPUMin UBO',
      size: this.UBOValues.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    })

    this.WK_SIZE = this.device.limits.maxComputeWorkgroupSizeX

    const shaderModule = this.device.createShaderModule({
      label: 'Shader Module',
      code: varEx(shaderCode, { WK_SIZE: this.WK_SIZE })
    })

    this.bindGroupLayout = this.device.createBindGroupLayout({
      label: 'GPUMin Bind Group Layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
      ]
    })

    this.initMinPipeline = this.device.createComputePipeline({
      label: 'Init Min Pipeline',
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
      compute: { module: shaderModule, entryPoint: 'init_min' }
    })

    this.initMaxPipeline = this.device.createComputePipeline({
      label: 'Init Max Pipeline',
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
      compute: { module: shaderModule, entryPoint: 'init_max' }
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

  compute(
    op: 'min' | 'max',
    inputBuffer: GPUBuffer,
    start: number,
    end: number,
    outputBuffer: GPUBuffer,
    writeAt: number = 0
  ) {
    const initPipeline = op === 'min' ? this.initMinPipeline : this.initMaxPipeline
    const pipeline = op === 'min' ? this.computeMinPipeline : this.computeMaxPipeline

    const bytesPerUnit = Float32Array.BYTES_PER_ELEMENT
    const bufferByteLength = inputBuffer.size
    const bufferUnitLength = bufferByteLength / bytesPerUnit

    this.UBOViews.start.set([start])
    this.UBOViews.end.set([end])
    this.device.queue.writeBuffer(this.bufUBO, 0, this.UBOValues)

    let encoder = this.device.createCommandEncoder({ label: 'Copy Min Encoder - Copy input' })
    encoder.copyBufferToBuffer(inputBuffer, 0, this.bufCopy, 0, inputBuffer.size)

    let pass = encoder.beginComputePass({ label: 'Init' })
    pass.setPipeline(initPipeline)
    pass.setBindGroup(0, this.bindGroup)
    pass.dispatchWorkgroups(Math.ceil(bufferUnitLength / this.WK_SIZE))
    pass.end()
    this.device.queue.submit([encoder.finish()])

    for (let size = 2; size <= bufferUnitLength; size *= 2) {
      this.UBOViews.size.set([size])
      this.device.queue.writeBuffer(this.bufUBO, 0, this.UBOValues)

      encoder = this.device.createCommandEncoder({ label: `Compute Min Encoder size=${size}` })
      pass = encoder.beginComputePass()
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, this.bindGroup)
      pass.dispatchWorkgroups(Math.ceil(bufferUnitLength / size / this.WK_SIZE))
      pass.end()
      this.device.queue.submit([encoder.finish()])
    }

    encoder = this.device.createCommandEncoder({ label: 'Copy Min Encoder - Write value' })
    encoder.copyBufferToBuffer(this.bufCopy, 0, outputBuffer, writeAt * bytesPerUnit, bytesPerUnit)
    this.device.queue.submit([encoder.finish()])
  }
}
