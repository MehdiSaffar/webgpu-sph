import { getBuffer } from '$lib/webgpu-utils'
import templateShaderCode from './sort.wgsl?raw'
import { varEx } from 'varex'

export class GPUSort {
  ubo: Uint32Array
  bufUBO: GPUBuffer
  WK_SIZE: number
  pipelineSortAll: GPUComputePipeline
  pipelineSortChunk: GPUComputePipeline
  bindGroupLayout: GPUBindGroupLayout
  bindGroup: GPUBindGroup

  constructor(private device: GPUDevice, private buffer: GPUBuffer) {
    this.ubo = new Uint32Array(2)
    this.bufUBO = this.device.createBuffer({
      label: 'GPUSort UBO',
      size: this.ubo.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    })

    this.WK_SIZE = this.device.limits.maxComputeWorkgroupSizeX

    const shaderCode = varEx(templateShaderCode, { WK_SIZE: this.WK_SIZE })
    const shaderModule = this.device.createShaderModule({ label: 'GPUSort Shader Module', code: shaderCode })

    this.bindGroupLayout = this.device.createBindGroupLayout({
      label: 'GPUSort Bind Group Layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
      ]
    })

    this.pipelineSortAll = this.device.createComputePipeline({
      label: 'GPUSort Sort All Pipeline',
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
      compute: { module: shaderModule, entryPoint: 'sort_all' }
    })

    this.pipelineSortChunk = this.device.createComputePipeline({
      label: 'GPUSort Sort Chunk Pipeline',
      layout: this.device.createPipelineLayout({
        label: 'GPUSort Sort Chunk Pipeline Layout',
        bindGroupLayouts: [this.bindGroupLayout]
      }),
      compute: { module: shaderModule, entryPoint: 'sort_chunk' }
    })

    this.bindGroup = this.device.createBindGroup({
      label: 'GPUSort Bind Group',
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.bufUBO } },
        { binding: 1, resource: { buffer } }
      ]
    })
  }

  sort() {
    const bytesPerUnit = Uint32Array.BYTES_PER_ELEMENT * 2
    const bufferByteLength = this.buffer.size
    const bufferUnitLength = bufferByteLength / bytesPerUnit

    const WK_COUNT = Math.max(1, bufferUnitLength / this.WK_SIZE)
    const OFFSET = Math.log2(bufferUnitLength) - Math.log2(this.WK_SIZE * 2)
    

    let encoder = this.device.createCommandEncoder({ label: 'GPUSort Command Encoder (All)' })
    let pass = encoder.beginComputePass({ label: 'GPUSort Compute Pass (All)' })
    pass.setPipeline(this.pipelineSortAll)
    pass.setBindGroup(0, this.bindGroup)
    pass.dispatchWorkgroups(WK_COUNT)
    pass.end()
    this.device.queue.submit([encoder.finish()])

    if (WK_COUNT > 1) {
      for (let k = WK_COUNT >> OFFSET; k <= bufferUnitLength; k = k << 1) {
        for (let j = k >> 1; j > 0; j = j >> 1) {
          encoder = this.device.createCommandEncoder({ label: `GPUSort Command Encoder ${[k, j]}` })
          this.device.queue.writeBuffer(this.bufUBO, 0, new Uint32Array([k, j]))

          const pass = encoder.beginComputePass({ label: `GPUSort Compute Pass ${[k, j]}` })
          pass.setPipeline(this.pipelineSortChunk)
          pass.setBindGroup(0, this.bindGroup)
          pass.dispatchWorkgroups(WK_COUNT)
          pass.end()

          this.device.queue.submit([encoder.finish()])
        }
      }
    }
  }
}
