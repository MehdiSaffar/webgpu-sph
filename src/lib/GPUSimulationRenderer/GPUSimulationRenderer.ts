import { GPUSimulation } from '$lib/GPUSimulation/GPUSimulation'
import renderShaderCode from './render.wgsl?raw'

export type GPUSimulationRendererSettings = {
  SCENE_SIZE: [number, number]
  VIEWPORT_SIZE: [number, number]
  SELECTED_PROPERTY: 'velocity' | 'density' | 'pressure' | 'force'
  MIN_COLOR: [number, number, number]
  MAX_COLOR: [number, number, number]
  BACKGROUND_COLOR: [number, number, number]
  RADIUS: number
}

export class GPUSimulationRenderer {
  device: GPUDevice
  simulation: GPUSimulation
  renderPipeline: GPURenderPipeline
  renderBindGroup: GPUBindGroup

  canvas: HTMLCanvasElement

  ctx: GPUCanvasContext

  RenderUBOValues = new ArrayBuffer(64)
  RenderUBOViews = {
    SCENE_SIZE: new Float32Array(this.RenderUBOValues, 0, 2),
    VIEWPORT_SIZE: new Float32Array(this.RenderUBOValues, 8, 2),
    SELECTED_PROPERTY: new Uint32Array(this.RenderUBOValues, 16, 1),
    MIN_COLOR: new Float32Array(this.RenderUBOValues, 32, 3),
    MAX_COLOR: new Float32Array(this.RenderUBOValues, 48, 3),
    RADIUS: new Float32Array(this.RenderUBOValues, 60, 1)
  }

  settings: GPUSimulationRendererSettings
  bufRenderUBO: GPUBuffer

  constructor(args: {
    device: GPUDevice
    canvas: HTMLCanvasElement
    simulation: GPUSimulation
    settings: GPUSimulationRendererSettings
  }) {
    this.device = args.device
    this.simulation = args.simulation
    this.canvas = args.canvas
    this.ctx = this.canvas.getContext('webgpu')!
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat()
    this.ctx.configure({ device: this.device, format: presentationFormat, alphaMode: 'premultiplied' })

    const renderShaderModule = this.device.createShaderModule({ code: renderShaderCode })

    this.bufRenderUBO = this.device.createBuffer({
      label: 'Render UBO',
      size: this.RenderUBOValues.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    })

    this.renderPipeline = this.device.createRenderPipeline({
      label: 'Render Pipeline',
      layout: 'auto',
      vertex: {
        module: renderShaderModule,
        entryPoint: 'vs',
        buffers: [
          {
            stepMode: 'instance',
            arrayStride: 2 * 4,
            attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }]
          }
        ]
      },
      fragment: {
        module: renderShaderModule,
        entryPoint: 'fs',
        targets: [
          {
            format: presentationFormat,
            blend: {
              color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }
            }
          }
        ]
      },
      primitive: {
        topology: 'triangle-list'
      }
    })

    this.renderBindGroup = this.device.createBindGroup({
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.bufRenderUBO } },
        { binding: 1, resource: { buffer: this.simulation.bufRanges } },
        { binding: 2, resource: { buffer: this.simulation.bufDensities } },
        { binding: 3, resource: { buffer: this.simulation.bufNearDensities } },
        { binding: 4, resource: { buffer: this.simulation.bufForcesMag } },
        { binding: 5, resource: { buffer: this.simulation.bufVelocitiesMag } }
      ]
    })

    this.settings = args.settings
    this.writeUniforms()
  }

  serializeSelectedProperty(property: GPUSimulationRendererSettings['SELECTED_PROPERTY']) {
    const _map = {
      density: 0,
      pressure: 1,
      force: 2,
      velocity: 3
    }

    return _map[property]
  }

  updateSettings(settings: Partial<GPUSimulationRendererSettings>) {
    this.settings = { ...this.settings, ...settings }
    this.writeUniforms()
  }

  writeUniforms() {
    this.RenderUBOViews.SCENE_SIZE.set(this.settings.SCENE_SIZE)
    this.RenderUBOViews.VIEWPORT_SIZE.set(this.settings.VIEWPORT_SIZE)
    this.RenderUBOViews.SELECTED_PROPERTY.set([this.serializeSelectedProperty(this.settings.SELECTED_PROPERTY)])
    this.RenderUBOViews.MIN_COLOR.set(this.settings.MIN_COLOR)
    this.RenderUBOViews.MAX_COLOR.set(this.settings.MAX_COLOR)
    this.RenderUBOViews.RADIUS.set([this.settings.RADIUS])
    this.device.queue.writeBuffer(this.bufRenderUBO, 0, this.RenderUBOValues)
  }

  render() {
    const renderPassDescriptor: GPURenderPassDescriptor = {
      label: 'renderPassDescriptor',
      colorAttachments: [
        {
          view: this.ctx.getCurrentTexture().createView(),
          clearValue: [...this.settings.BACKGROUND_COLOR, 1],
          loadOp: 'clear',
          storeOp: 'store'
        }
      ]
    }

    let encoder = this.device.createCommandEncoder({ label: 'pass encoder' })
    const pass = encoder.beginRenderPass(renderPassDescriptor)
    pass.setPipeline(this.renderPipeline)
    pass.setVertexBuffer(0, this.simulation.bufPositions)
    pass.setBindGroup(0, this.renderBindGroup)
    pass.draw(6, this.simulation.settings.N)
    pass.end()

    this.device.queue.submit([encoder.finish()])
  }
}
