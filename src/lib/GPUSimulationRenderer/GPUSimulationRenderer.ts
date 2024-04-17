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
  INTERACTION_RADIUS: number
}

export class GPUSimulationRenderer {
  device: GPUDevice
  simulation: GPUSimulation
  renderParticlesPipeline: GPURenderPipeline
  renderParticlesBindGroup: GPUBindGroup

  canvas: HTMLCanvasElement

  ctx: GPUCanvasContext

  RenderUBOValues = new ArrayBuffer(80)
  RenderUBOViews = {
    SCENE_SIZE: new Float32Array(this.RenderUBOValues, 0, 2),
    VIEWPORT_SIZE: new Float32Array(this.RenderUBOValues, 8, 2),
    SELECTED_PROPERTY: new Uint32Array(this.RenderUBOValues, 16, 1),
    MIN_COLOR: new Float32Array(this.RenderUBOValues, 32, 3),
    MAX_COLOR: new Float32Array(this.RenderUBOValues, 48, 3),
    INTERACTION_RADIUS: new Float32Array(this.RenderUBOValues, 60, 1),
    INTERACTION_AMOUNT: new Float32Array(this.RenderUBOValues, 64, 1),
    RADIUS: new Float32Array(this.RenderUBOValues, 68, 1)
  }

  settings: GPUSimulationRendererSettings
  bufRenderUBO: GPUBuffer
  renderInteractionPipeline: GPURenderPipeline
  bufInteraction: GPUBuffer
  renderInteractionBindGroup: GPUBindGroup

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

    this.renderParticlesPipeline = this.device.createRenderPipeline({
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

    this.renderInteractionPipeline = this.device.createRenderPipeline({
      label: 'Render Interaction Pipeline',
      layout: 'auto',
      vertex: {
        module: renderShaderModule,
        entryPoint: 'vs_interaction',
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
        entryPoint: 'fs_interaction',
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

    this.renderParticlesBindGroup = this.device.createBindGroup({
      layout: this.renderParticlesPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.bufRenderUBO } },
        { binding: 1, resource: { buffer: this.simulation.bufRanges } },
        { binding: 2, resource: { buffer: this.simulation.bufDensities } },
        { binding: 3, resource: { buffer: this.simulation.bufNearDensities } },
        { binding: 4, resource: { buffer: this.simulation.bufForcesMag } },
        { binding: 5, resource: { buffer: this.simulation.bufVelocitiesMag } }
      ]
    })

    this.renderInteractionBindGroup = this.device.createBindGroup({
      layout: this.renderInteractionPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.bufRenderUBO } }]
    })

    this.bufInteraction = this.device.createBuffer({
      label: 'Interaction Buffer',
      size: 2 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
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
    this.RenderUBOViews.INTERACTION_RADIUS.set([this.settings.INTERACTION_RADIUS])
    this.RenderUBOViews.INTERACTION_AMOUNT.set([this.simulation.settings.INTERACTION_AMOUNT])
    this.device.queue.writeBuffer(this.bufRenderUBO, 0, this.RenderUBOValues)
  }

  render(interactionType: 'add-fluid' | 'force') {
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

    this.writeUniforms()
    this.device.queue.writeBuffer(this.bufInteraction, 0, this.simulation.UBOViews.INTERACTION_POS)
    let encoder = this.device.createCommandEncoder({ label: 'pass encoder' })
    const pass = encoder.beginRenderPass(renderPassDescriptor)
    pass.setPipeline(this.renderParticlesPipeline)
    pass.setVertexBuffer(0, this.simulation.bufPositions)
    pass.setBindGroup(0, this.renderParticlesBindGroup)
    pass.draw(6, this.simulation.settings.N)

    if (this.simulation.settings.INTERACTION_AMOUNT !== 0.0) {
      pass.setPipeline(this.renderInteractionPipeline)
      pass.setVertexBuffer(0, this.bufInteraction)
      pass.setBindGroup(0, this.renderInteractionBindGroup)
      pass.draw(6, 1)
    }
    pass.end()

    this.device.queue.submit([encoder.finish()])
  }
}
