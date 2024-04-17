<script lang="ts">
  import { onMount } from 'svelte'
  import { GPUSimulation } from '../lib/GPUSimulation/GPUSimulation'
  import { GPUSimulationRenderer } from '$lib/GPUSimulationRenderer/GPUSimulationRenderer'
  import { AccordionItem, Accordion, Radio } from 'flowbite-svelte'
  import { Label, Input, Select, Range, Button } from 'flowbite-svelte'
  import Plotly from '@aknakos/sveltekit-plotly'
  import { writable } from 'svelte/store'
  import { hexToRGB, range, roundTo } from '$lib/utils'

  let canvas: HTMLCanvasElement

  const defaultSettings = {
    N: 4096,
    DENSITY_TO_PRESSURE_FACTOR: 2000,
    DENSITY_KERNEL: 'spiky' as 'spiky' | 'soft',
    SELECTED_PROPERTY: 'velocity' as 'velocity' | 'density' | 'force',
    NEAR_DENSITY_TO_NEAR_PRESSURE_FACTOR: 2000,
    PARTICLE_MASS: 1,
    BASE_DENSITY: 2,
    SMOOTHING_RADIUS: 5,
    DAMPING_COEFF: 0.95,
    TIME_STEP: 1 / 60,
    DYNAMIC_VISCOSITY: 0.8,
    GRAVITY: -9.81,
    RADIUS: 1,
    INTERACTION_STRENGTH: 30,
    SCENE_SIZE: [200, 200] as [number, number],
    VIEWPORT_SIZE: [1000, 1000] as [number, number],
    MIN_COLOR: '#c799c3',
    MAX_COLOR: '#d41111',
  }

  const settings = writable({ ...defaultSettings })

  let gpuSimulation: GPUSimulation
  let gpuSimulationRenderer: GPUSimulationRenderer

  function generateRandomPositions(N: number) {
    let positions = new Float32Array(N * 2)
    for (let i = 0; i < positions.length; i += 2) {
      positions[i] = Math.random() * $settings.SCENE_SIZE[0]
      positions[i + 1] = $settings.SCENE_SIZE[1] / 4 + (Math.random() * $settings.SCENE_SIZE[1] * 3) / 4
    }
    return positions
  }

  async function init() {
    if (!navigator.gpu) {
      alert('WebGPU is not supported in your browser. Please use a browser that supports WebGPU.')
      throw new Error('WebGPU is not supported in your browser.')
    }

    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) {
      alert('No appropriate GPUAdapter found.')
      throw new Error('No appropriate GPUAdapter found.')
    }

    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBuffersPerShaderStage: 10
      }
    })

    device.lost.then((...a) => {
      console.log('Device lost', ...a)
    })

    gpuSimulation = new GPUSimulation({
      device,
      settings: $settings,
      initialValues: { positions: generateRandomPositions($settings.N) }
    })

    gpuSimulationRenderer = new GPUSimulationRenderer({
      device,
      simulation: gpuSimulation,
      canvas,
      settings: {
        SCENE_SIZE: $settings.SCENE_SIZE,
        VIEWPORT_SIZE: $settings.VIEWPORT_SIZE,
        SELECTED_PROPERTY: $settings.SELECTED_PROPERTY,
        MIN_COLOR: hexToRGB($settings.MIN_COLOR),
        MAX_COLOR: hexToRGB($settings.MAX_COLOR),
        RADIUS: $settings.RADIUS
      }
    })
  }

  function loop() {
    requestAnimationFrame(loop)

    gpuSimulation.simulate()
    gpuSimulationRenderer.render()
  }

  onMount(async () => {
    await init()
    loop()
  })

  function onCanvasMouseMove(event: MouseEvent) {
    const type = event.buttons === 2 ? 'pull' : event.buttons === 1 ? 'push' : 'none'
    if (type != 'none') {
      const x = (event.offsetX / canvas.offsetWidth) * $settings.SCENE_SIZE[0]
      const y = (event.offsetY / canvas.offsetHeight) * $settings.SCENE_SIZE[1]

      // const [x, y] = mousePosToScenePos(event.clientX, event.clientY)
      gpuSimulation.setInteraction(x, y, type)
    }
  }

  function onCanvasMouseUp(event: MouseEvent) {
    gpuSimulation.setInteraction(0, 0, 'none')
  }


  $: ({N} = $settings)
  $: N, init()
  $: gpuSimulation?.updateSettings($settings)
  $: gpuSimulationRenderer?.updateSettings({
    SELECTED_PROPERTY: $settings.SELECTED_PROPERTY,
    MIN_COLOR: hexToRGB($settings.MIN_COLOR),
    MAX_COLOR: hexToRGB($settings.MAX_COLOR),
    RADIUS: $settings.RADIUS
  })

  function getSpikyKernelData() {
    const NORMALIZATION_DENSITY = 6 / Math.PI
    const x = range(-1, 1 + 0.01, 0.01)
    const y = x.map((x) => NORMALIZATION_DENSITY * (1 - Math.abs(x)) ** 2)

    return { name: 'Density Kernel', x, y }
  }

  function getSpikyNearKernelData() {
    const NORMALIZATION_NEAR_DENSITY = 10 / Math.PI
    const x = range(-1, 1 + 0.01, 0.01)
    const y = x.map((x) => NORMALIZATION_NEAR_DENSITY * (1 - Math.abs(x)) ** 3)

    return { name: 'Near Density Kernel', x, y }
  }

  function getSoftKernelData() {
    const NORMALIZATION_DENSITY = 6 / Math.PI
    const x = range(-1, 1 + 0.01, 0.01)
    const y = x.map((x) => NORMALIZATION_DENSITY * (1 - Math.abs(x) ** 2) ** 3)

    return { name: 'Density Kernel', x, y }
  }

  function getSoftNearKernelData() {
    const NORMALIZATION_NEAR_DENSITY = 10 / Math.PI
    const x = range(-1, 1 + 0.01, 0.01)
    const y = x.map((x) => NORMALIZATION_NEAR_DENSITY * (1 - Math.abs(x) ** 2) ** 4)

    return { name: 'Near Density Kernel', x, y }
  }

  $: kernelData = $settings.DENSITY_KERNEL === 'spiky' ? getSpikyKernelData() : getSoftKernelData()
  $: nearKernelData = $settings.DENSITY_KERNEL === 'spiky' ? getSpikyNearKernelData() : getSoftNearKernelData()

  const elements = [
    {
      label: 'Particle Count',
      name: 'N',
      values: [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576],
      description:
        'The number of particles in the simulation. It has to be a power of 2 because the algorithms used in the current implementation require it.'
    },
    {
      label: 'Density to Pressure Mult.',
      name: 'DENSITY_TO_PRESSURE_FACTOR',
      range: [100, 4000, 100],
      description:
        'This multiplier converts the local density gradient to a pressure force. A higher value means a higher pressure.'
    },
    {
      label: 'Near Density to Near Pressure Mult.',
      name: 'NEAR_DENSITY_TO_NEAR_PRESSURE_FACTOR',
      range: [100, 4000, 100],
      description:
        'This multiplier converts the local density gradient to a pressure force. A higher value means a higher pressure.'
    },
    {
      label: 'Particle Mass',
      name: 'PARTICLE_MASS',
      range: [0.1, 10, 0.1],
      description: 'The mass of each particle.'
    },
    {
      label: 'Base Density',
      name: 'BASE_DENSITY',
      range: [0.1, 4, 0.1],
      description: 'The base density of the fluid.'
    },
    {
      label: 'Smoothing Radius',
      name: 'SMOOTHING_RADIUS',
      range: [1, 10, 0.1],
      description: 'The radius of the smoothing kernel i.e. the radius of the interaction between particles.'
    },
    {
      label: 'Boundary Damping Coefficient',
      name: 'DAMPING_COEFF',
      range: [0, 1, 0.01],
      description:
        'The damping coefficient of boundary. 0 means the boundary absorbs all the energy. 1 means no damping.'
    },
    {
      label: 'Time Step',
      name: 'TIME_STEP',
      range: [0.001, 0.1, 0.001],
      description:
        'The time step of the simulation. A smaller value means a slower but more accurate simulation. A higher value means a faster but less accurate simulation. Very high values can cause instability.'
    },
    {
      label: 'Dynamic Viscosity',
      name: 'DYNAMIC_VISCOSITY',
      range: [0, 1, 0.01],
      description: 'Zero means no viscosity. One means full viscosity.'
    },
    {
      label: 'Gravity',
      name: 'GRAVITY',
      range: [-20, 20, 0.1],
      description: 'The gravity force experienced by the fluid.'
    },
    {
      label: 'Radius',
      name: 'RADIUS',
      range: [0.1, 10, 0.1],
      description: 'The radius of the particles.'
    }
  ]
</script>

<main class="flex h-screen">
  <canvas
    class="flex-shrink-0 self-start h-full"
    on:mouseup={onCanvasMouseUp}
    on:mousemove={onCanvasMouseMove}
    on:mousedown={onCanvasMouseMove}
    on:contextmenu={(e) => e.preventDefault()}
    bind:this={canvas}
    style="border: 1px solid black;"
    width={$settings.VIEWPORT_SIZE[0]}
    height={$settings.VIEWPORT_SIZE[1]}
  />
  <Accordion multiple class="flex-1 overflow-y-auto">
    <AccordionItem open>
      <span slot="header">Visualization</span>
      <div class="flex mb-2">
        <Label for="hs-color-input" class="block me-2">Color based on:</Label>
        <Select
          items={[
            { value: 'density', name: 'Density' },
            { value: 'velocity', name: 'Velocity' },
            { value: 'force', name: 'Force' }
          ]}
          bind:value={$settings.SELECTED_PROPERTY}
        />
      </div>
      <div class="flex gap-2">
        <div class="flex items-center">
          <Label for="hs-color-input" class="block me-2">Min color:</Label>
          <Input
            type="color"
            class="p-1 h-10 w-14 block bg-white border border-gray-200 cursor-pointer rounded-lg disabled:opacity-50 disabled:pointer-events-none"
            id="hs-color-input"
            bind:value={$settings.MIN_COLOR}
            title="Choose your color"
          />
        </div>
        <div class="flex items-center">
          <Label for="hs-color-input" class="block me-2">Max color:</Label>
          <Input
            type="color"
            class="p-1 h-10 w-14 block bg-white border border-gray-200 cursor-pointer rounded-lg disabled:opacity-50 disabled:pointer-events-none"
            id="hs-color-input"
            bind:value={$settings.MAX_COLOR}
            title="Choose your color"
          />
        </div>
      </div>
    </AccordionItem>
    <AccordionItem open>
      <span slot="header">Simulation parameters</span>
      <div class="flex">
        <Plotly
          id="id"
          class="m-2"
          data={[kernelData, nearKernelData]}
          layout={{
            height: 100,
            width: 300,
            autosize: false,
            margin: {
              autoexpand: false,
              t: 0,
              l: 0,
              b: 0,
              r: 0
            }
          }}
          config={{ staticPlot: true }}>Loading plotly..</Plotly
        >
        <div class="flex flex-col gap-2">
          <span>
            <Radio
              name="spiky"
              group={$settings.DENSITY_KERNEL}
              value="spiky"
              on:click={(e) => ($settings.DENSITY_KERNEL = 'spiky')}>Use Spiky Kernel (Recommended)</Radio
            >
            Spiky Kernel is better at pushing close particles away.
          </span>
          <span>
            <Radio
              name="soft"
              group={$settings.DENSITY_KERNEL}
              value="soft"
              on:click={(e) => ($settings.DENSITY_KERNEL = 'soft')}>Use Soft Kernel</Radio
            >
            When using a Soft Kernel, particles tend to clump together.
          </span>
        </div>
      </div>
      <section class="m-2 p-2">
        <div class="bg-red gap-8">
          <div class="grid grid-cols-2 gap-x-2">
            {#each elements as { name, label, range, values, description }}
              <Label for={name} class="self-center font-semibold">{label}: {roundTo($settings[name], 4)}</Label>
              {#if range}
                <Range
                  class="self-center"
                  {name}
                  min={range[0]}
                  max={range[1]}
                  step={range[2]}
                  bind:value={$settings[name]}
                />
              {:else if values}
                <Select
                  class="self-center"
                  {name}
                  items={values.map((value) => ({ value, name: value.toString() }))}
                  bind:value={$settings[name]}
                />
              {/if}
              <Button
                disabled={$settings[name] === defaultSettings[name]}
                on:click={() => ($settings[name] = defaultSettings[name])}>Reset</Button
              >
              <p class="col-span-3 mb-4">{description}</p>
            {/each}
          </div>
        </div>
      </section>
    </AccordionItem>
  </Accordion>
</main>
