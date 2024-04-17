<script lang="ts">
  import { onMount } from 'svelte'
  import { GPUSimulation } from '../lib/GPUSimulation/GPUSimulation'
  import { GPUSimulationRenderer } from '$lib/GPUSimulationRenderer/GPUSimulationRenderer'
  import { AccordionItem, Accordion, Radio, ButtonGroup, Label, Input, Select, Range, Button } from 'flowbite-svelte'
  import Plotly from '@aknakos/sveltekit-plotly'
  import { LinkedinSolid, GithubSolid, PlaySolid, PauseSolid } from 'flowbite-svelte-icons'

  import { writable } from 'svelte/store'
  import { hexToRGB, range, roundTo, sleep } from '$lib/utils'
  import { getBuffer } from '$lib/webgpu-utils'

  let canvas: HTMLCanvasElement

  let interactionType = 'add-fluid'

  const defaultSettings = {
    MAX_N: 8192 * 2,
    DENSITY_KERNEL: 'spiky' as 'spiky' | 'soft',
    SELECTED_PROPERTY: 'velocity' as 'velocity' | 'density' | 'force',
    DENSITY_TO_PRESSURE_FACTOR: 2000,
    NEAR_DENSITY_TO_NEAR_PRESSURE_FACTOR: 1500,
    PARTICLE_MASS: 1,
    BASE_DENSITY: 2.5,
    SMOOTHING_RADIUS: 5,
    DAMPING_COEFF: 0.95,
    TIME_STEP: 1 / 60,
    DYNAMIC_VISCOSITY: 0.5,
    GRAVITY: -9.81,
    RADIUS: 0.8,
    INTERACTION_STRENGTH: 75,
    INTERACTION_RADIUS: 25,
    SCENE_SIZE: [200, 200] as [number, number],
    VIEWPORT_SIZE: [1000, 1000] as [number, number],
    MIN_COLOR: '#FFE2DB',
    MAX_COLOR: '#FF3300',
    BACKGROUND_COLOR: '#ffffff',
    INTERACTION_ADD_FLUID_ADD_COUNT: 10
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
    running = false
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
      initialValues: { positions: generateRandomPositions($settings.MAX_N / 2) }
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
        INTERACTION_RADIUS: $settings.INTERACTION_RADIUS,
        BACKGROUND_COLOR: hexToRGB($settings.BACKGROUND_COLOR),
        RADIUS: $settings.RADIUS
      }
    })
    running = true
  }

  window.printall = async () => {
    const ranges = await getBuffer(gpuSimulation.bufRanges, gpuSimulation.device, Float32Array)
    // const pos = (await getBuffer(gpuSimulation.bufPositions, gpuSimulation.device, Float32Array)).slice(0, $settings.N)
    // const den = (await getBuffer(gpuSimulation.bufDensities, gpuSimulation.device, Float32Array)).slice(0, $settings.N * 2)
    // console.log('density', den)
    // console.log('positions', pos)
    console.log(ranges)
  }

  async function loop() {
    // await gpuSimulation.device.queue.onSubmittedWorkDone()
    if (running) {
      gpuSimulation.simulate()
      gpuSimulationRenderer.render()
    }

    // await sleep(100)
    requestAnimationFrame(loop)
  }

  onMount(async () => {
    await init()
    loop()
  })

  function onKeyDown(event: KeyboardEvent) {
    if (event.code === 'KeyA') {
      interactionType = 'add-fluid'
    } else if (event.code === 'KeyF') {
      interactionType = 'force'
    } else if (event.code === 'Space') {
      running = !running
    }
  }

  function onCanvasMouseMove(event: MouseEvent) {
    const x = (event.offsetX / canvas.offsetWidth) * $settings.SCENE_SIZE[0]
    const y = (event.offsetY / canvas.offsetHeight) * $settings.SCENE_SIZE[1]

    if (interactionType === 'force') {
      const type = event.buttons === 2 ? 'pull' : event.buttons === 1 ? 'push' : 'none'
      if (type !== 'none') {
        gpuSimulation.interactForce(x, y, type)
      }
    } else if (interactionType === 'add-fluid') {
      if (event.buttons === 1) {
        gpuSimulation.interactAddFluid(x, y, $settings.INTERACTION_ADD_FLUID_ADD_COUNT)
      }
    }
  }

  function onCanvasMouseUp(event: MouseEvent) {
    if (interactionType === 'force') {
      gpuSimulation.interactForce(0, 0, 'none')
    } else {
      onCanvasMouseMove(event)
    }
  }

  let running = true

  $: ({ MAX_N } = $settings)
  $: MAX_N, init()
  $: gpuSimulation?.updateSettings($settings)
  $: gpuSimulationRenderer?.updateSettings({
    SELECTED_PROPERTY: $settings.SELECTED_PROPERTY,
    MIN_COLOR: hexToRGB($settings.MIN_COLOR),
    MAX_COLOR: hexToRGB($settings.MAX_COLOR),
    BACKGROUND_COLOR: hexToRGB($settings.BACKGROUND_COLOR),
    INTERACTION_RADIUS: $settings.INTERACTION_RADIUS,
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

  const simulationParameters = [
    {
      label: 'Maximum Particle Count',
      name: 'MAX_N',
      values: [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576],
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
      range: [0.1, 2, 0.1],
      description: 'The radius of the particles.'
    }
  ]
  const interactionParameters = [
    {
      label: 'Force - Interaction Strength',
      name: 'INTERACTION_STRENGTH',
      range: [100, 10_000, 100],
      description: 'The intensity of the pushing-pulling force.'
    },
    {
      label: 'Force - Interaction Radius',
      name: 'INTERACTION_RADIUS',
      range: [1, 100, 1],
      description: 'The radius of the pushing-pulling force.'
    },
    {
      label: 'Add Fluid - Particle Count',
      name: 'INTERACTION_ADD_FLUID_ADD_COUNT',
      range: [1, 20, 1],
      description: 'Number of particles to add on every frame.'
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
    width={$settings.VIEWPORT_SIZE[0]}
    height={$settings.VIEWPORT_SIZE[1]}
  />
  <Accordion multiple class="flex-1 overflow-y-auto px-8" flush>
    <div class="flex flex-col gap-4 my-8">
      <h3 class="text-2xl font-medium">Welcome to the fluid simulator!</h3>
      <p>
        This simulator is based on the <a
          href="https://en.wikipedia.org/wiki/Smoothed-particle_hydrodynamics"
          target="_blank">Smoothed Particle Hydrodynamics (SPH)</a
        >
        technique. It runs on the
        <a href="https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API" target="_blank">WebGPU API</a>.
      </p>
      <p>
        To get started, choose the maximum number of desired particles under <span class="font-bold"
          >Simulation Parameters</span
        > and play around with the other settings to change the behavior of the fluid. You can also interact with the fluid
        by clicking and dragging on the canvas.
      </p>
      <p>
        For more information, check out the project's code on <a
          href="https://github.com/MehdiSaffar/webgpu-sph"
          target="_blank"><GithubSolid class="inline mb-1" /> GitHub</a
        >
        and connect with me on
        <a href="https://www.linkedin.com/in/MehdiSaffar" target="_blank"
          ><LinkedinSolid class="inline mb-1" /> LinkedIn</a
        >.
      </p>
      <p>Have fun!</p>
      <hr />
      <Button on:click={() => (running = !running)}>
        {#if running}
          Pause simulation (Spacebar) <PauseSolid />
        {:else}
          Play simulation (Spacebar) <PlaySolid />
        {/if}
      </Button>
      <div class="flex">
        <Label for="hs-color-input" class="flex-shrink-0 self-center me-2">Interaction mode:</Label>
        <ButtonGroup class="w-full">
          <Button
            class="w-full"
            outline={interactionType !== 'force'}
            on:click={() => (interactionType = 'force')}
            color="primary">Force (F)</Button
          >
          <Button
            class="w-full"
            outline={interactionType !== 'add-fluid'}
            on:click={() => (interactionType = 'add-fluid')}
            color="primary">Add fluid (A)</Button
          >
        </ButtonGroup>
      </div>

      <p>
        <span class="font-semibold">Force mode:</span> Left-click to pull the fluid towards the cursor. Right-click to push
        the fluid away.
      </p>
      <p>
        <span class="font-semibold">Add fluid mode:</span> Left-click to add fluid to the scene. You can set the number of
        particles to be added on every frame. You can only add up to the set Max Particle Count.
      </p>
    </div>
    <AccordionItem>
      <span slot="header">Visualization parameters</span>
      <div class="flex mb-2">
        <Label for="hs-color-input" class="flex-shrink-0 self-center me-2">Color based on:</Label>
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
          <Label for="background-color" class="block me-2">Background color:</Label>
          <Input
            type="color"
            class="p-1 h-10 w-14 block bg-white border border-gray-200 cursor-pointer rounded-lg disabled:opacity-50 disabled:pointer-events-none"
            id="background-color"
            bind:value={$settings.BACKGROUND_COLOR}
            title="Choose the background color"
          />
        </div>
        <div class="flex items-center">
          <Label for="min-color" class="block me-2">Minimum color:</Label>
          <Input
            type="color"
            class="p-1 h-10 w-14 block bg-white border border-gray-200 cursor-pointer rounded-lg disabled:opacity-50 disabled:pointer-events-none"
            id="min-color"
            bind:value={$settings.MIN_COLOR}
            title="Choose the minimum color"
          />
        </div>
        <div class="flex items-center">
          <Label for="max-color" class="block me-2">Maximum color:</Label>
          <Input
            type="color"
            class="p-1 h-10 w-14 block bg-white border border-gray-200 cursor-pointer rounded-lg disabled:opacity-50 disabled:pointer-events-none"
            id="max-color"
            bind:value={$settings.MAX_COLOR}
            title="Choose the maximum color"
          />
        </div>
      </div>
    </AccordionItem>
    <AccordionItem>
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
        <div class="parameters gap-x-2">
          {#each simulationParameters as { name, label, range, values, description }}
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
              class="ml-8"
              disabled={$settings[name] === defaultSettings[name]}
              on:click={() => ($settings[name] = defaultSettings[name])}>Reset</Button
            >
            <p class="col-span-3 mb-12">{description}</p>
          {/each}
        </div>
      </section>
    </AccordionItem>
    <AccordionItem>
      <span slot="header">Interaction parameters</span>
      <section class="m-2 p-2">
        <div class="parameters gap-x-2">
          {#each interactionParameters as { name, label, range, values, description }}
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
              class="ml-8"
              disabled={$settings[name] === defaultSettings[name]}
              on:click={() => ($settings[name] = defaultSettings[name])}>Reset</Button
            >
            <p class="col-span-3 mb-12">{description}</p>
          {/each}
        </div>
      </section>
    </AccordionItem>
  </Accordion>
</main>

<svelte:window on:keydown={onKeyDown} />

<style>
  a {
    @apply font-medium text-blue-600 hover:underline;
  }
  .parameters {
    @apply grid;
    grid-template-columns: auto 1fr auto;
  }
</style>
