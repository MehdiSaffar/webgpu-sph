# Fluid simulator app with WebGPU API

[![](./demo.mp4)](https://webgpu-sph.vercel.app)

This project is a fluid simulator application written in WebGPU, utilizing the [Smoothed Particle Hydrodynamics (SPH)](https://en.wikipedia.org/wiki/Smoothed-particle_hydrodynamics) technique for simulation. 

You can play around with the app at [webgpu-sph.vercel.app](https://webgpu-sph.vercel.app).

The frontend is developed using [SvelteKit](https://kit.svelte.dev/), [Flowbite](https://flowbite-svelte.com/), and [TailwindCSS](https://tailwindcss.com/). It is bundled with Vite. The simulation and rendering are entirely done in WebGPU, providing a fast and efficient fluid simulation experience.

This project was inspired by [Sebastian League's Coding Adventure video about Fluid Simulation](https://www.youtube.com/watch?v=rSKMYc1CQHE).

## Developing

```bash
# 1) Install dependencies
npm install
# 2) Start a development server
npm run dev
# 3) Open browser at the URL shown in the terminal
```

## Building

To create a production version of your app:

```bash
npm run build
```

You can preview the production build with `npm run preview` .

## Contact

For inquiries or collaborations, feel free to reach out to me on [LinkedIn](https://github.com/MehdiSaffar).

Developed by Mehdi Saffar.