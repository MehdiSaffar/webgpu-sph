import type { TypedArrayConstructor } from "webgpu-utils"

export async function getBuffer(buffer: GPUBuffer, device: GPUDevice, Type: TypedArrayConstructor) {
  const bufferCopy = device.createBuffer({
    label: 'Copy Buffer',
    size: buffer.size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
  })

  const encoder = device.createCommandEncoder()
  encoder.copyBufferToBuffer(buffer, 0, bufferCopy, 0, buffer.size)
  device.queue.submit([encoder.finish()])

  await bufferCopy.mapAsync(GPUMapMode.READ)
  const buf = bufferCopy.getMappedRange().slice(0)
  bufferCopy.unmap()
  bufferCopy.destroy()
  return new Type(buf)
}
