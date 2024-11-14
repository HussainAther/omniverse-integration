#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void shadowEffectKernel(float* output, const float* depth, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        output[idx] = depth[idx] * 0.5f;  // Example shadow intensity
    }
}

__global__ void reflectionEffectKernel(float* output, const float* image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        output[idx] = image[idx] * 0.8f;  // Reflection effect
    }
}

__global__ void particleEffectKernel(float* output, int numParticles, float time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        output[idx] = sinf(time + idx * 0.1f) * 0.5f;  // Simple particle oscillation
    }
}

extern "C" void launchShadowEffect(float* output, const float* depth, int width, int height) {
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
    shadowEffectKernel<<<blocks, threads>>>(output, depth, width, height);
}

extern "C" void launchReflectionEffect(float* output, const float* image, int width, int height) {
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
    reflectionEffectKernel<<<blocks, threads>>>(output, image, width, height);
}

extern "C" void launchParticleEffect(float* output, int numParticles, float time) {
    int threads = 256;
    int blocks = (numParticles + threads - 1) / threads;
    particleEffectKernel<<<blocks, threads>>>(output, numParticles, time);
}

