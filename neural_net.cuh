#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Hyperparameters
// ---------------------------------------------------------------------------
#define NN_LEAK       0.01f
#define NN_ADAM_BETA1 0.9f
#define NN_ADAM_BETA2 0.999f
#define NN_ADAM_EPS   1.0e-8f

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_err));             \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t _s = (call);                                            \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n",                       \
                    __FILE__, __LINE__, (int)_s);                              \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Struct
// ---------------------------------------------------------------------------
typedef struct {
    unsigned int size;
    unsigned int value_count;
    unsigned int weight_count;
    unsigned int bias_count;

    // Host — needed for the per-layer loop in C
    unsigned int* structure;       // [size]
    unsigned int* value_offsets;   // [size]
    unsigned int* weight_offsets;  // [size]
    unsigned int* bias_offsets;    // [size]

    // Device
    float* weights;    // [weight_count]
    float* bias;       // [bias_count]
    float* values;     // [value_count]  post-activation
    float* zvalues;    // [value_count]  pre-activation
    float* delta;      // [value_count]  dL/dz per neuron
    float* weight_m;   // [weight_count] Adam 1st moment
    float* weight_v;   // [weight_count] Adam 2nd moment
    float* bias_m;     // [bias_count]
    float* bias_v;     // [bias_count]

    unsigned int step; // Adam timestep (host scalar)
} neural_net;

// ---------------------------------------------------------------------------
// Global cuBLAS handle (define once in a .cu translation unit)
// ---------------------------------------------------------------------------
static cublasHandle_t nn_cublas_handle;
static const float    nn_alpha = 1.0f;
static const float    nn_beta  = 0.0f;

static inline void gpu_init() {
    CUBLAS_CHECK(cublasCreate(&nn_cublas_handle));
}

static inline void gpu_destroy() {
    cublasDestroy(nn_cublas_handle);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static inline float random_uniform(float range) {
    return ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * range;
}

// ---------------------------------------------------------------------------
// Init / Free
// ---------------------------------------------------------------------------
static void neural_net_free(neural_net* net) {
    free(net->structure);
    free(net->value_offsets);
    free(net->weight_offsets);
    free(net->bias_offsets);
    cudaFree(net->weights);
    cudaFree(net->bias);
    cudaFree(net->values);
    cudaFree(net->zvalues);
    cudaFree(net->delta);
    cudaFree(net->weight_m);
    cudaFree(net->weight_v);
    cudaFree(net->bias_m);
    cudaFree(net->bias_v);
    memset(net, 0, sizeof(*net));
}

static void neural_net_init(neural_net* a, unsigned int len, unsigned int structure[]) {
    memset(a, 0, sizeof(*a));
    a->size = len;

    // Host offset tables
    a->structure      = (unsigned int*)malloc(sizeof(unsigned int) * len);
    a->value_offsets  = (unsigned int*)malloc(sizeof(unsigned int) * len);
    a->weight_offsets = (unsigned int*)malloc(sizeof(unsigned int) * len);
    a->bias_offsets   = (unsigned int*)malloc(sizeof(unsigned int) * len);
    if (!(a->structure && a->value_offsets && a->weight_offsets && a->bias_offsets)) {
        fprintf(stderr, "Host alloc failed\n"); exit(1);
    }

    for (unsigned int i = 0; i < len; i++) a->structure[i] = structure[i];

    unsigned int weight_acc = 0, bias_acc = 0, value_acc = 0;
    for (unsigned int i = 0; i < len; i++) {
        a->value_offsets[i] = value_acc;
        value_acc += structure[i];
        if (i > 0) {
            a->weight_offsets[i] = weight_acc;
            weight_acc += structure[i - 1] * structure[i];
            a->bias_offsets[i] = bias_acc;
            bias_acc += structure[i];
        } else {
            a->weight_offsets[i] = 0;
            a->bias_offsets[i]   = 0;
        }
    }
    a->value_count  = value_acc;
    a->weight_count = weight_acc;
    a->bias_count   = bias_acc;

    // Init weights on host, upload to device
    float* h_weights = (float*)malloc(a->weight_count * sizeof(float));
    float* h_bias    = (float*)calloc(a->bias_count,   sizeof(float));
    if (!h_weights || !h_bias) { fprintf(stderr, "Host weight alloc failed\n"); exit(1); }

    for (unsigned int layer = 1; layer < len; layer++) {
        unsigned int fan_in  = structure[layer - 1];
        unsigned int fan_out = structure[layer];
        unsigned int w_off   = a->weight_offsets[layer];
        unsigned int n       = fan_in * fan_out;
        float range = (layer == len - 1)
            ? sqrtf(6.0f / (float)(fan_in + fan_out))                   // Glorot
            : sqrtf(6.0f / ((float)fan_in * (1.0f + NN_LEAK * NN_LEAK))); // He-leaky
        for (unsigned int i = 0; i < n; i++)
            h_weights[w_off + i] = random_uniform(range);
    }

    // Device allocs
    CUDA_CHECK(cudaMalloc((void**)&a->weights,  a->weight_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&a->bias,     a->bias_count   * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&a->values,   a->value_count  * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&a->zvalues,  a->value_count  * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&a->delta,    a->value_count  * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&a->weight_m, a->weight_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&a->weight_v, a->weight_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&a->bias_m,   a->bias_count   * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&a->bias_v,   a->bias_count   * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(a->weights, h_weights,
        a->weight_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(a->bias, h_bias,
        a->bias_count * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(a->weight_m, 0, a->weight_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(a->weight_v, 0, a->weight_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(a->bias_m,   0, a->bias_count   * sizeof(float)));
    CUDA_CHECK(cudaMemset(a->bias_v,   0, a->bias_count   * sizeof(float)));
    CUDA_CHECK(cudaMemset(a->values,   0, a->value_count  * sizeof(float)));
    CUDA_CHECK(cudaMemset(a->zvalues,  0, a->value_count  * sizeof(float)));
    CUDA_CHECK(cudaMemset(a->delta,    0, a->value_count  * sizeof(float)));

    free(h_weights);
    free(h_bias);
    a->step = 0;
}

// ---------------------------------------------------------------------------
// Kernels — forward
// ---------------------------------------------------------------------------

// Add bias vector into z in-place, then write sigmoid or leaky-relu into out.
// is_output: 1 = sigmoid (output layer), 0 = leaky-relu (hidden layer)
__global__ void k_bias_activate(
    float* __restrict__ out,
    float* __restrict__ z,
    const float* __restrict__ b,
    unsigned int n,
    int is_output)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = z[i] + b[i];
    z[i] = v;  // store pre-activation (needed for backprop delta of hidden layers)
    if (is_output) {
        // numerically stable sigmoid
        out[i] = (v >= 0.0f)
            ? 1.0f / (1.0f + expf(-v))
            : expf(v) / (1.0f + expf(v));
    } else {
        out[i] = v > 0.0f ? v : NN_LEAK * v;
    }
}

// ---------------------------------------------------------------------------
// Kernels — backward
// ---------------------------------------------------------------------------

// Output layer delta: sigmoid + BCE => dL/dz = a - target
__global__ void k_output_delta(
    float* __restrict__ delta,
    const float* __restrict__ values,
    const float* __restrict__ target,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        delta[i] = values[i] - target[i];
}

// Hidden layer delta: multiply incoming delta by W^T, then chain-rule through
// leaky-relu using the stored pre-activation z.
//
// Layout: W[layer] is (curr x prev) in row-major, i.e. W[j*prev + i].
// We compute delta_prev[i] = sum_j( W[j*prev+i] * delta_curr[j] ) * lrelu'(z_prev[i])
//
// One thread per previous-layer neuron.
__global__ void k_hidden_delta(
    float* __restrict__ delta_prev,       // out: [prev_size]
    const float* __restrict__ delta_curr, // in:  [curr_size]
    const float* __restrict__ W,          // in:  [curr_size * prev_size]
    const float* __restrict__ z_prev,     // in:  [prev_size]  pre-activation
    unsigned int prev_size,
    unsigned int curr_size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= prev_size) return;

    float sum = 0.0f;
    for (unsigned int j = 0; j < curr_size; j++)
        sum += W[j * prev_size + i] * delta_curr[j];

    float z = z_prev[i];
    delta_prev[i] = sum * (z > 0.0f ? 1.0f : NN_LEAK);
}

// Adam update for weights.
// One thread per weight: wi = j * prev_size + i
__global__ void k_adam_weights(
    float* __restrict__ W,
    float* __restrict__ m,
    float* __restrict__ v,
    const float* __restrict__ delta_curr, // [curr_size]
    const float* __restrict__ a_prev,     // [prev_size]  post-activation
    unsigned int prev_size,
    unsigned int curr_size,
    float adam_lr)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = curr_size * prev_size;
    if (idx >= total) return;

    unsigned int j = idx / prev_size;  // output neuron
    unsigned int i = idx % prev_size;  // input neuron

    float grad = delta_curr[j] * a_prev[i];
    float mi = NN_ADAM_BETA1 * m[idx] + (1.0f - NN_ADAM_BETA1) * grad;
    float vi = NN_ADAM_BETA2 * v[idx] + (1.0f - NN_ADAM_BETA2) * grad * grad;
    m[idx] = mi;
    v[idx] = vi;
    W[idx] -= adam_lr * mi / (sqrtf(vi) + NN_ADAM_EPS);
}

// Adam update for biases.
// One thread per output neuron.
__global__ void k_adam_biases(
    float* __restrict__ B,
    float* __restrict__ m,
    float* __restrict__ v,
    const float* __restrict__ delta,  // [n]
    unsigned int n,
    float adam_lr)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float grad = delta[i];
    float mi = NN_ADAM_BETA1 * m[i] + (1.0f - NN_ADAM_BETA1) * grad;
    float vi = NN_ADAM_BETA2 * v[i] + (1.0f - NN_ADAM_BETA2) * grad * grad;
    m[i] = mi;
    v[i] = vi;
    B[i] -= adam_lr * mi / (sqrtf(vi) + NN_ADAM_EPS);
}

// MSE reduction: each thread accumulates a partial sum, then atomicAdd into
// a single device float.
__global__ void k_mse(
    const float* __restrict__ values,
    const float* __restrict__ target,
    float* __restrict__ out,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float local = 0.0f;
    if (i < n) {
        float d = target[i] - values[i];
        local = d * d;
    }
    // warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        local += __shfl_down_sync(0xffffffff, local, offset);
    if ((threadIdx.x & 31) == 0)
        atomicAdd(out, local);
}

// ---------------------------------------------------------------------------
// Forward propagation
// ---------------------------------------------------------------------------
static void forward_prop(neural_net* a, const float* input) {
    unsigned int L = a->size;

    // Upload input into values[0] and zvalues[0]
    unsigned int input_size = a->structure[0];
    CUDA_CHECK(cudaMemcpy(a->values,  input, input_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(a->zvalues, input, input_size * sizeof(float), cudaMemcpyHostToDevice));

    for (unsigned int layer = 1; layer < L; layer++) {
        unsigned int prev = a->structure[layer - 1];
        unsigned int curr = a->structure[layer];

        float* prev_vals = a->values  + a->value_offsets[layer - 1];
        float* curr_vals = a->values  + a->value_offsets[layer];
        float* curr_z    = a->zvalues + a->value_offsets[layer];
        float* W         = a->weights + a->weight_offsets[layer];
        float* B         = a->bias    + a->bias_offsets[layer];

        // W is row-major (curr x prev): W[j*prev + i].
        // cuBLAS is column-major, so it sees W as (prev x curr) column-major.
        // CUBLAS_OP_T transposes that back to (curr x prev), giving curr_z = W * prev_vals.
        // Leading dimension is prev (the number of rows in the column-major view).
        CUBLAS_CHECK(cublasSgemv(
            nn_cublas_handle, CUBLAS_OP_T,
            (int)prev, (int)curr,
            &nn_alpha, W, (int)prev,
            prev_vals, 1,
            &nn_beta,  curr_z, 1));

        int threads = 256;
        int blocks  = ((int)curr + threads - 1) / threads;
        int is_out  = (layer == L - 1) ? 1 : 0;

        k_bias_activate<<<blocks, threads>>>(curr_vals, curr_z, B, curr, is_out);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ---------------------------------------------------------------------------
// Backward propagation + Adam  (returns MSE loss, host float)
// ---------------------------------------------------------------------------
static float backward_prop(neural_net* net, const float* d_target, float lr) {
    unsigned int L = net->size;
    net->step++;

    float bias_fix1 = 1.0f - powf(NN_ADAM_BETA1, (float)net->step);
    float bias_fix2 = 1.0f - powf(NN_ADAM_BETA2, (float)net->step);
    float adam_lr   = lr * sqrtf(bias_fix2) / bias_fix1;

    int threads = 256;

    // --- Output delta: dL/dz = a - target (sigmoid + BCE) ---
    {
        unsigned int out_size = net->structure[L - 1];
        float* d_vals  = net->values + net->value_offsets[L - 1];
        float* d_delta = net->delta  + net->value_offsets[L - 1];
        int blocks = ((int)out_size + threads - 1) / threads;
        k_output_delta<<<blocks, threads>>>(d_delta, d_vals, d_target, out_size);
        CUDA_CHECK(cudaGetLastError());
    }

    // --- Backpropagate deltas through hidden layers ---
    for (int layer = (int)L - 1; layer > 1; layer--) {
        unsigned int curr_size = net->structure[layer];
        unsigned int prev_size = net->structure[layer - 1];

        float* delta_curr = net->delta  + net->value_offsets[layer];
        float* delta_prev = net->delta  + net->value_offsets[layer - 1];
        float* z_prev     = net->zvalues+ net->value_offsets[layer - 1];
        float* W          = net->weights+ net->weight_offsets[layer];

        int blocks = ((int)prev_size + threads - 1) / threads;
        k_hidden_delta<<<blocks, threads>>>(
            delta_prev, delta_curr, W, z_prev, prev_size, curr_size);
        CUDA_CHECK(cudaGetLastError());
    }

    // --- Adam update for every layer ---
    for (unsigned int layer = 1; layer < L; layer++) {
        unsigned int curr_size = net->structure[layer];
        unsigned int prev_size = net->structure[layer - 1];
        unsigned int total_w   = curr_size * prev_size;

        float* delta_curr = net->delta    + net->value_offsets[layer];
        float* a_prev     = net->values   + net->value_offsets[layer - 1];
        float* W          = net->weights  + net->weight_offsets[layer];
        float* wm         = net->weight_m + net->weight_offsets[layer];
        float* wv         = net->weight_v + net->weight_offsets[layer];
        float* B          = net->bias     + net->bias_offsets[layer];
        float* bm         = net->bias_m   + net->bias_offsets[layer];
        float* bv         = net->bias_v   + net->bias_offsets[layer];

        int w_blocks = ((int)total_w   + threads - 1) / threads;
        int b_blocks = ((int)curr_size + threads - 1) / threads;

        k_adam_weights<<<w_blocks, threads>>>(
            W, wm, wv, delta_curr, a_prev, prev_size, curr_size, adam_lr);
        CUDA_CHECK(cudaGetLastError());

        k_adam_biases<<<b_blocks, threads>>>(
            B, bm, bv, delta_curr, curr_size, adam_lr);
        CUDA_CHECK(cudaGetLastError());
    }

    // --- MSE loss (device → host) ---
    float* d_mse;
    CUDA_CHECK(cudaMalloc((void**)&d_mse, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_mse, 0, sizeof(float)));

    unsigned int out_size = net->structure[L - 1];
    float* d_out_vals = net->values + net->value_offsets[L - 1];
    int blocks = ((int)out_size + threads - 1) / threads;
    k_mse<<<blocks, threads>>>(d_out_vals, d_target, d_mse, out_size);
    CUDA_CHECK(cudaGetLastError());

    float h_mse = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_mse, d_mse, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_mse);

    return h_mse / (float)out_size;
}

// ---------------------------------------------------------------------------
// get_layer — copies a layer's post-activation values to a new host buffer.
// Caller must free() the returned pointer.
// ---------------------------------------------------------------------------
static float* get_layer(neural_net* a, unsigned int n) {
    unsigned int sz = a->structure[n];
    float* ret = (float*)malloc(sz * sizeof(float));
    if (!ret) { fprintf(stderr, "get_layer malloc failed\n"); exit(1); }
    CUDA_CHECK(cudaMemcpy(ret,
        a->values + a->value_offsets[n],
        sz * sizeof(float),
        cudaMemcpyDeviceToHost));
    return ret;
}