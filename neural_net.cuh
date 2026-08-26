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
// MEMORY LAYOUT  — one convention, enforced everywhere
// ---------------------------------------------------------------------------
//
// WEIGHTS for layer l:
//   Shape  (prev x curr)  where prev = structure[l-1], curr = structure[l]
//   Stored column-major with lda = prev.
//   Element W[i][j] = W_ptr[ i + j*prev ]
//     i = input neuron  (0..prev-1)
//     j = output neuron (0..curr-1)
//
//   Forward:  z = W^T * A_prev
//     cuBLAS sees W as (prev x curr), OP_T makes it (curr x prev).
//     Multiply (curr x prev) * (prev x batch) → (curr x batch).  lda=prev ✓
//
//   Backward gradient:  dW = A_prev * dZ^T   →  shape (prev x curr)
//     A_prev: (prev x batch) OP_N,  dZ: (curr x batch) OP_T
//     Result: (prev x curr), ldc=prev  ← SAME shape as W ✓
//
//   Gradient back-prop:  dA_prev = W * dZ    →  shape (prev x batch)
//     W: (prev x curr) OP_N,  dZ: (curr x batch) OP_N
//     Result: (prev x batch), ldc=prev  ✓
//
// ACTIVATIONS / DELTAS for layer l:
//   Shape  (curr x batch), column-major, lda = curr.
//   Element at neuron n, sample s:  ptr[ n + s*curr ]
//   Block pointer:  values + value_offsets[l] * batch
//
// EXTERNAL TARGET BUFFER (d_target in backward_prop):
//   Shape  (out_size x batch), same layout as activations.
//   Packed by main.cu as contiguous per-sample blocks, which equals this
//   layout when lda = out_size.
//
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t _s = (call);                                             \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n",                        \
                    __FILE__, __LINE__, (int)_s);                               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Struct
// ---------------------------------------------------------------------------
typedef struct {
    unsigned int size;
    unsigned int value_count;    // neurons per sample = sum(structure[])
    unsigned int weight_count;
    unsigned int bias_count;

    unsigned int* structure;      // [size]  neurons per layer
    unsigned int* value_offsets;  // [size]  cumulative neuron count (layer 0 = 0)
    unsigned int* weight_offsets; // [size]  cumulative weight count (layer 0 = 0)
    unsigned int* bias_offsets;   // [size]  cumulative bias count   (layer 0 = 0)

    float* weights;   // device  [weight_count]  all layers packed; layer l at weight_offsets[l]
    float* bias;      // device  [bias_count]

    // Per-layer blocks, each of size  batch * value_count  floats total.
    // Layer l block starts at ptr + value_offsets[l] * batch.
    float* values;    // device  post-activation
    float* zvalues;   // device  pre-activation  (kept for backprop)
    float* delta;     // device  dZ

    float* Wgrad;     // device  [weight_count]  weight gradients, same layout as weights
    float* Bgrad;     // device  [bias_count]
    float* ones;      // device  [batch]  all 1.0f  (used to row-sum dZ → Bgrad)

    float* weight_m;  // device  [weight_count]  Adam first moment
    float* weight_v;  // device  [weight_count]  Adam second moment
    float* bias_m;    // device  [bias_count]
    float* bias_v;    // device  [bias_count]

    unsigned int batch;
    unsigned int step;  // Adam timestep, incremented each backward pass
} neural_net;

// ---------------------------------------------------------------------------
// cuBLAS handle
// ---------------------------------------------------------------------------
static cublasHandle_t nn_cublas_handle;
static const float    nn_alpha = 1.0f;
static const float    nn_beta = 0.0f;

static inline void gpu_init() { CUBLAS_CHECK(cublasCreate(&nn_cublas_handle)); }
static inline void gpu_destroy() { cublasDestroy(nn_cublas_handle); }

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
    if (!net) return;
    free(net->structure);
    free(net->value_offsets);
    free(net->weight_offsets);
    free(net->bias_offsets);
    cudaFree(net->weights);  cudaFree(net->bias);
    cudaFree(net->values);   cudaFree(net->zvalues); cudaFree(net->delta);
    cudaFree(net->Wgrad);    cudaFree(net->Bgrad);   cudaFree(net->ones);
    cudaFree(net->weight_m); cudaFree(net->weight_v);
    cudaFree(net->bias_m);   cudaFree(net->bias_v);
    memset(net, 0, sizeof(*net));
}

static void neural_net_init(neural_net* a, unsigned int len,
    unsigned int structure[], unsigned int batch)
{
    memset(a, 0, sizeof(*a));
    a->size = len;
    a->batch = batch > 0 ? batch : 1;

    a->structure = (unsigned int*)malloc(sizeof(unsigned int) * len);
    a->value_offsets = (unsigned int*)malloc(sizeof(unsigned int) * len);
    a->weight_offsets = (unsigned int*)malloc(sizeof(unsigned int) * len);
    a->bias_offsets = (unsigned int*)malloc(sizeof(unsigned int) * len);
    if (!(a->structure && a->value_offsets && a->weight_offsets && a->bias_offsets)) {
        fprintf(stderr, "Host alloc failed\n"); exit(1);
    }

    for (unsigned int i = 0; i < len; i++) a->structure[i] = structure[i];

    unsigned int w_acc = 0, b_acc = 0, v_acc = 0;
    for (unsigned int i = 0; i < len; i++) {
        a->value_offsets[i] = v_acc;  v_acc += structure[i];
        a->weight_offsets[i] = (i > 0) ? w_acc : 0;
        a->bias_offsets[i] = (i > 0) ? b_acc : 0;
        if (i > 0) {
            w_acc += structure[i - 1] * structure[i];
            b_acc += structure[i];
        }
    }
    a->value_count = v_acc;
    a->weight_count = w_acc;
    a->bias_count = b_acc;

    // He-leaky init for hidden layers, Glorot for output layer.
    // W[l] is (prev x curr); element order doesn't matter for random init.
    float* h_w = (float*)malloc(a->weight_count * sizeof(float));
    float* h_b = (float*)calloc(a->bias_count, sizeof(float));
    if (!h_w || !h_b) { fprintf(stderr, "OOM weights\n"); exit(1); }

    for (unsigned int l = 1; l < len; l++) {
        unsigned int prev = structure[l - 1];
        unsigned int curr = structure[l];
        unsigned int woff = a->weight_offsets[l];
        float range = (l == len - 1)
            ? sqrtf(6.0f / (float)(prev + curr))
            : sqrtf(6.0f / ((float)prev * (1.0f + NN_LEAK * NN_LEAK)));
        for (unsigned int i = 0; i < prev * curr; i++)
            h_w[woff + i] = random_uniform(range);
    }

    CUDA_CHECK(cudaMalloc((void**)&a->weights, a->weight_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&a->bias, a->bias_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&a->values, a->value_count * a->batch * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&a->zvalues, a->value_count * a->batch * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&a->delta, a->value_count * a->batch * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&a->Wgrad, a->weight_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&a->Bgrad, a->bias_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&a->ones, a->batch * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&a->weight_m, a->weight_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&a->weight_v, a->weight_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&a->bias_m, a->bias_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&a->bias_v, a->bias_count * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(a->weights, h_w, a->weight_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(a->bias, h_b, a->bias_count * sizeof(float), cudaMemcpyHostToDevice));

    float* h_ones = (float*)malloc(a->batch * sizeof(float));
    for (unsigned int i = 0; i < a->batch; i++) h_ones[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(a->ones, h_ones, a->batch * sizeof(float), cudaMemcpyHostToDevice));
    free(h_ones);

    CUDA_CHECK(cudaMemset(a->weight_m, 0, a->weight_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(a->weight_v, 0, a->weight_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(a->bias_m, 0, a->bias_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(a->bias_v, 0, a->bias_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(a->values, 0, a->value_count * a->batch * sizeof(float)));
    CUDA_CHECK(cudaMemset(a->zvalues, 0, a->value_count * a->batch * sizeof(float)));
    CUDA_CHECK(cudaMemset(a->delta, 0, a->value_count * a->batch * sizeof(float)));
    CUDA_CHECK(cudaMemset(a->Wgrad, 0, a->weight_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(a->Bgrad, 0, a->bias_count * sizeof(float)));

    free(h_w); free(h_b);
    a->step = 0;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Add bias and apply activation.
// ptr layout: (curr x batch) col-major, element [n,s] = ptr[n + s*curr].
// neuron index = idx % curr, which correctly picks the right bias.
__global__ void k_bias_activate(
    float* __restrict__       out,   // (curr x batch) post-activation
    float* __restrict__       z,     // (curr x batch) pre-activation — written back
    const float* __restrict__ bias,  // [curr]
    unsigned int curr,
    unsigned int batch,
    int is_output)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= curr * batch) return;

    unsigned int n = idx % curr;   // neuron index within layer
    float v = z[idx] + bias[n];
    z[idx] = v;  // store pre-activation for backprop

    if (is_output) {
        out[idx] = v;  // linear — caller clamps to [0,1] for display only
    }
    else {
        out[idx] = v > 0.0f ? v : NN_LEAK * v;  // leaky ReLU
    }
}

// Output-layer delta for linear output: dZ = A - Y.
// All three arrays are (out_size x batch) col-major, lda=out_size, so idx indexes them identically.
__global__ void k_output_delta(
    float* __restrict__       dZ,       // (out_size x batch)
    const float* __restrict__ A,        // (out_size x batch)
    const float* __restrict__ Y,        // (out_size x batch)
    unsigned int out_size,
    unsigned int batch)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size * batch) return;
    dZ[idx] = A[idx] - Y[idx];
}

// Multiply dZ in-place by leaky-ReLU derivative.
__global__ void k_lrelu_deriv(
    float* __restrict__       dZ,  // (curr x batch) — modified in-place
    const float* __restrict__ z,   // (curr x batch) pre-activations
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dZ[i] *= (z[i] > 0.0f) ? 1.0f : NN_LEAK;
}

__global__ void k_adam_update(
    float* __restrict__       param,
    float* __restrict__       m,
    float* __restrict__       v,
    const float* __restrict__ grad,
    unsigned int n,
    float adam_lr)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = grad[i];
    float mi = NN_ADAM_BETA1 * m[i] + (1.0f - NN_ADAM_BETA1) * g;
    float vi = NN_ADAM_BETA2 * v[i] + (1.0f - NN_ADAM_BETA2) * g * g;
    m[i] = mi; v[i] = vi;
    param[i] -= adam_lr * mi / (sqrtf(vi) + NN_ADAM_EPS);
}

// Warp-reduce MSE.  values and target must have identical layout.
__global__ void k_mse(
    const float* __restrict__ values,
    const float* __restrict__ target,
    float* __restrict__       out,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float local = 0.0f;
    if (i < n) { float d = target[i] - values[i]; local = d * d; }
    for (int off = 16; off > 0; off >>= 1)
        local += __shfl_down_sync(0xffffffff, local, off);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, local);
}

// ---------------------------------------------------------------------------
// Forward propagation
//
// input : host buffer  (input_size x batch_size) col-major, lda=input_size.
//         sample s starts at  input + s * input_size.
// batch_size : actual samples this call; must be <= a->batch.
// ---------------------------------------------------------------------------
static void forward_prop_batch(neural_net* a, const float* input, int batch_size)
{
    unsigned int L = a->size;
    unsigned int b = ((unsigned int)batch_size > 0 && (unsigned int)batch_size <= a->batch)
        ? (unsigned int)batch_size : a->batch;

    unsigned int in_sz = a->structure[0];
    // Upload: layer-0 block is at values + value_offsets[0]*batch = values + 0 = values.
    CUDA_CHECK(cudaMemcpy(
        a->values,   // value_offsets[0] == 0
        input,
        (size_t)b * in_sz * sizeof(float),
        cudaMemcpyHostToDevice));

    for (unsigned int l = 1; l < L; l++) {
        int prev = (int)a->structure[l - 1];
        int curr = (int)a->structure[l];

        // W[l]: (prev x curr) col-major, lda=prev.
        float* W = a->weights + a->weight_offsets[l];
        // A_prev: (prev x b) col-major, lda=prev.
        float* Ap = a->values + a->value_offsets[l - 1] * a->batch;
        // z[l]:   (curr x b) col-major, lda=curr.
        float* z = a->zvalues + a->value_offsets[l] * a->batch;
        float* act = a->values + a->value_offsets[l] * a->batch;
        float* B = a->bias + a->bias_offsets[l];

        // z = W^T * A_prev
        // W  is (prev x curr), OP_T → treated as (curr x prev), lda=prev.
        // Ap is (prev x b),    OP_N, lda=prev.
        // z  is (curr x b),         ldc=curr.
        // Result shape: (curr x b) ✓
        CUBLAS_CHECK(cublasGemmEx(
            nn_cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            curr, (int)b, prev,
            &nn_alpha,
            W, CUDA_R_32F, prev,   // lda=prev for (prev x curr) matrix
            Ap, CUDA_R_32F, prev,   // ldb=prev for (prev x b)    matrix
            &nn_beta,
            z, CUDA_R_32F, curr,   // ldc=curr for (curr x b)    result
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        int total = curr * (int)b;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        k_bias_activate << <blocks, threads >> > (act, z, B, (unsigned int)curr, b, l == L - 1);
        CUDA_CHECK(cudaGetLastError());
    }
}

static void forward_prop(neural_net* a, const float* input) {
    forward_prop_batch(a, input, (int)a->batch);
}

// ---------------------------------------------------------------------------
// Backward propagation + Adam
//
// d_target : device buffer  (out_size x batch) col-major, lda=out_size.
//            element [n,s] = d_target[ n + s*out_size ].
// Returns mean squared error.
// ---------------------------------------------------------------------------
static float backward_prop(neural_net* net, const float* d_target, float lr)
{
    unsigned int L = net->size;
    unsigned int batch = net->batch;
    net->step++;

    float fix1 = 1.0f - powf(NN_ADAM_BETA1, (float)net->step);
    float fix2 = 1.0f - powf(NN_ADAM_BETA2, (float)net->step);
    float adam_lr = lr * sqrtf(fix2) / fix1;

    const int threads = 256;

    // ------------------------------------------------------------------
    // 1) Output delta:  dZ = (A - Y) * sigmoid'(A)
    // ------------------------------------------------------------------
    {
        unsigned int out_sz = net->structure[L - 1];
        float* A = net->values + net->value_offsets[L - 1] * batch;
        float* dZ = net->delta + net->value_offsets[L - 1] * batch;
        int total = (int)(out_sz * batch);
        int blocks = (total + threads - 1) / threads;
        k_output_delta << <blocks, threads >> > (dZ, A, d_target, out_sz, batch);
        CUDA_CHECK(cudaGetLastError());
    }

    // ------------------------------------------------------------------
    // 2) Per-layer: grad, Adam update, propagate dZ
    // ------------------------------------------------------------------
    for (int l = (int)L - 1; l >= 1; l--) {
        int curr = (int)net->structure[l];
        int prev = (int)net->structure[l - 1];

        // dZ_curr: (curr x batch) col-major, lda=curr
        float* dZ_c = net->delta + net->value_offsets[l] * batch;
        // A_prev:  (prev x batch) col-major, lda=prev
        float* Ap = net->values + net->value_offsets[l - 1] * batch;
        // W[l]:    (prev x curr) col-major, lda=prev
        float* W = net->weights + net->weight_offsets[l];
        float* Wg = net->Wgrad + net->weight_offsets[l];
        float* Bg = net->Bgrad + net->bias_offsets[l];

        // Wgrad = A_prev * dZ_curr^T  →  (prev x curr), lda=prev
        // A_prev:  (prev x batch) OP_N,  lda=prev
        // dZ_curr: (curr x batch) OP_T → (batch x curr), lda=curr
        // Result:  (prev x curr),  ldc=prev   ← same shape/layout as W ✓
        CUBLAS_CHECK(cublasGemmEx(
            nn_cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            prev, curr, (int)batch,
            &nn_alpha,
            Ap, CUDA_R_32F, prev,   // lda=prev
            dZ_c, CUDA_R_32F, curr,   // lda=curr (transposed)
            &nn_beta,
            Wg, CUDA_R_32F, prev,   // ldc=prev  ← matches W's lda
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // Bgrad = dZ_curr * ones  →  (curr x 1)
        // dZ_curr: (curr x batch), ones: (batch x 1)
        CUBLAS_CHECK(cublasGemmEx(
            nn_cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            curr, 1, (int)batch,
            &nn_alpha,
            dZ_c, CUDA_R_32F, curr,
            net->ones, CUDA_R_32F, (int)batch,
            &nn_beta,
            Bg, CUDA_R_32F, curr,
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // Average over batch
        float invB = 1.0f / (float)batch;
        int   wcount = prev * curr;
        CUBLAS_CHECK(cublasSscal(nn_cublas_handle, wcount, &invB, Wg, 1));
        CUBLAS_CHECK(cublasSscal(nn_cublas_handle, curr, &invB, Bg, 1));

        // Adam — Wgrad and W are both (prev x curr) so element k matches element k ✓
        int wblocks = (wcount + threads - 1) / threads;
        k_adam_update << <wblocks, threads >> > (
            W,
            net->weight_m + net->weight_offsets[l],
            net->weight_v + net->weight_offsets[l],
            Wg, (unsigned int)wcount, adam_lr);

        int bblocks = (curr + threads - 1) / threads;
        k_adam_update << <bblocks, threads >> > (
            net->bias + net->bias_offsets[l],
            net->bias_m + net->bias_offsets[l],
            net->bias_v + net->bias_offsets[l],
            Bg, (unsigned int)curr, adam_lr);

        CUDA_CHECK(cudaGetLastError());

        // Propagate dZ to previous layer (not needed for layer 1 → layer 0 is input)
        if (l > 1) {
            float* dZ_p = net->delta + net->value_offsets[l - 1] * batch;  // (prev x batch)

            // dZ_prev = W * dZ_curr
            // W:       (prev x curr) OP_N, lda=prev
            // dZ_curr: (curr x batch) OP_N, lda=curr
            // Result:  (prev x batch), ldc=prev  ✓
            CUBLAS_CHECK(cublasGemmEx(
                nn_cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                prev, (int)batch, curr,
                &nn_alpha,
                W, CUDA_R_32F, prev,   // lda=prev
                dZ_c, CUDA_R_32F, curr,   // lda=curr
                &nn_beta,
                dZ_p, CUDA_R_32F, prev,   // ldc=prev
                CUBLAS_COMPUTE_32F_FAST_TF32,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));

            int total = prev * (int)batch;
            int blocks = (total + threads - 1) / threads;
            k_lrelu_deriv << <blocks, threads >> > (
                dZ_p,
                net->zvalues + net->value_offsets[l - 1] * batch,
                (unsigned int)total);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // ------------------------------------------------------------------
    // 3) MSE loss
    // ------------------------------------------------------------------
    float* d_mse;
    CUDA_CHECK(cudaMalloc(&d_mse, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_mse, 0, sizeof(float)));

    unsigned int out_sz = net->structure[L - 1];
    int total = (int)(out_sz * batch);
    int blocks = (total + threads - 1) / threads;
    k_mse << <blocks, threads >> > (
        net->values + net->value_offsets[L - 1] * batch,
        d_target, d_mse, (unsigned int)total);
    CUDA_CHECK(cudaGetLastError());

    float h_mse = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_mse, d_mse, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_mse);
    return h_mse / (float)total;
}

// ---------------------------------------------------------------------------
// get_layer — returns a host copy of sample 0's activations at layer n.
// Caller must free() the result.
// ---------------------------------------------------------------------------
static float* get_layer(neural_net* a, unsigned int n) {
    unsigned int sz = a->structure[n];
    float* ret = (float*)malloc(sz * sizeof(float));
    if (!ret) { fprintf(stderr, "get_layer OOM\n"); exit(1); }
    // Layer n block starts at value_offsets[n]*batch.
    // Sample 0 is the first sz floats (col-major, lda=sz, sample index 0).
    CUDA_CHECK(cudaMemcpy(ret,
        a->values + a->value_offsets[n] * a->batch,
        sz * sizeof(float),
        cudaMemcpyDeviceToHost));
    return ret;
}
