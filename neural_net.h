#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#define NN_PARALLEL_FOR _Pragma("omp parallel for")
#else
#define NN_PARALLEL_FOR
#endif

#define NN_LEAK 0.01f
#define NN_ADAM_BETA1 0.9f
#define NN_ADAM_BETA2 0.999f
#define NN_ADAM_EPS 1.0e-8f

typedef struct {
    unsigned int size;
    unsigned int value_count;
    unsigned int weight_count;
    unsigned int bias_count;
    unsigned int* structure;
    unsigned int* value_offsets;
    unsigned int* weight_offsets;
    unsigned int* bias_offsets;
    float* weights;
    float* bias;
    float* values;
    float* zvalues;
    float* delta;
    float* weight_m;
    float* weight_v;
    float* bias_m;
    float* bias_v;
    unsigned int step;
} neural_net;

static void neural_net_free(neural_net* net) {
    free(net->structure);
    free(net->value_offsets);
    free(net->weight_offsets);
    free(net->bias_offsets);
    free(net->weights);
    free(net->bias);
    free(net->values);
    free(net->zvalues);
    free(net->delta);
    free(net->weight_m);
    free(net->weight_v);
    free(net->bias_m);
    free(net->bias_v);
    memset(net, 0, sizeof(*net));
}

static inline float random_uniform(float range) {
    return ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * range;
}

static void neural_net_init(neural_net* a, unsigned int len, unsigned int structure[]) {
    memset(a, 0, sizeof(*a));

    a->size = len;
    a->structure = (unsigned int*)malloc(sizeof(unsigned int) * len);
    a->value_offsets = (unsigned int*)malloc(sizeof(unsigned int) * len);
    a->weight_offsets = (unsigned int*)malloc(sizeof(unsigned int) * len);
    a->bias_offsets = (unsigned int*)malloc(sizeof(unsigned int) * len);
    if (!(a->structure && a->value_offsets && a->weight_offsets && a->bias_offsets)) { exit(1); }

    for (unsigned int i = 0; i < len; i++) {
        a->structure[i] = structure[i];
    }

    unsigned int weight_acc = 0;
    unsigned int bias_acc = 0;
    unsigned int value_acc = 0;

    for (unsigned int i = 0; i < len; i++) {
        a->value_offsets[i] = value_acc;
        value_acc += structure[i];

        if (i > 0) {
            a->weight_offsets[i] = weight_acc;
            weight_acc += structure[i - 1] * structure[i];

            a->bias_offsets[i] = bias_acc;
            bias_acc += structure[i];
        }
        else {
            a->weight_offsets[i] = 0;
            a->bias_offsets[i] = 0;
        }
    }

    a->value_count = value_acc;
    a->weight_count = weight_acc;
    a->bias_count = bias_acc;

    a->weights = (float*)malloc((size_t)a->weight_count * sizeof(float));
    a->bias = (float*)malloc((size_t)a->bias_count * sizeof(float));
    a->values = (float*)malloc((size_t)a->value_count * sizeof(float));
    a->zvalues = (float*)malloc((size_t)a->value_count * sizeof(float));
    a->delta = (float*)malloc((size_t)a->value_count * sizeof(float));
    a->weight_m = (float*)calloc(a->weight_count, sizeof(float));
    a->weight_v = (float*)calloc(a->weight_count, sizeof(float));
    a->bias_m = (float*)calloc(a->bias_count, sizeof(float));
    a->bias_v = (float*)calloc(a->bias_count, sizeof(float));

    if (!(a->weights && a->bias && a->values && a->zvalues && a->delta &&
        a->weight_m && a->weight_v && a->bias_m && a->bias_v)) { exit(1); }

    for (unsigned int layer = 1; layer < a->size; layer++) {
        unsigned int fan_in = a->structure[layer - 1];
        unsigned int fan_out = a->structure[layer];
        unsigned int w_off = a->weight_offsets[layer];
        unsigned int b_off = a->bias_offsets[layer];
        unsigned int layer_weight_count = fan_in * fan_out;
        float range;

        if (layer == a->size - 1) {
            range = sqrtf(6.0f / (float)(fan_in + fan_out));
        }
        else {
            range = sqrtf(6.0f / ((float)fan_in * (1.0f + NN_LEAK * NN_LEAK)));
        }

        for (unsigned int i = 0; i < layer_weight_count; i++) {
            a->weights[w_off + i] = random_uniform(range);
        }

        for (unsigned int i = 0; i < fan_out; i++) {
            a->bias[b_off + i] = 0.0f;
        }
    }

    a->step = 0;
}

static float* get_layer(neural_net* a, unsigned int n) {
    float* ret = (float*)malloc(a->structure[n] * sizeof(float));
    if (!ret) exit(1);
    unsigned int value_offset = 0;
    for (unsigned int i = 0; i < n; i++) {
        value_offset += a->structure[i];
    }
    for (unsigned int i = 0; i < a->structure[n]; i++) {
        ret[i] = a->values[i + value_offset];
    }
    return ret;
}

static inline float activation(float x) {
    return x > 0.0f ? x : NN_LEAK * x;
}

static inline float activation_deriv(float a) {
    return a > 0.0f ? 1.0f : NN_LEAK;
}

static inline float output_activation(float x) {
    if (x >= 0.0f) {
        return 1.0f / (1.0f + expf(-x));
    }

    float ex = expf(x);
    return ex / (1.0f + ex);
}

static void forward_prop(neural_net* a, const float* input) {
    unsigned int L = a->size;

    unsigned int input_size = a->structure[0];
    for (unsigned int i = 0; i < input_size; i++) {
        a->values[i] = input[i];
        a->zvalues[i] = input[i];
    }

    for (unsigned int layer = 1; layer < L; layer++) {

        unsigned int prev = a->structure[layer - 1];
        unsigned int curr = a->structure[layer];

        float* prev_vals = &a->values[a->value_offsets[layer - 1]];
        float* curr_vals = &a->values[a->value_offsets[layer]];
        float* curr_z = &a->zvalues[a->value_offsets[layer]];

        float* W = &a->weights[a->weight_offsets[layer]];
        float* B = &a->bias[a->bias_offsets[layer]];

        NN_PARALLEL_FOR
        for (int j = 0; j < (int)curr; j++) {
            float sum = B[j];

            for (unsigned int i = 0; i < prev; i++) {
                sum += W[j * prev + i] * prev_vals[i];
            }

            curr_z[j] = sum;

            if (layer == L - 1)
                curr_vals[j] = output_activation(sum);
            else
                curr_vals[j] = activation(sum);
        }

    }
}

static float backward_prop(neural_net* net, const float* target, float lr) {
    unsigned int L = net->size;
    float* delta = net->delta;
    unsigned int* voff = net->value_offsets;
    unsigned int* woff = net->weight_offsets;
    unsigned int* boff = net->bias_offsets;
    net->step++;

    float bias_fix1 = 1.0f - powf(NN_ADAM_BETA1, (float)net->step);
    float bias_fix2 = 1.0f - powf(NN_ADAM_BETA2, (float)net->step);
    float adam_lr = lr * sqrtf(bias_fix2) / bias_fix1;

    unsigned int out = L - 1;
    unsigned int out_size = net->structure[out];
    unsigned int out_vo = voff[out];

    NN_PARALLEL_FOR
    for (int i = 0; i < (int)out_size; i++) {
        float a = net->values[out_vo + i];
        delta[out_vo + i] = (a - target[i]);
    }

    for (int layer = (int)L - 1; layer > 1; layer--) {
        unsigned int curr = (unsigned int)layer;
        unsigned int prev = curr - 1;

        unsigned int curr_size = net->structure[curr];
        unsigned int prev_size = net->structure[prev];

        unsigned int curr_vo = voff[curr];
        unsigned int prev_vo = voff[prev];
        unsigned int w_off = woff[curr];

        NN_PARALLEL_FOR
        for (int i = 0; i < (int)prev_size; i++) {
            float sum = 0.0f;

            for (unsigned int j = 0; j < curr_size; j++) {
                float w = net->weights[w_off + j * prev_size + i];
                float d = delta[curr_vo + j];
                sum += w * d;
            }

            float a_prev = net->values[prev_vo + i];
            float deriv = activation_deriv(a_prev);
            delta[prev_vo + i] = sum * deriv;
        }
    }

    for (unsigned int layer = 1; layer < L; layer++) {
        unsigned int curr_size = net->structure[layer];
        unsigned int prev_size = net->structure[layer - 1];

        unsigned int curr_vo = voff[layer];
        unsigned int prev_vo = voff[layer - 1];

        unsigned int w_off = woff[layer];
        unsigned int b_off = boff[layer];

        NN_PARALLEL_FOR
        for (int j = 0; j < (int)curr_size; j++) {
            float d = delta[curr_vo + j];

            for (unsigned int i = 0; i < prev_size; i++) {
                unsigned int wi = w_off + j * prev_size + i;
                float a_prev = net->values[prev_vo + i];
                float grad = d * a_prev;

                net->weight_m[wi] = NN_ADAM_BETA1 * net->weight_m[wi] + (1.0f - NN_ADAM_BETA1) * grad;
                net->weight_v[wi] = NN_ADAM_BETA2 * net->weight_v[wi] + (1.0f - NN_ADAM_BETA2) * grad * grad;
                net->weights[wi] -= adam_lr * net->weight_m[wi] / (sqrtf(net->weight_v[wi]) + NN_ADAM_EPS);
            }

            unsigned int bi = b_off + j;
            net->bias_m[bi] = NN_ADAM_BETA1 * net->bias_m[bi] + (1.0f - NN_ADAM_BETA1) * d;
            net->bias_v[bi] = NN_ADAM_BETA2 * net->bias_v[bi] + (1.0f - NN_ADAM_BETA2) * d * d;
            net->bias[bi] -= adam_lr * net->bias_m[bi] / (sqrtf(net->bias_v[bi]) + NN_ADAM_EPS);
        }
    }

    float error = 0.0f;
    unsigned int out_n = net->structure[net->size - 1];
    float* output_vals = &net->values[voff[net->size - 1]];

    for (unsigned int i = 0; i < out_n; i++) {
        float diff = target[i] - output_vals[i];
        error += (diff * diff) / (float)out_n;
    }

    return error;
}
