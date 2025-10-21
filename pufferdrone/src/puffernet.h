#pragma once
#include <stddef.h>

typedef struct Weights {
    const float* data;
    int size;
    int idx;
} Weights;

// --- kernels ---
void _relu(const float* input, float* output, int size);
void _gelu(const float* input, float* output, int size);
void _linear(const float* input, const float* weights, const float* bias, float* output,
             int batch_size, int input_dim, int output_dim);
void _linear_accumulate(const float* input, const float* weights, const float* bias, float* output,
                        int batch_size, int input_dim, int output_dim);
void _conv2d(const float* input, const float* weights, const float* bias, float* output,
             int batch_size, int in_width, int in_height, int in_channels,
             int out_channels, int kernel_size, int stride);
void _conv3d(const float* input, const float* weights, const float* bias, float* output,
             int batch_size, int in_width, int in_height, int in_depth, int in_channels,
             int out_channels, int kernel_size, int stride);
void _lstm(const float* input, float* state_h, float* state_c,
           const float* weights_input, const float* weights_state,
           const float* bias_input,   const float* bias_state,
           float* buffer, int batch_size, int input_size, int hidden_size);
void _embedding(const int* input, const float* weights, float* output,
                int batch_size, int num_embeddings, int embedding_dim);
void _layernorm(const float* input, const float* weights, const float* bias, float* output,
                int batch_size, int input_dim);
void _one_hot(const int* input, int* output, int batch_size, int input_size, int num_classes);
void _cat_dim1(const float* x, const float* y, float* output, int batch_size, int x_size, int y_size);
void _argmax_multidiscrete(const float* input, int* output, int batch_size,
                           const int logit_sizes[], int num_actions);
void _softmax_multidiscrete(const float* input, int* output, int batch_size,
                            const int logit_sizes[], int num_actions);
void _max_dim1(const float* input, float* output, int batch_size, int seq_len, int feature_dim);

// --- layers & wrappers ---
typedef struct Linear {
    float*       output;
    const float* weights;
    const float* bias;
    int batch_size, input_dim, output_dim;
} Linear;

Linear* make_linear(Weights* weights, int batch_size, int input_dim, int output_dim);
void    linear(Linear* layer, const float* input);
void    linear_accumulate(Linear* layer, const float* input);

typedef struct ReLU { float* output; int batch_size, input_dim; } ReLU;
ReLU* make_relu(int batch_size, int input_dim);
void  relu(ReLU* layer, const float* input);

typedef struct GELU { float* output; int batch_size, input_dim; } GELU;
GELU* make_gelu(int batch_size, int input_dim);
void  gelu(GELU* layer, const float* input);

typedef struct MaxDim1 { float* output; int batch_size, seq_len, feature_dim; } MaxDim1;
MaxDim1* make_max_dim1(int batch_size, int seq_len, int feature_dim);
void      max_dim1(MaxDim1* layer, const float* input);

typedef struct Conv2D {
    float*       output;
    const float* weights;
    const float* bias;
    int batch_size, in_width, in_height, in_channels, out_channels, kernel_size, stride;
} Conv2D;
Conv2D* make_conv2d(Weights* weights, int batch_size, int in_width, int in_height,
                    int in_channels, int out_channels, int kernel_size, int stride);
void    conv2d(Conv2D* layer, const float* input);

typedef struct Conv3D {
    float*       output;
    const float* weights;
    const float* bias;
    int batch_size, in_width, in_height, in_depth, in_channels, out_channels, kernel_size, stride;
} Conv3D;
Conv3D* make_conv3d(Weights* weights, int batch_size, int in_width, int in_height, int in_depth,
                    int in_channels, int out_channels, int kernel_size, int stride);
void    conv3d(Conv3D* layer, const float* input);

typedef struct LSTM {
    float*       state_h;
    float*       state_c;
    const float* weights_input;
    const float* weights_state;
    const float* bias_input;
    const float* bias_state;
    float*       buffer;
    int batch_size, input_size, hidden_size;
} LSTM;
LSTM* make_lstm(Weights* weights, int batch_size, int input_size, int hidden_size);
void  lstm(LSTM* layer, const float* input);

typedef struct Embedding {
    float*       output;
    const float* weights;
    int batch_size, num_embeddings, embedding_dim;
} Embedding;
Embedding* make_embedding(Weights* weights, int batch_size, int num_embeddings, int embedding_dim);
void       embedding(Embedding* layer, const int* input);

typedef struct LayerNorm {
    float*       output;
    const float* weights;
    const float* bias;
    int batch_size, input_dim;
} LayerNorm;
LayerNorm* make_layernorm(Weights* weights, int batch_size, int input_dim);
void       layernorm(LayerNorm* layer, const float* input);

typedef struct OneHot { int* output; int batch_size, input_size, num_classes; } OneHot;
OneHot* make_one_hot(int batch_size, int input_size, int num_classes);
void    one_hot(OneHot* layer, const int* input);

typedef struct CatDim1 { float* output; int batch_size, x_size, y_size; } CatDim1;
CatDim1* make_cat_dim1(int batch_size, int x_size, int y_size);
void     cat_dim1(CatDim1* layer, const float* x, const float* y);

// Utility
const float* get_weights(Weights* weights, int num_weights);
