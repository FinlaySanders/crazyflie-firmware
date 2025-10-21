/**
 * out_of_tree_controller.c - App layer application of an out of tree controller.
 */
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>

#include "app.h"
#include "FreeRTOS.h"
#include "task.h"
#define DEBUG_MODULE "MYCONTROLLER"
#include "debug.h"

#include "controller.h"
#include "drone_weights_blob.h"   // extern const unsigned char drone_weights[]; extern const size_t drone_weights_len;
#include "portable.h"

// If your port lacks pvPortCalloc, provide it HERE (and remove from puffernet.c), or vice versa.
// (Keep it in ONE translation unit only.)
#ifndef pvPortCalloc
static inline void *pvPortCalloc(size_t n, size_t size) {
    size_t bytes = n * size;
    void *p = pvPortMalloc(bytes);
    if (p) memset(p, 0, bytes);
    return p;
}
#endif

// *** Do NOT include .c files here ***
// Use a header that declares the API instead:
#include "puffernet.h"            // declares: Weights, make_*(), forward_*(), etc.

// ---- Weights handling -------------------------------------------------------

static inline void init_weights(Weights* w) {
    // Assume drone_weights is a contiguous float blob in little-endian IEEE754.
    // Cast to const float*; size is in number of floats.
    w->data = (const float *)(const void *)drone_weights;
    // If you also have a length in bytes, prefer that:
    // w->size = (int)(drone_weights_len / sizeof(float));
    w->size = (int)(sizeof(drone_weights) / sizeof(float));
    w->idx  = 0;
}

static Weights g_weights_storage;
static Weights* g_weights = NULL;

// Safe float read helper (not strictly needed once we cast above)
static inline float weight_at(const Weights* w, size_t i) {
    // Bounds are the caller’s responsibility; add asserts if you want.
    return w->data[i];
}

// ---- Small RNG helper -------------------------------------------------------

// ---- This model wraps PufferNet pieces --------------------------------------

typedef struct LinearContLSTM {
    int num_agents;
    float *obs;
    const float *log_std;  // read-only params live in flash
    Linear *encoder;
    GELU *gelu1;
    LSTM *lstm;
    Linear *actor;
    Linear *value_fn;
    int num_actions;       // number of continuous actions we sample
} LinearContLSTM;

static LinearContLSTM *make_linearcontlstm(Weights *weights,
                                           int num_agents, int input_dim,
                                           const int logit_sizes[], int num_actions)
{
    LinearContLSTM *net = pvPortCalloc(1, sizeof(LinearContLSTM));
    net->num_agents = num_agents;
    net->obs = pvPortCalloc((size_t)num_agents * (size_t)input_dim, sizeof(float));

    // First 'num_actions' entries are log_std (design choice from your code)
    net->num_actions = num_actions;                 // use the argument, not logit_sizes[0]
    net->log_std = get_weights(weights, num_actions);

    // Feature extractor
    net->encoder = make_linear(weights, num_agents, input_dim, 128);
    net->gelu1   = make_gelu(num_agents, 128);
    net->lstm    = make_lstm(weights, num_agents, 128, 128);

    // Policy/value heads
    int atn_sum = 0;
    for (int i = 0; i < num_actions; i++) atn_sum += logit_sizes[i];
    net->actor   = make_linear(weights, num_agents, 128, atn_sum);
    net->value_fn= make_linear(weights, num_agents, 128, 1);
    return net;
}

// ---- App scaffolding --------------------------------------------------------

static int *buffer_;
static float a[4];
static LinearContLSTM *net = NULL;

void appMain(void) {
  DEBUG_PRINT("Waiting for activation ...\n");
  while (1) {
    vTaskDelay(M2T(2000));
  }
}

static void generate_dummy_actions(float* actions) {
    for (int i = 0; i < 4; i++) {
      actions[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
}

void controllerOutOfTreeInit(void) {
    buffer_ = (int*)pvPortCalloc(1, sizeof(int));
    if (buffer_) buffer_[0] = 10;

    if (!g_weights) {
        g_weights = &g_weights_storage;
        init_weights(g_weights);
    }

    // Define your action layout for the actor head: e.g., 4 independent Gaussians
    static const int logit_sizes[] = { 1, 1, 1, 1 }; // 4 dims -> atn_sum == 4
    const int num_actions = (int)(sizeof(logit_sizes)/sizeof(logit_sizes[0]));

    net = make_linearcontlstm(g_weights, /*num_agents*/1, /*input_dim*/25,
                              logit_sizes, num_actions);
}

bool controllerOutOfTreeTest(void) {
  return true;
}

void controllerOutOfTree(control_t *control, const setpoint_t *setpoint,
                         const sensorData_t *sensors, const state_t *state,
                         const uint32_t tick) {
  if (!g_weights) {
      g_weights = &g_weights_storage;
      init_weights(g_weights);
  }

  if (!net) {
      DEBUG_PRINT("Error: controllerOutOfTree called before init!\n");
      return;
  }

  // for now, use dummy actions (or build obs and call forward_*)
  generate_dummy_actions(a);

  control->controlMode = controlModeForce;
  control->normalizedForces[0] = a[0] / 3.0f + weight_at(g_weights, 0);
  control->normalizedForces[1] = a[1] / 3.0f + weight_at(g_weights, 100);
  control->normalizedForces[2] = a[2] / 3.0f + weight_at(g_weights, 200);
  control->normalizedForces[3] = a[3] / 3.0f + weight_at(g_weights, 1000);
}
