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
#include "puffer_drone_swarm_weights.h"   // extern const unsigned char drone_weights[]; extern const size_t drone_weights_len;
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
    w->data = (const float *)(const void *)src_puffer_drone_swarm_weights_bin;
    // If you also have a length in bytes, prefer that:
    // w->size = (int)(drone_weights_len / sizeof(float));
    w->size = 16420/4;
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

float randn(float mean, float std) {
    static int has_spare = 0;
    static float spare;

    if (has_spare) {
        has_spare = 0;
        return mean + std * spare;
    }

    has_spare = 1;
    float u, v, s;
    do {
        u = 2.0f * rand() / RAND_MAX - 1.0f;
        v = 2.0f * rand() / RAND_MAX - 1.0f;
        s = u * u + v * v;
    } while (s >= 1.0f || s == 0.0f);

    s = sqrtf(-2.0f * logf(s) / s);
    spare = v * s;
    return mean + std * (u * s);
}


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

LinearContLSTM *make_linearcontlstm(Weights *weights, int num_agents, int input_dim,
                                    const int logit_sizes[], int num_actions) {
    LinearContLSTM *net = pvPortCalloc(1, sizeof(LinearContLSTM));
    net->num_agents = num_agents;
    net->obs = pvPortCalloc(num_agents * input_dim, sizeof(float));
    net->num_actions = logit_sizes[0];
    net->log_std = weights->data;
    weights->idx += net->num_actions;
    net->encoder = make_linear(weights, num_agents, input_dim, 128);
    net->gelu1 = make_gelu(num_agents, 128);
    int atn_sum = 0;
    for (int i = 0; i < num_actions; i++) {
        atn_sum += logit_sizes[i];
    }
    net->actor = make_linear(weights, num_agents, 128, atn_sum);
    net->value_fn = make_linear(weights, num_agents, 128, 1);
    //net->lstm = make_lstm(weights, num_agents, 128, 128);
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

//static void generate_dummy_actions(float* actions) {
//    for (int i = 0; i < 4; i++) {
//      actions[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
//    }
//}

void controllerOutOfTreeInit(void) {
    buffer_ = (int*)pvPortCalloc(1, sizeof(int));
    if (buffer_) buffer_[0] = 10;

    if (!g_weights) {
        g_weights = &g_weights_storage;
        init_weights(g_weights);
    }

    // Define your action layout for the actor head: e.g., 4 independent Gaussians
    static const int logit_sizes[1] = {4};
    const int num_actions = 1;

    net = make_linearcontlstm(g_weights, /*num_agents*/1, /*input_dim*/26,
                              logit_sizes, num_actions);
}

void forward_linearcontlstm(LinearContLSTM *net, float *observations, float *actions) {
    linear(net->encoder, observations);
    gelu(net->gelu1, net->encoder->output);
    //lstm(net->lstm, net->gelu1->output);
    linear(net->actor, net->gelu1->output);
    //linear(net->value_fn, net->lstm->state_h);
    for (int i = 0; i < net->num_actions; i++) {
        float std = expf(net->log_std[i]);
        float mean = net->actor->output[i];
        actions[i] = randn(mean, std);
    }
}

bool controllerOutOfTreeTest(void) {
  return true;
}

float clampf(float val, float min, float max) {
    if (val < min) return min;
    if (val > max) return max;
    return val;
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
  //generate_dummy_actions(a);

  // Build observation vector
  net->obs[0] = state->velocity.x / 50.0f;
  net->obs[1] = state->velocity.y / 50.0f;
  net->obs[2] = state->velocity.z / 50.0f;

  net->obs[3] = sensors->gyro.x / 50.0f / 180.0f * 3.14159265f;
  net->obs[4] = sensors->gyro.y / 50.0f / 180.0f * 3.14159265f;
  net->obs[5] = sensors->gyro.z / 50.0f / 180.0f * 3.14159265f;

  net->obs[6] = state->attitudeQuaternion.w;
  net->obs[7] = state->attitudeQuaternion.x;
  net->obs[8] = state->attitudeQuaternion.y;
  net->obs[9] = state->attitudeQuaternion.z;

  net->obs[10] = control->normalizedForces[0];
  net->obs[11] = control->normalizedForces[1];
  net->obs[12] = control->normalizedForces[2];
  net->obs[13] = control->normalizedForces[3];
  
  net->obs[14] = (setpoint->position.x - state->position.x) / 30.0f;
  net->obs[15] = (setpoint->position.y - state->position.y) / 30.0f;
  net->obs[16] = (setpoint->position.z - state->position.z) / 10.0f;

  net->obs[17] = clampf(setpoint->position.x - state->position.x, -1.0f, 1.0f);
  net->obs[18] = clampf(setpoint->position.y - state->position.y, -1.0f, 1.0f);
  net->obs[19] = clampf(setpoint->position.z - state->position.z, -1.0f, 1.0f);

  net->obs[20] = 0.0f;
  net->obs[21] = 0.0f;
  net->obs[22] = 0.0f;

  net->obs[23] = 0.0f;
  net->obs[24] = 0.0f;
  net->obs[25] = 0.0f;

  forward_linearcontlstm(net, net->obs, a);

  a[0] = clampf(a[0], -1.0f, 1.0f);
  a[1] = clampf(a[1], -1.0f, 1.0f);
  a[2] = clampf(a[2], -1.0f, 1.0f);
  a[3] = clampf(a[3], -1.0f, 1.0f);

  control->controlMode = controlModeForce;
  control->normalizedForces[0] = (a[0] + 1.0f) * 0.5f;
  control->normalizedForces[1] = (a[1] + 1.0f) * 0.5f;
  control->normalizedForces[2] = (a[2] + 1.0f) * 0.5f;
  control->normalizedForces[3] = (a[3] + 1.0f) * 0.5f;

  //(actions[i] + 1.0f) * 0.5f;
}