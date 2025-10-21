/**
 * ,---------,       ____  _ __
 * |  ,-^-,  |      / __ )(_) /_______________ _____  ___
 * | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
 * | / ,--´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
 *    +------`   /_____/_/\__/\___/_/   \__,_/ /___/\___/
 *
 * Crazyflie control firmware
 *
 * out_of_tree_controller.c - App layer application of an out of tree controller.
 */

#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

#include "app.h"
#include "FreeRTOS.h"
#include "task.h"

// Edit the debug name to get nice debug prints
#define DEBUG_MODULE "MYCONTROLLER"
#include "debug.h"

// The new controller goes here --------------------------------------------
#include "controller.h"
#include "drone_weights_blob.h"   // provides: extern const unsigned char drone_weights[];

// FreeRTOS memory (ensure your include paths are set)
#include "portable.h"

#include "puffernet.c"

// Provide pvPortCalloc if your port doesn't have it
#ifndef pvPortCalloc
static inline void *pvPortCalloc(size_t n, size_t size) {
    size_t bytes = n * size;
    void *p = pvPortMalloc(bytes);
    if (p) memset(p, 0, bytes);
    return p;
}
#endif

// Keep a tiny handle in RAM; read floats from Flash with memcpy (alignment/aliasing-safe)
typedef struct Weights {
    const unsigned char *bytes;  // Flash-resident raw bytes
    int size;                    // number of floats
    int idx;
} Weights;

static inline const Weights* load_weights(void) {
  static const Weights w = {
    .bytes = drone_weights,
    .size  = (int)(sizeof(drone_weights) / 4),
    .idx   = 0,
  };
  return &w;
}

// Read one float from Flash safely
static inline float weight_at(const Weights* w, size_t i) {
  float f;
  memcpy(&f, w->bytes + i * 4, 4);
  return f;
}

double randn(double mean, double std) {
    static int has_spare = 0;
    static double spare;

    if (has_spare) {
        has_spare = 0;
        return mean + std * spare;
    }

    has_spare = 1;
    double u, v, s;
    do {
        u = 2.0 * rand() / RAND_MAX - 1.0;
        v = 2.0 * rand() / RAND_MAX - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return mean + std * (u * s);
}

typedef struct LinearContLSTM LinearContLSTM;
struct LinearContLSTM {
    int num_agents;
    float *obs;
    float *log_std;
    Linear *encoder;
    GELU *gelu1;
    LSTM *lstm;
    Linear *actor;
    Linear *value_fn;
    int num_actions;
};

LinearContLSTM *make_linearcontlstm(Weights *weights, int num_agents, int input_dim,
                                    int logit_sizes[], int num_actions) {
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
    net->lstm = make_lstm(weights, num_agents, 128, 128);
    return net;
}

void forward_linearcontlstm(LinearContLSTM *net, float *observations, float *actions) {
    linear(net->encoder, observations);
    gelu(net->gelu1, net->encoder->output);
    lstm(net->lstm, net->gelu1->output);
    linear(net->actor, net->lstm->state_h);
    linear(net->value_fn, net->lstm->state_h);
    for (int i = 0; i < net->num_actions; i++) {
        float std = expf(net->log_std[i]);
        float mean = net->actor->output[i];
        actions[i] = randn(mean, std);
    }
}

// Globals
static const Weights *g_weights = NULL;
static int *buffer;
static float a[4];
LinearContLSTM *net;

// We still need an appMain() function, but we will not really use it. Just let it quietly sleep.
void appMain(void) {
  DEBUG_PRINT("Waiting for activation ...\n");
  while (1) {
    vTaskDelay(M2T(2000));
  }
}

static void generate_dummy_actions(float* actions) {
    // Generate random floats in [-1, 1] range
    for (int i = 0; i < 4; i++) {
      actions[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
}

void controllerOutOfTreeInit(void) {
    buffer = (int*)pvPortCalloc(1, sizeof(int));
    if (buffer) buffer[0] = 10;

    g_weights = load_weights();
    // Optional sanity log (safe read from Flash)
    // DEBUG_PRINT("weights[0]=%f, count=%d\n", weight_at(g_weights, 0), g_weights->size);
    net = make_linearcontlstm(weights, 1, 25, logit_sizes, 1)
}

bool controllerOutOfTreeTest(void) {
  return true;
}

void controllerOutOfTree(control_t *control, const setpoint_t *setpoint,
                         const sensorData_t *sensors, const state_t *state,
                         const uint32_t tick) {
  if (!g_weights) g_weights = load_weights();  // lazy guard

  generate_dummy_actions(a);

  control->controlMode = controlModeForce;
  control->normalizedForces[0] = a[0] / 3.0f + weight_at(g_weights, 0);
  control->normalizedForces[1] = a[1] / 3.0f + weight_at(g_weights, 100);
  control->normalizedForces[2] = a[2] / 3.0f + weight_at(g_weights, 200);
  control->normalizedForces[3] = a[3] / 3.0f + weight_at(g_weights, 1000);
}
