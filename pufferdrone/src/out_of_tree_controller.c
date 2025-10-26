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
#include "puffer_drone_swarm_weights.h"   // src_puffer_drone_swarm_weights_bin, _len
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
    // Binary blob is a contiguous float buffer (IEEE754, little-endian)
    w->data = (const float *)(const void *)src_puffer_drone_swarm_weights_bin;
    w->size = (int)(src_puffer_drone_swarm_weights_bin_len / sizeof(float));
    w->idx  = 0;
}

static Weights g_weights_storage;
static Weights* g_weights = NULL;

// ---- Small helpers -----------------------------------------------------------

typedef struct { float w,x,y,z; } Quat;
typedef struct { float x,y,z;   } Vec3;

static inline float clampf(float val, float min, float max) {
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

// Normalize (w,x,y,z) with guard + early exit
static inline void quat_normalize_safe(float *w, float *x, float *y, float *z) {
    const float eps = 1e-12f, tol = 1e-6f;
    float n2 = (*w)*(*w) + (*x)*(*x) + (*y)*(*y) + (*z)*(*z);
    if (n2 < eps) { *w = 1.0f; *x = *y = *z = 0.0f; return; }
    if (fabsf(n2 - 1.0f) < tol) return;
    float invn = 1.0f / sqrtf(n2);
    *w *= invn; *x *= invn; *y *= invn; *z *= invn;
}

// Rotate a world-frame vector into body frame using q (body→world)
static inline Vec3 world_to_body_vec(Quat q_body_to_world, Vec3 v_world) {
    float qw = q_body_to_world.w;
    float qx = q_body_to_world.x;
    float qy = q_body_to_world.y;
    float qz = q_body_to_world.z;
    quat_normalize_safe(&qw,&qx,&qy,&qz);

    // q^{-1} for world→body
    float w =  qw, x = -qx, y = -qy, z = -qz;

    // u = 2 * (q_vec × v)
    float ux = 2.0f*(y*v_world.z - z*v_world.y);
    float uy = 2.0f*(z*v_world.x - x*v_world.z);
    float uz = 2.0f*(x*v_world.y - y*v_world.x);

    // v_body = v_world + w*u + q_vec × u
    Vec3 v_body;
    v_body.x = v_world.x + w*ux + (y*uz - z*uy);
    v_body.y = v_world.y + w*uy + (z*ux - x*uz);
    v_body.z = v_world.z + w*uz + (x*uy - y*ux);
    return v_body;
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
    net->obs = pvPortCalloc((size_t)num_agents * (size_t)input_dim, sizeof(float));
    net->num_actions = logit_sizes[0];

    // weights layout: [log_std(4)] [Linear(26->128)] [Actor(128->4)] [Value(128->1)]
    net->log_std = weights->data;
    weights->idx += net->num_actions;

    net->encoder = make_linear(weights, num_agents, input_dim, 128);
    net->gelu1 = make_gelu(num_agents, 128);

    int atn_sum = 0;
    for (int i = 0; i < num_actions; i++) atn_sum += logit_sizes[i];

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

void controllerOutOfTreeInit(void) {
    buffer_ = (int*)pvPortCalloc(1, sizeof(int));
    if (buffer_) buffer_[0] = 10;

    if (!g_weights) {
        g_weights = &g_weights_storage;
        init_weights(g_weights);
    }

    // Define your action layout for the actor head: e.g., 4 independent outputs
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

    // Minimal change: use deterministic mean (no sampling)
    for (int i = 0; i < net->num_actions; i++) {
        actions[i] = net->actor->output[i];
    }
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

  // ---------------- Build observation vector (match training) ----------------

  // (0..2) Linear velocity in BODY frame / 50
  Vec3 v_world = (Vec3){ state->velocity.x, state->velocity.y, state->velocity.z };
  Quat q = (Quat){ state->attitudeQuaternion.w, state->attitudeQuaternion.x,
                   state->attitudeQuaternion.y, state->attitudeQuaternion.z };
  Vec3 v_body = world_to_body_vec(q, v_world);
  net->obs[0] = v_body.x / 50.0f;
  net->obs[1] = v_body.y / 50.0f;
  net->obs[2] = v_body.z / 50.0f;

  // (3..5) Body rates: deg/s -> rad/s, then /50
  const float DEG2RAD = 3.14159265358979323846f / 180.0f;
  net->obs[3] = (sensors->gyro.x * DEG2RAD) / 50.0f;
  net->obs[4] = (sensors->gyro.y * DEG2RAD) / 50.0f;
  net->obs[5] = (sensors->gyro.z * DEG2RAD) / 50.0f;

  // (6..9) Quaternion w,x,y,z
  // (q already normalized in world_to_body_vec; safe to push as-is)
  net->obs[6] = q.w;
  net->obs[7] = q.x;
  net->obs[8] = q.y;
  net->obs[9] = q.z;

  // (10..13) Motor feedback (keep as your normalizedForces proxy, unchanged)
  net->obs[10] = control->normalizedForces[0];
  net->obs[11] = control->normalizedForces[1];
  net->obs[12] = control->normalizedForces[2];
  net->obs[13] = control->normalizedForces[3];

  // (14..16) Target deltas / (30,30,10)
  net->obs[14] = (0.0f - state->position.x) / 30.0f;
  net->obs[15] = (0.0f - state->position.y) / 30.0f;
  net->obs[16] = (0.5f - state->position.z) / 10.0f;

  // (17..19) Clamped copies
  net->obs[17] = clampf(0.0f - state->position.x, -1.0f, 1.0f);
  net->obs[18] = clampf(0.0f - state->position.y, -1.0f, 1.0f);
  net->obs[19] = clampf(0.5f - state->position.z, -1.0f, 1.0f);

  // (20..22) Task-normal (unused on hardware; keep zeros)
  net->obs[20] = 0.0f;
  net->obs[21] = 0.0f;
  net->obs[22] = 0.0f;

  // (23..25) Multi-agent features (single drone: zeros)
  net->obs[23] = 1.0f;
  net->obs[24] = 1.0f;
  net->obs[25] = 1.0f;

  // ---------------- Forward pass ----------------
  forward_linearcontlstm(net, net->obs, a);

  // Clamp actions and map to [0,1] (unchanged behavior)
  a[0] = clampf(a[0], -1.0f, 1.0f);
  a[1] = clampf(a[1], -1.0f, 1.0f);
  a[2] = clampf(a[2], -1.0f, 1.0f);
  a[3] = clampf(a[3], -1.0f, 1.0f);

  control->controlMode = controlModeForce;
  control->normalizedForces[0] = (a[0] + 1.0f) * 0.5f;
  control->normalizedForces[1] = (a[1] + 1.0f) * 0.5f;
  control->normalizedForces[2] = (a[2] + 1.0f) * 0.5f;
  control->normalizedForces[3] = (a[3] + 1.0f) * 0.5f;
}
