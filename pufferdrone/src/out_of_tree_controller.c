/**
 * ,---------,       ____  _ __
 * |  ,-^-,  |      / __ )(_) /_______________ _____  ___
 * | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
 * | / ,--´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
 *    +------`   /_____/_/\__/\___/_/   \__,_/ /___/\___/
 *
 * Crazyflie control firmware
 *
 * Copyright (C) 2024 Bitcraze AB
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, in version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 *
 * out_of_tree_controller.c - App layer application of an out of tree controller.
 */

#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "app.h"

#include "FreeRTOS.h"
#include "task.h"

// Edit the debug name to get nice debug prints
#define DEBUG_MODULE "MYCONTROLLER"
#include "debug.h"


// We still need an appMain() function, but we will not really use it. Just let it quietly sleep.
void appMain() {
  DEBUG_PRINT("Waiting for activation ...\n");

  while(1) {
    vTaskDelay(M2T(2000));
  }
}

// The new controller goes here --------------------------------------------
// Move the includes to the the top of the file if you want to
#include "controller.h"

#include <stdlib.h>

#include "puffernet.h"
#include "drone_weights_blob.h"

float a[4];

void generate_dummy_actions(float* actions) {
    // Generate random floats in [-1, 1] range
    actions[0] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    actions[1] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    actions[2] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    actions[3] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

void controllerOutOfTreeInit() {}

bool controllerOutOfTreeTest() {
  return true;
}

void controllerOutOfTree(control_t *control, const setpoint_t *setpoint, const sensorData_t *sensors, const state_t *state, const uint32_t tick) {
  //forward_linearcontlstm(net, env->observations, env->actions);
  generate_dummy_actions(a);

  // Default all motors to 0
  control->controlMode = controlModeForce;
  control->normalizedForces[0] = a[0]/3;
  control->normalizedForces[1] = a[1]/3;
  control->normalizedForces[2] = a[2]/3;
  control->normalizedForces[3] = a[3]/3;
}