#include <math.h>
#include <stdio.h>
#include "ddpgc.h"


#define EPISODE_LENGTH  200
#define EPISODE_COUNT   100

#define STATE_SIZE      2
#define ACTION_SIZE     3

#define LAYER_SIZE      2

#define STEP_CONTROL    .025

double vector_largest_index(double* vector, int vector_size)
{
    int     max_index   = 0;
    double  max_size    = vector[0];

    for (int i = 1; i < vector_size; i++)
    {
        if (vector[i] > max_size)
        {
            max_index   = i;
            max_size    = vector[i];
        }
    }

    return max_index;
}

#define ACTION_UP   0
#define ACTION_DOWN 1
#define ACTION_IDLE 2

double state_step(double* state, double* action)
{
    double cost = fabs(state[0] - state[1]);

    int target_index = vector_largest_index(action, ACTION_SIZE);

    if (target_index == ACTION_UP)
        state[0] = state[0] + STEP_CONTROL;
    else if (target_index == ACTION_DOWN)
        state[0] = state[0] - STEP_CONTROL;

    return -cost;
}

int main()
{
    int     layers[LAYER_SIZE]  = {128, 64};
    double  state[STATE_SIZE]   = {0};
    double* action              = NULL;

    ddpg_init();

    DDPG *ddpg = ddpg_create(STATE_SIZE, ACTION_SIZE, NULL, LAYER_SIZE, layers, LAYER_SIZE, layers, 100000, 32, 1);

    for (int episode = 0; episode < EPISODE_COUNT; episode++)
    {
        double episodeReward = 0;

        state[0] = deepc_random_double(0, 1);
        state[1] = deepc_random_double(0, 1);

        ddpg_new_episode(ddpg);

        for (int step = 0; step < EPISODE_LENGTH; step++)
        {
            action = ddpg_action(ddpg, state);

            double reward = state_step(state, action);

            episodeReward += reward;

            ddpg_observe(ddpg, action, &reward, state, 0);

            ddpg_train(ddpg, 0.99);
        }

        ddpg_update_target_networks(ddpg);
        

        printf("%d %f\n", episode, episodeReward / EPISODE_LENGTH);
    }

    ddpg_destroy(ddpg);

    return 0;
}