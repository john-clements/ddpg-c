#include <math.h>
#include <stdio.h>
#include "ddpgc.h"


#define EPISODE_LENGTH  200
#define EPISODE_COUNT   100

#define STATE_SIZE      2
#define ACTION_SIZE     3
#define REWARD_SIZE     2

#define LAYER_SIZE      2

#define STEP_CONTROL    .025

#define MEASURE_QUALITY_START 25

int vector_largest_index(double* vector, int vector_size)
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

int is_double_zero(float val)
{
    double epsilon = 1e-9; // A small tolerance value

    if (fabs(val) < epsilon)
        return 1;

    return 0;
}

double double_constrain(float val)
{
    int val_int = (int)val;

    val = val * 1000;

    int dec_int_mult = ((int)val)/250;

    return ((double)val_int) + ((double)dec_int_mult)*250/1000;
}

#define ACTION_UP   0
#define ACTION_DOWN 1
#define ACTION_IDLE 2

const char* g_action_str[ACTION_SIZE] = {"UP", "DOWN", "IDLE"};

int is_action_correct(double* state, double* action)
{
    double diff = state[0] - state[1];

    int target_index = vector_largest_index(action, ACTION_SIZE);

    if (target_index == ACTION_UP)
    {
        if (diff < 0)
            return 1;
    }
    else if (target_index == ACTION_DOWN)
    {
        if (diff > 0)
            return 1;
    }
    else if (target_index == ACTION_IDLE)
    {
        if (is_double_zero(diff))
            return 1;
    }

    return 0;
}

void state_step(double* state, double* action, double* reward)
{
    double cost = fabs(state[0] - state[1]);

    int target_index = vector_largest_index(action, ACTION_SIZE);

    if (target_index == ACTION_UP)
        state[0] = state[0] + STEP_CONTROL;
    else if (target_index == ACTION_DOWN)
        state[0] = state[0] - STEP_CONTROL;

    if (state[0] < 0)
        state[0] = 0;
    else if (state[0] > 1)
        state[0] = 1;

    reward[0] = -cost;

    if (is_action_correct(state, action))
        reward[1] += .01;
    else
        reward[1] -= .01;

    if (reward[1] > 0)
        reward[1] = 0;
    else if (reward[1] < -1)
        reward[1] = -1;
}

void print_double_vector(double* vector, int vector_size)
{
    printf("[");
    for (int i = 0; i < vector_size - 1; i++)
        printf("%f ", vector[i]);
    printf("%f]", vector[vector_size - 1]);
}

int main()
{
    int     layers[LAYER_SIZE]  = {128, 64};
    double  state[STATE_SIZE]   = {0};
    double  reward[REWARD_SIZE] = {0};
    double* action              = NULL;

    double  reward_quality[REWARD_SIZE] = {0};

    ddpg_init();

    DDPG *ddpg = ddpg_create(STATE_SIZE, ACTION_SIZE, NULL, LAYER_SIZE, layers, LAYER_SIZE, layers, 100000, 32, REWARD_SIZE);

    for (int episode = 0; episode < EPISODE_COUNT; episode++)
    {
        double episode_reward[REWARD_SIZE] = {0};

        state[0] = double_constrain(deepc_random_double(0, 1));
        state[1] = double_constrain(deepc_random_double(0, 1));

        ddpg_new_episode(ddpg);

        for (int step = 0; step < EPISODE_LENGTH; step++)
        {
            action = ddpg_action(ddpg, state);

            printf("%d:%d -> ", episode, step);
            print_double_vector(reward, REWARD_SIZE);
            printf(" -> ");
            print_double_vector(state, STATE_SIZE);
            printf(" -> ");
            print_double_vector(action, ACTION_SIZE);
            if (is_action_correct(state, action))
                printf(" -> \e[1;32m%s\e[0m\n", g_action_str[vector_largest_index(action, ACTION_SIZE)]);
            else
                printf(" -> \e[1;31m%s\e[0m\n", g_action_str[vector_largest_index(action, ACTION_SIZE)]);

            state_step(state, action, reward);

            for (int i = 0; i < REWARD_SIZE; i++)
                episode_reward[i] += reward[i];

            ddpg_observe(ddpg, action, reward, state, 0);

            ddpg_train(ddpg, 0.99);
        }

        ddpg_update_target_networks(ddpg);

        for (int i = 0; i < REWARD_SIZE; i++)
            episode_reward[i] /= EPISODE_LENGTH;

        printf("%d -> ", episode);
        print_double_vector(episode_reward, REWARD_SIZE);
        printf("\n");

        if (episode >= EPISODE_COUNT - MEASURE_QUALITY_START)
        {
            for (int i = 0; i < REWARD_SIZE; i++)
                reward_quality[i] = reward_quality[i] + episode_reward[i]/EPISODE_COUNT;
        }
    }

    ddpg_destroy(ddpg);

    printf("Final Reward Quality -> ");
    print_double_vector(reward_quality, REWARD_SIZE);
    printf("\n");

    return 0;
}