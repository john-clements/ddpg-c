#include <math.h>
#include <stdio.h>
#include "ddpg.h"

//#define MULTI_HEAD_EN

#define EPISODE_LENGTH  200
#define EPISODE_COUNT   200

#define STATE_SIZE      4
#define ACTION_SIZE     1
#define REWARD_SIZE     1

#define LAYER_SIZE      2

#define NOISE_EN

#define MULTI_HEAD_LAYER_SIZE 2

#define COLOR_NORMAL    "\033[0m"
#define COLOR_RED       "\033[0;31m"
#define COLOR_GREEN     "\033[0;32m"
#define COLOR_YELLOW    "\033[0;33m"

#define COLOR_ID_NORMAL 0
#define COLOR_ID_RED    1
#define COLOR_ID_GREEN  2
#define COLOR_ID_YELLOW 3

const char* g_color_map[] = {
    COLOR_NORMAL,
    COLOR_RED,
    COLOR_GREEN,
    COLOR_YELLOW,
};

int vector_largest_index(float* vector, int vector_size)
{
    int     max_index   = 0;
    float  max_size    = vector[0];

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

int is_float_zero(float val)
{
    float epsilon = .001; // A small tolerance value

    if (fabs(val) < epsilon)
        return 1;

    return 0;
}

float float_constrain(float val)
{
    int val_int = (int)val;

    val = val * 1000;

    int dec_int_mult = ((int)val)/250;

    return ((float)val_int) + ((float)dec_int_mult)*250/1000;
}

#define ACTION_OFF  0
#define ACTION_ON   1

//const char* g_action_str[ACTION_GROUP_SIZE] = {"UP", "DOWN", "IDLE"};

int get_action_index(float* action)
{
    if (action[0] > 0.5f)
        return ACTION_ON;

    return ACTION_OFF;
}

#define INDUCTOR_CAPACITY       10.0f
#define CAPACITOR_TARGET        50.0f
#define ENERGY_TRANSFER_RATE    0.5f
#define LOAD                    0.25f

// INPUT SPACE:
// 00 -> Output Voltage
// 01 -> Inductor Charge
// 02 -> Switch
// 03 -> Load
float state_step(float* state, float* action)
{
    int switch_state = get_action_index(action);

    state[2] = (float)switch_state;

    if (switch_state == ACTION_OFF)
    {
        if (state[1] < INDUCTOR_CAPACITY)
        {
            int multiplier = (int)state[1] + 1;

            for (int i = 0; i < multiplier; i++)
            {
                if (state[1] >= (float)i)
                    state[1] = state[1] + ENERGY_TRANSFER_RATE;
            }

            if (state[1] > INDUCTOR_CAPACITY)
                state[1] = INDUCTOR_CAPACITY;
        }
    }
    else if (switch_state == ACTION_ON)
    {
        if (state[1] > 0.0f)
        {
            state[0] = state[0] + state[1];
            state[1] = 0.0f;
        }
    }

    // TODO: CALCULATE OUTPUT BASED ON LOAD
    state[0] = state[0] - LOAD;
    if (state[0] < 0.0f)
        state[0] = 0.0f;

    float dist = (CAPACITOR_TARGET - state[0]) / CAPACITOR_TARGET;

    return -fabs(dist);
}

void normalize_state(float* state, float* input)
{
    input[0] = state[0] / (2 * CAPACITOR_TARGET);   // Optmial value at .5
    input[1] = state[1] / INDUCTOR_CAPACITY;
    input[2] = state[2];
    input[3] = 0;
}

void print_float_vector(float* vector, int vector_size)
{
    printf("[");
    for (int i = 0; i < vector_size - 1; i++)
        printf("%.3f ", vector[i]);
    printf("%.3f]", vector[vector_size - 1]);
}

void print_float_vector_highlight(float* vector, int vector_size, int* highlight_vector)
{
    int i = 0;
    printf("[");
    for (i = 0; i < vector_size - 1; i++)
        printf("%s%.1f, "COLOR_NORMAL, g_color_map[highlight_vector[i]], vector[i]);

    printf("%s%.1f"COLOR_NORMAL"]", g_color_map[highlight_vector[i]], vector[i]);
}

int main()
{
    int     layers[LAYER_SIZE]  = {32, 16};
    float  state[STATE_SIZE]   = {0};
    float  input[STATE_SIZE]   = {0};
    float  reward[REWARD_SIZE] = {0};
    float* action              = NULL;
#ifdef NOISE_EN
    float  noise[ACTION_SIZE]  = {0};
#else
    float* noise               = NULL;
#endif
#ifdef MULTI_HEAD_EN
    int     head_layers[MULTI_HEAD_LAYER_SIZE]  = {16, 8};
#endif

    int     highlight_vector[STATE_SIZE]    = {0};

    ddpg_init();

#ifdef NOISE_EN
    for (int i = 0; i < ACTION_SIZE; i++)
        noise[i] = .2;
#endif

#ifdef MULTI_HEAD_EN
    DDPG *ddpg = ddpg_multi_head_create(STATE_SIZE, ACTION_GROUP_SIZE, noise, LAYER_SIZE, layers, TARGET_CNT, MULTI_HEAD_LAYER_SIZE, head_layers, LAYER_SIZE, layers, 100000, 32, REWARD_SIZE);
#else
    DDPG *ddpg = ddpg_create(STATE_SIZE, ACTION_SIZE, noise, LAYER_SIZE, layers, LAYER_SIZE, layers, 100000, 32, REWARD_SIZE);
#endif

    for (int episode = 0; episode < EPISODE_COUNT; episode++)
    {
        float episode_reward[REWARD_SIZE] = {0};

        for (int i = 0; i < STATE_SIZE; i++)
        {
            state[i] = 0.0f;
            input[i] = 0.0f;
        }

        ddpg_new_episode(ddpg);

        for (int step = 0; step < EPISODE_LENGTH; step++)
        {
            action = ddpg_action(ddpg, input);

            reward[0] = state_step(state, action);

            for (int i = 0; i < REWARD_SIZE; i++)
                episode_reward[i] += reward[i];

            normalize_state(state, input);

            if (step == EPISODE_LENGTH - 1)
                ddpg_observe(ddpg, action, reward, input, 1);
            else
                ddpg_observe(ddpg, action, reward, input, 0);

            ddpg_train(ddpg, 0.99);

            for (int i = 0; i < STATE_SIZE; i++)
                highlight_vector[i] = COLOR_ID_NORMAL;

            if (state[2] > 0.5f)
                highlight_vector[2] = COLOR_ID_GREEN;
            else
                highlight_vector[2] = COLOR_ID_RED;

            if ((state[0] > (CAPACITOR_TARGET - 10.0f)) && (state[0] < (CAPACITOR_TARGET + 10.0f)))
            {
                if ((state[0] > (CAPACITOR_TARGET - 5.0f)) && (state[0] < (CAPACITOR_TARGET + 5.0f)))
                    highlight_vector[0] = COLOR_ID_GREEN;
                else
                    highlight_vector[0] = COLOR_ID_YELLOW;
            }
            else
                highlight_vector[0] = COLOR_ID_RED;

            printf("%3d:%3d -> ", episode, step);
            print_float_vector(reward, REWARD_SIZE);
            printf(" -> ");
            print_float_vector_highlight(state, STATE_SIZE, highlight_vector);
            printf("\n");
        }

        ddpg_update_target_networks(ddpg);

        for (int i = 0; i < REWARD_SIZE; i++)
            episode_reward[i] /= EPISODE_LENGTH;

        printf("%d -> ", episode);
        print_float_vector(episode_reward, REWARD_SIZE);
        printf("\n");
    }

    ddpg_destroy(ddpg);

    return 0;
}
