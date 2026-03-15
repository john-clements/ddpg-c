#include <math.h>
#include <stdio.h>
#include "ddpg.h"

//#define MULTI_HEAD_EN

#define EPISODE_LENGTH  50
#define EPISODE_COUNT   800

#define STATE_SIZE      4
#define ACTION_SIZE     2
#define REWARD_SIZE     2

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

// Action Group 1
#define ACTION_DOWN     0
#define ACTION_UP       1
#define ACTION_IDLE     2

// Action Group 2
#define ACTION_TRIGGER_OFF  0
#define ACTION_TRIGGER_ON   1

//const char* g_action_str[ACTION_GROUP_SIZE] = {"UP", "DOWN", "IDLE"};

void get_action_index(float* action, int action_group[2])
{
    if (action[0] > 0.1f)
        action_group[0] =  ACTION_UP;
    else if (action[0] < -0.1f)
        action_group[0] =  ACTION_DOWN;
    else
        action_group[0] =  ACTION_IDLE;
        

    if (action[1] >= 0)
        action_group[1] = ACTION_TRIGGER_ON;
    else
        action_group[1] = ACTION_TRIGGER_OFF;
}

#define ANGLE_MIN   (0.0f + (3.14f / 4.0f))
#define ANGLE_MAX   (3.14f - (3.14f / 4.0f))
#define ACTION_STEP 0.1f
#define X_SPAN      100.0f
#define Y_SPAN      100.0f
#define X_SHOOTER   (X_SPAN / 2.0f)

// INPUT SPACE:
// 00 -> Target X Pos
// 01 -> Target Y Pos
// 02 -> Target V Velocity
// 03 -> Armeture Angle (0 to 180 Deg)
void state_step(float* state, float* action, float* reward)
{
    int action_group[2] = {0};

    get_action_index(action, action_group);

    if (action_group[0] == ACTION_DOWN)
        state[3] = state[3] - ACTION_STEP;
    else if (action_group[0] == ACTION_UP)
        state[3] = state[3] + ACTION_STEP;

    if (state[3] < ANGLE_MIN)
        state[3] = ANGLE_MIN;
    else if (state[3] > ANGLE_MAX)
        state[3] = ANGLE_MAX;

    float theta = state[3] - (ANGLE_MAX / 2.0f);

    float projectile_x = state[1] * tan(theta);

    projectile_x = X_SHOOTER - projectile_x;

    float dist = (projectile_x - state[0]) / X_SPAN;

    // Advance target X position
    //state[0] = state[0] + state[2];

    reward[0] = -fabs(dist);

    if (reward[0] < -1.0f)
        reward[0] = -1.0f;

    if (action_group[1] == ACTION_TRIGGER_ON)
    {
        if (reward[0] > -0.1f)
            reward[1] = reward[1] + 0.1f;
        else
            reward[1] = reward[1] - 0.1f;

        if (reward[1] < -1.0f)
            reward[1] = -1.0f;
        else if (reward[1] > 0.0f)
            reward[1] = 0.0f;
    }
}

#define TARGET_V    0.1f
#define TARGET_Y    10.0f

void set_initial_state(float* state)
{
    state[0] = (float)deepc_random_int(0, (int)X_SPAN);
    state[1] = TARGET_Y;
    state[2] = (X_SPAN / EPISODE_LENGTH);
    state[3] = ANGLE_MAX / 2.0f;
}

void normalize_state(float* state, float* input)
{
    input[0] = state[0] / X_SPAN;
    input[1] = state[1] / Y_SPAN;
    input[2] = state[2];
    input[3] = input[3] / ANGLE_MAX;
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
    int     layers[LAYER_SIZE]  = {128, 64};
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

    int     highlight_vector[ACTION_SIZE]    = {0};

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

        set_initial_state(state);

        reward[1] = 0.5f;

        ddpg_new_episode(ddpg);

        for (int step = 0; step < EPISODE_LENGTH; step++)
        {
            action = ddpg_action(ddpg, input);

            state_step(state, action, reward);

            for (int i = 0; i < REWARD_SIZE; i++)
                episode_reward[i] += reward[i];

            normalize_state(state, input);

            if (step == EPISODE_LENGTH - 1)
                ddpg_observe(ddpg, action, reward, input, 1);
            else
                ddpg_observe(ddpg, action, reward, input, 0);

            ddpg_train(ddpg, 0.99);

            for (int i = 0; i < ACTION_SIZE; i++)
                highlight_vector[i] = COLOR_ID_NORMAL;

            int action_group[2] = {0};
            get_action_index(action, action_group);
            //highlight_vector[action_group[0]] = COLOR_ID_GREEN;

            if (action_group[1])
                highlight_vector[1] = COLOR_ID_GREEN;

            printf("%3d:%3d -> ", episode, step);
            print_float_vector(reward, REWARD_SIZE);
            printf(" -> ");
            print_float_vector(state, STATE_SIZE);
            printf(" -> ");
            print_float_vector_highlight(action, ACTION_SIZE, highlight_vector);
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
