#include <math.h>
#include <stdio.h>
#include "ddpg.h"


#define EPISODE_LENGTH  200
#define EPISODE_COUNT   200

#define TARGET_CNT      1

#define REWARD_PER_ACTION   4
#define MULTI_ACTION_CNT    10

#define STATE_SIZE      (2*TARGET_CNT)
#define ACTION_SIZE     (MULTI_ACTION_CNT*TARGET_CNT)
#define REWARD_SIZE     (REWARD_PER_ACTION*TARGET_CNT)

#define LAYER_SIZE      2

#define STEP_CONTROL    .025

#define MEASURE_QUALITY_START 25

#define PARSED_OUTPUTS  3

//#define NOISE_EN

#define COLOR_NORMAL    "\033[0m"
#define COLOR_RED       "\033[0;31m"
#define COLOR_GREEN     "\033[0;32m"

#define COLOR_ID_NORMAL 0
#define COLOR_ID_RED    1
#define COLOR_ID_GREEN  2

const char* g_color_map[] = {
    COLOR_NORMAL,
    COLOR_RED,
    COLOR_GREEN,
};

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

double random_target()
{
    int x = deepc_random_int(0, 40);

    return STEP_CONTROL * (double)x;
}

#define ACTION_UP   0
#define ACTION_DOWN 1
#define ACTION_IDLE 2

const char* g_action_str[PARSED_OUTPUTS] = {"UP", "DOWN", "IDLE"};

int get_action_index(double action)
{
    // Binary parsing
    //return vector_largest_index(action, ACTION_SIZE);

    // Custom Binary + continuous
    //if (action[0] <= 0 && action[1] <= 0)
    //    return ACTION_IDLE;
    //return vector_largest_index(action, ACTION_SIZE);


    // Trinary parsing

    if (action > 0.5f)
        return ACTION_UP;
    else if (action < -0.5f)
        return ACTION_DOWN;

    return ACTION_IDLE;
}

int get_action_inverse_index(double action)
{
    if (action > 0.5f)
        return ACTION_DOWN;
    else if (action <ACTION_UP-0.5f)
        return ACTION_DOWN;

    return ACTION_IDLE;
}

int is_action_correct(double* state, double* action)
{
    double diff = state[0] - state[1];

    int target_index[MULTI_ACTION_CNT] = {0};

    for (int i = 0; i < MULTI_ACTION_CNT; i++)
    {
        if (i%2 == 0)
            target_index[i] = get_action_index(action[i]);
        else
            target_index[i] = get_action_inverse_index(action[i]);
    }

    if (is_double_zero(diff))
        diff = 0.0f;

    double adjust = 0.0f;

    for (int i = 0; i < MULTI_ACTION_CNT; i++)
    {
        if (target_index[i] == ACTION_UP)
            adjust = adjust + STEP_CONTROL;
        else if (target_index[i] == ACTION_DOWN)
            adjust = adjust - STEP_CONTROL;
    }

    double new_diff = state[0] + adjust - state[1];

    if (is_double_zero(new_diff))
        new_diff = 0.0f;

    if ((new_diff == 0.0f) && (diff == 0.0f))
        return 1;

    if (new_diff < diff)
        return 1;

    return 0;
}

double calculate_tracker_reward(double reward, int is_correct)
{
    if (is_correct)
        reward = reward + .1;
    else
        reward = reward - .1;

    if (reward < -1)
        reward = -1;
    else if (reward > 1)
        reward = 1;

    return reward;
}

double calculate_stop_reward(double* state)
{
    double diff = state[0] - state[1];

    if (is_double_zero(diff))
        return 0;

    return -1;
}

double calculate_fine_grain_reward(double* state)
{
    double diff = state[0] - state[1];

    if (is_double_zero(diff))
        return 0;

    if (diff > .1)
        return -1;

    return diff*10;
}

double calculate_reward(double* state)
{
    double diff = state[0] - state[1];

    if (is_double_zero(diff))
        diff = 0.0f;

    return -fabs(diff);
}

double state_step(double* state, double* action)
{
    int target_index[MULTI_ACTION_CNT] = {0};

    for (int i = 0; i < MULTI_ACTION_CNT; i++)
    {
        if (i%2 == 0)
            target_index[i] = get_action_index(action[i]);
        else
            target_index[i] = get_action_inverse_index(action[i]);
    }

    for (int i = 0; i < MULTI_ACTION_CNT; i++)
    {
//printf("TEST %f -> %d\n", action[i], target_index[i]);
        if (target_index[i] == ACTION_UP)
            state[0] = state[0] + STEP_CONTROL;
        else if (target_index[i] == ACTION_DOWN)
            state[0] = state[0] - STEP_CONTROL;
    }

    if (state[0] < 0)
        state[0] = 0;
    else if (state[0] > 1)
        state[0] = 1;

    double diff = state[0] - state[1];

    if (is_double_zero(diff))
        diff = 0.0f;

    return -fabs(diff);
}

void print_double_vector(double* vector, int vector_size)
{
    printf("[");
    for (int i = 0; i < vector_size - 1; i++)
        printf("%.3f ", vector[i]);
    printf("%.3f]", vector[vector_size - 1]);
}

void print_double_vector_highlight(double* vector, int vector_size, int* highlight_vector)
{
    int i = 0;
    printf("[");
    for (i = 0; i < vector_size - 1; i++)
        printf("%s%.3f, "COLOR_NORMAL, g_color_map[highlight_vector[i]], vector[i]);

    printf("%s%.3f"COLOR_NORMAL"]", g_color_map[highlight_vector[i]], vector[i]);
}

int main()
{
    int     layers[LAYER_SIZE]  = {128, 64};
    double  state[STATE_SIZE]   = {0};
    double  reward[REWARD_SIZE] = {0};
    double* action              = NULL;
#ifdef NOISE_EN
    double  noise[ACTION_SIZE]  = {0};
#else
    double* noise               = NULL;
#endif

    double  reward_quality[REWARD_SIZE]     = {0};
    int     highlight_vector[STATE_SIZE]    = {0};

    ddpg_init();

#ifdef NOISE_EN
    for (int i = 0; i < ACTION_SIZE; i++)
        noise[i] = .01;
#endif

    DDPG *ddpg = ddpg_create(STATE_SIZE, ACTION_SIZE, noise, LAYER_SIZE, layers, LAYER_SIZE, layers, 100000, 32, REWARD_SIZE);

    for (int episode = 0; episode < EPISODE_COUNT; episode++)
    {
        double episode_reward[REWARD_SIZE] = {0};

        for (int i = 0; i < TARGET_CNT; i++)
        {
            state[2*i]      = .5;
            state[2*i + 1]  = random_target();
        }

        ddpg_new_episode(ddpg);

        for (int step = 0; step < EPISODE_LENGTH; step++)
        {
            action = ddpg_action(ddpg, state);

            for (int i = 0; i < TARGET_CNT; i++)
            {
                int is_correct = is_action_correct(&state[i*2], &action[i*MULTI_ACTION_CNT]);

                if (is_correct)
                    highlight_vector[i*2] = COLOR_ID_GREEN;
                else
                    highlight_vector[i*2] = COLOR_ID_RED;

                highlight_vector[i*2 + 1] = COLOR_ID_NORMAL;

                reward[i*REWARD_PER_ACTION+1] = calculate_tracker_reward(reward[i*REWARD_PER_ACTION+1], is_correct);
            }

            for (int i = 0; i < TARGET_CNT; i++)
            {
                reward[i*REWARD_PER_ACTION] = state_step(&state[i*REWARD_PER_ACTION], &action[i*MULTI_ACTION_CNT]);

                reward[i*REWARD_PER_ACTION+2] = calculate_stop_reward(&state[i*REWARD_PER_ACTION]);
                reward[i*REWARD_PER_ACTION+3] = calculate_fine_grain_reward(&state[i*REWARD_PER_ACTION]);
            }

            for (int i = 0; i < REWARD_SIZE; i++)
                episode_reward[i] += reward[i];

            ddpg_observe(ddpg, action, reward, state, 0);

            ddpg_train(ddpg, 0.99);

            printf("%3d:%3d -> ", episode, step);
            print_double_vector(reward, REWARD_SIZE);
            printf(" -> ");
            print_double_vector_highlight(state, STATE_SIZE, highlight_vector);
            printf(" -> ");
            print_double_vector(action, ACTION_SIZE);
            printf("\n");
        }

        ddpg_update_target_networks(ddpg);

        for (int i = 0; i < REWARD_SIZE; i++)
            episode_reward[i] /= EPISODE_LENGTH;

        printf("%d -> ", episode);
        print_double_vector(episode_reward, REWARD_SIZE);
        printf("\n");

        if (episode >= EPISODE_COUNT - MEASURE_QUALITY_START - 1)
        {
            for (int i = 0; i < REWARD_SIZE; i++)
                reward_quality[i] = reward_quality[i] + episode_reward[i]/MEASURE_QUALITY_START;
        }
    }

    ddpg_destroy(ddpg);

    printf("Final Reward Quality -> ");
    print_double_vector(reward_quality, REWARD_SIZE);
    printf("\n");

    return 0;
}
