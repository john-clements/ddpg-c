#include <math.h>
#include <stdio.h>
#include "ddpg.h"


#define SINGLE_ENDED_OUT
//#define MULTI_HEAD_EN

#define EPISODE_LENGTH  20
#define EPISODE_COUNT   2000

#define TARGET_CNT      6

#define STATE_SIZE      (2*TARGET_CNT)
#define ACTION_SIZE     (3*TARGET_CNT)
#define REWARD_SIZE     (1*TARGET_CNT)

#define LAYER_SIZE      2

#define STEP_CONTROL    0.1f

#define MEASURE_QUALITY_START 25

#ifdef SINGLE_ENDED_OUT
#define ACTION_GROUP_SIZE  1
#else
#define ACTION_GROUP_SIZE  3
#endif

//#define NOISE_EN

#define MULTI_HEAD_LAYER_SIZE 2

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
    double epsilon = .001; // A small tolerance value

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
    float range = 1.0f / STEP_CONTROL;

    int x = deepc_random_int(0, (int)range);

    return STEP_CONTROL * (double)x;
}

#define ACTION_UP   0
#define ACTION_DOWN 1
#define ACTION_IDLE 2

//const char* g_action_str[ACTION_GROUP_SIZE] = {"UP", "DOWN", "IDLE"};

int get_action_index(double* action)
{
#ifdef SINGLE_ENDED_OUT
    // Trinary parsing
    if (action[0] > 0.1f)
        return ACTION_UP;
    else if (action[0] < -0.1f)
        return ACTION_DOWN;

    return ACTION_IDLE;
#else
    // Binary parsing
    return vector_largest_index(action, ACTION_GROUP_SIZE);

    // Custom Binary + continuous
    //if (action[0] <= 0 && action[1] <= 0)
    //    return ACTION_IDLE;
    //return vector_largest_index(action, ACTION_GROUP_SIZE);
#endif
}

int is_action_correct(double* state, double* action)
{
    double diff = state[0] - state[1];

    int target_index = get_action_index(action);

    if (is_double_zero(diff))
        diff = 0.0f;

    if (target_index == ACTION_UP)
    {
        if (diff < 0.0f)
            return 1;
    }
    else if (target_index == ACTION_DOWN)
    {
        if (diff > 0.0f)
            return 1;
    }
    else if (target_index == ACTION_IDLE)
    {
        if (is_double_zero(diff))
            return 1;
    }

    return 0;
}

double state_step(double* state, double* action)
{
    int target_index = get_action_index(action);

    if (target_index == ACTION_UP)
        state[0] = state[0] + STEP_CONTROL;
    else if (target_index == ACTION_DOWN)
        state[0] = state[0] - STEP_CONTROL;

    if (state[0] < 0.0f)
        state[0] = 0.0f;
    else if (state[0] > 1.0f)
        state[0] = 1.0f;

    double diff = state[0] - state[1];

    if (is_double_zero(diff))
        diff = 0.0f;

    return -fabs(diff);
}

void print_double_vector(double* vector, int vector_size)
{
    printf("[");
    for (int i = 0; i < vector_size - 1; i++)
        printf("%f ", vector[i]);
    printf("%f]", vector[vector_size - 1]);
}

void print_double_vector_highlight(double* vector, int vector_size, int* highlight_vector)
{
    int i = 0;
    printf("[");
    for (i = 0; i < vector_size - 1; i++)
        printf("%s%.3f, "COLOR_NORMAL, g_color_map[highlight_vector[i]], vector[i]);

    printf("%s%.3f"COLOR_NORMAL"]", g_color_map[highlight_vector[i]], vector[i]);
}

#define TEST_TRIALS             10
#define TEST_EPISODE_LENGTH     25
void validate_target_seeker(DDPG* ddpg)
{
    double  state[STATE_SIZE]               = {0};
    double  reward[REWARD_SIZE]             = {0};
    double* action                          = NULL;
    double  reward_quality[REWARD_SIZE]     = {0};
    int     highlight_vector[STATE_SIZE]    = {0};

    for (int trial = 0; trial < TEST_TRIALS; trial++)
    {
        double episode_reward[REWARD_SIZE] = {0};

        for (int i = 0; i < TARGET_CNT; i++)
        {
            state[2*i]      = 0.5f;
            state[2*i + 1]  = random_target();
        }

        for (int step = 0; step < 25; step++)
        {
            action = ddpg_action(ddpg, state);

            for (int i = 0; i < TARGET_CNT; i++)
            {
                if (is_action_correct(&state[i*2], &action[i*ACTION_GROUP_SIZE]))
                    highlight_vector[i*2] = COLOR_ID_GREEN;
                else
                    highlight_vector[i*2] = COLOR_ID_RED;

                highlight_vector[i*2 + 1] = COLOR_ID_NORMAL;
            }

            for (int i = 0; i < TARGET_CNT; i++)
                reward[i] = state_step(&state[i*2], &action[i*ACTION_GROUP_SIZE]);

            for (int i = 0; i < REWARD_SIZE; i++)
                episode_reward[i] += reward[i];

            printf("%3d:%3d -> ", trial, step);
            print_double_vector(reward, REWARD_SIZE);
            printf(" -> ");
            print_double_vector_highlight(state, STATE_SIZE, highlight_vector);
            printf("\n");
        }

        for (int i = 0; i < REWARD_SIZE; i++)
            episode_reward[i] /= TEST_EPISODE_LENGTH;

        printf("%d -> ", trial);
        print_double_vector(episode_reward, REWARD_SIZE);
        printf("\n");

        for (int i = 0; i < REWARD_SIZE; i++)
            reward_quality[i] = reward_quality[i] + episode_reward[i]/TEST_TRIALS;
    }

    printf("Final Reward Quality -> ");
    print_double_vector(reward_quality, REWARD_SIZE);
    printf("\n");
}

int main()
{
    int     layers[LAYER_SIZE]  = {48, 48};
    double  state[STATE_SIZE]   = {0};
    double  reward[REWARD_SIZE] = {0};
    double* action              = NULL;
#ifdef NOISE_EN
    double  noise[ACTION_SIZE]  = {0};
#else
    double* noise               = NULL;
#endif
#ifdef MULTI_HEAD_EN
    int     head_layers[MULTI_HEAD_LAYER_SIZE]  = {16, 8};
#endif


    double  reward_quality[REWARD_SIZE]     = {0};
    int     highlight_vector[STATE_SIZE]    = {0};

    ddpg_init();

#ifdef NOISE_EN
    for (int i = 0; i < ACTION_SIZE; i++)
        noise[i] = .01;
#endif

#ifdef MULTI_HEAD_EN
    DDPG *ddpg = ddpg_multi_head_create(STATE_SIZE, ACTION_GROUP_SIZE, noise, LAYER_SIZE, layers, TARGET_CNT, MULTI_HEAD_LAYER_SIZE, head_layers, LAYER_SIZE, layers, 100000, 32, REWARD_SIZE);
#else
    DDPG *ddpg = ddpg_create(STATE_SIZE, ACTION_SIZE, noise, LAYER_SIZE, layers, LAYER_SIZE, layers, 100000, 32, REWARD_SIZE);
#endif

    for (int episode = 0; episode < EPISODE_COUNT; episode++)
    {
        double episode_reward[REWARD_SIZE] = {0};

        for (int i = 0; i < TARGET_CNT; i++)
        {
            state[2*i]      = 0.5f;
            state[2*i + 1]  = random_target();
        }

        ddpg_new_episode(ddpg);

        for (int step = 0; step < EPISODE_LENGTH; step++)
        {
            action = ddpg_action(ddpg, state);

            for (int i = 0; i < TARGET_CNT; i++)
            {
                if (is_action_correct(&state[i*2], &action[i*ACTION_GROUP_SIZE]))
                    highlight_vector[i*2] = COLOR_ID_GREEN;
                else
                    highlight_vector[i*2] = COLOR_ID_RED;

                highlight_vector[i*2 + 1] = COLOR_ID_NORMAL;
            }

            for (int i = 0; i < TARGET_CNT; i++)
                reward[i] = state_step(&state[i*2], &action[i*ACTION_GROUP_SIZE]);

            for (int i = 0; i < REWARD_SIZE; i++)
                episode_reward[i] += reward[i];

            ddpg_observe(ddpg, action, reward, state, 0);

            ddpg_train(ddpg, 0.99);

            printf("%3d:%3d -> ", episode, step);
            print_double_vector(reward, REWARD_SIZE);
            printf(" -> ");
            print_double_vector_highlight(state, STATE_SIZE, highlight_vector);
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

    printf("Final Reward Quality -> ");
    print_double_vector(reward_quality, REWARD_SIZE);
    printf("\n");

    validate_target_seeker(ddpg);

    ddpg_destroy(ddpg);

    return 0;
}
