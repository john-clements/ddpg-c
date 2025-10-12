#include <math.h>
#include <stdio.h>
#include "mlpc.h"


//#define SINGLE_ENDED_OUT

#ifdef SINGLE_ENDED_OUT
#define ACTION_GROUP_SIZE  1
#else
#define ACTION_GROUP_SIZE  3
#endif

#define EPISODE_LENGTH  5
#define EPISODE_COUNT   8000

#define TARGET_CNT      1

#define STATE_SIZE      (2*TARGET_CNT)
#define ACTION_SIZE     (ACTION_GROUP_SIZE*TARGET_CNT)
#define REWARD_SIZE     (1*TARGET_CNT)

#define BATCH_SIZE      1

#define LAYER_SIZE      2

#define STEP_CONTROL    0.025f

#define MEASURE_QUALITY_START 25


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
    double epsilon = 0.001f; // A small tolerance value

    if (fabs(val) < epsilon)
        return 1;

    return 0;
}

double double_constrain(float val)
{
    int val_int = (int)val;

    val = val * 1000;

    int dec_int_mult = ((int)val)/(STEP_CONTROL*10000);

    return ((double)val_int) + ((double)dec_int_mult)*(STEP_CONTROL*10);
}

#define STEP_RANGE (1/STEP_CONTROL)
double random_target()
{
    int x = deepc_random_int(0, STEP_RANGE);

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

void set_action(double* state, double* action)
{
    double diff = state[0] - state[1];

#ifdef SINGLE_ENDED_OUT
    if (is_double_zero(diff))
    {
        action[0] = 0.0f;
    }
    else if (diff < 0.0f)
    {
        action[0] = 1.0f;

    }
    else if (diff > 0.0f)
    {
        action[0] = -1.0f;
    }
#else
    if (is_double_zero(diff))
    {
        action[0] = 0.0f;
        action[1] = 0.0f;
        action[2] = 1.0f;
    }
    else if (diff < 0.0f)
    {
        action[0] = 1.0f;
        action[1] = 0.0f;
        action[2] = 0.0f;
    }
    else if (diff > 0.0f)
    {
        action[0] = 0.0f;
        action[1] = 1.0f;
        action[2] = 0.0f;
    }
#endif
}

void print_double_vector(double* vector, int vector_size)
{
    printf("[");
    for (int i = 0; i < vector_size - 1; i++)
        printf("%f ", vector[i]);
    printf("%f]", vector[vector_size - 1]);
}

void print_double_vector_short(double* vector, int vector_size)
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

#define TEST_TRIALS 10
void validate_target_seeker(MLP* mlp)
{
    double  state[STATE_SIZE]               = {0};
    double  reward[REWARD_SIZE]             = {0};
    double  action[ACTION_SIZE]             = {0};
    double  reward_quality[REWARD_SIZE]     = {0};
    int     highlight_vector[STATE_SIZE]    = {0};
    double  correct_action[ACTION_SIZE]     = {0};

    Matrix x = matrix_create(1, STATE_SIZE);

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
            for (int i = 0; i < STATE_SIZE; i++)
                MATRIX(x, 0, i) = state[i];

            Matrix y = mlp_feedforward(mlp, x);
        
            for (int i = 0; i < ACTION_SIZE; i++)
                action[i] = MATRIX(y, 0, i);

            for (int i = 0; i < TARGET_CNT; i++)
            {
                if (is_action_correct(&state[i*2], &action[i*ACTION_GROUP_SIZE]))
                    highlight_vector[i*2] = COLOR_ID_GREEN;
                else
                    highlight_vector[i*2] = COLOR_ID_RED;

                highlight_vector[i*2 + 1] = COLOR_ID_NORMAL;
            }

            for (int i = 0; i < TARGET_CNT; i++)
                set_action(&state[i*2], &correct_action[i*ACTION_GROUP_SIZE]);

            for (int i = 0; i < TARGET_CNT; i++)
                reward[i] = state_step(&state[i*2], &action[i*ACTION_GROUP_SIZE]);

            for (int i = 0; i < REWARD_SIZE; i++)
                episode_reward[i] += reward[i];

            printf("%3d:%3d -> ", trial, step);
            print_double_vector(reward, REWARD_SIZE);
            printf(" -> ");
            print_double_vector_highlight(state, STATE_SIZE, highlight_vector);
            printf(" -> ");
            print_double_vector_short(action, ACTION_SIZE);
            printf(" -> ");
            print_double_vector_short(correct_action, ACTION_SIZE);
            printf("\n");
        }

        for (int i = 0; i < REWARD_SIZE; i++)
            episode_reward[i] /= 25;

        printf("%d -> ", trial);
        print_double_vector(episode_reward, REWARD_SIZE);
        printf("\n");

        for (int i = 0; i < REWARD_SIZE; i++)
            reward_quality[i] = reward_quality[i] + episode_reward[i]/TEST_TRIALS;
    }

    printf("Final Reward Quality -> ");
    print_double_vector(reward_quality, REWARD_SIZE);
    printf("\n");

    matrix_destroy(x);
}

int main()
{
    int     layers[LAYER_SIZE]              = {128, 64};
    double  state[STATE_SIZE*BATCH_SIZE]    = {0};
    double  action[ACTION_SIZE*BATCH_SIZE]  = {0};

    mlp_init();

    MLP* mlp = mlp_create(STATE_SIZE, ACTION_SIZE, LAYER_SIZE, layers, ACTIVATION_RELU, ACTIVATION_TANH, BATCH_SIZE);

    Matrix x = matrix_create(BATCH_SIZE, STATE_SIZE);

    Matrix y = matrix_create(BATCH_SIZE, ACTION_SIZE);

    Adam *adam = adam_create(mlp);

    double loss = 0;

    for (int episode = 0; episode < EPISODE_COUNT; episode++)
    {
        for (int i = 0; i < TARGET_CNT*BATCH_SIZE; i++)
        {
            state[2*i]      = random_target();
            state[2*i + 1]  = random_target();
        }

        for (int step = 0; step < EPISODE_LENGTH; step++)
        {
            for (int i = 0; i < STATE_SIZE; i++)
                for (int j = 0; j < BATCH_SIZE; j++)
                    MATRIX(x, j, i) = state[i];

            for (int i = 0; i < ACTION_SIZE*BATCH_SIZE; i++)
                action[i] = 0.0f;

            for (int i = 0; i < TARGET_CNT*BATCH_SIZE; i++)
                set_action(&state[i*2], &action[i*ACTION_GROUP_SIZE]);

            for (int i = 0; i < ACTION_SIZE; i++)
                for (int j = 0; j < BATCH_SIZE; j++)
                    MATRIX(y, j, i) = action[i];

            mlp_feedforward(mlp, x);
        
            /* Back-propagate using the MSE loss function on the true values y. */
            loss += mlp_backpropagate(mlp, y, LOSS_MSE);
            
            /* Optimize the neural network with Adam. */
            adam_optimize(mlp, adam);

            if ((episode*EPISODE_LENGTH + step) % 100 == 0)
            {
                printf("%d %f\n", (episode*EPISODE_LENGTH + step), loss / 100);
                loss = 0;
            }

            for (int i = 0; i < TARGET_CNT; i++)
                state_step(&state[i*2], &action[i*ACTION_GROUP_SIZE]);
        }
    }
/*
    // Closeness rounds
    for (int i = 0; i < TARGET_CNT*BATCH_SIZE; i++)
    {
        state[2*i]      = 0;
        state[2*i + 1]  = STEP_CONTROL;
    }

    for (int episode = 0; episode < 10000; episode++)
    {
        for (int i = 0; i < TARGET_CNT*BATCH_SIZE; i++)
        {
            state[2*i]      = state[2*i] + STEP_CONTROL;
            state[2*i + 1]  = state[2*i + 1] + STEP_CONTROL;

            if (state[2*i] > 1.0f)
                state[2*i] = 0.0f;
            if (state[2*i + 1] > 1.0f)
                state[2*i + 1] = 0.0f;
        }

        for (int i = 0; i < STATE_SIZE; i++)
            for (int j = 0; j < BATCH_SIZE; j++)
                MATRIX(x, j, i) = state[i];

        for (int i = 0; i < ACTION_SIZE*BATCH_SIZE; i++)
            action[i] = 0.0f;

        for (int i = 0; i < TARGET_CNT*BATCH_SIZE; i++)
            set_action(&state[i*2], &action[i*ACTION_GROUP_SIZE]);

        for (int i = 0; i < ACTION_SIZE; i++)
            for (int j = 0; j < BATCH_SIZE; j++)
                MATRIX(y, j, i) = action[i];

        mlp_feedforward(mlp, x);
    
        loss += mlp_backpropagate(mlp, y, LOSS_MSE);
        
        adam_optimize(mlp, adam);

        if ((episode*EPISODE_LENGTH) % 100 == 0)
        {
            printf("%d %f\n", (episode*EPISODE_LENGTH), loss / 100);
            loss = 0;
        }

        for (int i = 0; i < TARGET_CNT; i++)
            state_step(&state[i*2], &action[i*ACTION_GROUP_SIZE]);
    }
*/
    validate_target_seeker(mlp);

    mlp_destroy(mlp);
    matrix_destroy(x);
    matrix_destroy(y);
    adam_destroy(adam);

    return 0;
}