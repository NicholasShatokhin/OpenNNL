#include <iostream>

#include "opennnl.h"

using namespace std;

#define INPUTS_COUNT 3
#define LAYERS_COUNT 4
#define OUTPUTS_COUNT 2

#define TRAINING_SAMPLES_COUNT 21
#define SPEED 0.01
#define ERROR 0.01

#define TEST_INPUTS_COUNT 5

int main()
{
    int neuronsInLayers[LAYERS_COUNT] = {3, 10, 10, 2};
    double trainingInputs[TRAINING_SAMPLES_COUNT*INPUTS_COUNT] = {1.0, 	1.0, 	1.0,
                                                                  0.5, 	1.0, 	1.0,
                                                                  1.0, 	1.0, 	0.5,
                                                                  1.0, 	0.5, 	1.0,
                                                                  0.5, 	1.0, 	0.5,
                                                                  0.0, 	0.0, 	0.0,
                                                                  0.5, 	0.5, 	0.5,
                                                                  0.0, 	1.0, 	1.0,
                                                                  1.0, 	1.0, 	0.0,
                                                                  1.0, 	0.0, 	1.0,
                                                                  0.0, 	1.0, 	0.0,
                                                                  0.0, 	0.0, 	1.0,
                                                                  1.0, 	0.0, 	0.0,
                                                                  0.3, 	0.4, 	0.1,
                                                                  0.1, 	0.4, 	0.3,
                                                                  0.0, 	0.1, 	0.2,
                                                                  0.2, 	0.1, 	0.0,
                                                                  0.0, 	0.3, 	0.6,
                                                                  0.6, 	0.3, 	0.0,
                                                                  0.2, 	0.3, 	0.4,
                                                                  0.4, 	0.3, 	0.2 };
    double trainingOutputs[TRAINING_SAMPLES_COUNT*OUTPUTS_COUNT] = {
                                                                           1.0, 	0.5,
                                                                           0.6, 	0.7,
                                                                           0.6, 	0.3,
                                                                           0.3, 	0.4,
                                                                           0.7, 	0.5,
                                                                           0.2, 	0.2,
                                                                           0.5, 	0.4,
                                                                           0.4, 	0.9,
                                                                           0.4, 	0.1,
                                                                           0.2, 	0.2,
                                                                           1.0, 	0.5,
                                                                           0.3, 	0.8,
                                                                           0.3, 	0.2,
                                                                           0.5, 	0.3,
                                                                           0.5, 	0.7,
                                                                           0.3, 	0.9,
                                                                           0.3, 	0.1,
                                                                           0.5, 	0.8,
                                                                           0.5, 	0.2,
                                                                           0.5, 	0.9,
                                                                           0.4, 	0.1
                                                                   };

    double testInputs[TEST_INPUTS_COUNT*INPUTS_COUNT] = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 1.0,
        0.0, 1.0, 0.5,
        0.0, 0.5, 1.0,
        0.2, 0.1, 0.3
    };

    OpenNNL * opennnl = new OpenNNL(INPUTS_COUNT, LAYERS_COUNT, neuronsInLayers);
    opennnl->randomizeWeightsAndBiases();

    opennnl->trainingBP(TRAINING_SAMPLES_COUNT, trainingInputs, trainingOutputs, 1000, SPEED, ERROR);

    opennnl->printDebugInfo();

    double inputs[INPUTS_COUNT];
    double outputs[OUTPUTS_COUNT];

    for(int i=0;i<TEST_INPUTS_COUNT;i++)
    {
        memcpy(inputs, testInputs+i*INPUTS_COUNT, INPUTS_COUNT*sizeof(double));

        opennnl->calculate(inputs);
        opennnl->getOutputs(outputs);

        cout << "test sample #" << i+1 << ":" << endl;
        for(int j=0;j<INPUTS_COUNT;j++)
            cout << inputs[j] << " ";
        cout << " --> ";
        for(int j=0;j<OUTPUTS_COUNT;j++)
            cout << outputs[j] << " ";
        cout << endl;
    }

    delete opennnl;

    return 0;
}

