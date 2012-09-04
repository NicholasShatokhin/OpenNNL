#include <iostream>
#include <cmath>

#include "opennnl.h"
#include "mnistfile.h"

using namespace std;

#define INPUTS_COUNT 3
#define LAYERS_COUNT 4
#define OUTPUTS_COUNT 2

#define TRAINING_SAMPLES_COUNT 21
#define SPEED 0.015
#define ERROR 0.01

#define TEST_INPUTS_COUNT 9

void testNetwork1();
void testNetwork2();

int main()
{
    testNetwork1();
    testNetwork2();

    return 0;
}

void testNetwork1()
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
        0.2, 0.1, 0.3,
        0.0, 1.0, 0.0,
        1.0, 1.0, 0.0,
        1.0, 0.5, 0.0,
        1.0, 1.0, 1.0
    };

    OpenNNL * opennnl = new OpenNNL(INPUTS_COUNT, LAYERS_COUNT, neuronsInLayers);
    opennnl->randomizeWeightsAndBiases();

    opennnl->trainingIDBD(TRAINING_SAMPLES_COUNT, trainingInputs, trainingOutputs, 1000, SPEED, ERROR);

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

    cout << endl;
    cout << endl;
}

void testNetwork2()
{
    cout << "Creating network..." << endl;
    const int layers_count = 3;
    const int inputs_count = 784;
    int neuronsInLayers[layers_count] = {300, 100, 1};

    OpenNNL * opennnl = new OpenNNL(inputs_count, layers_count, neuronsInLayers);
    opennnl->randomizeWeightsAndBiases();

    cout << "Preparing train data..." << endl;

    MnistFile images;
    MnistFile labels;
    if(!images.openFile("../mnist/data/train-images.idx3-ubyte"))
    {
        cout << "Couldn't find train images file" << endl;
        return;
    }

    if(!labels.openFile("../mnist/data/train-labels.idx1-ubyte"))
    {
        cout << "Couldn't find train labels file" << endl;
        return;
    }

    cout << "Files opened. Reading..." << endl;

    unsigned char * image = new unsigned char[images.getRows()*images.getCols()];
    unsigned char label;

    const int trainingSamplesCount = images.getLength();

    double * trainingInputs = new double[trainingSamplesCount*inputs_count];
    double * trainingOutputs = new double[trainingSamplesCount];

    for(int i=0;i<trainingSamplesCount;i++)
    {
        images.readRecord(image);
        labels.readRecord(&label);

        for(int j=0;j<inputs_count;j++)
        {
            trainingInputs[i*inputs_count+j] = ((double) image[j] - 127.5) / 127.5;
        }

        trainingOutputs[i] = ((double) label - 4.5) / 4.5;
    }

    images.closeFile();
    labels.closeFile();

    cout << "Training..." << endl;

    opennnl->trainingIDBD(trainingSamplesCount, trainingInputs, trainingOutputs, 1000, SPEED, ERROR);

    delete trainingInputs;
    delete trainingOutputs;

    if(!images.openFile("../mnist/data/t10k-images.idx3-ubyte"))
    {
        cout << "Couldn't find test images file" << endl;
        return;
    }

    if(!labels.openFile("../mnist/data/t10k-labels.idx1-ubyte"))
    {
        cout << "Couldn't find test labels file" << endl;
        return;
    }

    const int testSamplesCount = images.getLength();

    double * testInputs = new double[inputs_count];
    double * testOutputs = new double[1];
    int outputLabel, correctAnswers=0;

    cout << "Testing..." << endl;

    for(int i=0;i<testSamplesCount;i++)
    {
        images.readRecord(image);
        labels.readRecord(&label);

        for(int j=0;j<inputs_count;j++)
        {
            testInputs[j] = ((double) image[j] - 127.5) / 127.5;
        }

        opennnl->calculate(testInputs);
        opennnl->getOutputs(testOutputs);

        outputLabel = round(testOutputs[0] * 4.5 + 4.5);
        if(outputLabel == label)
            correctAnswers++;
        else
        {
            cout << "Incorrect answer: " << outputLabel << " instead: " << label << endl;
        }
    }

    images.closeFile();
    labels.closeFile();

    delete image;

    delete testInputs;
    delete testOutputs;

    cout << endl;
    cout << "Correct answers " << correctAnswers << " from " << testSamplesCount << " labels" << endl;
    cout << "Error rate: " << correctAnswers / testSamplesCount * 100 << "%" << endl;

    delete opennnl;
}
