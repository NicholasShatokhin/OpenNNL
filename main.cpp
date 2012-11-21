#include <iostream>
#include <cmath>
#include <iomanip>
#include <time.h>

#include "opennnl.h"
#include "mnistfile.h"

using namespace std;

#define REAL double

int maxArrayElementsIndex(REAL array[], int count);

void testNetwork1();
void testNetworkMnist(int epochCount);
void prepareMnistDataAndTestNetwork();

void startTimer(struct timespec * tp)
{
    clock_gettime(CLOCK_MONOTONIC_RAW, tp);
}

void printTimerValue(struct timespec * tp, const char * message = "")
{

    long int startTime = tp->tv_sec * 1000000000 + tp->tv_nsec;

    clock_gettime(CLOCK_MONOTONIC_RAW, tp);

    cout << message << ": time: " << tp->tv_sec * 1000000000 + tp->tv_nsec - startTime << " ns" << endl;

    ofstream fout;
    fout.open("timing.txt", ios_base::out | ios_base::app);
    fout << message << ": time: " << tp->tv_sec * 1000000000 + tp->tv_nsec - startTime << " ns" << endl;
    fout.close();
}

int main()
{
    prepareMnistDataAndTestNetwork();

    return 0;
}

void testNetwork1()
{
    struct timespec tp;

    const int INPUTS_COUNT = 3;
    const int LAYERS_COUNT = 4;
    const int OUTPUTS_COUNT = 2;

    const int TRAINING_SAMPLES_COUNT = 21;
    const REAL SPEED = 0.015;
    const REAL ERROR = 0.005;

    const int TEST_INPUTS_COUNT = 9;

    int neuronsInLayers[LAYERS_COUNT] = {3, 10, 10, 2};
    REAL trainingInputs[TRAINING_SAMPLES_COUNT*INPUTS_COUNT] = {1.0, 	1.0, 	1.0,
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
    REAL trainingOutputs[TRAINING_SAMPLES_COUNT*OUTPUTS_COUNT] = {
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

    REAL testInputs[TEST_INPUTS_COUNT*INPUTS_COUNT] = {
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

    REAL weights[159];
    REAL biases[25];

    for(int i=0;i<159;i++)
        weights[i] = 0.1;

    for(int i=0;i<25;i++)
        biases[i] = 0.1;


    startTimer(&tp);

    cout << "Creating object..." << endl;
    OpenNNL * opennnl = new OpenNNL(INPUTS_COUNT, LAYERS_COUNT, neuronsInLayers);

    printTimerValue(&tp);

    cout << "Randomizing weights..." << endl;
    //opennnl->randomizeWeights();
    opennnl->setWeights(weights);

    cout << "Randomizing biases..." << endl;
    //opennnl->randomizeBiases();
    opennnl->setBiases(biases);

    cout << "Training..." << endl;

    printTimerValue(&tp);

    opennnl->trainingBP(TRAINING_SAMPLES_COUNT, trainingInputs, trainingOutputs, 1, SPEED, ERROR);

    printTimerValue(&tp);

    opennnl->printDebugInfo();

    cout << "Calculations..." << endl;

    REAL inputs[INPUTS_COUNT];
    REAL outputs[OUTPUTS_COUNT];

    printTimerValue(&tp);

    for(int i=0;i<TEST_INPUTS_COUNT;i++)
    {
        memcpy(inputs, testInputs+i*INPUTS_COUNT, INPUTS_COUNT*sizeof(REAL));

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

    printTimerValue(&tp);

    cout << "Deleting object..." << endl;
    delete opennnl;

    printTimerValue(&tp);

    cout << "Done!" << endl;
}

void testNetworkMnist(int epochCount, int trainingSamplesCount, REAL * trainingInputs, REAL * trainingOutputs, int testSamplesCount, REAL * testInputs, REAL * testOutputs, unsigned char * labels, int layers_count, int inputs_count, int outputs_count, int * neuronsInLayers)
{
    cout << "Creating network..." << endl;

    struct timespec tp;
    const REAL error = 0.005;
    const REAL speed = 1 / (REAL) trainingSamplesCount;
    int outputLabel, correctAnswers = 0;

    startTimer(&tp);

    OpenNNL * opennnl = new OpenNNL(inputs_count, layers_count, neuronsInLayers);

    printTimerValue(&tp, "Creating network");

    opennnl->randomizeWeightsAndBiases();

    printTimerValue(&tp, "randomizeWeightsAndBiases");

    opennnl->trainingBP(trainingSamplesCount, trainingInputs, trainingOutputs, epochCount, speed, error);

    printTimerValue(&tp, "Training");

    opennnl->calculate(testInputs, testOutputs, testSamplesCount);

    printTimerValue(&tp, "Testing");

    for(int i=0;i<testSamplesCount;i++)
    {
        outputLabel = maxArrayElementsIndex(testOutputs+i*outputs_count, outputs_count);
        if(outputLabel == (int) labels[i])
            correctAnswers++;
        else
        {
            cout << i << ": Incorrect answer: " << outputLabel << " instead: " << (int) labels[i] << endl;

            cout << "Outputs: " << setprecision(6);
            for(int k=0;k<outputs_count;k++)
            {
                cout << testOutputs[i*outputs_count+k] << " ";
            }
            cout << endl;
        }
    }

    cout << endl;
    cout << "Correct answers " << correctAnswers << " from " << testSamplesCount << " labels" << endl;
    REAL error_rate = 100.0 - ((REAL) correctAnswers) / ((REAL) testSamplesCount) * 100.0;
    cout << "Error rate: " << error_rate << "%" << endl;

    ofstream fout;
    fout.open("timing.txt", ios_base::out | ios_base::app);
    cout << "Error rate: " << error_rate << "%" << endl;
    fout.close();

    printTimerValue(&tp, "Check");

    delete opennnl;

    printTimerValue(&tp, "Delete");
}

void prepareMnistDataAndTestNetwork()
{
    const int inputs_count = 784;
    const int outputs_count = 10;

    MnistFile images;
    MnistFile labels;
    if(!images.openFile("../OpenNNLCuda/data/mnist/train-images.idx3-ubyte"))
    {
        cout << "Couldn't find train images file" << endl;
        return;
    }

    if(!labels.openFile("../OpenNNLCuda/data/mnist/train-labels.idx1-ubyte"))
    {
        cout << "Couldn't find train labels file" << endl;
        return;
    }

    cout << "Files opened. Reading..." << endl;

    unsigned char * image = new unsigned char[images.getRows()*images.getCols()];
    unsigned char label;

    const int trainingSamplesCount = images.getLength();

    REAL * trainingInputs = new REAL[trainingSamplesCount*inputs_count];
    REAL * trainingOutputs = new REAL[trainingSamplesCount*outputs_count];

    for(int i=0;i<trainingSamplesCount;i++)
    {
        images.readRecord(image);
        labels.readRecord(&label);

        for(int j=0;j<inputs_count;j++)
        {
            trainingInputs[i*inputs_count+j] = ((REAL) image[j] - 127.5) / 127.5;
        }

        for(int k=0;k<label;k++)
            trainingOutputs[i*outputs_count+k] = -1;
        trainingOutputs[i*outputs_count+label] = 1;
        for(int k=label+1;k<outputs_count;k++)
            trainingOutputs[i*outputs_count+k] = -1;
    }

    images.closeFile();
    labels.closeFile();

    if(!images.openFile("../OpenNNLCuda/data/mnist/t10k-images.idx3-ubyte"))
    {
        cout << "Couldn't find test images file" << endl;
        return;
    }

    if(!labels.openFile("../OpenNNLCuda/data/mnist/t10k-labels.idx1-ubyte"))
    {
        cout << "Couldn't find test labels file" << endl;
        return;
    }

    const int testSamplesCount = images.getLength();

    unsigned char * testLabels = new unsigned char[testSamplesCount];

    REAL * testInputs = new REAL[inputs_count*testSamplesCount];
    REAL * testOutputs = new REAL[outputs_count*testSamplesCount];


    for(int i=0;i<testSamplesCount;i++)
    {
        images.readRecord(image);
        labels.readRecord(&testLabels[i]);

        for(int j=0;j<inputs_count;j++)
        {
            testInputs[i*inputs_count+j] = ((REAL) image[j] - 127.5) / 127.5;
        }
    }

    images.closeFile();
    labels.closeFile();

    // 300
    int layers_count = 2;
    int neuronsInLayers1[2] = {300, outputs_count};

    ofstream fout;
    fout.open("timing.txt", ios_base::out | ios_base::app);
    fout << "Layers: " << layers_count << "; 1 hidden layer: 300 neurons"  << endl;
    fout.close();

    for(int i=1;i<=10;i++)
    {
        fout.open("timing.txt", ios_base::out | ios_base::app);
        fout << "Epochs: " << i << endl;
        fout.close();

        testNetworkMnist(i, trainingSamplesCount, trainingInputs, trainingOutputs, testSamplesCount, testInputs, testOutputs, testLabels, layers_count, inputs_count, outputs_count, neuronsInLayers1);
    }

    // 300x100
    layers_count = 3;
    int neuronsInLayers2[3] = {300, 100, outputs_count};

    fout.open("timing.txt", ios_base::out | ios_base::app);
    fout << "Layers: " << layers_count << "; 2 hidden layers: 300 and 100 neurons"  << endl;
    fout.close();

    for(int i=1;i<=10;i++)
    {
        fout.open("timing.txt", ios_base::out | ios_base::app);
        fout << "Epochs: " << i << endl;
        fout.close();

        testNetworkMnist(i, trainingSamplesCount, trainingInputs, trainingOutputs, testSamplesCount, testInputs, testOutputs, testLabels, layers_count, inputs_count, outputs_count, neuronsInLayers2);
    }

    // 500x150
    layers_count = 3;
    int neuronsInLayers3[3] = {500, 150, outputs_count};

    fout.open("timing.txt", ios_base::out | ios_base::app);
    fout << "Layers: " << layers_count << "; 2 hidden layers: 500 and 150 neurons"  << endl;
    fout.close();

    for(int i=1;i<=10;i++)
    {
        fout.open("timing.txt", ios_base::out | ios_base::app);
        fout << "Epochs: " << i << endl;
        fout.close();

        testNetworkMnist(i, trainingSamplesCount, trainingInputs, trainingOutputs, testSamplesCount, testInputs, testOutputs, testLabels, layers_count, inputs_count, outputs_count, neuronsInLayers3);
    }

    // 500x300
    layers_count = 3;
    int neuronsInLayers4[3] = {500, 300, outputs_count};

    fout.open("timing.txt", ios_base::out | ios_base::app);
    fout << "Layers: " << layers_count << "; 2 hidden layers: 500 and 300 neurons"  << endl;
    fout.close();

    for(int i=1;i<=10;i++)
    {
        fout.open("timing.txt", ios_base::out | ios_base::app);
        fout << "Epochs: " << i << endl;
        fout.close();

        testNetworkMnist(i, trainingSamplesCount, trainingInputs, trainingOutputs, testSamplesCount, testInputs, testOutputs, testLabels, layers_count, inputs_count, outputs_count, neuronsInLayers4);
    }

    delete trainingInputs;
    delete trainingOutputs;

    delete image;
    delete testLabels;

    delete testInputs;
    delete testOutputs;
}

int maxArrayElementsIndex(REAL array[], int count)
{
    REAL maxElement = -10;
    int index = 0;

    for(int i=0;i<count;i++)
    {
        if(array[i] > maxElement)
        {
            maxElement = array[i];
            index = i;
        }
    }

    return index;
}
