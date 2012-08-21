#ifndef _OPENNNL_H_
#define _OPENNNL_H_

#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>

#include "utils.h"

using namespace std;

typedef enum {LIN, SIG} TActivationKind;

class OpenNNL
{

    private:
        int _layersCount; // num of layers
        int * _neuronsPerLayerCount; // num of neurons in each layer
        int _inputsCount;    // num of network inputs
        int _weightsCount;   // num of weights of all neurons in network
        int _neuronsCount;   // num of all neurons in network (and also num of biases count)
        int _outputsCount;   // num of network outputs

        int * _neuronsInPreviousLayers; // the sum of the number of neurons in previous layers
        int * _inputsInPreviousLayers; // the sum of the inputs of each neuron in previous layers
        int * _inputsInCurrentLayer; // the inputs of each neuron in current layer (not sum)

        double * _neuronsInputsWeights;  // weights of neurons inputs
        double * _neuronsBiases; // biases of neurons

        double * _inputs;    // inputs of network
        double * _outputs;   // outputs of network
        double * _derivatives; // derivatives of output of each neuron

        double * _Bs;    // B for IDBD training
        double * _BsForBias;    // B for IDBD training
        double * _Hs;    // H for IDBD training
        double * _HsForBias;    // H for IDBD training

        /*double activation(double output, int type); // activation function
        double derivative(double x, int type);  // derivative of activation function

        inline double sigmoid(double output, double a);
        inline double sigmoid_simple(double output);*/

        void calculateNeuronsOutputsAndDerivatives(double * inputs=NULL, double * outputs=NULL, double * derivatives=NULL); // calculates neurons outputs and derivatives for training functions

        inline void setB(int layer, int neuron, int input, double value);  // set B for current neuron's input
        inline double getB(int layer, int neuron, int input);  // get B of current neuron's input

        inline void setBForBias(int layer, int neuron, double value);  // set B for current neuron's bias
        inline double getBForBias(int layer, int neuron);  // get B of current neuron's bias

        inline void setH(int layer, int neuron, int input, double value); // set H for current neuron's input
        inline double getH(int layer, int neuron, int input);  // get H of current neuron's input

        inline void setHForBias(int layer, int neuron, double value); // set H for current neuron's input
        inline double getHForBias(int layer, int neuron);  // get H of current neuron's input

        inline void setWeight(int layer, int neuron, int input, double value); // set weight for current neuron's input
        inline double getWeight(int layer, int neuron, int input); // get weight current neuron's input

        inline void setBias(int layer, int neuron, double value);  // set bias for current neuron
        inline double getBias(int layer, int neuron);  // get bias of current neuron

        inline void setDerivative(int layer, int neuron, double value); // sets neuron's derivative value
        inline double getDerivative(int layer, int neuron); // gets neuron's derivative value

        void resetHs();
        void resetHsForBias();
        void resetHsAndHsForBias();

        void randomizeBs();
        void randomizeBsForBias();
        void randomizeBsAndBsForBias();

        inline int indexByLayerAndNeuron(int layer, int neuron);
        inline int indexByLayerNeuronAndInput(int layer, int neuron, int input);

        inline double activation(double x, TActivationKind kind=SIG);
        inline double activation_derivative(double x, TActivationKind kind=SIG);

        double * _calculateWorker(double * inputs); // worker for calculation network outputs
        double _changeWeightsByBP(double * trainingInputs, double * trainingOutputs, double speed, double sample_weight=1.0);
        double _changeWeightsByIDBD(double * trainingInputs, double *trainingOutputs, double speed, double sample_weight=1.0);

        double _doEpochBP(int samplesCount, double * trainingInputs, double * trainingOutputs, int numEpoch, double speed);
        double _doEpochIDBD(int samplesCount, double * trainingInputs, double * trainingOutputs, int numEpoch, double speed);
        void _trainingBP(int samplesCount, double * trainingInputs, double * trainingOutputs, int maxEpochsCount, double speed);
        void _trainingIDBD(int samplesCount, double * trainingInputs, double * trainingOutputs, int maxEpochsCount, double speed);

    public:
        OpenNNL(const int inptCount, const int lrCount, const int * neuronsInLayer);
        OpenNNL(const char * filename); // creates object and loads network and its parameters from file
        ~OpenNNL();

        void printDebugInfo();
        void randomizeWeights();    // randomizes neurons weights
        void randomizeBiases(); // randomizes neurons biases
        void randomizeWeightsAndBiases();   // randomizes weights and biases

        inline void setInput(int index, double value);  // sets value to input by index
        inline double getOutput(int index); // gets output by index

        void setWeights(double * weights);  // sets neurons weights from argument
        void setWeightsRef(double * weights);  // sets neurons weights by ref in argument (data must be alive while OpenNNL's object lives)
        void setBiases(double * biases);    // sets neurons biases from argument
        void setBiasesRef(double * biases);    // sets neurons biases by ref in argument (data must be alive while OpenNNL's object lives)
        void setWeightsAndBiases(double * weights, double * biases);    // sets neurons weights and biases from argument
        void setWeightsAndBiasesRef(double * weights, double * biases);    // sets neurons weights and biases by refs in arguments (data must be alive while OpenNNL's object lives)

        bool loadWeights(const char * filename);

        void loadNetwork(const char * filename);    // this function loads network and its parameters from file
        void saveNetwork(const char * filename);    // this function stores network and its parameters to file

        double * calculate(double * inputs=NULL);   // calculates network outputs and returns pointer to outputs array (copy 'inputs' data )
        double * calculateRef(double * inputs=NULL);    // calculates network outputs and returns pointer to outputs array (sets internal inputs array by 'inputs' ref - data must be alive while OpenNNL's object lives)

        /*void training(int trainingSetSize, double ** trainingInputs, double **trainingOutputs, double speed, double error, int maxEpochs);
        void trainingByFile(const char * filename, double speed, double error, int maxEpochs);
        void trainingByFileBatch(const char * filename, double speed, double error, int maxEpochs, int batchSize=0, int offset=0);*/

        void trainingBP(int samplesCount, double * trainingInputs, double *trainingOutputs, int maxEpochsCount, double speed, double error);
        void trainingIDBD(int samplesCount, double * trainingInputs, double *trainingOutputs, int maxEpochsCount, double speed, double error);

        void setInputs(double * in);    // copies in to inputs
        void setInputsRef(double * in);    // sets inputs by ref in argument (data must be alive while OpenNNL's object lives)

        void getOutputs(double * out);  // copies network outputs to out
        double * getOutputs();  // returns pointer to outputs array

};

#endif // _OPENNNL_H_
