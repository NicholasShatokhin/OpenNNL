#include "opennnl.h"

OpenNNL::OpenNNL(const int inputsCount, const int layersCount, const int * neuronsPerLayerCount)
{
    _inputsCount = inputsCount;
    _layersCount = layersCount;
    _weightsCount = 0;
    _neuronsCount = 0;

    _neuronsPerLayerCount = new int[_layersCount];
    _neuronsInPreviousLayers = new int[_layersCount];
    _inputsInPreviousLayers = new int[_layersCount];
    _inputsInCurrentLayer = new int[_layersCount];

    _inputs = new double[_inputsCount];

    int inputs = _inputsCount;

    for(int i=0;i<_layersCount;i++)
    {
        _neuronsInPreviousLayers[i] = _neuronsCount;
        _inputsInPreviousLayers[i] = _weightsCount;

        _inputsInCurrentLayer[i] = inputs;

        _weightsCount += neuronsPerLayerCount[i] * inputs;
        _neuronsCount += neuronsPerLayerCount[i];

        inputs = _neuronsPerLayerCount[i] = neuronsPerLayerCount[i];
    }

    _outputsCount = inputs;
    _outputs = new double[_outputsCount];

    _derivatives = new double[_neuronsCount];

    _neuronsInputsWeights = new double[_weightsCount];
    _neuronsBiases = new double[_neuronsCount];

    _Bs = new double[_weightsCount];
    _Hs = new double[_weightsCount];
}

OpenNNL::~OpenNNL()
{
    delete[] _neuronsPerLayerCount;
    delete[] _neuronsInPreviousLayers;
    delete[] _inputsInPreviousLayers;
    delete[] _inputsInCurrentLayer;
    delete[] _inputs;
    delete[] _outputs;
    delete[] _derivatives;
    delete[] _neuronsInputsWeights;
    delete[] _neuronsBiases;
    delete[] _Bs;
    delete[] _Hs;
}

void OpenNNL::printDebugInfo()
{
    printf("inputsCount=%d\n", _inputsCount);
    printf("outputsCount=%d\n", _outputsCount);
    printf("layersCount=%d\n", _layersCount);
    printf("neuronsCount=%d\n", _neuronsCount);
    printf("weightsCount=%d\n", _weightsCount);

    for(int i=0;i<_layersCount;i++)
    {
        printf("neurons in layer %d: %d\n", i, _neuronsPerLayerCount[i]);
        printf("neurons in all layers before %d: %d\n", i, _neuronsInPreviousLayers[i]);
        printf("inputs in all layers before %d: %d\n", i, _inputsInPreviousLayers[i]);
        printf("inputs of each neuron in layer %d: %d\n", i, _inputsInCurrentLayer[i]);
    }
}

inline void OpenNNL::setB(int layer, int neuron, int input, double value)
{
    _Bs[_inputsInPreviousLayers[layer] + neuron*_inputsInCurrentLayer[layer] + input] = value;
}

inline double OpenNNL::getB(int layer, int neuron, int input)
{
    return _Bs[_inputsInPreviousLayers[layer] + neuron*_inputsInCurrentLayer[layer] + input];
}

inline void OpenNNL::setH(int layer, int neuron, int input, double value)
{
    _Hs[_inputsInPreviousLayers[layer] + neuron*_inputsInCurrentLayer[layer] + input] = value;
}

inline double OpenNNL::getH(int layer, int neuron, int input)
{
    return _Hs[_inputsInPreviousLayers[layer] + neuron*_inputsInCurrentLayer[layer] + input];
}

inline void OpenNNL::setWeight(int layer, int neuron, int input, double value)
{
    _neuronsInputsWeights[_inputsInPreviousLayers[layer] + neuron*_inputsInCurrentLayer[layer] + input] = value;
}

inline double OpenNNL::getWeight(int layer, int neuron, int input)
{
    return _neuronsInputsWeights[_inputsInPreviousLayers[layer] + neuron*_inputsInCurrentLayer[layer] + input];
}

inline void OpenNNL::setBias(int layer, int neuron, double value)
{
    _neuronsBiases[_neuronsInPreviousLayers[layer] + neuron] = value;
}

inline double OpenNNL::getBias(int layer, int neuron)
{
    return _neuronsBiases[_neuronsInPreviousLayers[layer] + neuron];
}

inline void OpenNNL::setDerivative(int layer, int neuron, double value)
{
    _derivatives[_neuronsInPreviousLayers[layer] + neuron] = value;
}

inline double OpenNNL::getDerivative(int layer, int neuron)
{
    return _derivatives[_neuronsInPreviousLayers[layer] + neuron];
}

inline void OpenNNL::setInput(int index, double value)
{
    _inputs[index] = value;
}

inline double OpenNNL::getOutput(int index)
{
    return _outputs[index];
}

void OpenNNL::randomizeWeights()
{
    initialize_random_generator();

    for(int i=0;i<_weightsCount;i++)
    {
        _neuronsInputsWeights[i] = unified_random();
    }

}

void OpenNNL::randomizeBiases()
{
    initialize_random_generator();

    for(int i=0;i<_neuronsCount;i++)
    {
        _neuronsBiases[i] = unified_random();
    }
}

void OpenNNL::randomizeWeightsAndBiases()
{
    this->randomizeWeights();
    this->randomizeBiases();
}

double * OpenNNL::calculate(double * inputs)
{

}

void OpenNNL::calculateNeuronsOutputsAndDerivatives(double * inputs, double * outputs, double * derivatives)
{

}
