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

inline int OpenNNL::indexByLayerAndNeuron(int layer, int neuron)
{
    return _neuronsInPreviousLayers[layer] + neuron;
}

inline int OpenNNL::indexByLayerNeuronAndInput(int layer, int neuron, int input)
{
    return _inputsInPreviousLayers[layer] + neuron*_inputsInCurrentLayer[layer] + input;
}

inline void OpenNNL::setB(int layer, int neuron, int input, double value)
{
    _Bs[indexByLayerNeuronAndInput(layer, neuron, input)] = value;
}

inline double OpenNNL::getB(int layer, int neuron, int input)
{
    return _Bs[indexByLayerNeuronAndInput(layer, neuron, input)];
}

inline void OpenNNL::setBForBias(int layer, int neuron, double value)
{
    _BsForBias[indexByLayerAndNeuron(layer, neuron)] = value;
}

inline double OpenNNL::getBForBias(int layer, int neuron)
{
    return _BsForBias[indexByLayerAndNeuron(layer, neuron)];
}

inline void OpenNNL::setH(int layer, int neuron, int input, double value)
{
    _Hs[indexByLayerNeuronAndInput(layer, neuron, input)] = value;
}

inline double OpenNNL::getH(int layer, int neuron, int input)
{
    return _Hs[indexByLayerNeuronAndInput(layer, neuron, input)];
}

inline void OpenNNL::setHForBias(int layer, int neuron, double value)
{
    _HsForBias[indexByLayerAndNeuron(layer, neuron)] = value;
}

inline double OpenNNL::getHForBias(int layer, int neuron)
{
    return _HsForBias[indexByLayerAndNeuron(layer, neuron)];
}

inline void OpenNNL::setWeight(int layer, int neuron, int input, double value)
{
    _neuronsInputsWeights[indexByLayerNeuronAndInput(layer, neuron, input)] = value;
}

inline double OpenNNL::getWeight(int layer, int neuron, int input)
{
    return _neuronsInputsWeights[indexByLayerNeuronAndInput(layer, neuron, input)];
}

inline void OpenNNL::setBias(int layer, int neuron, double value)
{
    _neuronsBiases[indexByLayerAndNeuron(layer, neuron)] = value;
}

inline double OpenNNL::getBias(int layer, int neuron)
{
    return _neuronsBiases[indexByLayerAndNeuron(layer, neuron)];
}

inline void OpenNNL::setDerivative(int layer, int neuron, double value)
{
    _derivatives[indexByLayerAndNeuron(layer, neuron)] = value;
}

inline double OpenNNL::getDerivative(int layer, int neuron)
{
    return _derivatives[indexByLayerAndNeuron(layer, neuron)];
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

    /*for(int i=0;i<_weightsCount;i++)
    {
        _neuronsInputsWeights[i] = unified_random();
    }*/

    int inputs = _inputsCount;

    for(int i=0;i<_layersCount;i++)
    {
        for(int j=0;j<inputs*_neuronsPerLayerCount[i];j++)
        {
            _neuronsInputsWeights[_inputsInPreviousLayers[i]+j] = unified_random() / sqrt(inputs);
        }

        inputs = _neuronsPerLayerCount[i];
    }

}

void OpenNNL::randomizeBiases()
{
    initialize_random_generator();

    /*for(int i=0;i<_neuronsCount;i++)
    {
        _neuronsBiases[i] = unified_random();
    }*/
    int inputs = _inputsCount;

    for(int i=0;i<_layersCount;i++)
    {
        for(int j=0;j<_neuronsPerLayerCount[i];j++)
        {
            _neuronsBiases[_neuronsInPreviousLayers[i]+j] = unified_random() / sqrt(inputs);
        }

        inputs = _neuronsPerLayerCount[i];
    }
}

void OpenNNL::randomizeWeightsAndBiases()
{
    this->randomizeWeights();
    this->randomizeBiases();
}

/*****************************************************************************/
/* Вычислить активационную функцию y(x) = 2x / (1 + abs(x)). */
/*****************************************************************************/
inline double OpenNNL::activation(double x, TActivationKind kind)
{
    return ((kind == SIG) ? (2.0 * x / (1 + fabs(x))):x);
}

/*****************************************************************************/
/* Вычислить производную активационной функции y(x) по формуле:
   dy(x)         2.0
   ----- = ---------------.
    dx     (1 + abs(x))^2
*/
/*****************************************************************************/
inline double OpenNNL::activation_derivative(double x, TActivationKind kind)
{
    double temp = 1.0 + fabs(x);
    return ((kind == SIG) ? (2.0 / (temp * temp)):1.0);
}

double * OpenNNL::_calculateSingle(double *inputs)
{
    _doCalculation(inputs, _outputs);

    return _outputs;
}

void OpenNNL::_doCalculation(double *inpt, double * outputs)
{
    int inputsCount;
    double * temp;
    double * inputs = new double[_inputsCount];

    memcpy(inputs, inpt, sizeof(*inputs)*_inputsCount);

    inputsCount = _inputsCount;

    for(int i=0;i<_layersCount;i++)
    {
        temp = new double[_neuronsPerLayerCount[i]*inputsCount];
        for(int j=0;j<_neuronsPerLayerCount[i];j++)
        {
            for(int k=0;k<inputsCount;k++)
            {
                temp[j*inputsCount+k] = inputs[k] * _neuronsInputsWeights[indexByLayerNeuronAndInput(i, j, k)];
            }
        }

        delete[] inputs;

        inputs = new double[_neuronsPerLayerCount[i]];

        for(int j=0;j<_neuronsPerLayerCount[i];j++)
        {
            inputs[j] = 0;

            for(int k=0;k<inputsCount;k++)
            {
                inputs[j] += temp[j*inputsCount+k];
            }

            inputs[j]  -= _neuronsBiases[indexByLayerAndNeuron(i, j)];

            inputs[j] = activation(inputs[j]);

        }

        inputsCount = _neuronsPerLayerCount[i];
        delete[] temp;
    }

    memcpy(outputs, inputs, sizeof(*_outputs)*inputsCount);

    delete[] inputs;
}

double * OpenNNL::calculate(double *inputs)
{
    if(inputs)
    {
        memcpy(_inputs, inputs, _inputsCount*sizeof(*_inputs));
    }

    return _calculateSingle(_inputs);
}

double * OpenNNL::calculateRef(double *inputs)
{
    if(!inputs)
        inputs = _inputs;

    return _calculateSingle(inputs);
}

void OpenNNL::calculate(double * inputs, double * outputs, int samplesCount)
{
    for(int i=0;i<samplesCount;i++)
    {
        _doCalculation(inputs+i*_inputsCount, outputs+i*_outputsCount);
    }
}

void OpenNNL::calculateNeuronsOutputsAndDerivatives(double *inpt, double *outputs, double *derivatives)
{
    int inputsCount, neuronIndex = 0;
    double * temp;
    double * inputs = new double[_inputsCount];

    memcpy(inputs, inpt, sizeof(*inputs)*_inputsCount);

    inputsCount = _inputsCount;

    for(int i=0;i<_layersCount;i++)
    {
        temp = new double[_neuronsPerLayerCount[i]*inputsCount];
        for(int j=0;j<_neuronsPerLayerCount[i];j++)
        {
            for(int k=0;k<inputsCount;k++)
            {
                temp[j*inputsCount+k] = inputs[k] * _neuronsInputsWeights[indexByLayerNeuronAndInput(i, j, k)];
            }
        }

        delete[] inputs;

        inputs = new double[_neuronsPerLayerCount[i]];

        for(int j=0;j<_neuronsPerLayerCount[i];j++)
        {
            double tempInput = 0;

            for(int k=0;k<inputsCount;k++)
            {
                tempInput += temp[j*inputsCount+k];
            }

            tempInput -= _neuronsBiases[indexByLayerAndNeuron(i, j)];

            outputs[neuronIndex] = inputs[j] = activation(tempInput);
            derivatives[neuronIndex] = activation_derivative(tempInput);

            neuronIndex++;
        }

        inputsCount = _neuronsPerLayerCount[i];

        delete[] temp;
    }

    delete[] inputs;
}

double OpenNNL::_changeWeightsByBP(double * trainingInputs, double *trainingOutputs, double speed, double sample_weight)
{
    double error = 0, current_error;
    double * localGradients = new double[_neuronsCount];
    double * outputs = new double[_neuronsCount];
    double * derivatives = new double[_neuronsCount];

    calculateNeuronsOutputsAndDerivatives(trainingInputs, outputs, derivatives);

    for(int j=0;j<_neuronsPerLayerCount[_layersCount-1];j++) // cuda kernel
    {
        current_error = trainingOutputs[j] - outputs[indexByLayerAndNeuron(_layersCount-1, j)];
        localGradients[indexByLayerAndNeuron(_layersCount-1, j)] = current_error * sample_weight * derivatives[indexByLayerAndNeuron(_layersCount-1, j)];

        error += current_error * current_error;
    }

    if(_layersCount > 1)
    {
        for(int i=_layersCount-2;i>=0;i--)
        {
            for(int j=0;j<_neuronsPerLayerCount[i];j++) // cuda kernel
            {
                localGradients[indexByLayerAndNeuron(i, j)] = 0;

                for(int k=0;k<_neuronsPerLayerCount[i+1];k++)
                {
                    localGradients[indexByLayerAndNeuron(i, j)] += _neuronsInputsWeights[indexByLayerNeuronAndInput(i+1, k, j)]
                                                                    * localGradients[indexByLayerAndNeuron(i+1, k)];
                }

                localGradients[indexByLayerAndNeuron(i, j)] *= derivatives[indexByLayerAndNeuron(i, j)];
            }
        }
    }

    for(int j=0;j<_neuronsPerLayerCount[0];j++) // this and next cicle for cuda kernel (j*k threads)
    {
        for(int k=0;k<_inputsCount;k++)
        {
            _neuronsInputsWeights[indexByLayerNeuronAndInput(0, j, k)] += speed * localGradients[indexByLayerAndNeuron(0, j)] * trainingInputs[k];
        }

        _neuronsBiases[indexByLayerAndNeuron(0, j)] -= speed * localGradients[indexByLayerAndNeuron(0, j)];
    }

    for(int i=1;i<_layersCount;i++) // try to parallelize all three cicles in one kernel. If it's impossible, only two inner
    {
        for(int j=0;j<_neuronsPerLayerCount[i];j++)
        {
            for(int k=0;k<_neuronsPerLayerCount[i-1];k++)
            {
                _neuronsInputsWeights[indexByLayerNeuronAndInput(i, j, k)] += speed * localGradients[indexByLayerAndNeuron(i, j)] * outputs[indexByLayerAndNeuron(i-1, k)];
            }

            _neuronsBiases[indexByLayerAndNeuron(i, j)] -= speed * localGradients[indexByLayerAndNeuron(i, j)];
        }
    }

    delete[] localGradients;
    delete[] outputs;
    delete[] derivatives;

    error /= 2;
    return error;
}

double OpenNNL::_changeWeightsByIDBD(double * trainingInputs, double *trainingOutputs, double speed, double sample_weight)
{
    double error = 0, current_error;
    double cur_rate, delta, deltaB, deltaH;
    double * localGradients = new double[_neuronsCount];
    double * outputs = new double[_neuronsCount];
    double * derivatives = new double[_neuronsCount];

    calculateNeuronsOutputsAndDerivatives(trainingInputs, outputs, derivatives);

    for(int j=0;j<_neuronsPerLayerCount[_layersCount-1];j++)
    {
        current_error = trainingOutputs[j] - outputs[indexByLayerAndNeuron(_layersCount-1, j)];
        localGradients[indexByLayerAndNeuron(_layersCount-1, j)] = current_error * sample_weight * derivatives[indexByLayerAndNeuron(_layersCount-1, j)];

        error += current_error * current_error;
    }

    if(_layersCount > 1)
    {
        for(int i=_layersCount-2;i>=0;i--)
        {
            for(int j=0;j<_neuronsPerLayerCount[i];j++)
            {
                localGradients[indexByLayerAndNeuron(i, j)] = 0;

                for(int k=0;k<_neuronsPerLayerCount[i+1];k++)
                {
                    localGradients[indexByLayerAndNeuron(i, j)] += _neuronsInputsWeights[indexByLayerNeuronAndInput(i+1, k, j)]
                                                                    * localGradients[indexByLayerAndNeuron(i+1, k)];
                }

                localGradients[indexByLayerAndNeuron(i, j)] *= derivatives[indexByLayerAndNeuron(i, j)];
            }
        }
    }

    for(int j=0;j<_neuronsPerLayerCount[0];j++)
    {
        for(int k=0;k<_inputsCount;k++)
        {
            deltaB = speed * localGradients[indexByLayerAndNeuron(0, j)] * trainingInputs[k] * getH(0, j, k);

            if (deltaB > 2.0)
            {
                deltaB = 2.0;
            }
            else
            {
                if (deltaB < -2.0)
                {
                    deltaB = -2.0;
                }
            }

            setB(0, j, k, getB(0, j, k) + deltaB);
            cur_rate = exp(getB(0, j, k));
            delta = cur_rate * localGradients[indexByLayerAndNeuron(0, j)] * trainingInputs[k];

            _neuronsInputsWeights[indexByLayerNeuronAndInput(0, j, k)] += delta;

            deltaH = 1 - cur_rate * trainingInputs[k] * trainingInputs[k];
            if(deltaH <= 0)
                setH(0, j, k, delta);
            else
                setH(0, j, k, getH(0, j, k) * deltaH + delta);
        }

        deltaB = speed * localGradients[indexByLayerAndNeuron(0, j)] * getHForBias(0, j);

        if (deltaB > 2.0)
        {
            deltaB = 2.0;
        }
        else
        {
            if (deltaB < -2.0)
            {
                deltaB = -2.0;
            }
        }

        setBForBias(0, j, getBForBias(0, j) - deltaB);
        cur_rate = exp(getBForBias(0, j));
        delta = cur_rate * localGradients[indexByLayerAndNeuron(0, j)];

        _neuronsBiases[indexByLayerAndNeuron(0, j)] -= delta;

        deltaH = 1 - cur_rate;
        if(deltaH <= 0)
            setHForBias(0, j, -delta);
        else
            setHForBias(0, j, getHForBias(0, j) * deltaH - delta);
    }

    for(int i=1;i<_layersCount;i++)
    {
        for(int j=0;j<_neuronsPerLayerCount[i];j++)
        {
            for(int k=0;k<_neuronsPerLayerCount[i-1];k++)
            {
                deltaB = speed * localGradients[indexByLayerAndNeuron(i, j)] * outputs[indexByLayerAndNeuron(i-1, k)] * getH(i, j, k);

                if (deltaB > 2.0)
                {
                    deltaB = 2.0;
                }
                else
                {
                    if (deltaB < -2.0)
                    {
                        deltaB = -2.0;
                    }
                }

                setB(i, j, k, getB(i, j, k) + deltaB);
                cur_rate = exp(getB(i, j, k));
                delta = cur_rate * localGradients[indexByLayerAndNeuron(i, j)] * outputs[indexByLayerAndNeuron(i-1, k)];

                _neuronsInputsWeights[indexByLayerNeuronAndInput(i, j, k)] += delta;

                deltaH = 1 - cur_rate * outputs[indexByLayerAndNeuron(i-1, k)] * outputs[indexByLayerAndNeuron(i-1, k)];
                if(deltaH <= 0)
                    setH(i, j, k, delta);
                else
                    setH(i, j, k, getH(i, j, k) * deltaH + delta);
            }

            deltaB = speed * localGradients[indexByLayerAndNeuron(i, j)] * getHForBias(i, j);

            if (deltaB > 2.0)
            {
                deltaB = 2.0;
            }
            else
            {
                if (deltaB < -2.0)
                {
                    deltaB = -2.0;
                }
            }

            setBForBias(i, j, getBForBias(i, j) - deltaB);
            cur_rate = exp(getBForBias(i, j));
            delta = cur_rate * localGradients[indexByLayerAndNeuron(i, j)];

            _neuronsBiases[indexByLayerAndNeuron(i, j)] -= delta;

            deltaH = 1 - cur_rate;
            if(deltaH <= 0)
                setHForBias(i, j, -delta);
            else
                setHForBias(i, j, getHForBias(i, j) * deltaH - delta);
        }
    }

    delete[] localGradients;
    delete[] outputs;
    delete[] derivatives;

    error /= 2;
    return error;
}

bool OpenNNL::_doEpochBP(int samplesCount, double * trainingInputs, double * trainingOutputs, int numEpoch, double speed, double minError)
{
    double error = 0;
    double * currentSampleInputs = new double[_inputsCount];
    double * currentSampleOutputs = new double[_outputsCount];

    for(int sample=0;sample<samplesCount;sample++)
    {
        cout << "Epoch: " << numEpoch << ", Sample: " << sample << endl;
        memcpy(currentSampleInputs, trainingInputs+sample*_inputsCount, _inputsCount*sizeof(*currentSampleInputs));
        memcpy(currentSampleOutputs, trainingOutputs+sample*_outputsCount, _outputsCount*sizeof(*currentSampleOutputs));

        error = _changeWeightsByBP(currentSampleInputs, currentSampleOutputs, speed, 1);

        cout << "Error: " << error << endl;
    }

    delete[] currentSampleInputs;
    delete[] currentSampleOutputs;

    return (error <= minError);
}

bool OpenNNL::_doEpochIDBD(int samplesCount, double * trainingInputs, double * trainingOutputs, int numEpoch, double speed, double minError)
{
    double error = 0;
    double * currentSampleInputs = new double[_inputsCount];
    double * currentSampleOutputs = new double[_outputsCount];

    for(int sample=0;sample<samplesCount;sample++)
    {
        cout << "Sample: " << sample << endl;
        memcpy(currentSampleInputs, trainingInputs+sample*_inputsCount, _inputsCount*sizeof(*currentSampleInputs));
        memcpy(currentSampleOutputs, trainingOutputs+sample*_outputsCount, _outputsCount*sizeof(*currentSampleOutputs));

        error = _changeWeightsByIDBD(currentSampleInputs, currentSampleOutputs, speed, 1);
    }

    delete[] currentSampleInputs;
    delete[] currentSampleOutputs;

    return (error <= minError);
}

void OpenNNL::_trainingBP(int samplesCount, double * trainingInputs, double * trainingOutputs, int maxEpochsCount, double speed, double error)
{
    for(int i=0;i<maxEpochsCount;i++)
    {
        if(_doEpochBP(samplesCount, trainingInputs, trainingOutputs, i, speed, error))
        {
            break;
        }
    }
}


void OpenNNL::_trainingIDBD(int samplesCount, double * trainingInputs, double * trainingOutputs, int maxEpochsCount, double speed, double error)
{
    for(int i=0;i<maxEpochsCount;i++)
    {
        if(_doEpochIDBD(samplesCount, trainingInputs, trainingOutputs, i, speed, error))
        {
            break;
        }
    }
}

void OpenNNL::trainingBP(int samplesCount, double * trainingInputs, double *trainingOutputs, int maxEpochsCount, double speed, double error)
{
    _trainingBP(samplesCount, trainingInputs, trainingOutputs, maxEpochsCount, speed, error);
}

void OpenNNL::trainingIDBD(int samplesCount, double * trainingInputs, double *trainingOutputs, int maxEpochsCount, double speed, double error)
{
    _Bs = new double[_weightsCount];
    _Hs = new double[_weightsCount];

    _BsForBias = new double[_neuronsCount];
    _HsForBias = new double[_neuronsCount];

    resetHsAndHsForBias();
    randomizeBsAndBsForBias();

    _trainingIDBD(samplesCount, trainingInputs, trainingOutputs, maxEpochsCount, speed, error);

    delete[] _Bs;
    delete[] _Hs;
    delete[] _BsForBias;
    delete[] _HsForBias;
}

void OpenNNL::getOutputs(double * out)
{
    memcpy(out, _outputs, sizeof(*out)*_outputsCount);
}

void OpenNNL::resetHs()
{
    for(int i=0;i<_weightsCount;i++)
        _Hs[i] = 0;
}

void OpenNNL::resetHsForBias()
{
    for(int i=0;i<_neuronsCount;i++)
        _HsForBias[i] = 0;
}

void OpenNNL::resetHsAndHsForBias()
{
    resetHs();
    resetHsForBias();
}

void OpenNNL::randomizeBs()
{
    initialize_random_generator();
    for(int i=0;i<_weightsCount;i++)
        _Bs[i] = unified_random();
}

void OpenNNL::randomizeBsForBias()
{
    initialize_random_generator();
    for(int i=0;i<_neuronsCount;i++)
        _BsForBias[i] = unified_random();
}

void OpenNNL::randomizeBsAndBsForBias()
{
    randomizeBs();
    randomizeBsForBias();
}

void OpenNNL::setWeights(double *weights)
{
    memcpy(_neuronsInputsWeights, weights, _weightsCount*sizeof(double));
}

void OpenNNL::setBiases(double *biases)
{
    memcpy(_neuronsBiases, biases, _neuronsCount*sizeof(double));
}

void OpenNNL::setWeightsAndBiases(double *weights, double *biases)
{
    setWeights(weights);
    setBiases(biases);
}
