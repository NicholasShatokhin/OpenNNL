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

double OpenNNL::_changeWeightsByIDBD(double * trainingInputs, double *trainingOutputs, double sample_weight, double speed)
{
    int i, j, k, nInputsCount;
    double cur_output, cur_input, cur_error;
    double delta_bias, delta_weight;
    double cur_rate, dB, newH;

    double * localGradients = new double[_neuronsCount];
    double * outputs = new double[_neuronsCount];
    double * derivatives = new double[_neuronsCount];

    calculateNeuronsOutputsAndDerivatives(trainingInputs, outputs, derivatives);

    if(_layersCount > 1)
    {
        i = _layersCount-1;
        nInputsCount = _inputsInCurrentLayer[i];

        for (j = 0; j < _neuronsPerLayerCount[i]; j++)
        {
            cur_error = (trainingOutputs[j] - outputs[indexByLayerAndNeuron(i, j)]) * sample_weight;

            localGradients[indexByLayerAndNeuron(i, j)] = cur_error * derivatives[indexByLayerAndNeuron(i, j)];

            dB = speed * localGradients[indexByLayerAndNeuron(i, j)] * getHForBias(i, j);
            if (dB > 2.0)
            {
                dB = 2.0;
            }
            else
            {
                if (dB < -2.0)
                {
                    dB = -2.0;
                }
            }
            setBForBias(i, j, getBForBias(i, j) + dB);
            cur_rate = exp(getBForBias(i, j));

            delta_bias = cur_rate * localGradients[indexByLayerAndNeuron(i, j)];
            setBias(i, j, getBias(i, j) + delta_bias);

            newH = 1.0 - cur_rate;
            if (newH <= 0.0)
            {
                newH = delta_bias;
            }
            else
            {
                newH = getHForBias(i ,j) * newH + delta_bias;
            }
            setHForBias(i, j, newH);
        }

        // Цикл по всем скрытым слоям от последнего до первого
        for (i = _layersCount-2; i >= 0; i--)
        {
            nInputsCount = _inputsInCurrentLayer[i];

            /* Для каждого нейрона i-го слоя (цикл по j):
               1) вычисляем локальный градиент (по формуле для скрытого слоя);
               2) записываем его в соответствующее место массива
            m_aLocalGradients1[];
               3) корректируем веса связей между данным нейроном и всеми
            нейронами следующего, (i+1)-го слоя (теперь эти веса уже можно
            менять):
                  3.1) корректируем параметр Betta для каждой межнейронной
                       связи;
                  3.2) вычисляем коэффициент скорости обучения для веса этой
                       межнейронной связи;
                  3.3) корректируем вес межнейронной связи;
                  3.4) корректируем параметр H для межнейронной связи;
               4) корректируем параметр Betta для смещения нейрона;
               5) вычисляем коэффициент скорости обучения для смещения нейрона;
               6) корректируем смещение данного нейрона;
               7) корректируем параметр H для смещения нейрона. */
            for (j = 0; j < _neuronsPerLayerCount[i]; j++)
            {
                /* с помощью обратного распространения вычисляем ошибку
                   нейрона скрытого слоя */
                cur_output = outputs[indexByLayerAndNeuron(i, j)];
                cur_error = 0.0;
                for (k = 0; k < _neuronsPerLayerCount[i+1]; k++)
                {
                    cur_error += getWeight(i+1,k,j) * localGradients[indexByLayerAndNeuron(i, k)];

                    /* вычисляем новое значение параметра Betta для связи между
                       j-м нейроном i-го слоя и k-м нейроном (i+1)-го слоя */
                    dB = speed * localGradients[indexByLayerAndNeuron(i, k)] * getH(i+1,k,j) * cur_output;
                    if (dB > 2.0)
                    {
                        dB = 2.0;
                    }
                    else
                    {
                        if (dB < -2.0)
                        {
                            dB = -2.0;
                        }
                    }
                    setB(i+1,k,j, getB(i+1,k,j) + dB);

                    /* на основе нового значения Betta вычисляем коэффициент
                       скорости обучения */
                    cur_rate = exp(getB(i+1,k,j));
                    //cur_rate = m_rate;

                    /* вычисляем новое значение веса связи между j-м нейроном
                       i-го слоя и k-м нейроном (i+1)-го слоя */
                    delta_weight = cur_rate*cur_output*localGradients[indexByLayerAndNeuron(i, k)];
                    setWeight(i+1,k,j, getWeight(i+1,k,j) + delta_weight);

                    /* вычисляем новое значение параметра H для связи между
                       j-м нейроном i-го слоя и k-м нейроном (i+1)-го слоя */
                    newH = 1.0 - cur_rate * cur_output * cur_output;
                    if (newH <= 0.0)
                    {
                        newH = delta_weight;
                    }
                    else
                    {
                        newH = getH(i+1,k,j) * newH + delta_weight;
                    }
                    setH(i+1,k,j, newH);
                }

                // на основе ошибки вычисляем локальный градиент
                localGradients[indexByLayerAndNeuron(i, j)] = cur_error * derivatives[indexByLayerAndNeuron(i, j)];

                /* вычисляем новое значение параметра Betta для смещения, и на
                   его основе - значение коэффициента скорости обучения для
                   этого же смещения */
                dB = speed * localGradients[indexByLayerAndNeuron(i, j)] * getHForBias(i, j);
                if (dB > 2.0)
                {
                    dB = 2.0;
                }
                else
                {
                    if (dB < -2.0)
                    {
                        dB = -2.0;
                    }
                }
                setBForBias(i, j, getBForBias(i,j)+dB);
                cur_rate = exp(getBForBias(i, j));

                // корректируем смещение нейрона
                delta_bias = cur_rate * localGradients[indexByLayerAndNeuron(i, j)];

                setBias(i, j, getBias(i,j) + delta_bias);

                // вычисляем новое значение параметра H для смещения
                newH = 1.0 - cur_rate;
                if (newH <= 0.0)
                {
                    newH = delta_bias;
                }
                else
                {
                    newH = getHForBias(i, j) * newH + delta_bias;
                }
                setHForBias(i, j, newH);
            }
        }

        /* Сейчас в массиве m_aLocalGradients2[] содержатся локальные градиенты
        для нейронов первого слоя. Поэтому самое время скорректировать веса
        всех нейронов первого слоя, и на этом завершить обратный проход. */
        for (j = 0; j < _neuronsPerLayerCount[0]; j++)
        {
            for (k = 0; k < nInputsCount; k++)
            {
                /* вычисляем новое значение параметра Betta для связи между
                   k-м входом и j-м нейроном 1-го слоя */
                dB = speed * localGradients[indexByLayerAndNeuron(0, j)] * getH(0,j,k) * trainingInputs[k];
                if (dB > 2.0)
                {
                    dB = 2.0;
                }
                else
                {
                    if (dB < -2.0)
                    {
                        dB = -2.0;
                    }
                }
                setB(0,j,k, getB(0,j,k) + dB);

                /* на основе нового значения Betta вычисляем коэффициент
                   скорости обучения */
                cur_rate = exp(getB(0,j,k));
                //cur_rate = m_rate;

                cur_input = trainingInputs[k];

                /* вычисляем новое значение веса связи между k-м входом и
                   j-м нейроном 1-го слоя */
                delta_weight = cur_rate * cur_input * localGradients[indexByLayerAndNeuron(0, j)];
                setWeight(0, j, k, getWeight(0, j, k) + delta_weight);

                /* вычисляем новое значение параметра H для связи между
                   k-м входом и j-м нейроном 1-го слоя */
                newH = 1.0 - cur_rate * cur_input * cur_input;
                if (newH <= 0.0)
                {
                    newH = delta_weight;
                }
                else
                {
                    newH = getH(0,j,k) * newH + delta_weight;
                }
                setH(0,j,k, newH);
            }
        }
    }
    else
    {
        nInputsCount = _inputsInCurrentLayer[0];
        // Для каждого нейрона слоя (цикл по j)
        for (j = 0; j < _neuronsPerLayerCount[0]; j++)
        {
            /* вычисляем ошибку нейрона слоя как разность между реальным и
               желаемым выходами */
            cur_error = (trainingOutputs[j] - outputs[indexByLayerAndNeuron(0, j)]) * sample_weight;

            // вычисляем локальный градиент
            localGradients[indexByLayerAndNeuron(0, j)] = cur_error * derivatives[indexByLayerAndNeuron(0, j)];

            /* вычисляем новое значение параметра Betta для смещения, и на его
               основе - значение коэффициента скорости обучения для этого же
               смещения */
            dB = speed * localGradients[indexByLayerAndNeuron(0, j)] * getHForBias(0,j);
            if (dB > 2.0)
            {
                dB = 2.0;
            }
            else
            {
                if (dB < -2.0)
                {
                    dB = -2.0;
                }
            }
            setBForBias(0,j, getBForBias(0,j) + dB);
            cur_rate = exp(getBForBias(0,j));

            // корректируем смещение нейрона
            delta_bias = cur_rate * localGradients[indexByLayerAndNeuron(0, j)];
            setBias(0, j, getBias(0, j) + delta_bias);

            // вычисляем новое значение параметра H для смещения
            newH = 1.0 - cur_rate;
            if (newH <= 0.0)
            {
                newH = delta_bias;
            }
            else
            {
                newH = getHForBias(0, j) * newH + delta_bias;
            }
            setHForBias(0, j, newH);

            // Для всех входов j-го нейрона (цикл по k)
            for (k = 0; k < nInputsCount; k++)
            {
                /* вычисляем новое значение параметра Betta для связи между
                   k-м входом и j-м нейроном  */
                dB = speed * localGradients[indexByLayerAndNeuron(0, j)] * getH(0,j,k) * trainingInputs[k];
                if (dB > 2.0)
                {
                    dB = 2.0;
                }
                else
                {
                    if (dB < -2.0)
                    {
                        dB = -2.0;
                    }
                }
                setB(0, j, k, getB(0, j, k) + dB);

                /* на основе нового значения Betta вычисляем коэффициент
                   скорости обучения */
                cur_rate = exp(getB(0, j, k));

                cur_input = trainingInputs[k];

                /* вычисляем новое значение веса связи между k-м входом и
                   j-м нейроном */
                delta_weight = cur_rate * cur_input * localGradients[indexByLayerAndNeuron(0, j)];
                setWeight(0,j,k, getWeight(0,j,k) + delta_weight);

                /* вычисляем новое значение параметра H для связи между
                   k-м входом и j-м нейроном */
                newH = 1.0 - cur_rate * cur_input * cur_input;
                if (newH <= 0.0)
                {
                    newH = delta_weight;
                }
                else
                {
                    newH = getH(0,j,k) * newH + delta_weight;
                }
                setH(0,j,k, newH);
            }
        }
    }

    delete[] localGradients;
    delete[] outputs;
    delete[] derivatives;
}

double OpenNNL::_doEpoch(int samplesCount, double * trainingInputs, double * trainingOutputs, int numEpoch, double speed, bool isAdaptive)
{
    double * currentSampleInputs = new double[_inputsCount];
    double * currentSampleOutputs = new double[_outputsCount];

    for(int sample=0;sample<samplesCount;sample++)
    {
        memcpy(currentSampleInputs, trainingInputs+sample*_inputsCount, _inputsCount);
        memcpy(currentSampleOutputs, trainingOutputs+sample*_outputsCount, _outputsCount);

        if(isAdaptive)
        {
            _changeWeightsByIDBD(currentSampleInputs, currentSampleOutputs, 1, speed);
        }
        else
        {
            /*long double x_left = 1.0, x_center = getMaxEpochsCount();
            long double y_left = m_startRate, y_center = m_finalRate;
            long double a = (y_left - y_center)
                            / ((x_left - x_center) * (x_left - x_center));
            m_rate = y_center + a * ((numEpoch - x_center) * (numEpoch - x_center));*/
        }
    }

    delete[] currentSampleInputs;
    delete[] currentSampleOutputs;
}

void OpenNNL::_training(int samplesCount, double * trainingInputs, double * trainingOutputs, int nMaxEpochsCount, double speed, bool isAdaptive)
{
    for(int i=0;i<nMaxEpochsCount;i++)
    {
        if(!_doEpoch(samplesCount, trainingInputs, trainingOutputs, i, speed, isAdaptive))
        {
            break;
        }
    }
}
