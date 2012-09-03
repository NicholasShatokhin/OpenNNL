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

    _BsForBias = new double[_neuronsCount];
    _HsForBias = new double[_neuronsCount];
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
    delete[] _BsForBias;
    delete[] _HsForBias;
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

double * OpenNNL::_calculateWorker(double *inpt)
{
    int inputsCount;
    double * temp;
    double * inputs = new double[_inputsCount];

    memcpy(inputs, inpt, sizeof(double)*_inputsCount);

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

            //inputs[j]  += _neuronsBiases[indexByLayerAndNeuron(i, j)];

            inputs[j] = activation(inputs[j]);

        }

        inputsCount = _neuronsPerLayerCount[i];
        delete[] temp;
    }

    memcpy(_outputs, inputs, sizeof(double)*inputsCount);

    delete[] inputs;

    return _outputs;
}

double * OpenNNL::calculate(double *inputs)
{
    if(inputs)
    {
        memcpy(_inputs, inputs, _inputsCount*sizeof(double));
    }

    return _calculateWorker(_inputs);
}

double * OpenNNL::calculateRef(double *inputs)
{
    if(!inputs)
        inputs = _inputs;

    return _calculateWorker(inputs);
}

void OpenNNL::calculateNeuronsOutputsAndDerivatives(double *inpt, double *outputs, double *derivatives)
{
    int inputsCount, neuronIndex = 0;
    double * temp;
    double * inputs = new double[_inputsCount];

    memcpy(inputs, inpt, sizeof(double)*_inputsCount);

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

            //inputs[j] += _neuronsBiases[indexByLayerAndNeuron(i, j)];

            outputs[neuronIndex] = inputs[j] = activation(inputs[j]);
            derivatives[neuronIndex] = activation_derivative(inputs[j]);

            neuronIndex++;
        }

        inputsCount = _neuronsPerLayerCount[i];

        delete[] temp;
    }

    delete[] inputs;
}

double OpenNNL::_changeWeightsByBP(double * trainingInputs, double *trainingOutputs, double speed, double sample_weight)
{
    double * localGradients = new double[_neuronsCount];
    double * outputs = new double[_neuronsCount];
    double * derivatives = new double[_neuronsCount];

    calculateNeuronsOutputsAndDerivatives(trainingInputs, outputs, derivatives);

    for(int j=0;j<_neuronsPerLayerCount[_layersCount-1];j++)
    {
        localGradients[indexByLayerAndNeuron(_layersCount-1, j)] = trainingOutputs[j] - outputs[indexByLayerAndNeuron(_layersCount-1, j)];
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
            }
        }
    }

    for(int j=0;j<_neuronsPerLayerCount[0];j++)
    {
        for(int k=0;k<_inputsCount;k++)
        {
            _neuronsInputsWeights[indexByLayerNeuronAndInput(0, j, k)] += speed * localGradients[indexByLayerAndNeuron(0, j)]
                    * derivatives[indexByLayerAndNeuron(0, j)] * trainingInputs[k];
        }
    }

    for(int i=1;i<_layersCount;i++)
    {
        for(int j=0;j<_neuronsPerLayerCount[i];j++)
        {
            for(int k=0;k<_neuronsPerLayerCount[i-1];k++)
            {
                _neuronsInputsWeights[indexByLayerNeuronAndInput(i, j, k)] += speed * localGradients[indexByLayerAndNeuron(i, j)]
                        * derivatives[indexByLayerAndNeuron(i, j)] * outputs[indexByLayerAndNeuron(i-1, k)];
            }
        }
    }

    delete[] localGradients;
    delete[] outputs;
    delete[] derivatives;
}

double OpenNNL::_changeWeightsByIDBD(double * trainingInputs, double *trainingOutputs, double speed, double sample_weight)
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

double OpenNNL::_doEpochBP(int samplesCount, double * trainingInputs, double * trainingOutputs, int numEpoch, double speed)
{
    double * currentSampleInputs = new double[_inputsCount];
    double * currentSampleOutputs = new double[_outputsCount];

    for(int sample=0;sample<samplesCount;sample++)
    {

        memcpy(currentSampleInputs, trainingInputs+sample*_inputsCount, _inputsCount*sizeof(double));
        memcpy(currentSampleOutputs, trainingOutputs+sample*_outputsCount, _outputsCount*sizeof(double));

        _changeWeightsByBP(currentSampleInputs, currentSampleOutputs, speed, 1);
    }

    delete[] currentSampleInputs;
    delete[] currentSampleOutputs;
}

double OpenNNL::_doEpochIDBD(int samplesCount, double * trainingInputs, double * trainingOutputs, int numEpoch, double speed)
{
    double * currentSampleInputs = new double[_inputsCount];
    double * currentSampleOutputs = new double[_outputsCount];

    for(int sample=0;sample<samplesCount;sample++)
    {

        memcpy(currentSampleInputs, trainingInputs+sample*_inputsCount, _inputsCount*sizeof(double));
        memcpy(currentSampleOutputs, trainingOutputs+sample*_outputsCount, _outputsCount*sizeof(double));

        _changeWeightsByIDBD(currentSampleInputs, currentSampleOutputs, speed, 1);
    }

    delete[] currentSampleInputs;
    delete[] currentSampleOutputs;
}

void OpenNNL::_trainingBP(int samplesCount, double * trainingInputs, double * trainingOutputs, int maxEpochsCount, double speed)
{
    for(int i=0;i<maxEpochsCount;i++)
    {
        if(!_doEpochBP(samplesCount, trainingInputs, trainingOutputs, i, speed))
        {
            break;
        }
    }
}


void OpenNNL::_trainingIDBD(int samplesCount, double * trainingInputs, double * trainingOutputs, int maxEpochsCount, double speed)
{
    for(int i=0;i<maxEpochsCount;i++)
    {
        if(!_doEpochIDBD(samplesCount, trainingInputs, trainingOutputs, i, speed))
        {
            break;
        }
    }
}

void OpenNNL::trainingBP(int samplesCount, double * trainingInputs, double *trainingOutputs, int maxEpochsCount, double speed, double error)
{
    resetHsAndHsForBias();
    randomizeBsAndBsForBias();

    _trainingBP(samplesCount, trainingInputs, trainingOutputs, maxEpochsCount, speed);
}

void OpenNNL::trainingIDBD(int samplesCount, double * trainingInputs, double *trainingOutputs, int maxEpochsCount, double speed, double error)
{
    resetHsAndHsForBias();
    randomizeBsAndBsForBias();

    _trainingIDBD(samplesCount, trainingInputs, trainingOutputs, maxEpochsCount, speed);
}

void OpenNNL::getOutputs(double * out)
{
    memcpy(out, _outputs, sizeof(double)*_outputsCount);
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
