using System;
using System.Collections.Generic;
using System.Text;

namespace SelfGorwingNN
{
    public class BackProp
    {
        private Neuron _hiddenNeuron1;
        private Neuron _hiddenNeuron2;
        private Neuron _outputNeuron;

        public BackProp()
        {
            _hiddenNeuron1 = new Neuron();
            _hiddenNeuron2 = new Neuron();
            _outputNeuron = new Neuron();
            _hiddenNeuron1.RandomizeWeights();
            _hiddenNeuron2.RandomizeWeights();
            _outputNeuron.RandomizeWeights();
        }

        public void Train()
        {
            double[,] inputs =
            {
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
            };

            // desired results
            double[] results = { 0, 1, 1, 0 };
            Train(inputs, results);
        }

        public double Test(double in1, double in2)
        {
            _hiddenNeuron1.Inputs = new double[] { in1, in2 };
            _hiddenNeuron2.Inputs = new double[] { in1, in2 };

            _outputNeuron.Inputs = new double[] { _hiddenNeuron1.Output, _hiddenNeuron2.Output };

            return _outputNeuron.Output;
        }

        public void Train(double[,] inputs, double[] results)
        {
            var epoch = 0;

            Retry:
            epoch++;
            for (int i = 0; i < 4; i++) 
            {
                // 1) forward propagation (calculates output)
                //_hiddenNeuron1.Inputs = new double[] { inputs[i, 0], inputs[i, 1] };
                //_hiddenNeuron2.Inputs = new double[] { inputs[i, 0], inputs[i, 1] };

                //_outputNeuron.Inputs = new double[] { _hiddenNeuron1.Output, _hiddenNeuron2.Output };

                //Console.WriteLine("{0} xor {1} = {2}", inputs[i, 0], inputs[i, 1], _outputNeuron.Output);
                Console.WriteLine("{0} xor {1} = {2}", inputs[i, 0], inputs[i, 1], Test(inputs[i, 0], inputs[i, 1]));

                // 2) back propagation (adjusts weights)

                // adjusts the weight of the output neuron, based on its error
                _outputNeuron.Error = Sigmoid.Derivative(_outputNeuron.Output) * (results[i] - _outputNeuron.Output);
                _outputNeuron.AdjustWeights();

                // then adjusts the hidden neurons' weights, based on their errors
                _hiddenNeuron1.Error = Sigmoid.Derivative(_hiddenNeuron1.Output) * _outputNeuron.Error *
                                      _outputNeuron.Weights[0];
                _hiddenNeuron2.Error = Sigmoid.Derivative(_hiddenNeuron2.Output) * _outputNeuron.Error *
                                      _outputNeuron.Weights[1];

                _hiddenNeuron1.AdjustWeights();
                _hiddenNeuron2.AdjustWeights();
            }

            if (epoch < 200)
                goto Retry;

            Console.ReadLine();
        }

        class Sigmoid
        {
            public static double Output(double x)
            {
                return 1.0 / (1.0 + Math.Exp(-x));
            }

            public static double Derivative(double x)
            {
                return x * (1 - x);
            }
        }


        public class Neuron
        {
            public double[] Inputs = new double[2];
            public double[] Weights = new double[2];
            public double Error;

            private double _biasWeight;

            private readonly Random r = new Random();

            public double Output => Sigmoid.Output(Weights[0] * Inputs[0] + Weights[1] * Inputs[1] + _biasWeight);

            public void RandomizeWeights()
            {
                Weights[0] = 0.51; //r.NextDouble();
                Weights[1] = 0.49; //r.NextDouble();
                _biasWeight = 0.5;//r.NextDouble();
            }

            public void AdjustWeights()
            {
                Weights[0] += Error * Inputs[0];
                Weights[1] += Error * Inputs[1];
                _biasWeight += Error;
            }
        }
    }
}
