using System;

namespace SelfGorwingNN
{
    public class Network
    {
        // i1=>h1, i2=>h1
        public double[][] ihWeights = new double[2][] {
            new[] {0.15, 0.25},
            new[] {0.20, 0.30}
        };
        public double[] hBiases = new[] { 0.35, 0.35 };
        public double[][] hoWeights = new double[2][]
        {
            new[] { 0.40, 0.50},
            new[] { 0.45, 0.55}
        };
        public double[] oBiases = new[] { 0.60, 0.60 };
        public double learnRate = 1;



        public void Train(double[] inputs, double[] otargets)
        {
            // i1=>h1, i2=>h1
            var hOutputs = TestHidden(inputs);
            var output = TestOutput(hOutputs);

            var error = 0.0;
            for (var i = 0; i < output.Length; i++)
            {
                error = +Math.Pow(otargets[i] - output[i], 2);
            }


            Console.Out.WriteLine("Error:" + error);

            var oSignals = CalculateOutputErrorSignals(otargets, output);
            var hSignals = CalculateHiddenErrorSignals(hOutputs, oSignals);

            hoWeights = CalculateWeigths(oSignals, hOutputs);

            oBiases = CalculateBias(oSignals);

            ihWeights = CalculateWeigths(hSignals, inputs);

            hBiases = CalculateBias(hSignals);
        }

        private double[] CalculateBias(double[] oSignals)
        {
            var obGrads = new double[2];
            obGrads[0] = oSignals[0] * 1.0;
            obGrads[1] = oSignals[1] * 1.0;
            var oBiases_1 = new double[2];
            oBiases_1[0] = oBiases[0] + obGrads[0] * learnRate;
            oBiases_1[1] = oBiases[1] + obGrads[1] * learnRate;
            return oBiases_1;
        }

        private double[][] CalculateWeigths(double[] oSignals, double[] hOutputs)
        {
            var hoGrads = new double[2][] { new double[2], new double[2] };
            hoGrads[0][0] = oSignals[0] * hOutputs[0];
            hoGrads[0][1] = oSignals[1] * hOutputs[0];
            hoGrads[1][0] = oSignals[0] * hOutputs[1];
            hoGrads[1][1] = oSignals[1] * hOutputs[1];
            var hoWeights_1 = new double[2][] { new double[2], new double[2] };
            ;
            hoWeights_1[0][0] = hoWeights[0][0] + hoGrads[0][0] * learnRate;
            hoWeights_1[0][1] = hoWeights[0][1] + hoGrads[0][1] * learnRate;
            hoWeights_1[1][0] = hoWeights[1][0] + hoGrads[1][0] * learnRate;
            hoWeights_1[1][1] = hoWeights[1][1] + hoGrads[1][1] * learnRate;
            return hoWeights_1;
        }

        private double[] CalculateHiddenErrorSignals(double[] hOutputs, double[] oSignals)
        {
            var hSignals = new double[2];
            hSignals[0] = (1 + hOutputs[0]) * (1 - hOutputs[0]) *
                          (oSignals[0] * hoWeights[0][0] + oSignals[1] * hoWeights[0][1]);
            hSignals[1] = (1 + hOutputs[1]) * (1 - hOutputs[1]) *
                          (oSignals[0] * hoWeights[1][0] + oSignals[1] * hoWeights[1][1]);
            return hSignals;
        }

        private static double[] CalculateOutputErrorSignals(double[] otargets, double[] output)
        {
            var oSignals = new double[2];
            oSignals[0] = (otargets[0] - output[0]) * (1 - output[0]) * output[0];
            oSignals[1] = (otargets[1] - output[1]) * (1 - output[1]) * output[1];
            //Console.Out.WriteLine($"Signals: {oSignals[0]} {oSignals[1]}");
            return oSignals;
        }

        private double[] TestHidden(double[] inputs)
        {
            var hOutputs = new double[2];
            hOutputs[0] = Hypertan(inputs[0] * ihWeights[0][0] + inputs[1] * ihWeights[0][1] + hBiases[0]); // .05 * .15 + .10 *.25
            hOutputs[1] = Hypertan(inputs[0] * ihWeights[1][0] + inputs[1] * ihWeights[1][1] + hBiases[1]); // .05 * .20 + .10 *.30

            return hOutputs;
        }

        private double[] TestOutput(double[] hOutputs)
        {
            var oSums = new double[2];
            oSums[0] = hOutputs[0] * hoWeights[0][0] + hOutputs[1] * hoWeights[0][1] + oBiases[0]; // ? * .40 + ? *.50
            oSums[1] = hOutputs[0] * hoWeights[1][0] + hOutputs[1] * hoWeights[1][1] + oBiases[1]; // ? * .45 + ? *.55

            return Softmax(oSums);
        }

        public double[] Test(double[] inputs)
        {
            var hOutputs = TestHidden(inputs);
            return TestOutput(hOutputs);
        }

        private double Hypertan(double x)
        {
            if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
            else if (x > 20.0) return 1.0;
            else return Math.Tanh(x);
        }


        private double[] Softmax(double[] oSums)
        {
            double sum = 0.0;
            for (int i = 0; i < oSums.Length; ++i)
            {
                sum += Math.Exp(oSums[i]);
            }

            double[] result = new double[oSums.Length];
            for (int i = 0; i < oSums.Length; ++i)
            {
                result[i] = Math.Exp(oSums[i]) / sum;
            }

            return result; // now scaled so that xi sum to 1.0
        }
    }
}
