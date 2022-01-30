using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;

namespace SelfGorwingNN.Tests
{
    // https://visualstudiomagazine.com/Articles/2015/04/01/Back-Propagation-Using-C.aspx?Page=2
    [TestClass]
    public class VSMagBackProp
    {
        // inputs[0]				hsums[0]	hOutputs[0]					oSums[0] Softmax() results[0]
        // 			ihWeights								hoWeights	
        // inputs[1]				hsums[1]	hOutputs[1]					oSums[1] Softmax() results[1]
        // 			bias									bias
        [TestMethod]
        public void Train1Example_2_2_2()
        {
            // i1=>h1, i2=>h1
            var ihWeights = new double[2][] {
                new[] {0.15, 0.25},
                new[] {0.20, 0.30}
            };
            var hBiases = new[] { 0.35, 0.35 };
            var hoWeights = new double[2][]
            {
                new[] { 0.40, 0.50},
                new[] { 0.45, 0.55}
            };
            var oBiases = new[] { 0.60, 0.60 };
            var inputs = new[] { 0.05, 0.1 };
            var otargets = new[] { 0.01, 0.99 };
            var learnRate = 1;

            var hsums = new double[2];
            hsums[0] = inputs[0] * ihWeights[0][0] + inputs[1] * ihWeights[0][1] + hBiases[0]; // .05 * .15 + .10 *.25
            hsums[1] = inputs[0] * ihWeights[1][0] + inputs[1] * ihWeights[1][1] + hBiases[1]; // .05 * .20 + .10 *.30
            var hOutputs = new double[2];
            hOutputs[0] = Hypertan(hsums[0]);
            hOutputs[1] = Hypertan(hsums[1]);
            var oSums = new double[2];
            oSums[0] = hOutputs[0] * hoWeights[0][0] + hOutputs[1] * hoWeights[0][1] + oBiases[0]; // ? * .40 + ? *.50
            oSums[1] = hOutputs[0] * hoWeights[1][0] + hOutputs[1] * hoWeights[1][1] + oBiases[1]; // ? * .45 + ? *.55
            var output = Softmax(oSums);
            Console.Out.WriteLine(output[0] + "," + output[1]);

            var oSignals = new double[2];
            oSignals[0] = (otargets[0] - output[0]) * (1 - output[0]) * output[0];
            oSignals[1] = (otargets[1] - output[1]) * (1 - output[1]) * output[1];
            var hoGrads = new double[2][] { new double[2], new double[2] };
            hoGrads[0][0] = oSignals[0] * hOutputs[0];
            hoGrads[0][1] = oSignals[1] * hOutputs[0];
            hoGrads[1][0] = oSignals[0] * hOutputs[1];
            hoGrads[1][1] = oSignals[1] * hOutputs[1];
            var obGrads = new double[2];
            obGrads[0] = oSignals[0] * 1.0;
            obGrads[1] = oSignals[1] * 1.0;
            var hSignals = new double[2];
            hSignals[0] = (1 + hOutputs[0]) * (1 - hOutputs[0]) *
                          (oSignals[0] * hoWeights[0][0] + oSignals[1] * hoWeights[0][1]);
            hSignals[1] = (1 + hOutputs[1]) * (1 - hOutputs[1]) *
                          (oSignals[0] * hoWeights[1][0] + oSignals[1] * hoWeights[1][1]);
            var ihGrads = new double[2][] { new double[2], new double[2] }; ;
            ihGrads[0][0] = hSignals[0] * inputs[0];
            ihGrads[0][1] = hSignals[1] * inputs[0];
            ihGrads[1][0] = hSignals[0] * inputs[1];
            ihGrads[1][1] = hSignals[1] * inputs[1];
            var hbGrads = new double[2];
            hbGrads[0] = hSignals[0] * 1.0;
            hbGrads[1] = hSignals[1] * 1.0;
            var ihWeights_1 = new double[2][] { new double[2], new double[2] }; ;
            ihWeights_1[0][0] = ihGrads[0][0] * learnRate;
            ihWeights_1[0][1] = ihGrads[0][1] * learnRate;
            ihWeights_1[1][0] = ihGrads[1][0] * learnRate;
            ihWeights_1[1][1] = ihGrads[1][1] * learnRate;
            var hBiases_1 = new double[2];
            hBiases_1[0] = hbGrads[0] * learnRate;
            hBiases_1[1] = hbGrads[1] * learnRate;
            var hoWeights_1 = new double[2][] { new double[2], new double[2] }; ;
            hoWeights_1[0][0] = hoGrads[0][0] * learnRate;
            hoWeights_1[0][1] = hoGrads[0][1] * learnRate;
            hoWeights_1[1][0] = hoGrads[1][0] * learnRate;
            hoWeights_1[1][1] = hoGrads[1][1] * learnRate;
            var oBiases_1 = new double[2];
            oBiases_1[0] = obGrads[0] * learnRate;
            oBiases_1[1] = obGrads[1] * learnRate;

            hsums = new double[2];
            hsums[0] = inputs[0] * ihWeights_1[0][0] + inputs[1] * ihWeights_1[0][1] + hBiases_1[0]; // .05 * .15 + .10 *.25
            hsums[1] = inputs[0] * ihWeights_1[1][0] + inputs[1] * ihWeights_1[1][1] + hBiases_1[1]; // .05 * .20 + .10 *.30
            hOutputs = new double[2];
            hOutputs[0] = Hypertan(hsums[0]);
            hOutputs[1] = Hypertan(hsums[1]);
            oSums = new double[2];
            oSums[0] = hOutputs[0] * hoWeights_1[0][0] + hOutputs[1] * hoWeights_1[0][1] + oBiases_1[0]; // ? * .40 + ? *.50
            oSums[1] = hOutputs[0] * hoWeights_1[1][0] + hOutputs[1] * hoWeights_1[1][1] + oBiases_1[1]; // ? * .45 + ? *.55
            var output_1 = Softmax(oSums);
            Console.Out.WriteLine(output_1[0] + "," + output_1[1]);

            Assert.IsTrue(
                otargets[0] < output[0] && output_1[0] < output[0],
                "since target is lower trained should be lower");
            Assert.IsTrue(
                otargets[1] > output[1] && output_1[1] > output[1],
                "since target is higher trained should be higher");

        }

        [TestMethod]
        public void NetworkTest()
        {
            var inputs = new[] { 0.05, 0.1 };
            var otargets = new[] { 0.01, 0.99 };
            Console.Out.WriteLine("Targets: " + otargets[0] + "," + otargets[1]);

            var nn = new Network();

            nn.Train(inputs, otargets);
            var output = nn.Test(inputs).ToArray();

            Console.Out.WriteLine("After: " + output[0] + "," + output[1]);

            Assert.IsTrue(
                otargets[0] < output[0],
                "since target is lower trained should be lower");
            Assert.IsTrue(
                otargets[1] > output[1],
                "since target is higher trained should be higher");
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

        private double Hypertan(double x)
        {
            if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
            else if (x > 20.0) return 1.0;
            else return Math.Tanh(x);
        }
    }
}