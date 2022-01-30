using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using System.Net.Http.Headers;

namespace SelfGorwingNN.Tests
{
    // https://visualstudiomagazine.com/Articles/2015/04/01/Back-Propagation-Using-C.aspx?Page=2
    [TestClass]
    public class Network1Test
    {
        // inputs[0]				hsums[0]	hOutputs[0]					oSums[0] Softmax() results[0]
        // 			ihWeights								hoWeights	
        // inputs[1]				hsums[1]	hOutputs[1]					oSums[1] Softmax() results[1]
        // 			bias									bias
        [TestMethod]
        public void Backward_Test()
        {
            var inputs = new[] {0.05, 0.1};
            var otargets = new[] {0.01, 0.99};
            var nn = new Network1();
            var hOutputs = nn.TestHidden(inputs);
            var output = nn.TestOutput(hOutputs);

            var eo = new double[2];
            eo[0] = 0.5 * Math.Pow(otargets[0] - output[0], 2);
            eo[1] = 0.5 * Math.Pow(otargets[1] - output[1], 2);
            var eTotal = eo[0] + eo[1];
            Assert.AreEqual(0.29837110876000272, eTotal);

            var eTotal_out = new double[2];
            eTotal_out[0] = -(otargets[0] - output[0]);
            eTotal_out[1] = -(otargets[1] - output[1]);
            Assert.AreEqual(eTotal_out[0], 0.7413650695523157);
            Assert.AreEqual(eTotal_out[1], -0.21707153467853746);

            var out_net = new double[2];
            out_net[0] = Network1.SigmoidDerivative(output[0]);
            out_net[1] = Network1.SigmoidDerivative(output[1]);
            Assert.AreEqual(out_net[0], 0.18681560180895948);
            Assert.AreEqual(out_net[1], 0, 17551005281727122);

            double neto1_w5 = hOutputs[0];
            double eTotal_w5 = eTotal_out[0] * out_net[0] * neto1_w5;
            Assert.AreEqual(0.08216704056423078, eTotal_w5);
            var w5_1 = nn.hoWeights[0][0] - nn.learnRate * eTotal_w5;
            Assert.AreEqual(0.35891647971788465, w5_1);

            double neto1_w6 = hOutputs[1];
            double eTotal_w6 = eTotal_out[0] * out_net[0] * neto1_w6;
            var w6_1 = nn.hoWeights[1][0] - nn.learnRate * eTotal_w6;
            Assert.AreEqual(w6_1, 0.4086661860762334);

            double neto1_w7 = hOutputs[0];
            double eTotal_w7 = eTotal_out[1] * out_net[1] * neto1_w7;
            var w7_1 = nn.hoWeights[0][1] - nn.learnRate * eTotal_w7;
            Assert.AreEqual(w7_1, 0.5113012702387375);

            double neto1_w8 = hOutputs[1];
            double eTotal_w8 = eTotal_out[1] * out_net[1] * neto1_w8;
            var w8_1 = nn.hoWeights[1][1] - nn.learnRate * eTotal_w8;
            Assert.AreEqual(w8_1, 0.5613701211079891);

            var eo1_neto1 = eTotal_out[0] * out_net[0];
            Assert.AreEqual(eo1_neto1, 0.13849856162855698);
            var neto1_outh1 = nn.hoWeights[0][0];
            var eo1_outh1 = eo1_neto1 * neto1_outh1;
            Assert.AreEqual(eo1_outh1, 0, 05539942465142279);
            var eo2_neto2 = eTotal_out[1] * out_net[1];
            var neto2_outh1 = nn.hoWeights[0][1]; // w7 = 0.5 
            var eo2_outh1 = eo2_neto2 * neto2_outh1;
            Assert.AreEqual(eo2_outh1, -0, 019049118258278114);

            var neth1_w1 = inputs[0];
            var outh1_neth1 = Network1.SigmoidDerivative(hOutputs[0]);
            var eTotal_outh1 = eo1_outh1 + eo2_outh1;
            var etotal_w1 = neth1_w1 * eTotal_outh1 * outh1_neth1;
            Assert.AreEqual(etotal_w1, 0, 00043856773447434685);
            var w1_1 = nn.ihWeights[0][0] - nn.learnRate * etotal_w1;
            Assert.AreEqual(w1_1, 0.1497807161327628);

            var neth1_w2 = inputs[1];
            var etotal_w2 = eTotal_outh1 * outh1_neth1 * neth1_w2;
            Assert.AreEqual(etotal_w2, 0.0008771354689486937);
            var w2_1 = nn.ihWeights[1][0] - nn.learnRate * etotal_w2;
            Assert.AreEqual(w2_1, 0.19956143226552567);

            var neto1_outh2 = nn.hoWeights[1][0]; //w6=.45
            var eo1_outh2 = neto1_outh2 * eo1_neto1;
            var neto2_outh2 = nn.hoWeights[1][1]; //w8=.55
            var eo2_outh2 = neto2_outh2 * eo2_neto2;

            var eTotal_outh2 = eo1_outh2 + eo2_outh2;
            var neth2_w3 = inputs[0]; //nn.ihWeights[0][1]; // w3 =.25
            var outh2_neth2 = Network1.SigmoidDerivative(hOutputs[1]);
            var etotal_w3 = neth2_w3 * eTotal_outh2 * outh2_neth2;
            Assert.AreEqual(etotal_w3, 0.00049771273526086);
            var w3_1 = nn.ihWeights[0][1] - nn.learnRate * etotal_w3; // .25
            Assert.AreEqual(w3_1, 0.24975114363236958);

            var neth2_w4 = inputs[1];
            var etotal_w4 = neth2_w4 * eTotal_outh2 * outh2_neth2;
            Assert.AreEqual(etotal_w4, 0.00099542547052172);
            var w4_1 = nn.ihWeights[1][1] - nn.learnRate * etotal_w4; // .30
            Assert.AreEqual(0.29950228726473915, w4_1);

            // TestError
            nn.hoWeights[0][0] = w5_1;
            nn.hoWeights[1][0] = w6_1;
            nn.hoWeights[0][1] = w7_1;
            nn.hoWeights[1][1] = w8_1;
            nn.ihWeights[0][0] = w1_1;
            nn.ihWeights[1][0] = w2_1;
            nn.ihWeights[0][1] = w3_1;
            nn.ihWeights[1][1] = w4_1;

            hOutputs = nn.TestHidden(inputs);
            output = nn.TestOutput(hOutputs);
            eo = new double[2];
            eo[0] = 0.5 * Math.Pow(otargets[0] - output[0], 2);
            eo[1] = 0.5 * Math.Pow(otargets[1] - output[1], 2);
            eTotal = eo[0] + eo[1];
            Assert.AreEqual(0.29102777369359933, eTotal);
        }

        [TestMethod]
        public void Backward1_Test()
        {
            var inputs = new[] {0.05, 0.1};
            var oTargets = new[] {0.01, 0.99};
            var nn = new Network1();
            var hOutputs = nn.TestHidden(inputs);
            var outputs = nn.TestOutput(hOutputs);

            var eTotal = nn.Error(outputs, oTargets);
            Assert.AreEqual(eTotal, 0.29837110876000272);

            var eTotal_out = nn.Errors(outputs, oTargets).ToArray();
            Assert.AreEqual(eTotal_out[0], 0.7413650695523157);
            Assert.AreEqual(eTotal_out[1], -0.21707153467853746);

            var out_net = new double[2];
            out_net[0] = Network1.SigmoidDerivative(outputs[0]);
            out_net[1] = Network1.SigmoidDerivative(outputs[1]);
            Assert.AreEqual(out_net[0], 0.18681560180895948);
            Assert.AreEqual(out_net[1], 0.17551005281727122);

            var newOutWeights = nn.Backward(hOutputs, out_net, eTotal_out);
            Assert.AreEqual(newOutWeights[0][0], 0.35891647971788465);
            Assert.AreEqual(newOutWeights[1][0], 0.4086661860762334);
            Assert.AreEqual(newOutWeights[0][1], 0.5113012702387375);
            Assert.AreEqual(newOutWeights[1][1], 0.5613701211079891);

            var newHiddenWeights = nn.BakwardHidden(inputs, hOutputs, out_net, eTotal_out);
            Assert.AreEqual(newHiddenWeights[0][0], 0.1497807161327628);
            Assert.AreEqual(newHiddenWeights[1][0], 0.19956143226552567);
            Assert.AreEqual(newHiddenWeights[0][1], 0.24975114363236958);
            Assert.AreEqual(newHiddenWeights[1][1], 0.29950228726473915);

            nn.ihWeights = newHiddenWeights;
            nn.hoWeights = newOutWeights;
            hOutputs = nn.TestHidden(inputs);
            outputs = nn.TestOutput(hOutputs);

            eTotal = nn.Error(outputs, oTargets);
            Assert.AreEqual(eTotal, 0.29102777369359933);
        }

        [TestMethod]
        public void Backward2_Test()
        {
            var inputs = new[] {0.05, 0.1};
            var oTargets = new[] {0.01, 0.99};
            var nn = new Network1();
            nn.Activation = Network1.Sigmoid;
            nn.InvertedActivation = Network1.SigmoidDerivative;

            var outputs = nn.Test(inputs);
            var eTotal = nn.Error(outputs, oTargets);
            Assert.AreEqual(0.29837110876000272, eTotal);

            nn.Train(inputs, oTargets);

            outputs = nn.Test(inputs);
            eTotal = nn.Error(outputs, oTargets);
            Assert.AreEqual(0.29102777369359933, eTotal);
        }

        [TestMethod]
        public void BackwardNone_Test()
        {
            var inputs = new[] {0.0, 0.0};
            var oTargets = new[] {1.0, 1.0};
            var nn = new Network1();
            nn.learnRate = 1;
            nn.Activation = Network1.None;
            nn.InvertedActivation = Network1.None;
            var hOutputs = nn.TestHidden(inputs);
            var outputs = nn.TestOutput(hOutputs);

            var eTotal_out = nn.Error(outputs, oTargets);

            var out_net = new double[2];
            out_net[0] = Network1.None(outputs[0]);
            out_net[1] = Network1.None(outputs[1]);
            nn.Print(new[] {1.0, 0.0}, new[] {1.0, 0.00});
            Console.Out.WriteLine("---");
            for (var i = 0; i < 20; i++)
            {
                nn.Train(new[] {1.0, 0.0}, new[] {1.0, 0.00});
                //nn.Print(new[] { 1.0, 0.0 }, new[] { 1.0, 0.00 });
                eTotal_out = nn.Error(nn.Test(new[] {1.0, 0.00}), new[] {1.0, 0.00});
                Console.Out.WriteLine(eTotal_out);
                //Console.Out.WriteLine();
                nn.Train(new[] {0.0, 1.0}, new[] {0.0, 1.0});
                //nn.Print(new[] { 0.0, 1.0 }, new[] { 0.0, 1.0 });
                eTotal_out = nn.Error(nn.Test(new[] {0.0, 1.0}), new[] {0.0, 1.00});
                Console.Out.WriteLine(eTotal_out);
                //Console.Out.WriteLine("---");
            }
        }

        [TestMethod]
        public void Backward1_And_Test()
        {
            var nn = new Network1();
            nn.InvertedActivation = Network1.SigmoidDerivative;
            nn.Activation = Network1.Sigmoid;
            for (var i = 0; i < 10000; i++)
            {
                nn.Train(new[] {0.01, 0.99}, new[] {0.01, 0.99});
                nn.Train(new[] {0.99, 0.01}, new[] {0.99, 0.01});
            }

            var outputs = nn.Test(new[] {0.01, 0.99});
            Console.Out.WriteLine($"0 1 => {outputs[0]} {outputs[1]}");
            outputs = nn.Test(new[] {0.99, 0.01});
            Console.Out.WriteLine($"1 0 => {outputs[0]} {outputs[1]}");
        }

        [TestMethod]
        public void Backward1_Xor_Test()
        {
            var nn = new Network1();
            nn.learnRate = 0.01;
            nn.InvertedActivation = Network1.SigmoidDerivative;
            nn.Activation = Network1.Sigmoid;
            for (var i = 0; i < 1000000; i++)
            {
                nn.Train(new[] {0.01, 0.01}, new[] {0.01, 0.99});
                nn.Train(new[] {0.99, 0.01}, new[] {0.99, 0.01});
                nn.Train(new[] {0.01, 0.99}, new[] {0.99, 0.01});
                nn.Train(new[] {0.99, 0.99}, new[] {0.01, 0.99});
            }

            var outputs = nn.Test(new[] {0.01, 0.01});
            Console.Out.WriteLine($"0 0 => {outputs[0]} {outputs[1]}");
            outputs = nn.Test(new[] {0.99, 0.01});
            Console.Out.WriteLine($"1 0 => {outputs[0]} {outputs[1]}");
            outputs = nn.Test(new[] {0.01, 0.99});
            Console.Out.WriteLine($"0 1 => {outputs[0]} {outputs[1]}");
            outputs = nn.Test(new[] {0.99, 0.99});
            Console.Out.WriteLine($"1 1 => {outputs[0]} {outputs[1]}");
        }


        [TestMethod]
        public void Forward_Test()
        {
            var inputs = new[] {0.05, 0.1};
            var otargets = new[] {0.01, 0.99};
            Console.Out.WriteLine("Targets: " + otargets[0] + "," + otargets[1]);

            var nn = new Network1();
            var hOutputs = nn.TestHidden(inputs);
            Console.Out.WriteLine("hidden: " + hOutputs[0] + "," + hOutputs[1]);
            var output = nn.TestOutput(hOutputs);
            Console.Out.WriteLine("output: " + output[0] + "," + output[1]);

            var eo = new double[2];
            eo[0] = 0.5 * Math.Pow(otargets[0] - output[0], 2);
            eo[1] = 0.5 * Math.Pow(otargets[1] - output[1], 2);
            var eTotal = eo[0] + eo[1];
            Console.Out.WriteLine("etotal: " + eTotal);
        }
    }
}