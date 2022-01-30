using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace SelfGorwingNN.Tests
{
    // https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    [TestClass]
    public class BackPropagationNetworkTests
    {
        // inputs[0]				hsums[0]	hOutputs[0]					oSums[0] Softmax() results[0]
        // 			ihWeights								hoWeights	
        // inputs[1]				hsums[1]	hOutputs[1]					oSums[1] Softmax() results[1]
        // 			bias									bias

        [TestMethod]
        public void Backward1_Test()
        {
            var inputs = new Vector(0.05, 0.1);
            var oTargets = new Vector(0.01, 0.99);
            var nn = new BackPropagationNetwork();
            var hOutputs = nn.TestHidden(inputs);
            var outputs = nn.TestOutput(hOutputs);

            var eTotal = nn.Error(outputs, oTargets);
            Assert.AreEqual(eTotal, 0.29837110876000272);

            var eTotal_out = nn.Errors(outputs.ToVector(), oTargets.ToVector());
            Assert.AreEqual(0.7413650695523157, eTotal_out[0]);
            Assert.AreEqual(-0.21707153467853746, eTotal_out[1]);

            var out_net = Matrix.SigmoidDerivative(outputs).ToVector();
            Assert.AreEqual(out_net[0], 0.18681560180895948);
            Assert.AreEqual(out_net[1], 0.17551005281727122);

            var newOutWeights = nn.Backward(hOutputs.ToVector(), out_net, eTotal_out);
            Assert.AreEqual(newOutWeights[0][0], 0.35891647971788465);
            Assert.AreEqual(newOutWeights[1][0], 0.4086661860762334);
            Assert.AreEqual(newOutWeights[0][1], 0.5113012702387375);
            Assert.AreEqual(newOutWeights[1][1], 0.5613701211079891);

            var newHiddenWeights = nn.BakwardHidden(inputs.ToVector(), hOutputs.ToVector(), out_net.ToVector(), eTotal_out.ToVector());
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
            var inputs = new Vector(0.05, 0.1);
            var oTargets = new Vector(0.01, 0.99);
            var nn = new BackPropagationNetwork();
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
        public void BackwardOutput_Test()
        {
            var nn = new BackPropagationNetwork();
            nn.learnRate = 1;
            nn.Activation = None;
            nn.InvertedActivation = None;
            nn.hoWeights = new Matrix(new[]
            {
                new[] { 0, 0},
                new[] { 0, 0}
            });

            var eTotal_out = new Vector(1, 2);
            var out_net = new Vector(3, 4);
            var hOutputs = new Vector(5, 6);
            
            var w = nn.Backward(hOutputs, out_net, eTotal_out);

            Assert.AreEqual(-(1 * 3 * 5), w[0][0]);
            Assert.AreEqual(-(1 * 3 * 6), w[1][0]);
            Assert.AreEqual(-(2 * 4 * 5), w[0][1]);
            Assert.AreEqual(-(2 * 4 * 6), w[1][1]);

            var t1 = eTotal_out.Transpose().Mul(out_net);
        }

        [TestMethod]
        public void BackwardNone_Test()
        {
            var inputs = new Vector(0, 0);
            var oTargets = new Vector(1, 1);
            var nn = new BackPropagationNetwork();
            nn.learnRate = 1;
            nn.Activation = None;
            nn.InvertedActivation = None;
            var hOutputs = nn.TestHidden(inputs);
            var outputs = nn.TestOutput(hOutputs);

            var eTotal_out = nn.Error(outputs, oTargets);

            var out_net = new double[2];
            out_net[0] = Network1.None(outputs[0]);
            out_net[1] = Network1.None(outputs[1]);
            nn.Print(new Vector(1, 0), new Vector(1, 0));
                Console.Out.WriteLine("---");
            for (var i = 0; i < 20; i++)
            {
                nn.Train(new Vector(1, 0), new Vector(1, 0));
                //nn.Print(new[] { 1.0, 0.0 }, new[] { 1.0, 0.00 });
                eTotal_out = nn.Error(nn.Test(new Vector(1, 0)), new Vector(1, 0));
                Console.Out.WriteLine(eTotal_out);
                //Console.Out.WriteLine();
                nn.Train(new Vector(0, 1), new Vector(0, 1));
                //nn.Print(new[] { 0.0, 1.0 }, new[] { 0.0, 1.0 });
                eTotal_out = nn.Error(nn.Test(new Vector(0, 1)), new Vector(0, 1));
                Console.Out.WriteLine(eTotal_out);
                //Console.Out.WriteLine("---");
            }
        }

        private Matrix None(Matrix x)
        {
            return x;
        }

        [TestMethod]
        public void Backward1_And_Test()
        {
            var nn = new Network1();
            nn.InvertedActivation = Network1.SigmoidDerivative;
            nn.Activation = Network1.Sigmoid;
            for (var i = 0; i < 10000; i++)
            {
                nn.Train(new[] { 0.01, 0.99 }, new[] { 0.01, 0.99 });
                nn.Train(new[] { 0.99, 0.01 }, new[] { 0.99, 0.01 });
            }

            var outputs = nn.Test(new[] { 0.01, 0.99 });
            Console.Out.WriteLine($"0 1 => {outputs[0]} {outputs[1]}");
            outputs = nn.Test(new[] { 0.99, 0.01 });
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
                nn.Train(new[] { 0.01, 0.01 }, new[] { 0.01, 0.99 });
                nn.Train(new[] { 0.99, 0.01 }, new[] { 0.99, 0.01 });
                nn.Train(new[] { 0.01, 0.99 }, new[] { 0.99, 0.01 });
                nn.Train(new[] { 0.99, 0.99 }, new[] { 0.01, 0.99 });
            }

            var outputs = nn.Test(new[] { 0.01, 0.01 });
            Console.Out.WriteLine($"0 0 => {outputs[0]} {outputs[1]}");
            outputs = nn.Test(new[] { 0.99, 0.01 });
            Console.Out.WriteLine($"1 0 => {outputs[0]} {outputs[1]}");
            outputs = nn.Test(new[] { 0.01, 0.99 });
            Console.Out.WriteLine($"0 1 => {outputs[0]} {outputs[1]}");
            outputs = nn.Test(new[] { 0.99, 0.99 });
            Console.Out.WriteLine($"1 1 => {outputs[0]} {outputs[1]}");
        }

        [TestMethod]
        public void ForwardMatrix_Test()
        {
            var inputs = new Vector(new[] { 0.05, 0.1 });
            var otargets = new Vector(new[] { 0.01, 0.99 });
            Console.Out.WriteLine("Targets: " + otargets[0] + "," + otargets[0]);

            var nn = new BackPropagationNetwork();
            var hOutputs = nn.TestHidden(inputs);
            Console.Out.WriteLine("hidden: " + hOutputs[0] + "," + hOutputs[0]);
            var output = nn.TestOutput(hOutputs);
            Console.Out.WriteLine("output: " + output[0] + "," + output[0]);

            var eo = new double[2];
            eo[0] = 0.5 * Math.Pow(otargets[0] - output[0], 2);
            eo[1] = 0.5 * Math.Pow(otargets[1] - output[1], 2);
            var eTotal = eo[0] + eo[1];
            Console.Out.WriteLine("etotal: " + eTotal);
        }
    }
}