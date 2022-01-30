using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Transactions;

namespace SelfGorwingNN
{
    public class Network1
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
        public double learnRate = 0.5;
        public Func<double, double> Activation { get; set; }
        public Func<double, double> InvertedActivation { get; set; }

        public Network1()
        {
            Activation = Sigmoid;
            InvertedActivation = InvertedActivation;
        }

        public double[] Test(double[] inputs)
        {
            var hOutputs = TestHidden(inputs);
            return TestOutput(hOutputs);
        }

        public double Error(IEnumerable<double> outputs, IEnumerable<double> oTargets)
        {
            var sum = 0.0;
            var count = 0;
            using var o = outputs.GetEnumerator();
            using var t = oTargets.GetEnumerator();
            while (o.MoveNext() && t.MoveNext())
            {
                sum += 0.5 *Math.Pow(t.Current - o.Current, 2);
                count++;
            }

            return sum; // / count;
        }

        public IEnumerable<double> Errors(IEnumerable<double> outputs, IEnumerable<double> oTargets)
        {
            using var o = outputs.GetEnumerator();
            using var t = oTargets.GetEnumerator();
            while (o.MoveNext() && t.MoveNext())
            {
                yield return -(t.Current - o.Current);
            }
        }

        public void Train(double[] inputs, double[] oTargets)
        {
            var hOutputs = TestHidden(inputs);
            var outputs = TestOutput(hOutputs);

            var out_net = new double[2];
            out_net[0] = InvertedActivation(outputs[0]);
            out_net[1] = InvertedActivation(outputs[1]);

            var eTotal_out = Errors(outputs, oTargets).ToArray();

            var newOutWeights = Backward(hOutputs, out_net, eTotal_out);

            var newHiddenWeights = BakwardHidden(inputs, hOutputs, out_net, eTotal_out);

            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    ihWeights[i][j] =  newHiddenWeights[i][j];
                    hoWeights[i][j] =  newOutWeights[i][j];
                }
            }
        }

        public double[] TestHidden(double[] inputs)
        {
            var hOutputs = new double[2];
            hOutputs[0] = Activation(inputs[0] * ihWeights[0][0] + inputs[1] * ihWeights[1][0] + hBiases[0]); // .05 * .15 + .10 *.20
            hOutputs[1] = Activation(inputs[0] * ihWeights[0][1] + inputs[1] * ihWeights[1][1] + hBiases[1]); // .05 * .20 + .10 *.30

            // Activation(inputs * ihWeights + hBiases)

            return hOutputs;
        }

        public double[] TestOutput(double[] hOutputs)
        {
            var oSums = new double[2];
            oSums[0] = Activation(hOutputs[0] * hoWeights[0][0] + hOutputs[1] * hoWeights[1][0] + oBiases[0]); // ? * .40 + ? *.50
            oSums[1] = Activation(hOutputs[0] * hoWeights[0][1] + hOutputs[1] * hoWeights[1][1] + oBiases[1]); // ? * .45 + ? *.55

            return oSums;
        }

        public static Matrix Sigmoid(Matrix x)
        {
            var result = new Matrix(x.Rows, x.Cols);
            for (var row = 0; row < x.Rows; row++)
            {
                for (var column = 0; column < x.Cols; column++)
                {
                    result[row][column] = Sigmoid(x[row][column]);
                }

            }

            return result;
        }

        public static double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        public static Matrix SigmoidDerivative(Matrix x)
        {
            var result = new Matrix(x.Rows, x.Cols);
            for (var row = 0; row < x.Rows; row++)
            {
                for (var column = 0; column < x.Cols; column++)
                {
                    result[row][column] = SigmoidDerivative(x[row][column]);
                }

            }

            return result;
        }

        public static double SigmoidDerivative(double x)
        {
            return x * (1 - x);
        }

        public static double None(double x)
        {
            return x;
        }

        //eTotal_out[0] * out_net[0] är alltid sammanslagen
        public double[][] Backward(double[] hOutputs, double[] out_net, double[] eTotal_out)
        {
            var w = new double[2][]
            {
                new double[2],
                new double[2]
            };
            double neto1_w5 = hOutputs[0];
            double eTotal_w5 = eTotal_out[0] * out_net[0] * neto1_w5;
            w[0][0] = hoWeights[0][0] - learnRate * eTotal_w5;

            double neto1_w6 = hOutputs[1];
            double eTotal_w6 = eTotal_out[0] * out_net[0] * neto1_w6;
            w[1][0] = hoWeights[1][0] - learnRate * eTotal_w6;

            double neto1_w7 = hOutputs[0];
            double eTotal_w7 = eTotal_out[1] * out_net[1] * neto1_w7;
            w[0][1] = hoWeights[0][1] - learnRate * eTotal_w7;

            double neto1_w8 = hOutputs[1];
            double eTotal_w8 = eTotal_out[1] * out_net[1] * neto1_w8;
            w[1][1] = hoWeights[1][1] - learnRate * eTotal_w8;

            return w;
        }

        public double[][] BakwardHidden(double[] inputs, double[] hOutputs, double[] out_net, double[] eTotal_out)
        {
            var w = new double[2][]
            {
                new double[2],
                new double[2]
            };
            var eo1_neto1 = eTotal_out[0] * out_net[0];
            var neto1_outh1 = hoWeights[0][0];
            var eo1_outh1 = eo1_neto1 * neto1_outh1;
            var eo2_neto2 = eTotal_out[1] * out_net[1];
            var neto2_outh1 = hoWeights[0][1]; // w7 = 0.5 
            var eo2_outh1 = eo2_neto2 * neto2_outh1;

            var neth1_w1 = inputs[0];
            var outh1_neth1 = Network1.SigmoidDerivative(hOutputs[0]);
            var eTotal_outh1 = eo1_outh1 + eo2_outh1;
            var etotal_w1 = neth1_w1 * eTotal_outh1 * outh1_neth1;
            w[0][0] = ihWeights[0][0] - learnRate * etotal_w1;

            var neth1_w2 = inputs[1];
            var etotal_w2 = eTotal_outh1 * outh1_neth1 * neth1_w2;
            w[1][0] = ihWeights[1][0] - learnRate * etotal_w2;

            var neto1_outh2 = hoWeights[1][0]; //w6=.45
            var eo1_outh2 = neto1_outh2 * eo1_neto1;
            var neto2_outh2 = hoWeights[1][1]; //w8=.55
            var eo2_outh2 = neto2_outh2 * eo2_neto2;

            var eTotal_outh2 = eo1_outh2 + eo2_outh2;
            var neth2_w3 = inputs[0];  // w3 =.25
            var outh2_neth2 = Network1.SigmoidDerivative(hOutputs[1]);
            var etotal_w3 = neth2_w3 * eTotal_outh2 * outh2_neth2;
            w[0][1] = ihWeights[0][1] - learnRate * etotal_w3; // .25

            var neth2_w4 = inputs[1];
            var etotal_w4 = neth2_w4 * eTotal_outh2 * outh2_neth2;
            w[1][1] = ihWeights[1][1] - learnRate * etotal_w4; // .30

            return w;
        }

        public void Print(double[] inputs = null, double[] targets = null)
        {
            //  -i1   0.15    h1  0.40  o1
            //      \ 0.25        0.45
            //      / 0.20        0.50
            //  -i2   0.30    h2  0.55  o2
            var format = "F2";
            var i1 = inputs?[0].ToString(format) ?? " i1 ";
            var i2 = inputs?[1].ToString(format) ?? " i2 ";
            var o1 = " o1 ";
            var o2 = " o2 ";
            if (inputs != null)
            {
                var o = Test(inputs);
                o1 = o[0].ToString(format) + $" ({targets?[0].ToString(format)})";
                o2 = o[1].ToString(format) + $" ({targets?[1].ToString(format)})";
            }
            Console.Out.WriteLine($" {i1}    {ihWeights[0][0].ToString(format)}   h1  {hoWeights[0][0].ToString(format)}   {o1}");
            Console.Out.WriteLine($"         {ihWeights[1][0].ToString(format)}       {hoWeights[1][0].ToString(format)}     ");
            Console.Out.WriteLine($"         {ihWeights[0][1].ToString(format)}       {hoWeights[0][1].ToString(format)}     ");
            Console.Out.WriteLine($" {i2}    {ihWeights[1][1].ToString(format)}   h1  {hoWeights[1][1].ToString(format)}   {o2}");
        }

        public void PrintGraph()
        {
            var size = 80.0;
            for (int row = 0; row < size; row++)
            {
                for (int column = 0; column < size; column++)
                {
                    if (Test(new[] { row / size, column / size }).First() > 0.5)
                    {
                        Console.Out.Write('+');
                    }
                    else
                    {
                        Console.Out.Write('o');
                    }
                }

                Console.Out.WriteLine();
            }
        }
    }
}
