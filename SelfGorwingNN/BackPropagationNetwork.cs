using System;

namespace SelfGorwingNN
{
    public class BackPropagationNetwork
    {
        // i1=>h1, i2=>h1
        public Matrix ihWeights = new Matrix(new[] {
            new[] {0.15, 0.25},
            new[] {0.20, 0.30}
        });
        public Vector hBiases = new Vector(0.35, 0.35);
        public Matrix hoWeights = new Matrix(new[]
        {
            new[] { 0.40, 0.50},
            new[] { 0.45, 0.55}
        });
        public Vector oBiases = new Vector(0.60, 0.60);
        public double learnRate = 0.5;

        public Func<Matrix, Matrix> Activation { get; set; }
        public Func<Matrix, Matrix> InvertedActivation { get; set; }

        public BackPropagationNetwork()
        {
            Activation = Matrix.Sigmoid;
            InvertedActivation = InvertedActivation;
        }

        public Vector Test(Vector inputs)
        {
            var hOutputs = TestHidden(inputs);
            return TestOutput(hOutputs);
        }

        public double Error(Vector outputs, Vector oTargets)
        {
            return (0.5 * (oTargets - outputs).Pow(2)).Sum();
        }

        public Vector Errors(Vector outputs, Vector oTargets)
        {
            return -(oTargets - outputs);
        }

        public void Train(Vector inputs, Vector oTargets)
        {
            var hOutputs = TestHidden(inputs);
            var outputs = TestOutput(hOutputs);

            var out_net = InvertedActivation(outputs).ToVector();

            var eTotal_out = Errors(outputs, oTargets);

            var newOutWeights = Backward(hOutputs, out_net, eTotal_out);

            var newHiddenWeights = BakwardHidden(inputs, hOutputs, out_net, eTotal_out);

            ihWeights = newHiddenWeights;
            hoWeights = newOutWeights;

        }

        public Vector TestHidden(Vector inputs)
        {
            return Activation(inputs.Mul(ihWeights) + hBiases).ToVector(); // .05 * .15 + .10 *.20
        }

        public Vector TestOutput(Vector hOutputs)
        {
            return Activation(hOutputs.Mul(hoWeights) + oBiases).ToVector(); // ? * .40 + ? *.50
        }

        //eTotal_out[0] * out_net[0] är alltid sammanslagen
        public Matrix Backward(Vector hOutputs, Vector out_net, Vector eTotal_out)
        {
            return hoWeights - learnRate * hOutputs.Transpose().Mul(eTotal_out * out_net);
        }

        public Matrix BakwardHidden(Vector inputs, Vector hOutputs, Vector out_net, Vector eTotal_out)
        {
            //var neth1_w1 = inputs[0];
            //var neth1_w2 = inputs[1];
            //var neth2_w3 = inputs[0];  
            //var neth2_w4 = inputs[1];

            //var neto1_outh1 = hoWeights[0][0];
            //var neto2_outh1 = hoWeights[0][1]; 
            //var neto1_outh2 = hoWeights[1][0]; 
            //var neto2_outh2 = hoWeights[1][1]; 

            var w = new Matrix(2, 2);
            var eo1_neto = (eTotal_out * out_net).ToVector();
            //var eo1_neto1 = eTotal_out[0] * out_net[0];
            //var eo2_neto2 = eTotal_out[1] * out_net[1];

            var eo1_outh1 = hoWeights[0][0] * eo1_neto[0];
            var eo2_outh1 = hoWeights[0][1] * eo1_neto[1];
            var eo1_outh2 = hoWeights[1][0] * eo1_neto[0];
            var eo2_outh2 = hoWeights[1][1] * eo1_neto[1];

            //var outh1_neth1 = Network1.SigmoidDerivative(hOutputs[0]);
            //var outh2_neth2 = Network1.SigmoidDerivative(hOutputs[1]);
            var outh_neth = Matrix.SigmoidDerivative(hOutputs).ToVector();

            var eTotal_outh1 = eo1_outh1 + eo2_outh1;
            var eTotal_outh2 = eo1_outh2 + eo2_outh2;
            //var eTotal_outh1 = eo_outh[0][0] + eo_outh[1][0];
            //var eTotal_outh2 = eo_outh[0][1] + eo_outh[1][1];

            var etotal_w1 = inputs[0] * eTotal_outh1 * outh_neth[0];
            var etotal_w2 = inputs[1] * eTotal_outh1 * outh_neth[0];
            var etotal_w3 = inputs[0] * eTotal_outh2 * outh_neth[1];
            var etotal_w4 = inputs[1] * eTotal_outh2 * outh_neth[1];
           
            w[0][0] = ihWeights[0][0] - learnRate * etotal_w1;
            w[1][0] = ihWeights[1][0] - learnRate * etotal_w2;
            w[0][1] = ihWeights[0][1] - learnRate * etotal_w3; // .25
            w[1][1] = ihWeights[1][1] - learnRate * etotal_w4; // .30

            return w;
        }

        public void Print(Vector inputs = null, Vector targets = null)
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
                    var t = Test(new Vector(new[] { row / size, column / size }));
                    if (t[0] > 0.5)
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
