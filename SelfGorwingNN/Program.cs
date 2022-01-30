using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SelfGorwingNN
{
    class Program
    {
        static NeuralNetwork nn = new NeuralNetwork();

        static void Main(string[] args)
        {
            TestNetworkXor();
            //Test1_1_1Layer();
            //CodingBackProp.BackPropProgram.Test();
            //CodingBackProp.BackPropProgram.TestXor();
            //CodingBackProp.BackPropProgram.Test1_1_1();
            //Not();
            //And();
            //new BackProp().Train();
            //TestBackPropagation();
            //MyExample();
            //TestExample();
        }

        private static void TestNetworkXor()
        {
            var andTestSet = new[]
            {
                new { Indata = new[] { 0.99, 0.99 }, Expected = new[] { 0.01, 0.99 } },
                new { Indata = new[] { 0.99, 0.01 }, Expected = new[] { 0.99, 0.01 } },
                new { Indata = new[] { 0.01, 0.99 }, Expected = new[] { 0.99, 0.01 } },
                new { Indata = new[] { 0.01, 0.01 }, Expected = new[] { 0.01, 0.99 } }
            };

            var nn = new Network1();
            nn.learnRate = 0.01;
            nn.InvertedActivation = Network1.SigmoidDerivative;
            nn.Activation = Network1.Sigmoid;
            for (var i = 0; i < 1000000; i++)
            {
                foreach (var test in andTestSet)
                {
                    nn.Train(test.Indata, test.Expected);
                }
            }
            Console.Out.WriteLine("--- Test ---");
            foreach (var test in andTestSet)
            {
                var result = nn.Test(test.Indata).ToArray();
                Console.Out.WriteLine($"{test.Indata[0]} and {test.Indata[1]} = {result[0] + "," + result[1]}");
            }

            nn.PrintGraph();
        }


        private static void PrintGraph(Network1 nn)
        {
            //nn.Verbose = false;

            var size = 80.0;
            for (int row = 0; row < size; row++)
            {
                for (int column = 0; column < size; column++)
                {
                    if (nn.Test(new[] { row / size, column / size }).First() > 0.5)
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

        private static void And()
        {
            Console.WriteLine("And");
            var i = new List<double> { 1, 1 };
            var o = new List<double> { 1 };
            var error = nn.Test(i, o);
            Display(error);
            nn.Train(i, error, o);
            error = nn.Test(i, o);
            Display(error);

            i = new List<double> { 0, 1 };
            o = new List<double> { 0 };
            error = nn.Test(i, o);
            Display(error);
            nn.Train(i, error, o);
            error = nn.Test(i, o);
            Display(error);

            i = new List<double> { 1, 0 };
            o = new List<double> { 0 };
            error = nn.Test(i, o);
            Display(error);
            nn.Train(i, error, o);
            error = nn.Test(i, o);
            Display(error);

            i = new List<double> { 0, 0 };
            o = new List<double> { 0 };
            error = nn.Test(i, o);
            Display(error);
            nn.Train(i, error, o);
            error = nn.Test(i, o);
            Display(error);
        }

        private static void Not()
        {
            Console.WriteLine("Not");
            var i = new List<double> { 1 };
            var o = new List<double> { -1 };
            var error = nn.Test(i, o);
            Display(error);
            nn.Train(i, error, o);
            error = nn.Test(i, o);
            Display(error);

            i = new List<double> { -1 };
            o = new List<double> { 1 };
            error = nn.Test(i, o);
            Display(error);
            nn.Train(i, error, o);
            error = nn.Test(i, o);
            Display(error);
        }

        private static void Display(List<double> result)
        {
            var line = new StringBuilder();
            foreach (var d in result)
            {
                line.Append(d.ToString("0.00")).Append(" ");
            }
            Console.Out.WriteLine(line);
        }
    }

    internal class NeuralNetwork
    {
        // i[0] * nn[0][0] = o[0]
        Dictionary<int, Dictionary<int, double>> nn = new Dictionary<int, Dictionary<int, double>>();

        public void Train(List<double> input, List<double> error, List<double> output)
        {
            for (int outputNode = 0; outputNode < output.Count; outputNode++)
            {
                if (error[outputNode] == 0.0) continue;

                for (int inputNode = 0; inputNode < input.Count; inputNode++)
                {
                    if (!nn.ContainsKey(inputNode))
                    {
                        nn.Add(inputNode, new Dictionary<int, double>());
                    }

                    if (!nn[inputNode].ContainsKey(outputNode))
                    {
                        nn[inputNode][outputNode] = error[outputNode];
                    }
                    else
                    {
                        throw new Exception("Can't train!");
                    }
                }
            }
        }

        public List<double> Test(List<double> input, List<double> output)
        {
            var message = new StringBuilder();
            var result = new List<double>();
            for (int outputNode = 0; outputNode < output.Count; outputNode++)
            {
                var currentResult = 0.0;
                for (int inputNode = 0; inputNode < input.Count; inputNode++)
                {
                    if (nn.ContainsKey(inputNode) && nn[inputNode].ContainsKey(outputNode))
                    {
                        message.Append(input[inputNode]).Append(" * ").Append(nn[inputNode][outputNode]).Append("+");
                        currentResult += input[inputNode] * nn[inputNode][outputNode];
                    }
                }

                if (message.Length > 0)
                {
                    message.Remove(message.Length - 1, 1);
                }
                message.Append("=");
                message.Append(currentResult);
                Console.Out.WriteLine(message);
                result.Add(output[outputNode] - currentResult);
            }

            return result;
        }
    }
}
