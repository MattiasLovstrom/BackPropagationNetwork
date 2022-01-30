using Microsoft.VisualStudio.TestTools.UnitTesting;
using SelfGorwingNN;
using System;
using System.Collections.Generic;
using System.Text;

namespace SelfGorwingNN.Tests
{
    [TestClass()]
    public class MatrixTests
    {
        [TestMethod]
        public void Constructor()
        {
            //var a = new Matrix(2, 4,
            //    1.0,2.0,
            //    3.0,4.0,
            //    5.0,6.0,
            //    7.0,8.0
            //);
        }

        [TestMethod]
        public void Vector()
        {
            var a = new Vector(new [] {1,2});
            var b = new Matrix(new []
            {
                new []{1}, 
                new []{2}
            });
            var c = a.Mul(b);

        }

        [TestMethod]
        public void Multiply4_2x2_3()
        {
            var a = new Matrix(new[]
            {
                    new [] {1.0,2.0},
                    new [] {3.0,4.0},
                    new [] {5.0,6.0},
                    new [] {7.0,8.0}
                });
            var b = new Matrix(new[]
            {
                new [] {11.0,12.0,13.0},
                new [] {21.0,22.0,23.0}
            });

            Assert.AreEqual(4, (a.Mul(b)).Rows);
            Assert.AreEqual(3, (a.Mul(b)).Cols);
        }

        [TestMethod]
        public void Multiply2_1x1_2()
        {
            var a = new Matrix(new[]
            {
                new [] {1},
                new [] {2}
            });
            var b = new Matrix(new[]
            {
                new [] {3,4}
            });

            Assert.AreEqual(2, (a.Mul(b)).Rows);
            Assert.AreEqual(2, (a.Mul(b)).Cols);
            Assert.AreEqual(1*3, (a.Mul(b))[0][0]);
            Assert.AreEqual(1*4, (a.Mul(b))[0][1]);
            Assert.AreEqual(2*3, (a.Mul(b))[1][0]);
            Assert.AreEqual(2*4, (a.Mul(b))[1][1]);
        }

        [TestMethod]
        public void MultiplyTree()
        {
            var eTotal_out = new Vector(1, 2);
            var out_net = new Vector(3, 4);
            var hOutputs = new Vector(5, 6);

            var w = hOutputs.Transpose().Mul(eTotal_out * out_net);
            Assert.AreEqual(1 * 3 * 5, w[0][0]);
            Assert.AreEqual(1 * 3 * 6, w[1][0]);
            Assert.AreEqual(2 * 4 * 5, w[0][1]);
            Assert.AreEqual(2 * 4 * 6, w[1][1]);
        }
    }
}