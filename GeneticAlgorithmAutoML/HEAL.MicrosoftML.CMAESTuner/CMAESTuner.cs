using Microsoft.ML.AutoML;
using Microsoft.ML.SearchSpace;

namespace HEAL.MicrosoftML.CMAESTuner
{

    public class CMAESTuner : ITuner
    {
        private readonly SearchSpace _searchSpace;


        public Random Random { get; init; } = new();
        public int ProblemSize { get; init; } = 0;
        public int Lambda { get; init; } = 30;
        public int MaxGenerations { get; init; } = 1_00;
        public double InitialSigma { get; init; } = 0.5;
        public int Digits { get; init; } = 10; // orig: 10



        public CMAESTuner
        (
            SearchSpace searchSpace
        ) 
        {
            _searchSpace = searchSpace;
            _normalRandom = new(0.0, 1.0, Random);

            // Create initial solution and assign fitness
            _optimizedSolution = new(CreateRandomSolution(), 0.0);

            _parameters = new(_optimizedSolution.Values.Length, Lambda, InitialSigma, MaxGenerations);
            ProblemSize = _optimizedSolution.Values.Length;
        }


        private Solution Mutate(Solution solution, NormalDistributedRandom normalRandom, CMAParameters p)
        {
            double[] values = solution.Values.ToArray();
            double[] d = p.D.Select(x => x * normalRandom.NextDouble()).ToArray();

            for (int i = 0; i < ProblemSize; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < ProblemSize; j++)
                {
                    sum += p.B[i, j] * d[j];
                }

                double val = double.Round(values[i] + p.Sigma * sum, Digits);  // m + sig * Normal(0,C)
                val = val switch
                {
                    > 1.0 => 1.0,
                    < 0.0 => 0.0,
                    _ => val
                };

                values[i] = val;
            }
            return new Solution(values, 0.0); // needs to be evaluated
        }

        private Solution Recombine(Solution[] children, CMAParameters p)
        {
            var ch = children.Take(p.Mu).Zip(p.Weights, (s, w) => s.Values.Select(x => x * w).ToArray());
            double[] values = Enumerable.Range(0, ProblemSize)
                                        .Select(i => double.Round(ch.Select(x => x[i]).Sum(), Digits))
                                        .ToArray();
            return new Solution(values, 0.0);
        }

        private bool UpdateParameters(CMAParameters p, int iteration, Solution oldSolution, Solution newSolution, Solution[] children)
        {
            p.QualityHistory.Enqueue(children[0].Quality);
            while (p.QualityHistory.Count > p.QualityHistorySize) p.QualityHistory.Dequeue();

            for (int i = 0; i < ProblemSize; i++)
            {
                p.BDz[i] = Math.Sqrt(p.MuEff) * (newSolution.Values[i] - oldSolution.Values[i]) / p.Sigma;
            }

            var artmp = new double[ProblemSize];
            for (int i = 0; i < ProblemSize; i++)
            {
                var sum = 0.0;
                for (int j = 0; j < ProblemSize; j++)
                {
                    sum += p.B[j, i] * p.BDz[j];
                }
                artmp[i] = sum / p.D[i];
            }
            for (int i = 0; i < ProblemSize; i++)
            {
                var sum = 0.0;
                for (int j = 0; j < ProblemSize; j++)
                {
                    sum += p.B[i, j] * artmp[j];
                }
                p.PS[i] = (1 - p.CS) * p.PS[i] + Math.Sqrt(p.CS * (2 - p.CS)) * sum;
            }

            var normPS = Math.Sqrt(p.PS.Select(x => x * x).Sum());
            var hsig = normPS / Math.Sqrt(1 - Math.Pow(1 - p.CS, 2 * iteration)) / p.ChiN < 1.4 + 2.0 / (ProblemSize + 1) ? 1.0 : 0.0;
            for (int i = 0; i < p.PC.Length; i++)
            {
                p.PC[i] = (1 - p.CC) * p.PC[i] + hsig * Math.Sqrt(p.CC * (2 - p.CC)) * p.BDz[i];
            }

            if (p.CCov > 0)
            {
                for (int i = 0; i < ProblemSize; i++)
                {
                    for (int j = 0; j < ProblemSize; j++)
                    {
                        p.C[i, j] = (1 - p.CCov) * p.C[i, j] + p.CCov * (1 / p.MuCov) * (p.PC[i] * p.PC[j] + (1 - hsig) * p.CC * (2 - p.CC) * p.C[i, j]);
                        for (int k = 0; k < p.Mu; k++)
                        {
                            p.C[i, j] += p.CCov * (1 - 1 / p.MuCov) * p.Weights[k] * (children[k].Values[i] - oldSolution.Values[i]) * (children[k].Values[j] - oldSolution.Values[j]) / (p.Sigma * p.Sigma);
                        }
                    }
                }
            }
            p.Sigma *= Math.Exp((p.CS / p.Damps) * (normPS / p.ChiN - 1));

            // testAndCorrectNumerics
            double fac = 1;
            if (p.D.Max() < 1e-6)
                fac = 1.0 / p.D.Max();
            else if (p.D.Min() > 1e4)
                fac = 1.0 / p.D.Min();

            if (fac != 1.0)
            {
                p.Sigma /= fac;
                for (int i = 0; i < ProblemSize; i++)
                {
                    p.PC[i] *= fac;
                    p.D[i] *= fac;
                    for (int j = 0; j < ProblemSize; j++)
                        p.C[i, j] *= fac * fac;
                }
            }
            // end testAndCorrectNumerics

            bool success;  // indicates if the algorithm is in a degenerated state and should be terminated
            success = alglib.smatrixevd(p.C, ProblemSize, 1, true, out double[] d, out double[,] b);
            p.D = d;
            p.B = b;

            // assign D to eigenvalue square roots
            for (int i = 0; i < ProblemSize; i++)
            {
                if (p.D[i] <= 0)
                { // numerical problem?
                    success = false;
                    p.D[i] = 0;
                }
                else p.D[i] = Math.Sqrt(p.D[i]);
            }

            if (p.D.Min() == 0.0) p.AxisRatio = double.PositiveInfinity;
            else p.AxisRatio = p.D.Max() / p.D.Min();

            return success;
        }

        private NormalDistributedRandom _normalRandom;
        private CMAParameters _parameters;
        private Solution _optimizedSolution;
        private Solution[] _children;
        private int _childrenIndex = 0;
        private bool _optimizedSolutionEvaluated = false;

        public Parameter Propose(TrialSettings settings)
        {
            if (_children != null && _children.Length > 0 && _childrenIndex < _children.Length)
            {
                return _searchSpace.SampleFromFeatureSpace(_children[_childrenIndex++].Values);
            }

            return _searchSpace.SampleFromFeatureSpace(_optimizedSolution.Values);
        }

        public void Update(TrialResult result)
        {
            if (!_optimizedSolutionEvaluated)
            {
                _optimizedSolution.Quality = 1 - result.Loss;
                Console.WriteLine("Optimized: Metric = " + result.Metric + "   Quality = " + _optimizedSolution.Quality + "    " + string.Join(' ', _optimizedSolution.Values) + "  -->  " + _children?.Select(c => c.Quality)?.Average());
                _optimizedSolutionEvaluated = true;
                _childrenIndex = 0;
                _children = Enumerable.Range(0, Lambda)
                                    .Select(x => Mutate(_optimizedSolution, _normalRandom, _parameters))
                                    .OrderBy(x => x.Quality) // orderbydescending
                                    .ToArray();

                return;
            }

            // Update fitness of the individual that was just evaluated
            _children[_childrenIndex - 1].Quality = 1 - result.Loss;

            if (_childrenIndex >= _children.Length)
            {
                // Recombine children to create new solution
                Solution newSolution = Recombine(_children, _parameters);

                UpdateParameters(_parameters, 0, _optimizedSolution, newSolution, _children);
                _optimizedSolution = newSolution;
                _optimizedSolutionEvaluated = false;
            }
        }


        private double[] CreateRandomSolution()
        {
            var d = _searchSpace.FeatureSpaceDim;
            return Enumerable.Repeat(0, d).Select(i => Random.NextDouble()).ToArray();
        }
    }
}
