using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HEAL.MicrosoftML.GATuner.Mutation
{
    public class GaussianMutator : IGeneticAlgorithmMutator
    {
        private double _mutationRate;
        private Random _rnd;
        private double _stdDeviation;

        public GaussianMutator(double mutationRate, double stdDeviation)
        {
            _mutationRate = mutationRate switch
            {
                < 0.0 => 0.0,
                > 1.0 => 1.0,
                _ => mutationRate
            };

            _rnd = new Random();
            _stdDeviation = stdDeviation;
        }

        public void Mutate(Individual individual)
        {
            for (int i = 0; i < individual.Parameters.Length; i++)
            {
                if (_rnd.NextDouble() < _mutationRate)
                {
                    double u1 = 1.0 - _rnd.NextDouble(); // uniform(0,1] random doubles
                    double u2 = 1.0 - _rnd.NextDouble();
                    double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                             Math.Sin(2.0 * Math.PI * u2); // random normal(0,1)
                    double gaussian = _stdDeviation * randStdNormal; // random normal(mean,stdDev^2)

                    individual.Parameters[i] += gaussian;

                    // Ensure parameters remain within specified bounds
                    individual.Parameters[i] = Math.Max(0.0, Math.Min(1.0, individual.Parameters[i]));
                }
            }
        }
    }
}
