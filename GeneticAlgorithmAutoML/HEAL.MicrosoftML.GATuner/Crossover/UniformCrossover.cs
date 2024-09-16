using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HEAL.MicrosoftML.GATuner.Crossover
{
    public class UniformCrossover : IGeneticAlgorithmCrossover
    {
        private Random _rnd;

        public UniformCrossover()
        {
            _rnd = new Random();
        }

        public Individual Crossover(Individual parent1, Individual parent2)
        {
            var child = new double[parent1.Parameters.Length];

            // Perform a uniform crossover
            for (int i = 0; i < parent1.Parameters.Length; i++)
            {
                child[i] = _rnd.NextDouble() < 0.5 ? parent1.Parameters[i] : parent2.Parameters[i];
            }

            return new Individual
            {
                Parameters = child,
                Fitness = 0
            };
        }
    }
}
