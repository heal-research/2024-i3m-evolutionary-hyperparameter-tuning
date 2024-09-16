using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HEAL.MicrosoftML.GATuner.Crossover
{
    public class MultiPointCrossover : IGeneticAlgorithmCrossover
    {
        private Random _rnd;
        private long _numPoints;

        public MultiPointCrossover(uint numPoints)
        {
            _rnd = new Random();
            _numPoints = numPoints;
        }

        public Individual Crossover(Individual parent1, Individual parent2)
        {
            int parametersLength = parent1.Parameters.Length;

            // Ensure numPoints doesn't exceed parameters length - 1
            _numPoints = Math.Min(_numPoints, parametersLength - 1);

            var childParameters = new double[parametersLength];

            // Generate N crossover points
            List<int> crossoverPoints = new List<int>();
            for (int i = 0; i < _numPoints; i++)
            {
                crossoverPoints.Add(_rnd.Next(1, parametersLength));
            }

            // Add start and end points and sort
            crossoverPoints.Add(0);
            crossoverPoints.Add(parametersLength);
            crossoverPoints.Sort();

            // Perform an N-point crossover
            for (int i = 0; i < crossoverPoints.Count - 1; i++)
            {
                var sourceParent = i % 2 == 0 ? parent1 : parent2;
                Array.Copy(sourceParent.Parameters, crossoverPoints[i], childParameters, crossoverPoints[i], crossoverPoints[i + 1] - crossoverPoints[i]);
            }

            return new Individual
            {
                Parameters = childParameters,
                Fitness = 0,
            };
        }
    }
}
