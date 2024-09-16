using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HEAL.MicrosoftML.GATuner.Selection
{
    public class RandomSelector : IGeneticAlgorithmSelector
    {
        private Random _rnd;

        public RandomSelector()
        {
            _rnd = new Random();
        }

        public Tuple<Individual, Individual> Select(List<Individual> population)
        {
            var index1 = _rnd.Next(population.Count);
            var index2 = _rnd.Next(population.Count);

            while (index2 == index1)
            {
                index2 = _rnd.Next(population.Count);
            }

            return new Tuple<Individual, Individual>(population[index1], population[index2]);
        }
    }
}
