using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HEAL.MicrosoftML.GATuner.Selection
{
    public class FitnessProportionalSelector : IGeneticAlgorithmSelector
    {
        private Random _rnd;

        public FitnessProportionalSelector()
        {
            _rnd = new Random();
        }

        public Tuple<Individual, Individual> Select(List<Individual> population)
        {
            // Sort the population based on fitness
            population = population.OrderByDescending(i => i.Fitness).ToList();

            // Calculate total fitness of population
            double totalFitness = population.Sum(i => i.Fitness);

            // Generate two random numbers between 0 and total fitness
            double rand1 = _rnd.NextDouble() * totalFitness;
            double rand2 = _rnd.NextDouble() * totalFitness;

            while (rand2 == rand1)
            {
                rand2 = _rnd.NextDouble() * totalFitness;
            }

            // Select first individual
            Individual selected1 = default!;
            double accumulatedFitness = 0;
            for (int i = 0; i < population.Count; i++)
            {
                accumulatedFitness += population[i].Fitness;
                if (accumulatedFitness >= rand1)
                {
                    selected1 = population[i];
                    break;
                }
            }

            // Select second individual
            Individual selected2 = default!;
            accumulatedFitness = 0;
            for (int i = 0; i < population.Count; i++)
            {
                accumulatedFitness += population[i].Fitness;
                if (accumulatedFitness >= rand2)
                {
                    selected2 = population[i];
                    break;
                }
            }

            // Return the pair of selected individuals
            return new Tuple<Individual, Individual>(selected1, selected2);
        }
    }
}
