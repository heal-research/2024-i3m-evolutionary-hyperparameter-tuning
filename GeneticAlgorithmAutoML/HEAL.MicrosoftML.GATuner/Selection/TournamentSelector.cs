using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HEAL.MicrosoftML.GATuner.Selection
{
    public class TournamentSelector : IGeneticAlgorithmSelector
    {
        private uint _tournamentSize;
        private Random _rnd;


        public TournamentSelector(uint tournamentSize)
        {
            _tournamentSize = tournamentSize;
            _rnd = new Random();
        }


        public Tuple<Individual, Individual> Select(List<Individual> population)
        {
            Individual parent1 = TournamentSelection(population);
            Individual parent2 = TournamentSelection(population);
            return new Tuple<Individual, Individual>(parent1, parent2);
        }

        private Individual TournamentSelection(List<Individual> population)
        {
            List<Individual> tournamentCompetitors = [];

            for (int i = 0; i < _tournamentSize; i++)
            {
                int randomIndex = _rnd.Next(population.Count);
                tournamentCompetitors.Add(population[randomIndex]);
            }

            // Select the fittest individual among the tournament competitors
            return tournamentCompetitors.OrderByDescending(individual => individual.Fitness).First();
        }
    }
}
