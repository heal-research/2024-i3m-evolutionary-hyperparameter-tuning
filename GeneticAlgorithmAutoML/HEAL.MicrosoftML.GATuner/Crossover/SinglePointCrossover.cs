namespace HEAL.MicrosoftML.GATuner.Crossover
{
    public class SinglePointCrossover : IGeneticAlgorithmCrossover
    {
        private Random _rnd;

        public SinglePointCrossover()
        {
            _rnd = new Random();
        }

        public Individual Crossover(Individual parent1, Individual parent2)
        {
            // Perform a single point crossover
            int crossoverPoint = _rnd.Next(parent1.Parameters.Length);

            return new Individual
            {
                Parameters = parent1.Parameters.Take(crossoverPoint).Concat(parent2.Parameters.Skip(crossoverPoint)).ToArray(),
                Fitness = 0
            };
        }
    }
}
