namespace HEAL.MicrosoftML.GATuner.Crossover
{
    public class AveragingCrossover : IGeneticAlgorithmCrossover
    {
        public Individual Crossover(Individual parent1, Individual parent2)
        {
            return new Individual
            {
                Parameters = parent1.Parameters.Zip(parent2.Parameters, (p1, p2) => (p1 + p2) / 2).ToArray(),
                Fitness = 0
            };
        }
    }
}
