namespace HEAL.MicrosoftML.GATuner.Crossover
{
    public interface IGeneticAlgorithmCrossover
    {
        public Individual Crossover(Individual parent1, Individual parent2);
    }
}
