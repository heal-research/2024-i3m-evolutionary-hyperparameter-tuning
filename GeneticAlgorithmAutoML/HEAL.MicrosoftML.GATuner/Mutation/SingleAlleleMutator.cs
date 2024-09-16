namespace HEAL.MicrosoftML.GATuner.Mutation
{
    public class SingleAlleleMutator : IGeneticAlgorithmMutator
    {
        private readonly double _mutationRate;
        private readonly Random _rnd;

        public SingleAlleleMutator(double mutationRate)
        {
            _mutationRate = mutationRate switch
            {
                < 0.0 => 0.0,
                > 1.0 => 1.0,
                _ => mutationRate
            };

            _rnd = new Random();
        }

        public void Mutate(Individual individual)
        {
            // copy original individual's parameters
            Individual mutatedIndividual = new Individual
            {
                Parameters = individual.Parameters.ToArray(),
                Fitness = 0.0
            };

            // Simple mutation process: just adjust some of the parameters randomly
            for (int i = 0; i < individual.Parameters.Length; i++)
            {
                // Check if we should perform a mutation
                if (_rnd.NextDouble() < _mutationRate)
                {
                    mutatedIndividual.Parameters[i] = _rnd.NextDouble(); // Mutate the value to a new random one
                }
            }

            //return mutatedIndividual;
        }
    }
}
