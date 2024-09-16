using HEAL.MicrosoftML.GATuner.Crossover;
using HEAL.MicrosoftML.GATuner.Mutation;
using HEAL.MicrosoftML.GATuner.Selection;
using Microsoft.ML.AutoML;
using Microsoft.ML.AutoML.CodeGen;
using Microsoft.ML.SearchSpace;

namespace HEAL.MicrosoftML.GATuner
{
    public class GeneticAlgorithmTuner : ITuner
    {
        #region MEMBERS

        private readonly SearchSpace _searchSpace;
        private List<Individual> _population;
        private Random _rnd = new Random();

        private int _populationSize;
        private int _elites;

        private IGeneticAlgorithmSelector _selector;
        private IGeneticAlgorithmCrossover _crossover;
        private IGeneticAlgorithmMutator _mutator;

        private int _currentIndividualIndex = 0;

        #endregion MEMBERS


        public GeneticAlgorithmTuner
        (
            SearchSpace searchSpace,
            uint populationSize,
            uint elites,
            IGeneticAlgorithmSelector selector, 
            IGeneticAlgorithmCrossover crossover,
            IGeneticAlgorithmMutator mutator
        )
        {
            _searchSpace = searchSpace;

            _populationSize = (int)populationSize;
            _elites = (int)elites;

            _selector = selector;
            _crossover = crossover;
            _mutator = mutator;

            _population = new List<Individual>();

            for (int i = 0; i < _populationSize; i++)
            {
                _population.Add(new Individual
                {
                    Parameters = CreateRandomIndividual(),
                    Fitness = 0.0
                });
            }   

            _currentIndividualIndex = 0;
        }


        public Parameter Propose(TrialSettings settings)
        {
            if (_currentIndividualIndex >= _populationSize)
            {
                // sort population by fitness   
                _population = _population.OrderByDescending(i => i.Fitness).ToList();

                // elitism
                List<Individual> newPopulation = _population.Take(_elites).ToList();

                // selection, crossover and mutation
                while (newPopulation.Count < _populationSize)
                {
                    (Individual p1, Individual p2) = _selector.Select(_population);
                    Individual child = _crossover.Crossover(p1, p2);
                    _mutator.Mutate(child);
                    newPopulation.Add(child);
                }

                // generational replacement
                _population = newPopulation;

                // skip elites and propose evolved individuals
                _currentIndividualIndex = _elites;
            }

            // propose new set of hyperparameters 
            return _searchSpace.SampleFromFeatureSpace(
              _population[_currentIndividualIndex++].Parameters
            );
        }

        public void Update(TrialResult result)
        {
            Console.WriteLine("Result = " + result.Metric + "    " + string.Join(' ', _population[_currentIndividualIndex - 1].Parameters));

            // Update fitness of the individual that was just evaluated
            _population[_currentIndividualIndex - 1].Fitness = 1 - result.Loss;
        }

        private double[] CreateRandomIndividual()
        {
            var d = _searchSpace.FeatureSpaceDim;
            return Enumerable.Repeat(0, d).Select(i => _rnd.NextDouble()).ToArray();
        }

        public void DebugSearchSpace()
        {
            foreach (KeyValuePair<string, Microsoft.ML.SearchSpace.Option.OptionBase> entry in _searchSpace)
            {
                Console.WriteLine(entry.Key + ": ");

                foreach (var entry2 in ((SearchSpace)entry.Value))
                {
                    Console.WriteLine("  " + entry2.Key + " = " + entry2.Value);

                    if (entry2.Value is SearchSpace<FastForestOption> option)
                    {
                        foreach (var entry3 in option)
                        {
                            Console.WriteLine("    " + entry3.Key + " = " + entry3.Value);

                            if (entry3.Value is Microsoft.ML.SearchSpace.Option.UniformDoubleOption doubleOpt)
                            {
                                Console.WriteLine("       " + doubleOpt.Min + " - " + doubleOpt.Max);
                            }
                        }
                    }
                }
            }
        }
    }
}
