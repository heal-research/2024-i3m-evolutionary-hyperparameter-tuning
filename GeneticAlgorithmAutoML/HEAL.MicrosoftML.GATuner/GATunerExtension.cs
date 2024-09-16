using HEAL.MicrosoftML.GATuner.Crossover;
using HEAL.MicrosoftML.GATuner.Mutation;
using HEAL.MicrosoftML.GATuner.Selection;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML.AutoML;
using static Microsoft.ML.AutoML.AutoMLExperiment;

namespace HEAL.MicrosoftML.GATuner
{
    public static class GATunerExtension
    {
        public static AutoMLExperiment SetGeneticAlgorithmTuner
        (
            this AutoMLExperiment experiment,
            uint populationSize,
            uint elites,
            IGeneticAlgorithmSelector selector,
            IGeneticAlgorithmCrossover crossover,
            IGeneticAlgorithmMutator mutator
        )
        {
            experiment.SetTuner((service) =>
            {
                //var settings = service.GetRequiredService<AutoMLExperimentSettings>();
                return new GeneticAlgorithmTuner(
                    service.GetRequiredService<AutoMLExperimentSettings>().SearchSpace, 
                    populationSize, elites, 
                    selector, 
                    crossover, 
                    mutator
                );

                //return tuner;
            });

            return experiment;
        }
    }
}
