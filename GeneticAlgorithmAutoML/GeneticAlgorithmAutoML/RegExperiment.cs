using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML;
using HEAL.MicrosoftML.GATuner;
using HEAL.MicrosoftML.GATuner.Selection;
using HEAL.MicrosoftML.GATuner.Crossover;
using HEAL.MicrosoftML.GATuner.Mutation;
using Microsoft.ML.AutoML.CodeGen;
using Microsoft.ML.SearchSpace.Option;
using HEAL.MicrosoftML.CMAESTuner;
using Microsoft.ML.Trainers;

namespace GeneticAlgorithmAutoML
{
    public class RegExperiment
    {
        public static async Task RunExperiment(MLContext mlContext, string trainPath, string testPath, string label, int run)
        {
            Console.WriteLine($"\nRUN #{run} - {DateTime.Now:yyyy-MM-dd:HH:mm:ss} - {trainPath}");

            // Infer column information
            ColumnInferenceResults columnInference = mlContext.Auto().InferColumns
            (
                path: trainPath,
                labelColumnName: label,
                groupColumns: true // changed to TRUE (default)
            );

            // Create text loader
            TextLoader loader = mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);

            // Load data into IDataView
            IDataView trainData = loader.Load(trainPath);
            IDataView testData = loader.Load(testPath);

            // Define ML pipeline
            SweepablePipeline pipeline = mlContext.Auto()
                .Featurizer(trainData, columnInformation: columnInference.ColumnInformation, outputColumnName: "Features")
                .Append(mlContext.Auto().Regression
                (
                    labelColumnName: columnInference.ColumnInformation.LabelColumnName,
                
                    useFastForest: true,
                    fastForestOption: ExperimentUtil.FastForestOption(columnInference),
                    fastForestSearchSpace: ExperimentUtil.FastForestSearchSpace(columnInference),

                    useLgbm: true,
                    lgbmOption: ExperimentUtil.LgbmOption(columnInference),
                    lgbmSearchSpace: ExperimentUtil.LgbmSearchSpace(columnInference),

                    useFastTree: true,
                    fastTreeOption: ExperimentUtil.FastTreeOption(columnInference),
                    fastTreeSearchSpace: ExperimentUtil.FastTreeSearchSpace(columnInference),

                    useLbfgsPoissonRegression: false,
                    lbfgsPoissonRegressionOption: ExperimentUtil.LbfgsOption(columnInference),
                    lbfgsPoissonRegressionSearchSpace: ExperimentUtil.LbfgsSearchSpace(columnInference),

                    useSdca: false,
                    sdcaOption: ExperimentUtil.SdcaOption(columnInference),
                    sdcaSearchSpace: ExperimentUtil.SdcaSearchSpace(columnInference)
                ));

            // Configure experiment
            AutoMLExperiment experiment = mlContext.Auto().CreateExperiment();

            experiment
                .SetPipeline(pipeline)
                .SetRegressionMetric(RegressionMetric.MeanAbsoluteError, labelColumn: columnInference.ColumnInformation.LabelColumnName)
                .SetTrainingTimeInSeconds(60 * 5)
                .SetMonitor(new AutoMLMonitor(pipeline))
                .SetDataset(trainData, 10)
                //.SetCMAESTuner();
                .SetGeneticAlgorithmTuner
                (
                    populationSize: 30, // 100
                    elites: 2, // 5
                    selector: new TournamentSelector(3),
                    crossover: new MultiPointCrossover(2),
                    mutator: new GaussianMutator(0.50, 0.25)
                );


            // Run experiment
            TrialResult experimentResult = await experiment.RunAsync();

            // Evaluate result
            Console.WriteLine(experimentResult.Metric);

            var predictions = experimentResult.Model.Transform(testData);
            var metrics = mlContext.Regression.Evaluate(data: predictions, labelColumnName: label, scoreColumnName: "Score");

            Console.WriteLine("\n Peak CPU = " + experimentResult.PeakCpu);
            Console.WriteLine("\n Peak Memory = " + experimentResult.PeakMemoryInMegaByte);

            Console.WriteLine("\n pipeline.CurrentParameter = " + pipeline.CurrentParameter.ToString());
            Console.WriteLine("\n trialSettings.Parameter[_pipeline_] = " + experimentResult.TrialSettings.Parameter["_pipeline_"]);

            Console.WriteLine("\n - RSquared = " + metrics.RSquared);
            Console.WriteLine(" - MeanAbsoluteError = " + metrics.MeanAbsoluteError);
            Console.WriteLine(" - MeanSquaredError = " + metrics.MeanSquaredError);
            Console.WriteLine(" - RootMeanSquaredError = " + metrics.RootMeanSquaredError);
            Console.WriteLine(" - LossFunction = " + metrics.LossFunction);
        }
    }
}
