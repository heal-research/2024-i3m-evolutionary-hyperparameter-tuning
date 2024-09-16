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

namespace GeneticAlgorithmAutoML
{
    public class MCExperiment
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
            SweepablePipeline pipeline = mlContext
                .Transforms.Conversion.MapValueToKey(inputColumnName: columnInference.ColumnInformation.LabelColumnName, outputColumnName: columnInference.ColumnInformation.LabelColumnName)
                .Append(mlContext.Auto().Featurizer(trainData, columnInformation: columnInference.ColumnInformation, outputColumnName: "Features"))
                .Append(mlContext.Auto().MultiClassification(
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

                    useLbfgsMaximumEntrophy: true,
                    lbfgsMaximumEntrophyOption: ExperimentUtil.LbfgsOption(columnInference),
                    lbfgsMaximumEntrophySearchSpace: ExperimentUtil.LbfgsSearchSpace(columnInference),

                    useLbfgsLogisticRegression: true,
                    lbfgsLogisticRegressionOption: ExperimentUtil.LbfgsOption(columnInference),
                    lbfgsLogisticRegressionSearchSpace: ExperimentUtil.LbfgsSearchSpace(columnInference),

                    useSdcaMaximumEntrophy: false,
                    sdcaMaximumEntrophyOption: ExperimentUtil.SdcaOption(columnInference),
                    sdcaMaximumEntorphySearchSpace: ExperimentUtil.SdcaSearchSpace(columnInference),

                    useSdcaLogisticRegression: false,
                    sdcaLogisticRegressionOption: ExperimentUtil.SdcaOption(columnInference),
                    sdcaLogisticRegressionSearchSpace: ExperimentUtil.SdcaSearchSpace(columnInference)
                ));

            // Configure experiment
            AutoMLExperiment experiment = mlContext.Auto().CreateExperiment();

            experiment
                .SetPipeline(pipeline)
                .SetMulticlassClassificationMetric(MulticlassClassificationMetric.MacroAccuracy, labelColumn: columnInference.ColumnInformation.LabelColumnName)
                .SetTrainingTimeInSeconds(60 * 10)
                .SetMonitor(new AutoMLMonitor(pipeline))
                .SetDataset(trainData, 10)
                .SetCMAESTuner();
                //.SetGeneticAlgorithmTuner
                //(
                //    populationSize: 30,
                //    elites: 1,
                //    selector: new TournamentSelector(3),
                //    crossover: new SinglePointCrossover(),
                //    mutator: new SingleAlleleMutator(mutationRate: 0.30)
                //);

            // Run experiment
            TrialResult experimentResult = await experiment.RunAsync();

            // Evaluate result
            Console.WriteLine(experimentResult.Metric);

            var predictions = experimentResult.Model.Transform(testData);
            var metrics = mlContext.MulticlassClassification.Evaluate(data: predictions, labelColumnName: label, scoreColumnName: "Score");

            Console.WriteLine("\n Peak CPU = " + experimentResult.PeakCpu);
            Console.WriteLine("\n Peak Memory = " + experimentResult.PeakMemoryInMegaByte);

            Console.WriteLine("\n pipeline.CurrentParameter = " + pipeline.CurrentParameter.ToString());

            Console.WriteLine("\n trialSettings.Parameter[_pipeline_] = " + experimentResult.TrialSettings.Parameter["_pipeline_"]);

            Console.WriteLine("\n - MacroAccuracy = " + metrics.MacroAccuracy);
            Console.WriteLine("\n - MicroAccuracy = " + metrics.MicroAccuracy);
            Console.WriteLine("\n - LogLoss = " + metrics.LogLoss);
            Console.WriteLine("\n - LogLossReduction = " + metrics.LogLossReduction);
            Console.WriteLine("\n - PerClassLogLoss = " + metrics.PerClassLogLoss);
            Console.WriteLine("\n - TopKAccuracy = " + metrics.TopKAccuracy);
            Console.WriteLine("\n - TopKAccuracyForAllK = " + metrics.TopKAccuracyForAllK);
            Console.WriteLine("\n - TopKPredictionCount = " + metrics.TopKPredictionCount);

            Console.WriteLine("\n" + metrics.ConfusionMatrix.GetFormattedConfusionTable());
        }
    }
}
