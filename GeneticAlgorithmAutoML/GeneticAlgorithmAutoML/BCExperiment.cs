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
    public class BCExperiment
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
                .Featurizer(trainData, columnInformation: columnInference.ColumnInformation)
                .Append(
                    mlContext.Auto().BinaryClassification
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

                        useLbfgsLogisticRegression: true,
                        lbfgsLogisticRegressionOption: ExperimentUtil.LbfgsOption(columnInference),
                        lbfgsLogisticRegressionSearchSpace: ExperimentUtil.LbfgsSearchSpace(columnInference),

                        useSdcaLogisticRegression: false,
                        sdcaLogisticRegressionOption: ExperimentUtil.SdcaOption(columnInference),
                        sdcaLogisticRegressionSearchSpace: ExperimentUtil.SdcaSearchSpace(columnInference)
                    )
                );

            // Configure experiment
            AutoMLExperiment experiment = mlContext.Auto().CreateExperiment();

            experiment
                .SetPipeline(pipeline)
                .SetBinaryClassificationMetric(BinaryClassificationMetric.AreaUnderRocCurve, labelColumn: columnInference.ColumnInformation.LabelColumnName)
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
            var metrics = mlContext.BinaryClassification.EvaluateNonCalibrated(data: predictions, labelColumnName: label, scoreColumnName: "Score");

            Console.WriteLine("\n Peak CPU = " + experimentResult.PeakCpu);
            Console.WriteLine("\n Peak Memory = " + experimentResult.PeakMemoryInMegaByte);

            Console.WriteLine("\n pipeline.CurrentParameter = " + pipeline.CurrentParameter.ToString());

            Console.WriteLine("\n trialSettings.Parameter[_pipeline_] = " + experimentResult.TrialSettings.Parameter["_pipeline_"]);

            Console.WriteLine("\n - AreaUnderRocCurve = " + metrics.AreaUnderRocCurve);
            Console.WriteLine(" - AreaUnderPrecisionRecallCurve = " + metrics.AreaUnderPrecisionRecallCurve);
            Console.WriteLine(" - F1Score = " + metrics.F1Score);
            Console.WriteLine(" - PositivePrecision = " + metrics.PositivePrecision);
            Console.WriteLine(" - NegativePrecision = " + metrics.NegativePrecision);
            Console.WriteLine(" - PositiveRecall = " + metrics.PositiveRecall);
            Console.WriteLine(" - NegativeRecall = " + metrics.NegativeRecall);
            Console.WriteLine(" - Accuracy = " + metrics.Accuracy);
            Console.WriteLine("\n" + metrics.ConfusionMatrix.GetFormattedConfusionTable());
        }
    }
}
