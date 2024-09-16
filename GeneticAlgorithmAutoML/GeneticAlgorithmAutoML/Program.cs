using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.AutoML.CodeGen;
using Microsoft.ML.SearchSpace.Option;
using HEAL.MicrosoftML.GATuner.Selection;
using HEAL.MicrosoftML.GATuner.Crossover;
using HEAL.MicrosoftML.GATuner.Mutation;
using HEAL.MicrosoftML.GATuner;
using HEAL.MicrosoftML.CMAESTuner;

namespace GeneticAlgorithmAutoML
{
    public class Program
    {
        public static async Task Main(string[] args)
        {
            Console.WriteLine($"Microsoft AutoML - Experiments");
            Console.WriteLine($"------------------------------");


            MLContext mlContext = new MLContext(seed: 123);

            //await BCExperiment.RunExperiment(
            //    mlContext,
            //    @".\datasets\bc\churn_train.csv",
            //    @".\datasets\bc\churn_test.csv",
            //    "class",
            //    1
            //);

            //await BCExperiment.RunExperiment(
            //    mlContext,
            //    @".\datasets\bc\credit_train.csv",
            //    @".\datasets\bc\credit_test.csv",
            //    "class",
            //    1
            //);

            //await BCExperiment.RunExperiment(
            //    mlContext,
            //    @".\datasets\bc\diabetes_train.csv",
            //    @".\datasets\bc\diabetes_test.csv",
            //    "class",
            //    1
            //);

            //await BCExperiment.RunExperiment(
            //    mlContext,
            //    @".\datasets\bc\qsar_train.csv",
            //    @".\datasets\bc\qsar_test.csv",
            //    "class",
            //    1
            //);


            //await MCExperiment.RunExperiment(
            //    mlContext,
            //    @".\datasets\mc\cmc_train.csv",
            //    @".\datasets\mc\cmc_test.csv",
            //    "Contraceptive_method_used",
            //    1
            //);

            //await MCExperiment.RunExperiment(
            //    mlContext,
            //    @".\datasets\mc\dmft_train.csv",
            //    @".\datasets\mc\dmft_test.csv",
            //    "Prevention",
            //    1
            //);

            //await MCExperiment.RunExperiment(
            //    mlContext,
            //    @".\datasets\mc\mfeat_train.csv",
            //    @".\datasets\mc\mfeat_test.csv",
            //    "class",
            //    1
            //);

            //await MCExperiment.RunExperiment(
            //    mlContext,
            //    @".\datasets\mc\vehicle_train.csv",
            //    @".\datasets\mc\vehicle_test.csv",
            //    "Class",
            //    1
            //);



            //await RegExperiment.RunExperiment(
            //    mlContext,
            //    @".\datasets\reg\cholesterol_train.csv",
            //    @".\datasets\reg\cholesterol_test.csv",
            //    "chol",
            //    1
            //);

            //await RegExperiment.RunExperiment(
            //    mlContext,
            //    @".\datasets\reg\cloud_train.csv",
            //    @".\datasets\reg\cloud_test.csv",
            //    "TE",
            //    1
            //);

            //await RegExperiment.RunExperiment(
            //    mlContext,
            //    @".\datasets\reg\liver_train.csv",
            //    @".\datasets\reg\liver_test.csv",
            //    "drinks",
            //    1
            //);

            await RegExperiment.RunExperiment(
                mlContext,
                @".\datasets\reg\plasma_train.csv",
                @".\datasets\reg\plasma_test.csv",
                "RETPLASMA",
                1
            );
        }
    }
}
