using Microsoft.ML.AutoML;
using Microsoft.ML.AutoML.CodeGen;
using Microsoft.ML.SearchSpace.Option;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GeneticAlgorithmAutoML
{
    public class ExperimentUtil
    {

        public static FastForestOption FastForestOption(ColumnInferenceResults columnInference)
        {
            return new FastForestOption
            {
                NumberOfLeaves = 4,     // min: 4,     max: 32768, init: 4,     logBase: true
                NumberOfTrees = 4,      // min: 4,     max: 32768, init: 4,     logBase: true
                FeatureFraction = 1F,   // min: 2E-10, max: 1,     init: 1F,    logBase: false
                LabelColumnName = columnInference.ColumnInformation.LabelColumnName,
                FeatureColumnName = "Features"
            };
        }

        public static Microsoft.ML.SearchSpace.SearchSpace<FastForestOption> FastForestSearchSpace(ColumnInferenceResults columnInference)
        {
            var fastForestSearchSpace = new Microsoft.ML.SearchSpace.SearchSpace<FastForestOption>(FastForestOption(columnInference));
            fastForestSearchSpace["NumberOfLeaves"] = new UniformIntOption(5, 100, true);
            fastForestSearchSpace["NumberOfTrees"] = new UniformIntOption(5, 100, true);
            fastForestSearchSpace["FeatureFraction"] = new UniformDoubleOption(0.5f, 1.0f, false);
            return fastForestSearchSpace;
        }



        public static LgbmOption LgbmOption(ColumnInferenceResults columnInference)
        {
            return new LgbmOption
            {
                NumberOfLeaves = 4,                 // min: 4,     max: 32768, init: 4,     logBase: false
                MinimumExampleCountPerLeaf = 20,    // min: 20,    max: 1024,  init: 20,    logBase: true
                LearningRate = 1d,                  // min: 2E-10, max: 1,     init: 1,     logBase: true
                NumberOfTrees = 4,                  // min: 4,     max: 32768, init: 4,     logBase: false
                SubsampleFraction = 1d,             // min: 2E-10, max: 1,     init: 1,     logBase: true
                MaximumBinCountPerFeature = 25,     // min: 8,     max: 1024,  init: 256,   logBase: true
                FeatureFraction = 1d,               // min: 2E-10, max: 1,     init: 1,     logBase: false
                L1Regularization = 2E-10,           // min: 2E-10, max: 1,     init: 2E-10, logBase: true
                L2Regularization = 1d,              // min: 2E-10, max: 1,     init: 1,     logBase: true
                LabelColumnName = columnInference.ColumnInformation.LabelColumnName,
                FeatureColumnName = "Features"
            };
        }

        public static Microsoft.ML.SearchSpace.SearchSpace<LgbmOption> LgbmSearchSpace(ColumnInferenceResults columnInference)
        {
            var lgbmSearchSpace = new Microsoft.ML.SearchSpace.SearchSpace<LgbmOption>(LgbmOption(columnInference));
            //lgbmSearchSpace["NumberOfLeaves"] = new UniformIntOption(5, 100, false);
            //lgbmSearchSpace["MinimumExampleCountPerLeaf"] = new UniformIntOption(5, 100, true);
            //lgbmSearchSpace["LearningRate"] = new UniformSingleOption(0.00001f, 1.0f, true);
            //lgbmSearchSpace["NumberOfTrees"] = new UniformIntOption(5, 100, false);
            //lgbmSearchSpace["SubsampleFraction"] = new UniformDoubleOption(0.00001d, 1.0d, true);
            //lgbmSearchSpace["MaximumBinCountPerFeature"] = new UniformIntOption(5, 100, true);
            //lgbmSearchSpace["FeatureFraction"] = new UniformDoubleOption(0.5d, 1.0d, false);
            //lgbmSearchSpace["L1Regularization"] = new UniformDoubleOption(0.00001d, 1.0d, true);
            //lgbmSearchSpace["L2Regularization"] = new UniformDoubleOption(0.00001d, 1.0d, true);
            lgbmSearchSpace["NumberOfLeaves"] = new UniformIntOption(5, 100, false);
            lgbmSearchSpace["MinimumExampleCountPerLeaf"] = new UniformIntOption(20, 100, true);
            lgbmSearchSpace["LearningRate"] = new UniformSingleOption(0.00001f, 1.0f, true);
            lgbmSearchSpace["NumberOfTrees"] = new UniformIntOption(5, 100, false);
            lgbmSearchSpace["SubsampleFraction"] = new UniformDoubleOption(0.00001d, 1.0d, true);
            lgbmSearchSpace["MaximumBinCountPerFeature"] = new UniformIntOption(8, 50, true);
            lgbmSearchSpace["FeatureFraction"] = new UniformDoubleOption(0.5d, 1.0d, false);
            lgbmSearchSpace["L1Regularization"] = new UniformDoubleOption(0.00001d, 1.0d, true);
            lgbmSearchSpace["L2Regularization"] = new UniformDoubleOption(0.00001d, 1.0d, true);
            return lgbmSearchSpace;
        }



        public static FastTreeOption FastTreeOption(ColumnInferenceResults columnInference)
        {
            return new FastTreeOption
            {
                NumberOfLeaves = 4,                 // min: 4,     max: 32768, init: 4,     logBase: true
                MinimumExampleCountPerLeaf = 20,    // min: 2,     max: 128,   init: 20,    logBase: true
                NumberOfTrees = 4,                  // min: 4,     max: 32768, init: 4,     logBase: true
                MaximumBinCountPerFeature = 25,     // min: 8,     max: 1024,  init: 256,   logBase: true
                FeatureFraction = 1d,               // min: 2E-10, max: 1,     init: 1,     logBase: false
                LearningRate = 0.1d,                // min: 2E-10, max: 1,     init: 0.1,   logBase: true
                LabelColumnName = columnInference.ColumnInformation.LabelColumnName,
                FeatureColumnName = "Features"
            };
        }

        public static Microsoft.ML.SearchSpace.SearchSpace<FastTreeOption> FastTreeSearchSpace(ColumnInferenceResults columnInference)
        {
            var fastTreeSearchSpace = new Microsoft.ML.SearchSpace.SearchSpace<FastTreeOption>(FastTreeOption(columnInference));
            fastTreeSearchSpace["NumberOfLeaves"] = new UniformIntOption(5, 100, true);
            fastTreeSearchSpace["MinimumExampleCountPerLeaf"] = new UniformIntOption(3, 25, true);
            fastTreeSearchSpace["NumberOfTrees"] = new UniformIntOption(5, 100, true);
            fastTreeSearchSpace["MaximumBinCountPerFeature"] = new UniformIntOption(8, 50, true);
            fastTreeSearchSpace["FeatureFraction"] = new UniformDoubleOption(0.5f, 1.0f, false);
            fastTreeSearchSpace["LearningRate"] = new UniformSingleOption(0.00001f, 1.0f, true);
            return fastTreeSearchSpace;
        }



        public static LbfgsOption LbfgsOption(ColumnInferenceResults columnInference)
        {
            return new LbfgsOption
            {
                L1Regularization = 1F, // min: 0.03125, max: 32768, init: 1F, logBase: true
                L2Regularization = 1F, // min: 0.03125, max: 32768, init: 1F, logBase: true
                LabelColumnName = columnInference.ColumnInformation.LabelColumnName,
                FeatureColumnName = "Features"
            };
        }

        public static Microsoft.ML.SearchSpace.SearchSpace<LbfgsOption> LbfgsSearchSpace(ColumnInferenceResults columnInferenceResults)
        {
            var lbfgsSearchSpace = new Microsoft.ML.SearchSpace.SearchSpace<LbfgsOption>(LbfgsOption(columnInferenceResults));
            lbfgsSearchSpace["L1Regularization"] = new UniformSingleOption(0.03125f, 15000f, true); // orig: 0.03125f, 32768f
            lbfgsSearchSpace["L2Regularization"] = new UniformSingleOption(0.03125f, 15000f, true); // orig: 0.03125f, 32768f
            return lbfgsSearchSpace;
        }



        public static SdcaOption SdcaOption(ColumnInferenceResults columnInference)
        {
            return new SdcaOption
            {
                L1Regularization = 1F,      // min: 0.03125, max: 32768, init: 1F, logBase: true
                L2Regularization = 0.1F,    // min: 0.03125, max: 32768, init: 0.1F, logBase: true
                LabelColumnName = columnInference.ColumnInformation.LabelColumnName,
                FeatureColumnName = "Features"
            };
        }

        public static Microsoft.ML.SearchSpace.SearchSpace<SdcaOption> SdcaSearchSpace(ColumnInferenceResults columnInference)
        {
            var sdcaSearchSpace = new Microsoft.ML.SearchSpace.SearchSpace<SdcaOption>(SdcaOption(columnInference));
            sdcaSearchSpace["L1Regularization"] = new UniformSingleOption(0.03125f, 0.5f, false);
            sdcaSearchSpace["L2Regularization"] = new UniformSingleOption(0.03125f, 0.5f, false);
            return sdcaSearchSpace;
        }



    }
}
