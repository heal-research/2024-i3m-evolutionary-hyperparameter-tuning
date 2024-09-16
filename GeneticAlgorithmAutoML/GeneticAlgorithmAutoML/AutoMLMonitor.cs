using Microsoft.ML;
using Microsoft.ML.AutoML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GeneticAlgorithmAutoML
{
    public class AutoMLMonitor : IMonitor
    {
        private readonly SweepablePipeline _pipeline;
        private readonly List<TrialResult> _completedTrials;

        public AutoMLMonitor(SweepablePipeline pipeline)
        {
            _pipeline = pipeline;
            _completedTrials = new List<TrialResult>();
        }

        public IEnumerable<TrialResult> GetCompletedTrials() => _completedTrials;

        public void ReportBestTrial(TrialResult result)
        {
            Console.WriteLine($"{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff} -- BEST: {result.Metric}");
            return;
        }

        public void ReportCompletedTrial(TrialResult result)
        {
            var trialId = result.TrialSettings.TrialId;
            var timeToTrain = result.DurationInMilliseconds;
            var pipeline = _pipeline.ToString(result.TrialSettings.Parameter);
            Console.WriteLine($"{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff} -- Trial {trialId} finished training in {timeToTrain}ms with pipeline {pipeline} : {result.Metric}");
            _completedTrials.Add(result);
        }

        public void ReportFailTrial(TrialSettings settings, Exception exception = null)
        {
            if (exception.Message.Contains("Operation was canceled."))
            {
                Console.WriteLine($"{settings.TrialId} cancelled. Time budget exceeded.");
            }
            Console.WriteLine($"{settings.TrialId} failed with exception {exception.Message}");
        }

        public void ReportRunningTrial(TrialSettings setting)
        {
            return;
        }
    }
}
