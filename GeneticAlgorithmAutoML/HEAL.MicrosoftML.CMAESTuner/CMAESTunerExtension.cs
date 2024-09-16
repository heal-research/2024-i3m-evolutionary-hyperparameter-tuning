using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML.AutoML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HEAL.MicrosoftML.CMAESTuner
{
    public static class CMAESTunerExtension
    {
        public static AutoMLExperiment SetCMAESTuner
        (
            this AutoMLExperiment experiment
        )
        {
            experiment.SetTuner((service) =>
            {
                var settings = service.GetRequiredService<AutoMLExperiment.AutoMLExperimentSettings>();
                var tuner = new CMAESTuner(settings.SearchSpace);

                return tuner;
            });

            return experiment;
        }
    }
}
