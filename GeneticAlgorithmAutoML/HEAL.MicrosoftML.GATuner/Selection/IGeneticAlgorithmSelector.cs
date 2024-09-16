using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HEAL.MicrosoftML.GATuner.Selection
{
    public interface IGeneticAlgorithmSelector
    {
        public Tuple<Individual, Individual> Select(List<Individual> population);
    }
}
