using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HEAL.MicrosoftML.GATuner
{
    public record Individual
    {
        public required double[] Parameters { get; set; }
        public required double Fitness { get; set; }
    }
}
