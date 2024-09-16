namespace HEAL.MicrosoftML.CMAESTuner
{
    public record Solution
    {
        public Solution(double[] values, double quality)
        {
            Values = values;
            Quality = quality;
        }

        public double[] Values { get; set; }
        public double Quality { get; set; }
    };
}
