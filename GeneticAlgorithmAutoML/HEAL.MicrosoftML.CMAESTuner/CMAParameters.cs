namespace HEAL.MicrosoftML.CMAESTuner {
  internal class CMAParameters {
    public double AxisRatio { get; internal set; }
    public double Sigma { get; internal set; }
    public int Mu { get; }
    public double[] Weights { get; }
    public double MuEff { get; }
    public double CC { get; }
    public double CS { get; }
    public double Damps { get; }
    public double MuCov { get; }
    public double CCov { get; }
    public double CCovSep { get; }
    public double[] PC { get; }                // evolution paths for C
    public double[] PS { get; }                // evolution paths for sigma
    public double[,] B { get; internal set; }  // B defines the coordinate system
    public double[] D { get; internal set; }   // diagnoal D defines the scaling
    public double[,] C { get; }                // covariance matrix C
    public double[] BDz { get; }
    public double ChiN { get; }
    public int QualityHistorySize { get; }
    public Queue<double> QualityHistory { get; }

    public CMAParameters(int problemSize, int lambda, double initialSigma, int maxGenerations) {
      Sigma = initialSigma;

      Mu = (int)Math.Floor(lambda / 2.0);

      double[] weights = Enumerable.Range(0, Mu)
                                   .Select(x => Math.Log((Mu + 1.0) / (x + 1.0)))
                                   .ToArray();
      double weightsSum = weights.Sum();
      Weights = weights.Select(x => x / weightsSum).ToArray();
      weightsSum = Weights.Sum();

      MuEff = weightsSum * weightsSum / Weights.Sum(x => x * x);

      CC = 4.0 / (problemSize + 4);

      CS = (MuEff + 2) / (problemSize + MuEff + 3);

      Damps = 2 * Math.Max(0, Math.Sqrt((MuEff - 1) / (problemSize + 1)) - 1) * Math.Max(0.3, 1 - problemSize / (1e-6 + maxGenerations)) + CS + 1;

      MuCov = MuEff;

      CCov = 2.0 / ((problemSize + 1.41) * (problemSize + 1.41) * MuCov) + (1 - (1.0 / MuCov)) * Math.Min(1, (2 * MuEff - 1) / (MuEff + (problemSize + 2) * (problemSize + 2)));

      CCovSep = Math.Min(1, CCov * (problemSize + 1.5) / 3);

      PC = new double[problemSize];

      PS = new double[problemSize];

      B = new double[problemSize, problemSize];

      D = new double[problemSize];

      C = new double[problemSize, problemSize];

      BDz = new double[problemSize];

      ChiN = Math.Sqrt(problemSize) * (1.0 - 1.0 / (4.0 * problemSize) + 1.0 / (21.0 * problemSize * problemSize));

      QualityHistorySize = 10 + 30 * problemSize / lambda;

      QualityHistory = new Queue<double>(QualityHistorySize + 1);

      for (int i = 0; i < problemSize; i++) {
        B[i, i] = 1;
        D[i] = 1;
        C[i, i] = 1;
      }

      AxisRatio = 1.0;
    }
  };
}
