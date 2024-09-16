namespace HEAL.MicrosoftML.CMAESTuner
{
  /// <summary>A pseudo random number generator which uses Marsaglia's polar method to create normally distributed random numbers.</summary>
  public sealed class NormalDistributedRandom {
    public double Mu { get; set; }
    public double Sigma { get; set; }
    public Random UniformRandom { get; init; }

    public NormalDistributedRandom() {
      Mu = 0.0;
      Sigma = 1.0;
      UniformRandom = new();
    }
    public NormalDistributedRandom(double mu, double sigma, Random? uniformRandom = null) {
      Mu = mu;
      Sigma = sigma;
      UniformRandom = uniformRandom ?? new();
    }

    /**
      * Polar method due to Marsaglia.
      *
      * Devroye, L. Non-Uniform Random Variates Generation. Springer-Verlag,
      * New York, 1986, Ch. V, Sect. 4.4.
      */
    public double NextDouble() {
      // we don't use spare numbers (efficency loss but easier for multi-threaded code)
      double u, v, s;
      do {
        u = UniformRandom.NextDouble() * 2 - 1;
        v = UniformRandom.NextDouble() * 2 - 1;
        s = u * u + v * v;
      } while (s > 1 || s == 0);
      s = Math.Sqrt(-2.0 * Math.Log(s) / s);
      return Mu + Sigma * u * s;
    }
  }
}
