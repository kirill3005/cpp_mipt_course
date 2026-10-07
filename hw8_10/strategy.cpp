#include "strategy.h"

#include <algorithm>
#include <cmath>
#include <random>

Init zeros() {
  return [] { return 0.0; };
}

Init constant(double c) {
  return [c] { return c; };
}

Init uniform(double lo, double hi, unsigned seed) {
  return [gen = std::mt19937(seed),
          dist = std::uniform_real_distribution<double>(lo, hi)]() mutable {
    return dist(gen);
  };
}

double mse(double pred, double target) {
  double d = pred - target;
  return d * d;
}

double mae(double pred, double target) { return std::abs(pred - target); }

Loss huber(double delta) {
  return [delta](double pred, double target) {
    double d = std::abs(pred - target);
    if (d <= delta) return 0.5 * d * d;
    return delta * (d - 0.5 * delta);
  };
}

double weightedSum(const std::vector<double>& w, const std::vector<double>& x,
                   double b) {
  double z = b;
  for (size_t i = 0; i < w.size(); i++) z += w[i] * x[i];
  return z;
}

double maxTerm(const std::vector<double>& w, const std::vector<double>& x,
               double b) {
  double best = w.empty() ? 0.0 : w[0] * x[0];
  for (size_t i = 1; i < w.size(); i++) best = std::max(best, w[i] * x[i]);
  return best + b;
}
