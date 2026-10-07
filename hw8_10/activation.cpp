#include "activation.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

double relu(double z) { return z > 0 ? z : 0.0; }
double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }
double tanhAct(double z) { return std::tanh(z); }

Activation withClip(Activation inner, double lo, double hi) {
  if (lo > hi) throw std::invalid_argument("withClip: lo > hi");
  return [inner, lo, hi](double z) { return std::clamp(inner(z), lo, hi); };
}

Activation withScale(Activation inner, double k) {
  return [inner, k](double z) { return k * inner(z); };
}

Activation withShift(Activation inner, double c) {
  return [inner, c](double z) { return inner(z) + c; };
}

Activation withNoise(Activation inner, double sigma, unsigned seed) {
  return [inner, gen = std::mt19937(seed),
          dist = std::normal_distribution<double>(0.0, sigma)](double z) mutable {
    return inner(z) + dist(gen);
  };
}

Activation withCounter(Activation inner, int& calls) {
  return [inner, &calls](double z) {
    ++calls;
    return inner(z);
  };
}

Activation withLogging(Activation inner, std::ostream& os) {
  return [inner, &os](double z) {
    double res = inner(z);
    os << "f(" << z << ") = " << res << "\n";
    return res;
  };
}
