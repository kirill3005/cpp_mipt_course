#include "activation.h"

#include <algorithm>
#include <cmath>
#include <iostream>

double relu(double z) { return z > 0 ? z : 0; }
double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }
double tanhAct(double z) { return std::tanh(z); }

Activation withClip(Activation inner, double lo, double hi) {
  return [inner, lo, hi](double z) { return std::clamp(inner(z), lo, hi); };
}
