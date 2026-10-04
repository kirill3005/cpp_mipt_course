#include "neuron.h"

Neuron::Neuron(std::vector<double> weights, double bias, Activation a)
    : w{weights}, b{bias}, act{a} {}

void Neuron::setActivation(Activation a) { act = a; }

double Neuron::forward(const std::vector<double>& x) {
  double z = b;
  for (size_t i = 0; i < w.size(); i++) z += w[i] * x[i];
  return act(z);
}
