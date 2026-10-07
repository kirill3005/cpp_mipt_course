#include "neuron.h"

#include <stdexcept>

Neuron::Neuron(std::vector<double> weights, double bias, Activation a,
               Aggregate g)
    : w{weights}, b{bias}, act{a}, agg{g} {}

Neuron::Neuron(std::size_t inputs, Init init, Activation a, Aggregate g)
    : w(inputs), b{init()}, act{a}, agg{g} {
  for (double& wi : w) wi = init();
}

void Neuron::setActivation(Activation a) { act = a; }

void Neuron::setAggregate(Aggregate g) { agg = g; }

double Neuron::forward(const std::vector<double>& x) const {
  if (x.size() != w.size()) {
    throw std::invalid_argument("Neuron::forward: wrong input size");
  }
  return act(agg(w, x, b));
}
