#pragma once
#include <cstddef>
#include <vector>

#include "activation.h"
#include "strategy.h"

class Neuron {
  std::vector<double> w;
  double b;
  Activation act;
  Aggregate agg;

 public:
  Neuron(std::vector<double> weights, double bias, Activation a,
         Aggregate g = weightedSum);
  Neuron(std::size_t inputs, Init init, Activation a,
         Aggregate g = weightedSum);
  void setActivation(Activation a);
  void setAggregate(Aggregate g);
  double forward(const std::vector<double>& x) const;
};
