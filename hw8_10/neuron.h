#pragma once
#include <vector>

#include "activation.h"

class Neuron {
  std::vector<double> w;
  double b;
  Activation act;

 public:
  Neuron(std::vector<double> weights, double bias, Activation a);
  void setActivation(Activation a);
  double forward(const std::vector<double>& x);
};
