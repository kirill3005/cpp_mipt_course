#include <iostream>
#include <vector>

#include "activation.h"
#include "neuron.h"

int main() {
  std::vector<double> x = {1, 2, 3};
  Neuron n({0.5, -1.0, 2.0}, 0.1, relu);

  std::cout << "ReLU: " << n.forward(x) << "\n";

  n.setActivation(sigmoid);
  std::cout << "Sigmoid: " << n.forward(x) << "\n";

  n.setActivation(withClip(relu, 0, 1));
  std::cout << "ReLU+clip(0-1): " << n.forward(x) << "\n";
}