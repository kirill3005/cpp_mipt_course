#include <iostream>
#include <vector>

#include "activation.h"
#include "neuron.h"
#include "strategy.h"

int main() {
  std::vector<double> x = {1, 2, 3};
  Neuron n({0.5, -1.0, 2.0}, 0.1, relu);

  std::cout << "ReLU: " << n.forward(x) << "\n";

  n.setActivation(sigmoid);
  std::cout << "Sigmoid: " << n.forward(x) << "\n";

  n.setActivation(withClip(relu, 0, 1));
  std::cout << "ReLU+clip(0-1): " << n.forward(x) << "\n";

  int calls = 0;
  n.setActivation(withLogging(
      withCounter(withShift(withScale(relu, 2.0), -1.0), calls), std::cout));
  n.forward(x);
  n.forward(x);
  std::cout << "calls: " << calls << "\n";

  n.setActivation(withNoise(relu, 0.1, 42));
  std::cout << "Noisy ReLU: " << n.forward(x) << " " << n.forward(x) << "\n";

  Neuron u(3, uniform(-1.0, 1.0, 7), tanhAct);
  std::cout << "Uniform init, tanh: " << u.forward(x) << "\n";
  Neuron z(3, zeros(), sigmoid);
  std::cout << "Zeros init, sigmoid: " << z.forward(x) << "\n";

  n.setActivation(relu);
  n.setAggregate(maxTerm);
  std::cout << "Max aggregate: " << n.forward(x) << "\n";

  double pred = n.forward(x);
  Loss losses[] = {mse, mae, huber(1.0)};
  for (const Loss& loss : losses) std::cout << loss(pred, 5.0) << " ";
  std::cout << "\n";
}
