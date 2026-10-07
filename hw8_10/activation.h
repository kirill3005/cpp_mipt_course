#pragma once
#include <functional>
#include <iostream>

using Activation = std::function<double(double)>;

double relu(double z);
double sigmoid(double z);
double tanhAct(double z);

Activation withClip(Activation inner, double lo, double hi);
Activation withScale(Activation inner, double k);
Activation withShift(Activation inner, double c);
Activation withNoise(Activation inner, double sigma, unsigned seed);
Activation withCounter(Activation inner, int& calls);
Activation withLogging(Activation inner, std::ostream& os);
