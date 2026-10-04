#pragma once
#include <functional>
#include <string>

using Activation = std::function<double(double)>;

double relu(double z);
double sigmoid(double z);
double tanhAct(double z);

Activation withClip(Activation inner, double lo, double hi);