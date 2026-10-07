#pragma once
#include <functional>
#include <vector>

using Init = std::function<double()>;

Init zeros();
Init constant(double c);
Init uniform(double lo, double hi, unsigned seed);

using Loss = std::function<double(double pred, double target)>;

double mse(double pred, double target);
double mae(double pred, double target);
Loss huber(double delta);

using Aggregate = std::function<double(const std::vector<double>& w,
                                       const std::vector<double>& x, double b)>;

double weightedSum(const std::vector<double>& w, const std::vector<double>& x,
                   double b);
double maxTerm(const std::vector<double>& w, const std::vector<double>& x,
               double b);
