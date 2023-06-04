#include <vector>
#include <cassert>
#include <iostream>
#include <limits>
#include <cmath>

class Interpolation {
public:
    virtual double interpolate(const std::vector<double>& xData, const std::vector<double>& yData, double x) = 0;
};

class NewtonInterpolation : public Interpolation {
public:
    double interpolate(const std::vector<double>& xData, const std::vector<double>& yData, double x) override {
        size_t n = xData.size();
        assert(n == yData.size());

        std::vector<double> f(n);
        for (size_t i = 0; i < n; i++) {
            f[i] = yData[i];
        }

        for (size_t i = 1; i < n; i++) {
            for (size_t j = n-1; j >= i; j--) {
                f[j] = (f[j] - f[j-1]) / (xData[j] - xData[j-i]);
            }
        }

        double result = f[n-1];
        for (int i = n - 2; i >= 0; i--) {
            result = result * (x - xData[i]) + f[i];
        }

        return result;
    }
};

class LagrangeInterpolation : public Interpolation {
public:
    double interpolate(const std::vector<double>& xData, const std::vector<double>& yData, double x) override {
        size_t n = xData.size();
        assert(n == yData.size());

        double result = 0;
        for (size_t i = 0; i < n; i++) {
            double term = yData[i];
            for (size_t j = 0; j < n; j++) {
                if (j != i) {
                    term = term * (x - xData[j]) / double(xData[i] - xData[j]);
                }
            }
            result += term;
        }

        return result;
    }
};

class Differentiation {
public:
    virtual double differentiate(const std::vector<double>& xData, const std::vector<double>& yData, double x) = 0;
};
class NewtonDifferentiation : public Differentiation {
public:
    double differentiate(const std::vector<double>& xData, const std::vector<double>& yData, double x) override {
        size_t n = xData.size();
        assert(n == yData.size() && n > 1);

        // Find two points closest to x
        double x0, x1, y0, y1;
        double minDiff = std::numeric_limits<double>::max();
        for (size_t i = 0; i < n - 1; i++) {
            double diff1 = fabs(x - xData[i]);
            double diff2 = fabs(x - xData[i+1]);
            if (diff1 < minDiff && diff2 < minDiff) {
                minDiff = std::max(diff1, diff2);
                x0 = xData[i];
                x1 = xData[i+1];
                y0 = yData[i];
                y1 = yData[i+1];
            }
        }

    // Дифференцирование Ньютона второго порядка
    return (2 * x - x0 - x1) * ((y1 - y0) / (x1 - x0));
    }
};

class FiniteDifferenceDifferentiation : public Differentiation {
public:
    double differentiate(const std::vector<double>& xData, const std::vector<double>& yData, double x) override {
        size_t n = xData.size();
        assert(n == yData.size() && n > 1);

        // Find two points closest to x
        double x0, x1, y0, y1;
        double minDiff = std::numeric_limits<double>::max();
        for (size_t i = 0; i < n - 1; i++) {
            double diff1 = fabs(x - xData[i]);
            double diff2 = fabs(x - xData[i+1]);
            if (diff1 < minDiff && diff2 < minDiff) {
                minDiff = std::max(diff1, diff2);
                x0 = xData[i];
                x1 = xData[i+1];
                y0 = yData[i];
                y1 = yData[i+1];
            }
        }

        // First order forward difference method
        return (y1 - y0) / (x1 - x0);
    }
};

int main() {
    std::vector<double> xData = {1, 2, 3, 4, 5};
    std::vector<double> yData = {1, 4, 9, 16, 25};

    double x = 3.6;

    NewtonInterpolation newtonInterpolation;
    double newYNewton = newtonInterpolation.interpolate(xData, yData, x);
    std::cout << "Newton interpolation at x=" << x << " is " << newYNewton << std::endl;

    LagrangeInterpolation lagrangeInterpolation;
    double newYLagrange = lagrangeInterpolation.interpolate(xData, yData, x);
    std::cout << "Lagrange interpolation at x=" << x << " is " << newYLagrange << std::endl;

    NewtonDifferentiation newtonDifferentiation;
    double newYDotNewton = newtonDifferentiation.differentiate(xData, yData, x);
    std::cout << "Newton differentiation at x=" << x << " is " << newYDotNewton << std::endl;

    FiniteDifferenceDifferentiation finiteDifferenceDifferentiation;
    double newYDotFiniteDifference = finiteDifferenceDifferentiation.differentiate(xData, yData, x);
    std::cout << "Finite difference differentiation at x=" << x << " is " << newYDotFiniteDifference << std::endl;

    return 0;
}