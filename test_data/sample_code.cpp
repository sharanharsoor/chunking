#include <iostream>
#include <vector>
#include <string>
#include <memory>

// Sample C++ code for testing C/C++ code chunking
namespace SampleCode {

class Calculator {
private:
    std::vector<double> history;

public:
    Calculator() {
        std::cout << "Calculator initialized" << std::endl;
    }

    double add(double a, double b) {
        double result = a + b;
        history.push_back(result);
        return result;
    }

    double multiply(double a, double b) {
        double result = a * b;
        history.push_back(result);
        return result;
    }

    void printHistory() {
        std::cout << "Calculation history:" << std::endl;
        for(size_t i = 0; i < history.size(); ++i) {
            std::cout << "Result " << i + 1 << ": " << history[i] << std::endl;
        }
    }

    ~Calculator() {
        std::cout << "Calculator destroyed" << std::endl;
    }
};

// Template function for generic operations
template<typename T>
T findMax(const std::vector<T>& values) {
    if(values.empty()) {
        throw std::invalid_argument("Empty vector");
    }

    T maxVal = values[0];
    for(const auto& val : values) {
        if(val > maxVal) {
            maxVal = val;
        }
    }
    return maxVal;
}

}  // namespace SampleCode

int main() {
    using namespace SampleCode;

    // Create calculator instance
    auto calc = std::make_unique<Calculator>();

    // Perform some calculations
    double sum = calc->add(10.5, 20.3);
    double product = calc->multiply(5.0, 7.0);

    std::cout << "Sum: " << sum << std::endl;
    std::cout << "Product: " << product << std::endl;

    // Test template function
    std::vector<int> numbers = {3, 7, 2, 9, 1, 5};
    int maxNumber = findMax(numbers);
    std::cout << "Max number: " << maxNumber << std::endl;

    // Print calculation history
    calc->printHistory();

    return 0;
}