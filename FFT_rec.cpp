#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <random>
#include <chrono>

using namespace std;
const double PI = 3.14159265358979323846;

// Recursive FFT function
void fft(vector<complex<double>>& X) {
    int n = X.size();
    if (n <= 1) return;

    // Split even and odd indexed elements
    vector<complex<double>> X_even(n / 2), X_odd(n / 2);
    for (int i = 0; i < n / 2; i++) {
        X_even[i] = X[i * 2];
        X_odd[i] = X[i * 2 + 1];
    }

    // Recursive FFT calls
    fft(X_even);
    fft(X_odd);

    // Combine results using complex exponentials
    for (int k = 0; k < n / 2; k++) {
        complex<double> t = polar(1.0, -2 * PI * k / n) * X_odd[k];
        X[k] = X_even[k] + t;
        X[k + n / 2] = X_even[k] - t;
    }
}

// Function to print complex arrays
void print_complex_array(const vector<complex<double>>& X) {
    for (size_t i = 0; i < X.size(); i++) {
        cout << "X[" << i << "] = " << X[i] << endl;
    }
}

vector<complex<double>> generateRandomSignal(int size) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-10.0, 10.0); // Random values between -10 and 10

    vector<complex<double>> signal;
    for (int i = 0; i < size; ++i) {
        signal.emplace_back(dist(gen), dist(gen));
    }
    return signal;
}

int main() {
    int n = 4096;  // FFT size (must be a power of 2)

    // Given complex input values
    /*vector<complex<double>> X = {
        {3.6, 2.6}, {2.9, 6.3}, {5.6, 4.0}, {4.8, 9.1},
        {3.3, 0.4}, {5.9, 4.8}, {5.0, 2.6}, {4.3, 4.1}
    };*/
    vector<complex<double>> X = generateRandomSignal(n);

    cout << "Input:\n";

    
    print_complex_array(X);

    auto start = std::chrono::high_resolution_clock::now();
    // Perform FFT
    fft(X);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    

    cout << "\nFFT Output:\n";
    print_complex_array(X);
    std::cout << "FFT recursive:\tnumber of points: " << n << "\ttime:\t" << duration.count() << "\tmicrosec\n";

    return 0;
}
