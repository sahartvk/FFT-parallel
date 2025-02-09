#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;
const double PI = 3.14159265358979323846;

// Function to reverse bits in an integer (for bit-reversal permutation)
int reverseBits(int index, int numBits) {
    int reversedIndex = 0;
    for (int i = 0; i < numBits; i++) {
        if (index & (1 << i))
            reversedIndex |= (1 << (numBits - 1 - i));
    }
    return reversedIndex;
}

// Iterative FFT function (Cooley-Tukey Algorithm)
void iterativeFFT(vector<complex<double>>& signal) {
    int dataSize = signal.size();
    int numStages = log2(dataSize);

    // Bit-reversal permutation
    vector<complex<double>> reorderedSignal(dataSize);
    for (int i = 0; i < dataSize; i++)
        reorderedSignal[reverseBits(i, numStages)] = signal[i];
    signal = reorderedSignal;

    // Iterative FFT computation
    for (int stage = 1; stage <= numStages; stage++) {
        int segmentSize = 1 << stage;  // 2^stage: Current FFT segment size
        complex<double> twiddleFactorRoot = polar(1.0, -2 * PI / segmentSize); // Twiddle factor
        for (int segmentStart = 0; segmentStart < dataSize; segmentStart += segmentSize) {
            complex<double> twiddleFactor = 1;
            for (int pairIndex = 0; pairIndex < segmentSize / 2; pairIndex++) {
                complex<double> temp = twiddleFactor * signal[segmentStart + pairIndex + segmentSize / 2];
                complex<double> upper = signal[segmentStart + pairIndex];
                signal[segmentStart + pairIndex] = upper + temp;
                signal[segmentStart + pairIndex + segmentSize / 2] = upper - temp;
                twiddleFactor *= twiddleFactorRoot; // Update twiddle factor
            }
        }
    }
}

// Function to print complex arrays
void printComplexArray(const vector<complex<double>>& signal) {
    for (size_t i = 0; i < signal.size(); i++) {
        cout << "X[" << i << "] = " << signal[i] << endl;
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
    int fftSize = 16; // FFT size (must be a power of 2)

    // Example input: 8-point sequence
    /*vector<complex<double>> inputSignal = {
        {3.6, 2.6}, {2.9, 6.3}, {5.6, 4.0}, {4.8, 9.1},
        {3.3, 0.4}, {5.9, 4.8}, {5.0, 2.6}, {4.3, 4.1}
    };*/
    vector<complex<double>> inputSignal = {
        {3.6, 2.6}, {2.9, 6.3}, {5.6, 4.0}, {4.8, 9.1},
        {3.3, 0.4}, {5.9, 4.8}, {5.0, 2.6}, {4.3, 4.1},
        {1.5, 0.9}, {2.2, 3.4}, {3.7, 6.0}, {1.9, 2.3},
        {6.4, 0.7}, {5.5, 2.0}, {4.6, 1.2}, {3.1, 5.6}
    };
    //vector<complex<double>> inputSignal = generateRandomSignal(fftSize);

    cout << "Input:\n";
    printComplexArray(inputSignal);

    auto start = std::chrono::high_resolution_clock::now();
    // Perform FFT
    iterativeFFT(inputSignal);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    cout << "\nFFT Output:\n";
    printComplexArray(inputSignal);
    std::cout << "FFT iterative:\tnumber of points: " << fftSize << "\ttime:\t" << duration.count() << "\tmicrosec\n";
    return 0;
}
