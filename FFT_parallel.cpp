#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <mpi.h>
#include <random>

using namespace std;
const double PI = 3.14159265358979323846;

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

// Function to reverse bits in an integer (for bit-reversal permutation)
int reverseBits(int index, int numBits) {
    int reversedIndex = 0;
    for (int i = 0; i < numBits; i++) {
        if (index & (1 << i))
            reversedIndex |= (1 << (numBits - 1 - i));
    }
    return reversedIndex;
}

// Iterative FFT function (Cooley-Tukey Algorithm) with MPI
void iterativeFFT(vector<complex<double>>& signal, int rank, int size) {
    int dataSize = signal.size();
    int numStages = log2(dataSize);

    // Bit-reversal permutation
    vector<complex<double>> reorderedSignal(dataSize);
    if (rank == 0) {
        for (int i = 0; i < dataSize; i++) {
            reorderedSignal[reverseBits(i, numStages)] = signal[i];
        }
    }

    // Broadcast reordered signal to all processes
    MPI_Bcast(reorderedSignal.data(), dataSize * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Iterative FFT computation
    for (int stage = 1; stage <= numStages; stage++) {
        int segmentSize = 1 << stage;  // 2^stage: Current FFT segment size
        complex<double> twiddleFactorRoot = polar(1.0, -2 * PI / segmentSize);

        // Determine the number of segments per process
        int numSegments = dataSize / segmentSize;
        int baseSegmentsPerProcess = numSegments / size;
        int extraSegments = numSegments % size;

        // Distribute extra segments to first 'extraSegments' processes
        int localNumSegments = baseSegmentsPerProcess + (rank < extraSegments ? 1 : 0);
        int segmentStart = rank * baseSegmentsPerProcess * segmentSize + min(rank, extraSegments) * segmentSize;

        vector<complex<double>> localData(segmentSize * localNumSegments);

        // Perform FFT on assigned segments
        for (int seg = 0; seg < localNumSegments; seg++) {
            int startIdx = segmentStart + seg * segmentSize;
            complex<double> twiddleFactor = 1;

            for (int pairIndex = 0; pairIndex < segmentSize / 2; pairIndex++) {
                complex<double> temp = twiddleFactor * reorderedSignal[startIdx + pairIndex + segmentSize / 2];
                complex<double> upper = reorderedSignal[startIdx + pairIndex];

                reorderedSignal[startIdx + pairIndex] = upper + temp;
                reorderedSignal[startIdx + pairIndex + segmentSize / 2] = upper - temp;
                twiddleFactor *= twiddleFactorRoot;
            }
        }

        // Gather results back to master process
        vector<int> recvCounts(size, 0);
        vector<int> displacements(size, 0);

        for (int i = 0; i < size; i++) {
            recvCounts[i] = (baseSegmentsPerProcess + (i < extraSegments ? 1 : 0)) * segmentSize * 2;
            displacements[i] = (i > 0) ? (displacements[i - 1] + recvCounts[i - 1]) : 0;
        }

        vector<complex<double>> gatheredData(dataSize);
        MPI_Gatherv(reorderedSignal.data() + segmentStart, localNumSegments * segmentSize * 2, MPI_DOUBLE,
            gatheredData.data(), recvCounts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            reorderedSignal = gatheredData;
        }

        // Broadcast updated data for next stage
        MPI_Bcast(reorderedSignal.data(), dataSize * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Copy the result back to the signal
    signal = reorderedSignal;
}

// Function to print complex arrays
void printComplexArray(const vector<complex<double>>& signal) {
    for (size_t i = 0; i < signal.size(); i++) {
        cout << "X[" << i << "] = " << signal[i] << endl;
    }
}

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double start_time, end_time;

    int fftSize = 1024 * 16; // FFT size (must be a power of 2)

    /*vector<complex<double>> inputSignal = {
        {3.6, 2.6}, {2.9, 6.3}, {5.6, 4.0}, {4.8, 9.1},
        {3.3, 0.4}, {5.9, 4.8}, {5.0, 2.6}, {4.3, 4.1},
        {1.5, 0.9}, {2.2, 3.4}, {3.7, 6.0}, {1.9, 2.3},
        {6.4, 0.7}, {5.5, 2.0}, {4.6, 1.2}, {3.1, 5.6}
    };*/

    vector<complex<double>> inputSignal = generateRandomSignal(fftSize);

    if (rank == 0) {
        cout << "Input:\n";
        //printComplexArray(inputSignal);
    }

    // Perform FFT
    if (rank == 0) {
        start_time = MPI_Wtime();
    }
    iterativeFFT(inputSignal, rank, size);

    if (rank == 0) {
        end_time = MPI_Wtime();
        cout << "\nFFT Output:\n";
        //printComplexArray(inputSignal);
        cout << "Execution time: " << end_time - start_time << " seconds" << std::endl;
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
