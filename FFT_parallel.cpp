#include <iostream>
#include <complex>
#include <cmath>
#include <mpi.h>
#include <random>

const double PI = 3.14159265358979323846;

std::complex<double>* generateRandomSignal(int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    std::complex<double>* signal = new std::complex<double>[size];
    for (int i = 0; i < size; ++i) {
        signal[i] = std::complex<double>(dist(gen), dist(gen));
    }
    return signal;
}

int reverseBits(int index, int numBits) {
    int reversedIndex = 0;
    for (int i = 0; i < numBits; i++) {
        if (index & (1 << i))
            reversedIndex |= (1 << (numBits - 1 - i));
    }
    return reversedIndex;
}

void iterativeFFT(std::complex<double>* signal, int size, int rank, int numProcs) {
    int numStages = std::log2(size);
    std::complex<double>* reorderedSignal = new std::complex<double>[size];

    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            reorderedSignal[reverseBits(i, numStages)] = signal[i];
        }
    }

    MPI_Bcast(reorderedSignal, size * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int stage = 1; stage <= numStages; stage++) {
        int segmentSize = 1 << stage;
        std::complex<double> twiddleFactorRoot = std::polar(1.0, -2 * PI / segmentSize);

        int numSegments = size / segmentSize;
        int baseSegmentsPerProcess = numSegments / numProcs;
        int extraSegments = numSegments % numProcs;
        int localNumSegments = baseSegmentsPerProcess + (rank < extraSegments ? 1 : 0);
        int segmentStart = rank * baseSegmentsPerProcess * segmentSize + std::min(rank, extraSegments) * segmentSize;

        for (int seg = 0; seg < localNumSegments; seg++) {
            int startIdx = segmentStart + seg * segmentSize;
            std::complex<double> twiddleFactor = 1;

            for (int pairIndex = 0; pairIndex < segmentSize / 2; pairIndex++) {
                std::complex<double> temp = twiddleFactor * reorderedSignal[startIdx + pairIndex + segmentSize / 2];
                std::complex<double> upper = reorderedSignal[startIdx + pairIndex];

                reorderedSignal[startIdx + pairIndex] = upper + temp;
                reorderedSignal[startIdx + pairIndex + segmentSize / 2] = upper - temp;
                twiddleFactor *= twiddleFactorRoot;
            }
        }

        int* recvCounts = new int[numProcs];
        int* displacements = new int[numProcs];

        for (int i = 0; i < numProcs; i++) {
            recvCounts[i] = (baseSegmentsPerProcess + (i < extraSegments ? 1 : 0)) * segmentSize * 2;
            displacements[i] = (i > 0) ? (displacements[i - 1] + recvCounts[i - 1]) : 0;
        }

        std::complex<double>* gatheredData = new std::complex<double>[size];
        MPI_Gatherv(reorderedSignal + segmentStart, localNumSegments * segmentSize * 2, MPI_DOUBLE,
            gatheredData, recvCounts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            for (int i = 0; i < size; i++) {
                reorderedSignal[i] = gatheredData[i];
            }
        }

        MPI_Bcast(reorderedSignal, size * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        delete[] recvCounts;
        delete[] displacements;
        delete[] gatheredData;
    }

    for (int i = 0; i < size; i++) {
        signal[i] = reorderedSignal[i];
    }

    delete[] reorderedSignal;
}

void printComplexArray(std::complex<double>* signal, int size) {
    for (int i = 0; i < size; i++) {
        std::cout << "X[" << i << "] = " << signal[i] << std::endl;
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double start_time, end_time;

    //int fftSize = 1024 * 16;
    //complex<double>* inputSignal = generateRandomSignal(fftSize);

    int fftSize = 16;
    std::complex<double>* inputSignal = new std::complex<double>[fftSize] {
        {3.6, 2.6}, { 2.9, 6.3 }, { 5.6, 4.0 }, { 4.8, 9.1 },
        { 3.3, 0.4 }, { 5.9, 4.8 }, { 5.0, 2.6 }, { 4.3, 4.1 },
        { 1.5, 0.9 }, { 2.2, 3.4 }, { 3.7, 6.0 }, { 1.9, 2.3 },
        { 6.4, 0.7 }, { 5.5, 2.0 }, { 4.6, 1.2 }, { 3.1, 5.6 }
    };

    if (rank == 0) {
        std::cout << "Input:\n";
        printComplexArray(inputSignal, fftSize);
    }

    if (rank == 0) {
        start_time = MPI_Wtime();
    }
    iterativeFFT(inputSignal, fftSize, rank, size);

    if (rank == 0) {
        end_time = MPI_Wtime();
        std::cout << "\nFFT Output:\n";
        printComplexArray(inputSignal, fftSize);
        std::cout << "Execution time: " << end_time - start_time << " seconds" << std::endl;
    }

    delete[] inputSignal;
    MPI_Finalize();

    return 0;
}
