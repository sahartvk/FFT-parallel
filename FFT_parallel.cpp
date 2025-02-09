#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <mpi.h>

//#include <chrono>

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

// Iterative FFT function (Cooley-Tukey Algorithm) with MPI
void iterativeFFT(vector<complex<double>>& signal, int rank, int size) {
    int dataSize = signal.size();
    int numStages = log2(dataSize);

    // Bit-reversal permutation
    vector<complex<double>> reorderedSignal(dataSize);
    for (int i = 0; i < dataSize; i++)
        reorderedSignal[reverseBits(i, numStages)] = signal[i];

    // Each process gets a portion of the reordered signal
    MPI_Bcast(reorderedSignal.data(), dataSize * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD); // Broadcasting reordered signal
    cout <<  rank <<" here1" << endl;

    // Iterative FFT computation
    for (int stage = 1; stage <= numStages; stage++) {
        cout << "satage: " << stage << endl;
        cout << rank << " here2" << endl;

        int segmentSize = 1 << stage;  // 2^stage: Current FFT segment size
        complex<double> twiddleFactorRoot = polar(1.0, -2 * PI / segmentSize); // Twiddle factor

        // Calculate the number of segments for each process
        int numSegmentsPerProcess = dataSize / segmentSize / size;
        int numSegments = dataSize / segmentSize;
        if ((numSegmentsPerProcess == 0 && rank < numSegments) || numSegmentsPerProcess > 0) {
            cout << rank << " here3" << endl;

            if (numSegmentsPerProcess == 0)
            {
                numSegmentsPerProcess = 1;
                cout << rank << " here4" << endl;

            }
            // main part of code
            int segmentStart = rank * numSegmentsPerProcess * segmentSize;

            // Compute FFT for the assigned segments
            for (int seg = 0; seg < numSegmentsPerProcess; seg++) {
                cout << rank << " here5" << endl;

                int startIdx = segmentStart + seg * segmentSize;
                complex<double> twiddleFactor = 1;

                for (int pairIndex = 0; pairIndex < segmentSize / 2; pairIndex++) {
                    cout << rank << " here6" << endl;

                    complex<double> temp = twiddleFactor * reorderedSignal[startIdx + pairIndex + segmentSize / 2];
                    complex<double> upper = reorderedSignal[startIdx + pairIndex];
                    reorderedSignal[startIdx + pairIndex] = upper + temp;
                    reorderedSignal[startIdx + pairIndex + segmentSize / 2] = upper - temp;
                    twiddleFactor *= twiddleFactorRoot; // Update twiddle factor
                }
            }


            // Gather results back to master process
            if ((dataSize / segmentSize / size) != 0){
            //if (segmentSize != dataSize) {

                cout << rank << " before here7" << endl;
                vector<complex<double>> recvBuffer(dataSize);
                MPI_Gather(reorderedSignal.data() + segmentStart, numSegmentsPerProcess * segmentSize * 2, MPI_DOUBLE,
                    recvBuffer.data(), numSegmentsPerProcess * segmentSize * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                reorderedSignal = recvBuffer;
                cout << rank << " here7" << endl;

                
            }
            /*else {
                MPI_Gather(reorderedSignal.data() + segmentStart, numSegmentsPerProcess * segmentSize * 2, MPI_DOUBLE,
                    nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }*/
            cout << rank <<" here8" << endl;

        }
        else {
            //MPI_Finalize();
            cout << rank << " elase" << endl;

        }

        
        // Distribute data back to processes for the next stage
        MPI_Bcast(reorderedSignal.data(), dataSize * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        cout << rank << " here9" << endl;


    }

    // Copy the result back to the signal
    signal = reorderedSignal;
    cout << rank << " here10" << endl;

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

    int fftSize = 16; // FFT size (must be a power of 2)

    /*vector<complex<double>> inputSignal = {
        {3.6, 2.6}, {2.9, 6.3}, {5.6, 4.0}, {4.8, 9.1},
        {3.3, 0.4}, {5.9, 4.8}, {5.0, 2.6}, {4.3, 4.1}
    };*/ 

    // Example input: 16-point sequence
    vector<complex<double>> inputSignal = {
        {3.6, 2.6}, {2.9, 6.3}, {5.6, 4.0}, {4.8, 9.1},
        {3.3, 0.4}, {5.9, 4.8}, {5.0, 2.6}, {4.3, 4.1},
        {1.5, 0.9}, {2.2, 3.4}, {3.7, 6.0}, {1.9, 2.3},
        {6.4, 0.7}, {5.5, 2.0}, {4.6, 1.2}, {3.1, 5.6}
    };

    if (rank == 0) {
        cout << "Input:\n";
        printComplexArray(inputSignal);
    }

    // Perform FFT
    iterativeFFT(inputSignal, rank, size);

    if (rank == 0) {
        cout << "\nFFT Output:\n";
        printComplexArray(inputSignal);
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
