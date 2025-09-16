#ifndef GEMPIC_MPI_H_
#define GEMPIC_MPI_H_

#include <atomic>
#include <chrono>
#include <iostream>
#include <mpi.h>
#include <string>
#include <thread>

#include "GEMPIC_Parameters.H"

void get_input_with_timeout (int timeoutSeconds)
{
    std::atomic<bool> inputReceived{false};

    std::cout << "MPI debug mode on\n";
    std::cout << "Please attach to process, then enter a number and hit ENTER\n";

    std::thread(
        [&] ()
        {
            int remaining = timeoutSeconds;
            std::string clearline(31, ' ');
            while (remaining > 0 && !inputReceived)
            {
                std::cout << clearline << "\rYou have " << remaining << " seconds remaining\r"
                          << std::flush;
                std::this_thread::sleep_for(std::chrono::seconds(1));
                --remaining;
            }

            if (!inputReceived)
            {
                std::terminate();
            }
        })
        .detach();

    std::string dummy;
    std::getline(std::cin, dummy);
    inputReceived = true;
}

enum class MpiDebugMode : bool
{
    True = true,  // enable mpi debugging mode
    False = false // disable mpi debugging mode
};

template <MpiDebugMode userMpiMode>
void set_mpi_debugging(int& myrank, int& nMPIranks);

template <>
void set_mpi_debugging<MpiDebugMode::True>(int& myrank, int& nMPIranks)
{
#ifdef AMREX_USE_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &nMPIranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (myrank == 0)
    {
        get_input_with_timeout(30);
    }

    MPI_Barrier(MPI_COMM_WORLD); // All threads will wait here until you give thread 0 an input
#endif
}
template <>

void set_mpi_debugging<MpiDebugMode::False>(int& myrank, int& nMPIranks)
{
    myrank = 0;
    nMPIranks = 1;
#ifdef AMREX_USE_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &nMPIranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
#endif
}

void mpi_debugging (int& myrank, int& nMPIranks)
{
    int userMpiMode;
    Gempic::Io::Parameters parameters{};
    Gempic::Io::Parameters params("Debugging");
    params.get("mpiDebuggingMode", userMpiMode);
    if (bool(userMpiMode))
    {
        set_mpi_debugging<MpiDebugMode::True>(myrank, nMPIranks);
    }
    else
    {
        set_mpi_debugging<MpiDebugMode::False>(myrank, nMPIranks);
    }
}

#endif