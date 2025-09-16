#include <chrono>
#include <iostream>
#include <vector>

#include <AMReX_Periodicity.H>

#include "GEMPIC_MaxwellInit.H"
#include "GEMPIC_MaxwellPDE.H"
#include "GEMPIC_NumTools.H"
#include "GEMPIC_RungeKutta.H"
#include "GEMPIC_Solvers.H"

static constexpr int timeDegree = 2;
static constexpr int myRKstages = timeDegree + 1; // polynomial degree of the reconstruction.
static constexpr int myTimeIntegration = 2;       // 0: explicit; 1: implicit, 2: imex ...

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
#ifdef MPIDEBUG
    int myrank, nMPIranks;
    nMPIranks = 1;
    myrank = 0;
    int temp;
    MPI_Comm_size(MPI_COMM_WORLD, &nMPIranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    if (myrank == 0)
    {
        std::cout << "MPI debug mode on";
        std::cout << "Please attach to process, then enter a number and hit RETURN.";
        std::cin >> temp;
    }
    MPI_Barrier(MPI_COMM_WORLD); // All threads will wait here until you give thread 0 an input
#endif
    {
        Gempic::Utils::Verbosity::set_level(1);
        std::string mIOcustomId{"HighResSubcellOutputProcessor"};
        // Add my new custom strategy
        Gempic::Io::add_output_processor<Gempic::Io::HighResSubcellOutputProcessor>(mIOcustomId);
        // Use TypeSelector to select the appropriate class
        using myRKTBtype =
            typename Gempic::TimeLoop::RKTypeSelector<myRKstages, 0>::selected_RK_ButcherTableau;
        // Create an instance of the selected type
        myRKTBtype rKreference;
        // Use TypeSelector to select the appropriate class
        using myIMEXRKTBtype = typename Gempic::TimeLoop::RKTypeSelector<
            myRKstages, myTimeIntegration>::selected_RK_ButcherTableau;
        // Create an instance of the selected type
        myIMEXRKTBtype imexrKreference;

        // labels for the output state in the diagnostics
        std::array<std::string, 2> outputFields = {"B", "E"};

        // the Maxwell numerical scheme class
        MaxwellNumericalScheme myMaxwell(outputFields);

        // the IMEX RK class
        Gempic::TimeLoop::ImexRk<MaxwellFieldsHandlerStruct, myRKstages> myImplicitExplicitRK(
            myMaxwell.m_disc.m_myfields, myMaxwell.m_disc.m_myfieldsTmpEx,
            myMaxwell.m_disc.m_myfieldsTmpIm, myMaxwell.m_disc.m_myfieldsDtEx,
            myMaxwell.m_disc.m_myfieldsDtIm, myMaxwell, myMaxwell.m_disc.m_drc, imexrKreference);

        {
            // set initial condition
            myMaxwell.set_initial_condition();
            // print initial condition
            myMaxwell.print_now();
            int nmaxSteps{myMaxwell.m_nmaxSteps};
            // time loop
            for (int tStep = 1; tStep < nmaxSteps; tStep++)
            {
                // eventually recompute dt:
                bool breakLoop = myMaxwell.is_finaltime_reached();
                if (breakLoop) break;

                myMaxwell.init_new_timestep();

                // use IMEX RK to update one time-step
                myImplicitExplicitRK.integrate_step(myMaxwell.m_disc.m_time, myMaxwell.m_disc.m_dt);

                myMaxwell.finalize_new_timestep();
                // print
                myMaxwell.check_and_print(tStep);
                amrex::Print() << "finished time-step: " << tStep
                               << " at time: " << myMaxwell.m_disc.m_time << std::endl;
            }
        }
    }
    amrex::Finalize();
}