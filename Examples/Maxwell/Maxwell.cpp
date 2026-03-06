/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <chrono>
#include <iostream>
#include <vector>

#include <AMReX_Periodicity.H>

#include "GEMPIC_CustomDiagnosticStrategies.H"
#include "GEMPIC_MaxwellInit.H"
#include "GEMPIC_MaxwellPDE.H"
#include "GEMPIC_NumTools.H"
#include "GEMPIC_RungeKutta.H"
#include "GEMPIC_Solvers.H"

static constexpr Gempic::TimeLoop::RungeKuttaTag rkTag = Gempic::TimeLoop::RungeKuttaTag::RK3;
static constexpr Gempic::TimeLoop::RungeKuttaTag imExRkTag =
    Gempic::TimeLoop::RungeKuttaTag::SASPImExRK3;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    Gempic::Utils::Verbosity::set_level(1);
    Gempic::Io::Parameters parameters{};

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
        std::string mIOcustomId{"HighResSubcellOutputProcessor"};
        // Add my new custom strategy
        Gempic::Io::add_output_processor<Gempic::Io::HighResSubcellOutputProcessor>(mIOcustomId);
        // Use TypeSelector to select the appropriate class
        using RKTBtype =
            typename Gempic::TimeLoop::RKTypeSelector<rkTag>::selected_RK_ButcherTableau;
        // Create an instance of the selected type
        RKTBtype rKreference;
        // Use TypeSelector to select the appropriate class
        using IMEXRKTBtype =
            typename Gempic::TimeLoop::RKTypeSelector<imExRkTag>::selected_RK_ButcherTableau;
        // Create an instance of the selected type
        IMEXRKTBtype imexrKreference;

        // the Maxwell numerical scheme class
        MaxwellNumericalScheme maxwell;

        // the IMEX RK class
        Gempic::TimeLoop::ImexRk<MaxwellFieldsHandlerStruct, imExRkTag, MaxwellDeRham>
            implicitExplicitRk(maxwell.m_disc.m_fields, maxwell, maxwell.m_disc.m_drc,
                               imexrKreference);
        {
            maxwell.set_initial_condition();
            // print initial condition
            maxwell.print_now();
            int nmaxSteps{maxwell.m_nmaxSteps};
            // time loop
            for (int tStep = 1; tStep < nmaxSteps; tStep++)
            {
                // break if final time is reached
                if (maxwell.is_finaltime_reached()) break;

                // eventually recompute dt:
                bool breakLoop = maxwell.is_finaltime_reached();
                if (breakLoop) break;
                maxwell.init_new_timestep();

                // use IMEX RK to update one time-step
                implicitExplicitRk.integrate_step(maxwell.m_disc.m_time, maxwell.m_disc.m_dt);

                maxwell.finalize_new_timestep();
                // print
                maxwell.check_and_print(tStep);
                amrex::Print() << "finished time-step: " << tStep
                               << " at time: " << maxwell.m_disc.m_time << std::endl;
            }
        }
    }
    amrex::Finalize();
}