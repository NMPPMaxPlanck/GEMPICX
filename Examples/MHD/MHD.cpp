#include <iostream>
#include <memory>
#include <mpi.h>
#include <random>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include "GEMPIC_AmrexInit.H"
#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_MHDPDE.H"
#include "GEMPIC_MHDStructDefinitions.H"
#include "GEMPIC_Mpi.H"
#include "GEMPIC_MultiFullDiagnostics.H"
#include "GEMPIC_MultiReducedDiagnostics.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_ParticleGroups.H" // necessary only for the output... (the plotters depends still on the particles)
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_RungeKutta.H"

using namespace Gempic;
using namespace Forms;

int main_new_mhdpde ()
{
    BL_PROFILE_VAR("mainNewMHDPDE()", pmain2);
    int myrank, nMPIranks;
    mpi_debugging(myrank, nMPIranks);

    Gempic::Utils::Verbosity::set_level(1);
    // Add my new custom strategy
    std::string mIOcustomId{"HighResSubcellOutputProcessor"};
    Gempic::Io::add_output_processor<Gempic::Io::HighResSubcellOutputProcessor>(mIOcustomId);

    // Use TypeSelector to select the appropriate class
    using myRKTBtype =
        typename Gempic::TimeLoop::RKTypeSelector<myRKTag>::selected_RK_ButcherTableau;
    // Create an instance of the selected type
    myRKTBtype rKreference;

    std::array<std::string, 2> outputFields = {"Q", "B"};
    MhdfvNumericalScheme myMHDscheme(myrank, nMPIranks, outputFields);

    amrex::Print() << "Initialization step 00" << std::endl;

    Gempic::TimeLoop::ExplicitRK<MHDFieldsHandlerStruct, myRKTag> rKintegrator(
        myMHDscheme.m_mhd.m_disc.m_myfields, myMHDscheme, myMHDscheme.m_mhd.m_disc.m_drc,
        rKreference);

    amrex::Print() << "Initialization step 01" << std::endl;
    // set initial condition
    myMHDscheme.set_initial_condition();

    amrex::Print() << "Initialization step 02" << std::endl;

    // compute initial norms and dt
    myMHDscheme.compute_norms_reduce();
    //

    amrex::Print() << "Initialization step 03" << std::endl;

    auto const strtTotal = amrex::second();

    // compute initial norms and dt
    myMHDscheme.print_now();
    int nMaxSteps{myMHDscheme.m_nmaxSteps};
    //bool break_loop{false};

    amrex::Print() << "Starting time loop 00" << std::endl;
    for (int tStep = 0; tStep < nMaxSteps; tStep++)
    {
        if (Gempic::Utils::Verbosity::level() > 1)
        {
            amrex::Print() << "    --- time loop 00" << std::endl;
        }
        // eventually recompute dt:
        bool breakLoop = myMHDscheme.is_finaltime_reached();
        if (breakLoop) break;
        if (Gempic::Utils::Verbosity::level() > 1)
        {
            amrex::Print() << "    --- time loop 01" << std::endl;
        }
        myMHDscheme.init_new_timestep();
        if (Gempic::Utils::Verbosity::level() > 1)
        {
            amrex::Print() << "    --- time loop 01 B" << std::endl;
        }
        rKintegrator.integrate_step(myMHDscheme.m_mhd.m_disc.m_time, myMHDscheme.m_mhd.m_disc.m_dt);

        if (Gempic::Utils::Verbosity::level() > 1)
        {
            amrex::Print() << "    --- time loop 02" << std::endl;
        }
        /*myMHDscheme.mhd.update_Qold_Q(myMHDscheme.mhd.m_disc, myMHDscheme.mhd.m_disc.m_compDom,
                                      myMHDscheme.mhd.m_disc.m_iopar);
        myMHDscheme.mhd.single_time_step(myMHDscheme.mhd.m_disc,
        myMHDscheme.mhd.m_disc.m_compDom,myMHDscheme.mhd.m_disc.m_iopar);
        myMHDscheme.mhd.update_Qnew(myMHDscheme.mhd.m_disc, myMHDscheme.mhd.m_disc.m_compDom,
                                    myMHDscheme.mhd.m_disc.m_iopar);*/

        myMHDscheme.finalize_new_timestep(); // mhd.m_disc.m_time += mhd.m_disc.m_dt;
        if (Gempic::Utils::Verbosity::level() > 1)
        {
            amrex::Print() << "    --- time loop 03" << std::endl;
        }
        myMHDscheme.check_and_print(tStep);
        if (Gempic::Utils::Verbosity::level() > 1)
        {
            amrex::Print() << "time-step " << tStep + 1
                           << " TIME = " << myMHDscheme.m_mhd.m_disc.m_time
                           << " DT = " << myMHDscheme.m_mhd.m_disc.m_dt << std::endl;
        }
        // compute new norms and dt
        myMHDscheme.compute_norms_reduce();
    }

    myMHDscheme.print_now();
    auto endTotal = amrex::second() - strtTotal;
    amrex::Print() << "\nTotal Time: " << endTotal << '\n';

    BL_PROFILE_VAR_STOP(pmain2);
    return 0;
}

int main (int argc, char* argv[])
{
    int runNewMHDPDE;
    amrex::Initialize(argc, argv);
    {
        BL_PROFILE_VAR("main()", pmain);
        runNewMHDPDE = main_new_mhdpde();

        BL_PROFILE_VAR_STOP(pmain);
    } // namespace amrex
    amrex::Finalize();
    return runNewMHDPDE;
}
