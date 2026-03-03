/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <ctime>
#include <iostream>
#include <random>

#include <AMReX.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include "GEMPIC_AmrexInit.H"
#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_FunctionParse.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_Particle.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_PoissonSolver.H"
#include "GEMPIC_Sampler.H"

using namespace Gempic;
using namespace Particle;
using namespace Forms;

int main (int argc, char* argv[])
{
    // amrex::Initialize(argc, argv);
    bool const buildParmParse = true;
    amrex::Initialize(argc, argv, buildParmParse, MPI_COMM_WORLD, overwrite_amrex_parser_defaults);
    BL_PROFILE_VAR("main()", pmain);
    // Initialize the main parameters instance and tell it to print output
    Io::Parameters::set_print_output();
    Io::Parameters parameters{};

    constexpr unsigned int vdim{3};
    // Spline degrees. Linear splines is ok
    constexpr int degx{1};
    constexpr int degy{1};
    constexpr int degz{1};
    constexpr int maxSplineDegree(std::max(std::max(degx, degy), degz));
    // lower dimension Hodge is good enough
    constexpr int hodgeDegree{2};

    {
        // Initialize computational_domain
        ComputationalDomain infra;

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, maxSplineDegree,
                                                        HodgeScheme::FDHodge);

        // parseB object needs to be kept alive at least as long as funcB
        auto [parseB, funcB] = Utils::parse_functions<3>({"Bx", "By", "Bz"});

        // Initialize fields that we'll definitely use, pinky swear
        [[maybe_unused]] DeRhamField<Grid::primal, Space::face> B(deRham, funcB);

        // Initialize particles
        std::vector<std::shared_ptr<ParticleSpecies<vdim>>> ions;
        init_particles(ions, infra);

        { // "Time Loop" scope. Should be a separate function
            Io::Parameters params("TimeLoop");

            amrex::Real dt;
            params.get("dt", dt);
            int nSteps;
            params.get("nSteps", nSteps);

            int saveFields = 0;
            params.get_or_set("saveFields", saveFields);

            // a time loop would be run here, using the variables just loaded
        }
    }
    BL_PROFILE_VAR_STOP(pmain);
    amrex::Finalize();
}
