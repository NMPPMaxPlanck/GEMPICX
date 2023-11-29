#include <AMReX.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_amrex_init.H>
#include <GEMPIC_parameters.H>
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_mesh_coupling.H>
#include <GEMPIC_sampler.H>
#include <GEMPIC_hs_zigzag.H>
#include <GEMPIC_Fields.H>
#include <GEMPIC_FDDeRhamComplex.H>
#include <GEMPIC_PoissonSolver.H>
#include <GEMPIC_FunctionParse.H>

#include <random>
#include <iostream>

using namespace Gempic;
using namespace CompDom;
using namespace Sampling;
using namespace GEMPIC_Fields;
using namespace GEMPIC_FDDeRhamComplex;

int main(int argc, char* argv[])
{
    //amrex::Initialize(argc, argv);
    const bool build_parm_parse = true;
    amrex::Initialize(argc, argv, build_parm_parse, MPI_COMM_WORLD,
		      overwrite_amrex_parser_defaults);

    // Initialize the main parameters instance and tell it to print output
    Parameters::setPrintOutput();
    Parameters parameters{};

    constexpr unsigned int vdim{3};
    constexpr unsigned int numspec{1};
    // Spline degrees. Linear splines is ok
    constexpr int degx{1};
    constexpr int degy{1};
    constexpr int degz{1};
    constexpr int maxSplineDegree(std::max(std::max(degx,degy),degz));
    // lower dimension Hodge is good enough
    constexpr int hodgeDegree{2};

{

    // Initialize computational_domain
    computational_domain infra;

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, maxSplineDegree);

    // parseB object needs to be kept alive at least as long as funcB
    auto [parseB, funcB] = Utils::parseFunctions<3>({"Bx", "By", "Bz"});

    // Initialize fields that we'll definitely use, pinky swear
    [[maybe_unused]] DeRhamField<Grid::primal, Space::face> B(deRham, funcB);

    // Initialize particle groups
    amrex::GpuArray<std::unique_ptr<particle_groups<vdim>>, numspec> ions;

    // initialize particles & loop preparation (probably should be one initializer):
    for (int spec = 0; spec < numspec; spec++)
    {
        ions[spec] =
            std::make_unique<particle_groups<vdim>>(spec, infra);
        init_particles_full_domain<vdim, numspec>(infra, ions, spec);
    }

    {// "Time Loop" scope. Should be a separate function
        Parameters params("time_loop");

        amrex::Real dt;
        params.get("dt", dt);
        int nSteps;
        params.get("n_steps", nSteps);

        int saveFields = 0;
        params.getOrSet("save_fields", saveFields);

        // a time loop would be run here, using the variables just loaded
    }
}
        amrex::Finalize();
}
