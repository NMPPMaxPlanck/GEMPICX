#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_Params.H>
#include <GEMPIC_amrex_init.H>
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_parameters.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_mesh_coupling.H>
#include <GEMPIC_particle_mesh_coupling_C2.H>
// #include <GEMPIC_PlotFile.H>
// #include <GEMPIC_profiling.H>
#include <GEMPIC_FDDeRhamComplex.H>
#include <GEMPIC_Fields.H>
#include <GEMPIC_PoissonSolver.H>
#include <GEMPIC_sampler.H>
#include <GEMPIC_time_loop_hs_zigzag.H>

using namespace amrex;
using namespace Gempic;

// using namespace Diagnostics_Output;
// using namespace Field_solvers;
using namespace Particles;
using namespace Sampling;
using namespace Time_Loop;
// using namespace Vlasov_Maxwell;
// using namespace Profiling;
using namespace GEMPIC_Fields;
using namespace GEMPIC_FDDeRhamComplex;
using namespace GEMPIC_PoissonSolver;

template <int vdim, int numspec, int degx, int degy, int degz, int degmw, int propagator>
void main_main()
{
    // const amrex::Real k = 0.6283185;
    const amrex::Real k = 0.4;
    /* Initialize the infrastructure */
    const amrex::RealBox realBox({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                 {AMREX_D_DECL(2.0 * M_PI / k, 2.0 * M_PI / k, 2.0 * M_PI / k)});
    const amrex::IntVect nCell = {AMREX_D_DECL(16, 2, 2)};
    const amrex::IntVect maxGridSize = {AMREX_D_DECL(16, 2, 2)};
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic = {1, 1, 1};
    const int hodgeDegree = 2;

    Parameters params(realBox, nCell, maxGridSize, isPeriodic, hodgeDegree);

    const amrex::Real dt = 0.02;

    /* Initialize particle parameters */
    const int strang_order = 2;
    bool ctest = true;
    gempic_parameters<vdim, numspec> VlMa;
    amrex::GpuArray<std::string, numspec>
        partDensity;         // species density, to be used in the sampler
    partDensity[0] = "1.0";  // first species
    // VlMa.init_Nghost(degx, degy, degz);
    amrex::GpuArray<int, numspec> nParticlesPerCell = {100};
    VlMa.set_params("test_vlasov_maxwell_hs_multispecies",  // sim_name
                    nCell,                                  // n_cell_vector
                    nParticlesPerCell,                      // n_part_per_cell
                    10,                                     // n_steps
                    10000,                                  // output freq
                    10000,                                  // output freq
                    10000,                                  // output freq
                    {AMREX_D_DECL(1, 1, 1)},                // periodicity
                    maxGridSize,                            // max_grid_size
                    dt,                                     // dt
                    {1.0},                                  // charge
                    {1.0},                                  // mass
                    k,                                      // k
                    partDensity,                            // density
                    "0.0",                                  // Bx
                    "0.0",                                  // By
                    "0.0",                                  // Bz
                    "0.0",                                  // Ex
                    "0.0",                                  // Ey
                    "0.0",                                  // Ez
                    "4 * 0.5 * cos(x)",                     // phirho
                    {1},                                    // num_gaussians
                    1);                                     // propagator
    VlMa.set_computed_params();

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(params);

    // Declare the fields
    DeRhamField<Grid::primal, Space::face> B(deRham);
    DeRhamField<Grid::primal, Space::edge> E(deRham);
    DeRhamField<Grid::dual, Space::cell> rho(deRham);
    DeRhamField<Grid::primal, Space::node> phi(deRham);

    // Parse analytical fields and and initialize parserEval. Has to be the same as Bx,By,Bz and Ex,
    // Ey, Ez
    const amrex::Array<std::string, 3> analyticalFuncB = {"0.0", "0.0", "1.0"};

    // Project B and D to a primal and dual two form respectively
    const int nVar = 4;  // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcB;
    amrex::Parser parser;

    for (int i = 0; i < 3; ++i)
    {
        parser.define(analyticalFuncB[i]);
        parser.registerVariables({"x", "y", "z", "t"});
        funcB[i] = parser.compile<4>();
    }

    deRham->projection(funcB, 0.0, B);

    // Initialize particle groups
    amrex::GpuArray<std::unique_ptr<particle_groups<vdim>>, numspec> partGr;

    // In order to use the particle sampler we need to also initialize computational_domain
    // This class does the same as GEMPIC_parameters.H and needs to be urgently redesigned.
    // {1, 1, 1} represents periodicity, has different types than Params and gempic_parameters.
    computational_domain infra;
    infra.initialize_computational_domain(nCell, maxGridSize, {1, 1, 1}, realBox);

    for (int spec = 0; spec < numspec; spec++)
    {
        partGr[spec] =
            std::make_unique<particle_groups<vdim>>(VlMa.charge[spec], VlMa.mass[spec], infra);
    }

    // initialize particles & loop preparation:
    // FIRST SPECIES
    amrex::Vector<amrex::Vector<amrex::Real>> meanVelocity = {{0.0, 0.0, 0.0}};
    amrex::Vector<amrex::Vector<amrex::Real>> vThermal = {{1.0, 1.0, 1.0}};
    amrex::Vector<amrex::Real> vWeight = {1.0};

    // Initializing the particles takes A LOT of time
    init_particles_full_domain<vdim, numspec>(infra, partGr, nParticlesPerCell, meanVelocity,
                                              vThermal, vWeight, 0, VlMa.densityEval[0]);

    // Deposit rho
    // rho.fillBoundary();
    const int ndata = 1;  // Needs to be 1 so that the correct ParIter type is defined. Putting 4
                          // gets a non-defined type
    const int spec = 0;
    amrex::Real charge = partGr[spec]->getCharge();
    amrex::Real chargemass = partGr[spec]->getCharge() / partGr[spec]->getMass();

    for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*partGr[spec], 0); pti.isValid(); ++pti)
    {
        const long np = pti.numParticles();
        const auto& particles = pti.GetArrayOfStructs();
        const auto partData = particles().data();
        const auto weight = pti.GetStructOfArrays().GetRealData(vdim).data();

        amrex::Array4<amrex::Real> const& rhoarr = rho.data[pti].array();

        amrex::ParallelFor(np,
                           [=] AMREX_GPU_DEVICE(long pp)
                           {
                               splines_at_particles<degx, degy, degz> spline;
                               amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
                               for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                                   position[d] = partData[pp].pos(d);
                               spline.init_particles(position, infra.plo, infra.dxi);
                               // Needs at least max(degx, degy, degz) ghost cells
                               gempic_deposit_rho_C3<degx, degy, degz>(
                                   spline, charge * infra.dxi[GEMPIC_SPACEDIM] * weight[pp],
                                   rhoarr);
                           });
    }

    // Needed for SumBoundary
    auto nGhost = deRham->getNGhost();

    (rho.data).SumBoundary(0, 1, {nGhost[0], nGhost[1], nGhost[2]}, {0, 0, 0},
                           params.geometry().periodicity());
    rho.averageSync();
    rho.fillBoundary();

    deRham->hodgeFD<degmw>(rho, phi);

    deRham->grad(phi, E);

    E *= -1.0;

    for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*partGr[spec], 0); pti.isValid(); ++pti)
    {
        amrex::Particle<0, 0>* AMREX_RESTRICT particles = &(pti.GetArrayOfStructs()[0]);
        // amrex::ArrayOfStructs<4, 0, amrex::DefaultAllocator>*  AMREX_RESTRICT particles =
        // pti.GetArrayOfStructs(); // get particles
        const long np = pti.numParticles();
        auto particle_attributes = &pti.GetStructOfArrays();
        amrex::ParticleReal* const AMREX_RESTRICT velx = particle_attributes->GetRealData(0).data();
        amrex::ParticleReal* const AMREX_RESTRICT vely = particle_attributes->GetRealData(1).data();
        amrex::ParticleReal* const AMREX_RESTRICT velz = particle_attributes->GetRealData(2).data();

        const auto weight = pti.GetStructOfArrays().GetRealData(vdim).data();

        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;
        for (int cc = 0; cc < vdim; cc++) eA[cc] = (E.data[cc])[pti].array();

        amrex::Array4<amrex::Real> const& rhoarr = rho.data[pti].array();
        amrex::ParallelFor(
            np,
            [=] AMREX_GPU_DEVICE(long pp)
            {
                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> particlePos;
                for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                    particlePos[d] = particles[pp].pos(d);
                splines_at_particles<degx, degy, degz> splineE;
                splineE.init_particles(particlePos, infra.plo, infra.dxi);

                amrex::GpuArray<amrex::Real, vdim> efield =
                    evaluate_efield<vdim, degx, degy, degz>(splineE, eA);

                amrex::GpuArray<amrex::Real, vdim> particleVel{velx[pp], vely[pp], velz[pp]};

                for (int comp = 0; comp < GEMPIC_SPACEDIM; ++comp)
                {
                    particleVel[comp] += (dt / 2.) * chargemass * efield[comp];
                    // Solve Crank-Nicolson magnetic field
                    particlePos[comp] += dt * particleVel[comp];
                }

                splines_at_particles<degx, degy, degz> splinePos;
                splinePos.init_particles(particlePos, infra.plo, infra.dxi);

                gempic_deposit_rho_C3<degx, degy, degz>(
                    splinePos, charge * infra.dxi[GEMPIC_SPACEDIM] * weight[pp], rhoarr);
            });
    }

    deRham->hodgeFD<degmw>(rho, phi);

    for (int tStep = 0; tStep < 200; ++tStep)
    {
        amrex::VisMF::Header::Version current_version = amrex::VisMF::GetHeaderVersion();
        amrex::VisMF::SetHeaderVersion(amrex::VisMF::Header::Version_v1);
        // save fields
        // MultiFab Info -------------------------------------------------------------
        std::string plotfilename{"Plotfiles/" + amrex::Concatenate("rho", tStep)};

        amrex::Vector<std::string> varnames{"rho"};

        amrex::WriteSingleLevelPlotfile(plotfilename, rho.data, varnames, infra.geom, tStep*dt, 0);

        for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*partGr[spec], 0); pti.isValid(); ++pti)
        {
            amrex::Particle<0, 0>* AMREX_RESTRICT particles = &(pti.GetArrayOfStructs()[0]);
            // amrex::ArrayOfStructs<4, 0, amrex::DefaultAllocator>*  AMREX_RESTRICT particles =
            // pti.GetArrayOfStructs(); // get particles
            const long np = pti.numParticles();
            auto particle_attributes = &pti.GetStructOfArrays();
            amrex::ParticleReal* const AMREX_RESTRICT velx =
                particle_attributes->GetRealData(0).data();
            amrex::ParticleReal* const AMREX_RESTRICT vely =
                particle_attributes->GetRealData(1).data();
            amrex::ParticleReal* const AMREX_RESTRICT velz =
                particle_attributes->GetRealData(2).data();

            const auto weight = pti.GetStructOfArrays().GetRealData(vdim).data();

            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;
            for (int cc = 0; cc < vdim; cc++) eA[cc] = (E.data[cc])[pti].array();

            amrex::Array4<amrex::Real> const& rhoarr = rho.data[pti].array();
            amrex::ParallelFor(
                np,
                [=] AMREX_GPU_DEVICE(long pp)
                {
                    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> particlePos;
                    for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                        particlePos[d] = particles[pp].pos(d);
                    splines_at_particles<degx, degy, degz> splineE;
                    splineE.init_particles(particlePos, infra.plo, infra.dxi);

                    amrex::GpuArray<amrex::Real, vdim> efield =
                        evaluate_efield<vdim, degx, degy, degz>(splineE, eA);

                    amrex::GpuArray<amrex::Real, vdim> particleVel{velx[pp], vely[pp], velz[pp]};

                    for (int comp = 0; comp < GEMPIC_SPACEDIM; ++comp)
                    {
                        particleVel[comp] += (dt / 2.) * chargemass * efield[comp];
                        // Solve Crank-Nicolson magnetic field
                        particlePos[comp] += dt * particleVel[comp];
                    }

                    splines_at_particles<degx, degy, degz> splinePos;
                    splinePos.init_particles(particlePos, infra.plo, infra.dxi);

                    gempic_deposit_rho_C3<degx, degy, degz>(
                        splinePos, charge * infra.dxi[GEMPIC_SPACEDIM] * weight[pp], rhoarr);
                });
        }
    }
}

int main(int argc, char* argv[])
{
    const bool build_parm_parse = true;
    /*
    amrex::Initialize(argc, argv, build_parm_parse, MPI_COMM_WORLD,
                      overwrite_amrex_parser_defaults);
    */

    amrex::Initialize(argc, argv);

    /* This ctest has a different output for each GEMPIC_SPACEDIM. Therefore, the expected_output
    file contains all outputs. For each dimension, apart from running the main_main for the
    dimension, the output for the other dimensions needs to be outputted, so that the comparison to
    the expected_output (which contains all dimensions) works The order of the outputs is:
    GEMPIC_SPACEDIM=1, GEMPIC_SPACEDIM=2, GEMPIC_SPACEDIM=3 */

    // Output for GEMPIC_SPACEDIM=3

    // Linear splines is ok, and lower dimension Hodge is good enough
    const int vdim = 3;
    const int numspec = 1;
    const int degx = 3;
    const int degy = 1;
    const int degz = 1;
    const int degmw = 2;
    const int propagator = 3;
    main_main<vdim, numspec, degx, degy, degz, degmw, propagator>();  // hs_zigzag

    amrex::Finalize();
}
