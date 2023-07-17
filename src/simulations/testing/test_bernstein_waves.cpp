#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_amrex_init.H>
#include <GEMPIC_parameters.H>
#include <GEMPIC_Params.H>
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_mesh_coupling.H>
#include <GEMPIC_particle_mesh_coupling_C2.H>
//#include <GEMPIC_PlotFile.H>
//#include <GEMPIC_profiling.H>
#include <GEMPIC_sampler.H>
//#include <GEMPIC_time_loop_hs_zigzag.H>
#include <GEMPIC_hs_zigzag.H>
#include <GEMPIC_Fields.H>
#include <GEMPIC_FDDeRhamComplex.H>
#include <GEMPIC_PoissonSolver.H>

#include <random>
#include <iostream>

using namespace Gempic;
using namespace Sampling;

void write_rho(DeRhamField<Grid::dual, Space::cell> &rho, computational_domain& infra, double time, int step) {
        // save fields
        // MultiFab Info -------------------------------------------------------------
        std::string plotfilename{"Plotfiles/" + amrex::Concatenate("rho", step)};
 
        amrex::Vector<std::string> varnames{"rho"};
 
        amrex::WriteSingleLevelPlotfile(plotfilename, rho.data, varnames, infra.geom, time, 0);
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    // Linear splines is ok, and lower dimension Hodge is good enough
    constexpr int vdim = 3;
    constexpr int numspec = 1;
    // Spline degrees
    constexpr int degx = 1;
    constexpr int degy = 1;
    constexpr int degz = 1;
    //
    constexpr int degmw = 2;
    constexpr int propagator = 3;

    const int hodgeDegree = 2;

{
    gempic_parameters<vdim, numspec> parametersBernstein;
    parametersBernstein.read_pp_params();
    parametersBernstein.set_computed_params();
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic = {AMREX_D_DECL(
                                                     parametersBernstein.is_periodic[0],
                                                     parametersBernstein.is_periodic[1],
                                                     parametersBernstein.is_periodic[2])};

    Parameters params(parametersBernstein.real_box,
                      parametersBernstein.n_cell,
                      parametersBernstein.max_grid_size,
                      isPeriodic,
                      hodgeDegree);

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(params);

	DeRhamField<Grid::primal, Space::face> B(deRham);
	DeRhamField<Grid::primal, Space::edge> E(deRham);
    DeRhamField<Grid::dual, Space::cell> rho(deRham);
    DeRhamField<Grid::primal, Space::node> phi(deRham); // Parse analytical fields and and initialize parserEval. Has to be the same as Bx,By,Bz and Ex,
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
    amrex::GpuArray<std::unique_ptr<particle_groups<vdim>>, numspec> ions;

    // In order to use the particle sampler we need to also initialize computational_domain
    // This class does the same as GEMPIC_parameters.H and needs to be urgently redesigned. 
    // {1, 1, 1} represents periodicity, has different types than Params and gempic_parameters.
    computational_domain infra;
    infra.initialize_computational_domain(parametersBernstein.n_cell, parametersBernstein.max_grid_size, parametersBernstein.is_periodic, parametersBernstein.real_box);

    for (int spec = 0; spec < numspec; spec++)
    {
        ions[spec] =
            std::make_unique<particle_groups<vdim>>(parametersBernstein.charge[spec], parametersBernstein.mass[spec], infra);
    }

    // initialize particles & loop preparation:
    // FIRST SPECIES
    amrex::Vector<amrex::Vector<amrex::Real>> meanVelocity = {{0.0, 0.0, 0.0}};
    amrex::Vector<amrex::Vector<amrex::Real>> vThermal = {{1.0, 1.0, 1.0}};
    amrex::Vector<amrex::Real> vWeight = {1.0};

    init_particles_full_domain<vdim, numspec>(infra,
                                              ions,
                                              parametersBernstein.n_part_per_cell,
                                              meanVelocity,
                                              vThermal, vWeight, 0,
                                              parametersBernstein.densityEval[0]);
    

    const int ndata = 1; // Needs to be 1 so that the correct ParIter type is defined. Putting 4 gets a non-defined type

    for (int spec = 0; spec < numspec; spec++) {

        amrex::Real charge = ions[spec]->getCharge();

        for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*ions[spec], 0); pti.isValid(); ++pti)
        {
            const long np = pti.numParticles();
            const auto& particles = pti.GetArrayOfStructs();
            const auto partData = particles().data();
            const auto weight = pti.GetStructOfArrays().GetRealData(vdim).data();

            amrex::Array4<amrex::Real> const& rhoarr = rho.data[pti].array();

            amrex::ParallelFor(np,
                            [=] AMREX_GPU_DEVICE(long pp)
                            {
                                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> positionParticle;
                                for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                                    positionParticle[d] = partData[pp].pos(d);
                                Spline::SplineBase<degx, degy, degz> spline(positionParticle, infra.plo, infra.dxi);
                                // Needs at least max(degx, degy, degz) ghost cells
                                gempic_deposit_rho_C3_new_splines<degx, degy, degz>(
                                    spline, charge * infra.dxi[GEMPIC_SPACEDIM] * weight[pp],
                                    rhoarr);
                            });
        }
    }

    // Needed for SumBoundary
    auto nGhost = deRham->getNGhost();

    (rho.data).SumBoundary(0, 1, {nGhost[0], nGhost[1], nGhost[2]}, {0, 0, 0},
                           params.geometry().periodicity());
    rho.averageSync();
    rho.fillBoundary();

    amrex::Real dt = parametersBernstein.dt;
    int nSteps = parametersBernstein.n_steps;

    for (int tStep = 0; tStep < nSteps; tStep++) {

        for (int spec = 0; spec < numspec; spec++)
        {
            amrex::Real charge = ions[spec]->getCharge();
            amrex::Real chargemass = charge / ions[spec]->getMass();
            amrex::Real a = 0.5 * chargemass * dt;

            for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*ions[spec], 0); pti.isValid(); ++pti)
            {
                amrex::Particle<0, 0>* AMREX_RESTRICT particles = &(pti.GetArrayOfStructs()[0]);
                const long np = pti.numParticles();
                auto particle_attributes = &pti.GetStructOfArrays();
                amrex::ParticleReal* const AMREX_RESTRICT velx =
                    particle_attributes->GetRealData(0).data();
                amrex::ParticleReal* const AMREX_RESTRICT vely =
                    particle_attributes->GetRealData(1).data();
                amrex::ParticleReal* const AMREX_RESTRICT velz =
                    particle_attributes->GetRealData(2).data();

                const auto weight = pti.GetStructOfArrays().GetRealData(vdim).data();

                amrex::Real Bz = 1.0;
                amrex::Real a = a * Bz;

                amrex::Array4<amrex::Real> const& rhoarr = rho.data[pti].array();

                amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp)
                {
                    // Read out particle position
                    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> positionParticle;
                    for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                        positionParticle[d] = particles[pp].pos(d);

                    amrex::Real vx = velx[pp];
                    amrex::Real vy = vely[pp];

                    velx[pp] = (vx*(1.-a*a) + 2.*a*vy)/(1.+a*a);
                    vely[pp] = (vy*(1.-a*a) - 2.*a*vx)/(1+a*a);

                    for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d) {
                        positionParticle[d] = positionParticle[d] + dt * 0.5 * velx[pp];
                        particles[pp].pos(d) = positionParticle[d];
                    }

                    Spline::SplineBase<degx, degy, degz> splinePos(positionParticle, infra.plo, infra.dxi);

                    gempic_deposit_rho_C3_new_splines<degx, degy, degz>(
                        splinePos, charge * infra.dxi[GEMPIC_SPACEDIM] * weight[pp], rhoarr);
                });
            }
            ions[spec] -> Redistribute();
        }
        if (tStep%parametersBernstein.save_particles == 0) {
            write_rho(rho, infra, tStep*parametersBernstein.dt, tStep);
        }
    }
}
    amrex::Finalize();
}
