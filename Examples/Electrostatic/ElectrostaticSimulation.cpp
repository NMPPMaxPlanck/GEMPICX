#include <iostream>
#include <random>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include "GEMPIC_AmrexInit.H"
#include "GEMPIC_BilinearFilter.H"
#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_Diagnostics.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_ParticleGroups.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_PoissonSolver.H"
#include "GEMPIC_Sampler.H"

using namespace Gempic;
using namespace Forms;
using namespace Particle;
using namespace ParticleMeshCoupling;
using namespace FieldSolvers;

int main (int argc, char *argv[])
{
    amrex::Initialize(argc, argv);

    // Linear splines is ok, and lower dimension Hodge is good enough
    constexpr int vdim{3};
    constexpr int ndata{1};  // Needs to be 1 so that the correct ParIter type is defined. Putting 4
                             // gets a non-defined type
    // Node spline degrees (smoothing spline degree is one less in each direction)
    constexpr int degx{3};
    constexpr int degy{3};
    constexpr int degz{3};
    constexpr int maxSplineDegree{std::max(std::max(degx, degy), degz)};

    constexpr int hodgeDegree{2};

    {
        BL_PROFILE("ElectrostaticMain()");
        // Parameters::setPrintOutput();  // uncomment to print an output file
        Io::Parameters parameters{};

        // Initialize computational_domain
        ComputationalDomain infra;

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, maxSplineDegree,
                                                        HodgeScheme::FDHodge);

        // Initialize simulation
        Io::Parameters paramsSim("Sim");
        std::string simType{"UNDEFINED"};
        paramsSim.get_or_set("Type", simType);

        auto poisson = std::make_shared<PoissonSolver>(deRham, infra);
#ifdef AMREX_USE_HYPRE
        HypreLinearSystem<DeRhamField<Grid::dual, Space::cell>,
                          DeRhamField<Grid::primal, Space::node>, Operator::poisson, hodgeDegree>
            hyprePoisson(&infra, deRham, poisson);
#else
        ConjugateGradient<DeRhamField<Grid::dual, Space::cell>,
                          DeRhamField<Grid::primal, Space::node>, Operator::poisson>
            cgPoisson(deRham, poisson);
#endif
        amrex::Real te{1.0};  // electron temperature (default 1.0)

        if (simType == "QuasiNeutral")
        {
            amrex::Print() << "****************************\n";
            amrex::Print() << "* Quasi-neutral simulation *\n";
            amrex::Print() << "****************************\n";
            paramsSim.get_or_set("Te", te);
        }
        else if (simType == "VlasovPoisson")
        {
            amrex::Print() << "*****************************\n";
            amrex::Print() << "* Vlasov-Poisson simulation *\n";
            amrex::Print() << "*****************************\n";
        }
        else
        {
            amrex::AllPrint() << "Simulation type " << simType << " is not implemented"
                              << std::endl;
            amrex::Abort();
        }

        // initialize fields
        auto [parseB, funcB] = Utils::parse_functions<3>({"Bx", "By", "Bz"});

        DeRhamField<Grid::primal, Space::face> B(deRham, funcB, "B");
        DeRhamField<Grid::dual, Space::edge> H(deRham, funcB, "H");
        DeRhamField<Grid::primal, Space::edge> E(deRham, "E");
        DeRhamField<Grid::dual, Space::face> D(deRham, "D");
        DeRhamField<Grid::dual, Space::cell> rho(deRham, "rho");
        DeRhamField<Grid::dual, Space::cell> divD(deRham, "divD");
        DeRhamField<Grid::dual, Space::cell> rhoFiltered(deRham);
        DeRhamField<Grid::primal, Space::node> phi(deRham, "phi");

        // Initialize particle groups
        std::vector<std::shared_ptr<ParticleGroups<vdim>>> ions;
        init_particles(ions, infra);

        // Initializing filter
        std::unique_ptr<Filter::Filter> biFilter = std::make_unique<Filter::BilinearFilter>();

        // For the moment we consider only a constant background field.
        amrex::Real Bz = funcB[zDir](AMREX_D_DECL(0., 0., 0.), 0.);

        {  // "Time Loop" scope. Should be a separate function

            // Initialize full diagnostics and write initial time step
            Io::Parameters params("TimeLoop");
            amrex::Real dt;
            params.get("dt", dt);
            int nSteps;
            params.get("nSteps", nSteps);
            auto diagnostics = Io::make_diagnostics<degx, degy, degz>(infra, deRham, ions);

            // Deposit initial charge
            for (auto &particleSpecies : ions)
            {
                amrex::Real charge = particleSpecies->get_charge();

                for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*particleSpecies, 0); pti.isValid();
                     ++pti)
                {
                    const long np = pti.numParticles();
                    auto *const particles = pti.GetArrayOfStructs()().data();
                    auto *const weight = pti.GetStructOfArrays().GetRealData(vdim).data();

                    amrex::Array4<amrex::Real> const &rhoarr = rho.m_data[pti].array();

                    amrex::ParallelFor(
                        np,
                        [=] AMREX_GPU_DEVICE(long pp)
                        {
                            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> positionParticle;
                            for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                            {
                                positionParticle[d] = particles[pp].pos(d);
                            }
                            SplineBase<degx, degy, degz> spline(positionParticle, infra.m_plo,
                                                                infra.m_dxi);
                            // Needs at least max(degx, degy, degz) ghost cells
                            deposit_rho(rhoarr, spline, charge * weight[pp]);
                        });
                }
            }

            rho.post_particle_loop_sync();

            // Add background charge (needs to be done after post_particle_loop_sync)
            amrex::Real rhoBackground = 1.0;
            rho +=
                rhoBackground * GEMPIC_D_MULT(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir]);

            // Apply filter and compute phi with filtered rho
            biFilter->apply_stencil(rho.m_data, rhoFiltered.m_data);
            if (simType == "QuasiNeutral")
            {
                deRham->hodge(phi, rhoFiltered);
                deRham->grad(E, phi);
                E *= -te;
            }
            else if (simType == "VlasovPoisson")
            {
                //poisson->solve_amrex(rhoFiltered, phi);
#ifdef AMREX_USE_HYPRE
                hyprePoisson.solve(phi, rhoFiltered);
#else
                cgPoisson.solve(phi, rhoFiltered);
#endif
                deRham->grad(E, phi);
                E *= -1.0;
            }

            // D is also needed to compute energy
            deRham->hodge(D, E);

            // Write initial time step
            amrex::Real simTime{0.0};
            diagnostics.compute_and_write_to_file(0, simTime);

            for (int tStep = 0; tStep < nSteps; tStep++)
            {
                rho.m_data.setVal(0.0);
                for (auto &particleSpecies : ions)
                {
                    amrex::Real charge = particleSpecies->get_charge();
                    amrex::Real chargemass = charge / particleSpecies->get_mass();
                    amrex::Real a = 0.5 * chargemass * dt * Bz;

                    for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*particleSpecies, 0);
                         pti.isValid(); ++pti)
                    {
                        const long np = pti.numParticles();
                        const auto &particles = pti.GetArrayOfStructs()().data();
                        auto *const velx = pti.GetStructOfArrays().GetRealData(0).data();
                        auto *const vely = pti.GetStructOfArrays().GetRealData(1).data();
                        auto *const velz = pti.GetStructOfArrays().GetRealData(2).data();
                        auto *const weight = pti.GetStructOfArrays().GetRealData(vdim).data();

                        amrex::Array4<amrex::Real> const &rhoarr = rho.m_data[pti].array();

                        amrex::ParallelFor(
                            np,
                            [=] AMREX_GPU_DEVICE(long pp)
                            {
                                // Local arrays for particle position and velocities
                                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> positionParticle;
                                amrex::GpuArray<amrex::Real, vdim> vel{velx[pp], vely[pp],
                                                                       velz[pp]};
                                for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                                {
                                    // positionParticle data structure needed for spline
                                    positionParticle[d] = particles[pp].pos(d) + 0.5 * dt * vel[d];
                                    particles[pp].pos(d) = positionParticle[d];
                                }

                                SplineBase<degx, degy, degz> spline(positionParticle, infra.m_plo,
                                                                    infra.m_dxi);

                                deposit_rho(rhoarr, spline, charge * weight[pp]);
                            });
                    }

                    particleSpecies->Redistribute();

                    rho.post_particle_loop_sync();

                    // Apply filter and compute phi with filtered rho
                    biFilter->apply_stencil(rho.m_data, rhoFiltered.m_data);
                    if (simType == "QuasiNeutral")
                    {
                        deRham->hodge(phi, rhoFiltered);
                        deRham->grad(E, phi);
                        E *= -te;
                    }
                    else if (simType == "VlasovPoisson")
                    {
                        // set initial guess phi to 0
                        phi.m_data.setVal(0.0);
#ifdef AMREX_USE_HYPRE
                        hyprePoisson.solve(phi, rhoFiltered);
#else
                        cgPoisson.solve(phi, rhoFiltered);
#endif
                        deRham->grad(E, phi);
                        E *= -1.0;
                    }
                    // D is also needed to compute energy
                    deRham->hodge(D, E);

                    rho.m_data.setVal(0.0);

                    for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*particleSpecies, 0);
                         pti.isValid(); ++pti)
                    {
                        const long np = pti.numParticles();
                        const auto &particles = pti.GetArrayOfStructs()().data();
                        auto *const velx = pti.GetStructOfArrays().GetRealData(0).data();
                        auto *const vely = pti.GetStructOfArrays().GetRealData(1).data();
                        auto *const velz = pti.GetStructOfArrays().GetRealData(2).data();
                        auto *const weight = pti.GetStructOfArrays().GetRealData(vdim).data();

                        amrex::Array4<amrex::Real> const &rhoarr = rho.m_data[pti].array();
                        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;

                        // Extract E
                        for (int cc = 0; cc < vdim; cc++)
                        {
                            eA[cc] = (E.m_data[cc])[pti].array();
                        }

                        amrex::ParallelFor(
                            np,
                            [=] AMREX_GPU_DEVICE(long pp)
                            {
                                // Read out particle position
                                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> positionParticle;
                                for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                                {
                                    positionParticle[d] = particles[pp].pos(d);
                                }

                                amrex::GpuArray<amrex::Real, vdim> vel{velx[pp], vely[pp],
                                                                       velz[pp]};

                                SplineBase<degx, degy, degz> spline(positionParticle, infra.m_plo,
                                                                    infra.m_dxi);

                                // evaluate the electric field
                                amrex::GpuArray<amrex::Real, vdim> efield =
                                    spline.template eval_spline_field<Field::PrimalOneForm>(eA);

                                // push v with the electric field over dt/2
                                for (int i = 0; i < vdim; i++)
                                {
                                    vel[i] += 0.5 * dt * chargemass * efield[i];
                                }
                                // rotate v with magnetic field over dt
                                amrex::Real vx = vel[xDir];
                                amrex::Real vy = vel[yDir];

                                // Crank Nicolson update
                                vel[xDir] = (vx * (1. - a * a) + 2. * a * vy) / (1. + a * a);
                                vel[yDir] = (vy * (1. - a * a) - 2. * a * vx) / (1. + a * a);

                                // push v with the electric field over dt/2
                                for (int i = 0; i < vdim; i++)
                                {
                                    vel[i] += 0.5 * dt * chargemass * efield[i];
                                }

                                // update global particle velocities arrays
                                velx[pp] = vel[xDir];
                                vely[pp] = vel[yDir];
                                velz[pp] = vel[zDir];

                                for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                                {
                                    positionParticle[d] = particles[pp].pos(d) + 0.5 * dt * vel[d];
                                    particles[pp].pos(d) = positionParticle[d];
                                }

                                SplineBase<degx, degy, degz> splineNew(positionParticle,
                                                                       infra.m_plo, infra.m_dxi);

                                deposit_rho(rhoarr, splineNew, charge * weight[pp]);
                            });
                    }
                    particleSpecies->Redistribute();
                }
                rho.post_particle_loop_sync();

                simTime = dt * (tStep + 1);
                diagnostics.compute_and_write_to_file(tStep + 1, simTime);

                if (tStep % 10 == 0)
                {
                    amrex::Print() << "Time Step: " << tStep + 1 << '\n';
                }
            }
        }  // end of "time loop" scope
    }
    amrex::Finalize();
}
