/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <iostream>
#include <random>

#include <AMReX.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include "GEMPIC_AmrexInit.H"
#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_Diagnostics.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_FieldRegistry.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Filter.H"
#include "GEMPIC_NumericalIntegrationDifferentiation.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_Particle.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_PoissonSolver.H"
#include "GEMPIC_Sampler.H"

using namespace Gempic;
using namespace Forms;
using namespace Particle;
using namespace ParticleMeshCoupling;
using namespace FieldSolvers;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    // Linear splines is ok, and lower dimension Hodge is good enough
    constexpr int vdim{3};
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
        Gempic::Io::FieldRegistry fieldRegistry;
        // Initialize simulation
        Io::Parameters paramsSim("Sim");
        std::string simType{"UNDEFINED"};
        paramsSim.get_or_set("Type", simType);

        auto poisson{make_poisson_solver(deRham, infra)};

        amrex::Real te{1.0}; // electron temperature (default 1.0)

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
            GEMPIC_ERROR("Simulation type " + simType + " is not implemented");
        }

        // initialize fields
        auto [parseB, funcB] = Utils::parse_functions<3>({"Bx", "By", "Bz"});

        auto B = Gempic::Io::registered_form<DeRhamField<Grid::primal, Space::face>>(
            fieldRegistry, "B", deRham, funcB);
        auto H = Gempic::Io::registered_form<DeRhamField<Grid::dual, Space::edge>>(
            fieldRegistry, "H", deRham, funcB);
        auto E = Gempic::Io::registered_form<DeRhamField<Grid::primal, Space::edge>>(fieldRegistry,
                                                                                     "E", deRham);
        auto D = Gempic::Io::registered_form<DeRhamField<Grid::dual, Space::face>>(fieldRegistry,
                                                                                   "D", deRham);
        auto rho = Gempic::Io::registered_form<DeRhamField<Grid::dual, Space::cell>>(fieldRegistry,
                                                                                     "rho", deRham);
        auto divD = Gempic::Io::registered_form<DeRhamField<Grid::dual, Space::cell>>(
            fieldRegistry, "divD", deRham);
        DeRhamField<Grid::dual, Space::cell> rhoFiltered(deRham);
        auto phi = Gempic::Io::registered_form<DeRhamField<Grid::primal, Space::node>>(
            fieldRegistry, "phi", deRham);

        // Initialize particles
        std::vector<std::shared_ptr<ParticleSpecies<vdim>>> ions;
        init_particles(ions, infra);

        // Initializing filter
        std::unique_ptr<Filter::Filter> filter = Filter::make_filter(infra);

        // For the moment we consider only a constant background field.
        amrex::Real Bz = funcB[zDir](AMREX_D_DECL(0., 0., 0.), 0.);

        { // "Time Loop" scope. Should be a separate function

            // Initialize full diagnostics and write initial time step
            Io::Parameters params("TimeLoop");
            amrex::Real dt;
            params.get("dt", dt);
            int nSteps;
            params.get("nSteps", nSteps);
            auto diagnostics =
                Io::make_diagnostics<degx, degy, degz>(infra, deRham, fieldRegistry, ions);

            // Deposit initial charge
            deposit_particle_density<degx, degy, degz>(rho, ions, infra);

            // Add background charge (needs to be done after post_particle_loop_sync)
            amrex::Real rhoBackground =
                get_and_apply_neutralizing_background(rho, infra, parameters);

            // Apply filter and compute phi with filtered rho
            filter->apply(rhoFiltered, rho);
            if (simType == "QuasiNeutral")
            {
                hodge(phi, rhoFiltered);
                grad(E, phi);
                E *= -te;
            }
            else if (simType == "VlasovPoisson")
            {
                //poisson->solve_amrex(rhoFiltered, phi);
                poisson->solve(phi, rhoFiltered);
                grad(E, phi);
                E *= -1.0;
            }

            // D is also needed to compute energy
            hodge(D, E);

            // Write initial time step
            amrex::Real simTime{0.0};
            diagnostics.compute_and_write_to_file(0, simTime);

            for (int tStep = 0; tStep < nSteps; tStep++)
            {
                rho.m_data.setVal(0.0);
                for (auto& particleSpecies : ions)
                {
                    amrex::Real charge = particleSpecies->get_charge();

                    for (auto& particleGrid : *particleSpecies)
                    {
                        long const np = particleGrid.numParticles();
                        auto const ptd = particleGrid.GetParticleTile().getParticleTileData();
                        auto const ii = particleSpecies->get_data_indices();

                        amrex::Array4<amrex::Real> const& rhoarr = rho.m_data[particleGrid].array();
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const plo{
                            infra.geometry().ProbLoArray()};

                        amrex::ParallelFor(
                            np,
                            [=] AMREX_GPU_DEVICE(long pp)
                            {
                                // Local arrays for particle position and velocities
                                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle{
                                    AMREX_D_DECL(ptd.rdata(ii.m_iposx)[pp] +
                                                     0.5 * dt * ptd.rdata(ii.m_ivelx)[pp],
                                                 ptd.rdata(ii.m_iposy)[pp] +
                                                     0.5 * dt * ptd.rdata(ii.m_ively)[pp],
                                                 ptd.rdata(ii.m_iposz)[pp] +
                                                     0.5 * dt * ptd.rdata(ii.m_ivelz)[pp])};
                                AMREX_D_EXPR(ptd.rdata(ii.m_iposx)[pp] = positionParticle[xDir],
                                             ptd.rdata(ii.m_iposy)[pp] = positionParticle[yDir],
                                             ptd.rdata(ii.m_iposz)[pp] = positionParticle[zDir]);

                                SplineBase<degx, degy, degz> spline(positionParticle, plo,
                                                                    infra.inv_cell_size_array());

                                deposit_rho(rhoarr, spline, charge * ptd.rdata(ii.m_iweight)[pp]);
                            });
                    }

                    particleSpecies->Redistribute();
                }

                rho.post_particle_loop_sync();
                // add neutralizing background charge
                rho += rhoBackground * infra.cell_volume();

                // Apply filter and compute phi with filtered rho
                filter->apply(rhoFiltered, rho);
                if (simType == "QuasiNeutral")
                {
                    hodge(phi, rhoFiltered);
                    grad(E, phi);
                    E *= -te;
                }
                else if (simType == "VlasovPoisson")
                {
                    // set initial guess phi to 0
                    phi.m_data.setVal(0.0);
                    poisson->solve(phi, rhoFiltered);
                    grad(E, phi);
                    E *= -1.0;
                }
                // D is also needed to compute energy
                hodge(D, E);

                rho.m_data.setVal(0.0);

                for (auto& particleSpecies : ions)
                {
                    amrex::Real charge = particleSpecies->get_charge();
                    amrex::Real chargemass = charge / particleSpecies->get_mass();
                    amrex::Real a = 0.5 * chargemass * dt * Bz;

                    for (auto& particleGrid : *particleSpecies)
                    {
                        long const np = particleGrid.numParticles();
                        auto const ptd = particleGrid.GetParticleTile().getParticleTileData();
                        auto const ii = particleSpecies->get_data_indices();

                        amrex::Array4<amrex::Real> const& rhoarr = rho.m_data[particleGrid].array();
                        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;

                        // Extract E
                        for (int cc = 0; cc < vdim; cc++)
                        {
                            eA[cc] = (E.m_data[cc])[particleGrid].array();
                        }
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const plo{
                            infra.geometry().ProbLoArray()};

                        amrex::ParallelFor(
                            np,
                            [=] AMREX_GPU_DEVICE(long pp)
                            {
                                // Read out particle position
                                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle{
                                    AMREX_D_DECL(ptd.rdata(ii.m_iposx)[pp],
                                                 ptd.rdata(ii.m_iposy)[pp],
                                                 ptd.rdata(ii.m_iposz)[pp])};
                                amrex::GpuArray<amrex::Real, vdim> vel{ptd.rdata(ii.m_ivelx)[pp],
                                                                       ptd.rdata(ii.m_ively)[pp],
                                                                       ptd.rdata(ii.m_ivelz)[pp]};

                                SplineBase<degx, degy, degz> spline(positionParticle, plo,
                                                                    infra.inv_cell_size_array());

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
                                ptd.rdata(ii.m_ivelx)[pp] = vel[xDir];
                                ptd.rdata(ii.m_ively)[pp] = vel[yDir];
                                ptd.rdata(ii.m_ivelz)[pp] = vel[zDir];

                                AMREX_D_EXPR(ptd.rdata(ii.m_iposx)[pp] +=
                                             0.5 * dt * ptd.rdata(ii.m_ivelx)[pp],
                                             ptd.rdata(ii.m_iposy)[pp] +=
                                             0.5 * dt * ptd.rdata(ii.m_ively)[pp],
                                             ptd.rdata(ii.m_iposz)[pp] +=
                                             0.5 * dt * ptd.rdata(ii.m_ivelz)[pp]);
                                AMREX_D_EXPR(positionParticle[xDir] = ptd.rdata(ii.m_iposx)[pp],
                                             positionParticle[yDir] = ptd.rdata(ii.m_iposy)[pp],
                                             positionParticle[zDir] = ptd.rdata(ii.m_iposz)[pp]);

                                SplineBase<degx, degy, degz> splineNew(positionParticle, plo,
                                                                       infra.inv_cell_size_array());

                                deposit_rho(rhoarr, splineNew,
                                            charge * ptd.rdata(ii.m_iweight)[pp]);
                            });
                    }
                    particleSpecies->Redistribute();
                }
                rho.post_particle_loop_sync();
                // add neutralizing background charge
                rho += rhoBackground * infra.cell_volume();

                simTime = dt * (tStep + 1);
                diagnostics.compute_and_write_to_file(tStep + 1, simTime);

                if (tStep % 10 == 0)
                {
                    amrex::Print() << "Time Step: " << tStep + 1 << std::endl;
                }
            }
        } // end of "time loop" scope
    }
    amrex::Finalize();
}
