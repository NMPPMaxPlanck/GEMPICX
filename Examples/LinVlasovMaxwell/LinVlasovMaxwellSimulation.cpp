/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <git.h>
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
#include "GEMPIC_Parameters.H"
#include "GEMPIC_Particle.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_ParticleUtils.H"
#include "GEMPIC_PoissonSolver.H"
#include "GEMPIC_Sampler.H"
#include "GEMPIC_SplineClass.H"
#include "GEMPIC_Splitting.H"

using namespace Gempic;
using namespace Forms;
using namespace Particle;
using namespace ParticleMeshCoupling;
using namespace TimeLoop;
using namespace Utils;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    if (git::AnyUncommittedChanges())
    {
        std::cerr << "WARN: there were uncommitted changes at build-time." << std::endl;
    }
    amrex::Print() << "GEMPIC commit " << git::CommitSHA1() << " (" << git::Branch() << ")\n"
                   << "describe " << git::Describe() << "\n"
                   << std::endl;

    constexpr int vDim{3};
    constexpr int nData{3};

    // Node spline degrees (smoothing spline degree is one less in each direction)
    constexpr int degx{1};
    constexpr int degy{1};
    constexpr int degz{1};
    constexpr int maxSplineDegree{std::max(std::max(degx, degy), degz)};
    //
    constexpr int hodgeDegree{2};

    {
        Io::Parameters parameters{};
        // Initialize computational_domain
        ComputationalDomain infra;

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, maxSplineDegree,
                                                        HodgeScheme::FDHodge);
        Gempic::Io::FieldRegistry fieldRegistry;
        auto [parseBBackground, funcBBackground] =
            Utils::parse_functions<3>({"BBackgroundX", "BBackgroundY", "BBackgroundZ"});

        auto E = Gempic::Io::registered_form<DeRhamField<Grid::primal, Space::edge>>(fieldRegistry,
                                                                                     "E", deRham);
        auto D = Gempic::Io::registered_form<DeRhamField<Grid::dual, Space::face>>(fieldRegistry,
                                                                                   "D", deRham);
        auto B = Gempic::Io::registered_form<DeRhamField<Grid::primal, Space::face>>(fieldRegistry,
                                                                                     "B", deRham);
        DeRhamField<Grid::primal, Space::face> bBackground(deRham, funcBBackground);
        auto H = Gempic::Io::registered_form<DeRhamField<Grid::dual, Space::edge>>(fieldRegistry,
                                                                                   "H", deRham);
        auto J = Gempic::Io::registered_form<DeRhamField<Grid::dual, Space::face>>(fieldRegistry,
                                                                                   "J", deRham);
        auto rho = Gempic::Io::registered_form<DeRhamField<Grid::dual, Space::cell>>(fieldRegistry,
                                                                                     "rho", deRham);
        auto phi = Gempic::Io::registered_form<DeRhamField<Grid::primal, Space::node>>(
            fieldRegistry, "phi", deRham);

        // Initialize needed propagators/solvers
        auto poisson{Gempic::FieldSolvers::make_poisson_solver(deRham, infra)};
        TimeLoop::OperatorHamilton<vDim, degx, degy, degz> operatorHamilton;

        // Initialize particles
        Io::Parameters params("Particle");
        amrex::Vector<std::string> speciesNames;
        params.get("speciesNames", speciesNames);

        std::vector<std::shared_ptr<ParticleSpeciesLinVlasov<vDim, nData>>> particlesLinVlasov;
        std::vector<std::shared_ptr<ParticleSpecies<vDim, nData>>> particles;
        init_particles(particlesLinVlasov, particles, infra);

        // Domain volume and number of cells to normalize s0
        amrex::Real domainVolume = infra.geometry().ProbDomain().volume();
        int nCells = infra.box().length3d().product();

        {
            // Initialize diagnostics and write initial time step
            Io::Parameters params("TimeLoop");
            amrex::Real dt;
            params.get("dt", dt);
            int nSteps;
            params.get("nSteps", nSteps);
            auto diagnostics =
                Io::make_diagnostics<degx, degy, degz>(infra, deRham, fieldRegistry, particles);

            // Deposit initial charge and compute s0
            for (auto& particleSpecies : particlesLinVlasov)
            {
                amrex::Real charge = particleSpecies->get_charge();

                // Get species information for s0
                std::string specString = "Particle." + particleSpecies->get_name();
                Io::Parameters params(specString);
                int nPartPerCell;
                params.get("nPartPerCell", nPartPerCell);
                int nPart = nCells * nPartPerCell;

                int numGaussians = 1;
                params.get("numGaussians", numGaussians);
                GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE(
                    numGaussians == 1,
                    "Number of gaussians must be 1 for linearized Vlasov Model!");
                // Multiple Maxwellians not possible since for the evaluation of s0 it cannot be
                // reconstructed from which Maxwellian a particle was drawn
                amrex::Vector<amrex::Real> vThermal;
                amrex::Vector<amrex::Real> vMean;
                params.get("G0.vMean", vMean);
                params.get("G0.vThermal", vThermal);
                std::array<amrex::Real, vDim> vThermalGPU{vThermal[xDir], vThermal[yDir],
                                                          vThermal[zDir]};
                std::array<amrex::Real, vDim> vMeanGPU{vMean[xDir], vMean[yDir], vMean[zDir]};

                amrex::Real vThermalBackground = particleSpecies->get_v_thermal_background();

                for (auto& particleGrid : *particleSpecies)
                {
                    long const np = particleGrid.numParticles();
                    auto const ptd = particleGrid.GetParticleTile().getParticleTileData();
                    auto const ii = particleSpecies->get_data_indices();

                    amrex::Array4<amrex::Real> const& rhoarr = rho.m_data[particleGrid].array();
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{
                        infra.geometry().ProbLoArray()};

                    //compile functions on device
                    auto funcDensityBackground =
                        Utils::compile_function(particleSpecies->get_density_background());

                    amrex::ParallelFor(
                        np,
                        [=] AMREX_GPU_DEVICE(long pp)
                        {
                            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> pos{
                                AMREX_D_DECL(ptd.rdata(ii.m_iposx)[pp], ptd.rdata(ii.m_iposy)[pp],
                                             ptd.rdata(ii.m_iposz)[pp])};
                            std::array<amrex::Real, vDim> vel{ptd.rdata(ii.m_ivelx)[pp],
                                                              ptd.rdata(ii.m_ively)[pp],
                                                              ptd.rdata(ii.m_ivelz)[pp]};

                            ptd.rdata(ii.m_isqrtf0)[pp] = eval_sqrt_maxwellian(
                                vel,
                                funcDensityBackground(AMREX_D_DECL(ptd.rdata(ii.m_iposx)[pp],
                                                                   ptd.rdata(ii.m_iposy)[pp],
                                                                   ptd.rdata(ii.m_iposz)[pp]),
                                                      0.),
                                vThermalBackground);

                            // Rescale weights to remove f0 from energy computation
                            ptd.rdata(ii.m_iweight)[pp] /= ptd.rdata(ii.m_isqrtf0)[pp];

                            SplineBase<degx, degy, degz> spline(pos, plo,
                                                                infra.inv_cell_size_array());

                            deposit_rho(
                                rhoarr, spline,
                                ptd.rdata(ii.m_isqrtf0)[pp] * charge * ptd.rdata(ii.m_iweight)[pp]);

                            // Compute s0 multiplied by number of particles as needed for electric
                            // field update and particle energy
                            ptd.rdata(ii.m_is0)[pp] =
                                nPart / domainVolume *
                                eval_maxwellian(vel, 1.0, vThermalGPU, vMeanGPU);
                        });
                }
            }

            rho.post_particle_loop_sync();
            // Add background charge (needs to be done after post_particle_loop_sync)
            std::ignore =
                FieldSolvers::get_and_apply_neutralizing_background(rho, infra, parameters);

            poisson->solve(phi, rho);
            grad(E, phi);
            E *= -1.0;
            hodge(D, E);

            // Write initial time step
            amrex::Real simTime{0.0};
            diagnostics.compute_and_write_to_file(0, simTime);

            for (int tStep = 0; tStep < nSteps; tStep++)
            {
                // Hb
                operatorHamilton.apply_h_b(D, deRham, B, H, 0.5 * dt);
                // He,field (also computes E from D, needed in He,particle)
                operatorHamilton.apply_h_e_field(B, E, deRham, D, 0.5 * dt);

                // Deposit particles in J and push particles: H_p = H_p1 + H_p2 + H_p3
                for (int comp = 0; comp < 3; ++comp)
                {
                    J.m_data[comp].setVal(0.0);
                }

                for (auto& particleSpecies : particlesLinVlasov)
                {
                    amrex::Real charge = particleSpecies->get_charge();
                    amrex::Real chargeMass = charge / particleSpecies->get_mass();
                    amrex::Real vThermalBackground = particleSpecies->get_v_thermal_background();
                    amrex::Real vThermalBackground2 = vThermalBackground * vThermalBackground;

                    for (auto& pti : *particleSpecies)
                    {
                        long const np = pti.numParticles();
                        auto const ptd = pti.GetParticleTile().getParticleTileData();
                        auto const ii = particleSpecies->get_data_indices();

                        amrex::GpuArray<amrex::Array4<amrex::Real>, 3> eA;
                        for (int cc = 0; cc < 3; cc++)
                        {
                            eA[cc] = (E.m_data[cc])[pti].array();
                        }

                        amrex::GpuArray<amrex::Array4<amrex::Real>, 3> bA;
                        for (int cc = 0; cc < 3; cc++)
                        {
                            bA[cc] = (bBackground.m_data[cc])[pti].array();
                        }

                        amrex::GpuArray<amrex::Array4<amrex::Real>, 3> jA;
                        for (int cc = 0; cc < 3; cc++)
                        {
                            jA[cc] = (J.m_data[cc])[pti].array();
                        }
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{
                            infra.geometry().ProbLoArray()};

                        amrex::ParallelFor(
                            np,
                            [=] AMREX_GPU_DEVICE(long pp)
                            {
                                // Local arrays for particle position and velocities
                                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> pos{AMREX_D_DECL(
                                    ptd.rdata(ii.m_iposx)[pp], ptd.rdata(ii.m_iposy)[pp],
                                    ptd.rdata(ii.m_iposz)[pp])};
                                amrex::GpuArray<amrex::Real, vDim> vel{ptd.rdata(ii.m_ivelx)[pp],
                                                                       ptd.rdata(ii.m_ively)[pp],
                                                                       ptd.rdata(ii.m_ivelz)[pp]};

                                ParticleMeshCoupling::SplineWithPrimitive<degx, degy, degz> spline(
                                    pos, plo, infra.inv_cell_size_array());

                                // He,particle
                                amrex::GpuArray<amrex::Real, 3> efield =
                                    spline.template eval_spline_field<Field::PrimalOneForm>(eA);
                                ptd.rdata(ii.m_iweight)[pp] +=
                                    0.5 * dt * chargeMass / vThermalBackground2 *
                                    ptd.rdata(ii.m_isqrtf0)[pp] *
                                    (efield[xDir] * vel[xDir] + efield[yDir] * vel[yDir] +
                                     efield[zDir] * vel[zDir]) /
                                    ptd.rdata(ii.m_is0)[pp];

                                // Push particle and integrate current
                                operatorHamilton.apply_h_p_exact(
                                    pos, vel, infra, spline, infra.cell_size_array(), jA, bA,
                                    chargeMass,
                                    ptd.rdata(ii.m_isqrtf0)[pp] * charge *
                                        ptd.rdata(ii.m_iweight)[pp],
                                    dt);

                                // Write position and velocities
                                AMREX_D_EXPR(ptd.rdata(ii.m_iposx)[pp] = pos[xDir],
                                             ptd.rdata(ii.m_iposy)[pp] = pos[yDir],
                                             ptd.rdata(ii.m_iposz)[pp] = pos[zDir]);
                                ptd.rdata(ii.m_ivelx)[pp] = vel[xDir];
                                ptd.rdata(ii.m_ively)[pp] = vel[yDir];
                                ptd.rdata(ii.m_ivelz)[pp] = vel[zDir];
                            });
                    }
                    particleSpecies->Redistribute();
                }

                J.post_particle_loop_sync();
                D -= J;
                hodge(E, D);

                // He,particle
                for (auto& particleSpecies : particlesLinVlasov)
                {
                    amrex::Real charge = particleSpecies->get_charge();
                    amrex::Real chargeMass = charge / particleSpecies->get_mass();
                    amrex::Real vThermalBackground = particleSpecies->get_v_thermal_background();
                    amrex::Real vThermalBackground2 = vThermalBackground * vThermalBackground;

                    for (auto& pti : *particleSpecies)
                    {
                        long const np = pti.numParticles();
                        auto const ptd = pti.GetParticleTile().getParticleTileData();
                        auto const ii = particleSpecies->get_data_indices();

                        amrex::GpuArray<amrex::Array4<amrex::Real>, 3> eA;
                        for (int cc = 0; cc < 3; cc++)
                        {
                            eA[cc] = (E.m_data[cc])[pti].array();
                        }
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{
                            infra.geometry().ProbLoArray()};

                        //compile functions on device
                        auto funcDensityBackground =
                            Utils::compile_function(particleSpecies->get_density_background());

                        amrex::ParallelFor(
                            np,
                            [=] AMREX_GPU_DEVICE(long pp)
                            {
                                // Local arrays for particle position and velocities
                                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> pos{AMREX_D_DECL(
                                    ptd.rdata(ii.m_iposx)[pp], ptd.rdata(ii.m_iposy)[pp],
                                    ptd.rdata(ii.m_iposz)[pp])};
                                std::array<amrex::Real, vDim> vel{ptd.rdata(ii.m_ivelx)[pp],
                                                                  ptd.rdata(ii.m_ively)[pp],
                                                                  ptd.rdata(ii.m_ivelz)[pp]};

                                ptd.rdata(ii.m_isqrtf0)[pp] = eval_sqrt_maxwellian(
                                    vel,
                                    funcDensityBackground(
                                        AMREX_D_DECL(pos[xDir], pos[yDir], pos[zDir]), 0.),
                                    vThermalBackground);

                                ParticleMeshCoupling::SplineBase<degx, degy, degz> spline(
                                    pos, plo, infra.inv_cell_size_array());

                                amrex::GpuArray<amrex::Real, 3> efield =
                                    spline.template eval_spline_field<Field::PrimalOneForm>(eA);

                                ptd.rdata(ii.m_iweight)[pp] +=
                                    0.5 * dt * chargeMass / vThermalBackground2 *
                                    ptd.rdata(ii.m_isqrtf0)[pp] *
                                    (efield[xDir] * vel[xDir] + efield[yDir] * vel[yDir] +
                                     efield[zDir] * vel[zDir]) /
                                    ptd.rdata(ii.m_is0)[pp];
                            });
                    }
                    // no redistribute since particle positions do not change
                }

                //He,field
                operatorHamilton.apply_h_e_field(B, E, deRham, D, 0.5 * dt);
                //Hb
                operatorHamilton.apply_h_b(D, deRham, B, H, 0.5 * dt);

                // Update primal electric field for energy computation
                hodge(E, D);

                //write outputs
                simTime = dt * (tStep + 1);
                diagnostics.compute_and_write_to_file(tStep + 1, simTime);

                if (tStep % 10 == 0)
                {
                    std::cout << "Time Step: " << tStep + 1 << std::endl;
                }
            }
        } // end of "time loop" scope
    }
    amrex::Finalize();
}
