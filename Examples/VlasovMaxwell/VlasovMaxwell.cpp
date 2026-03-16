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
#include "GEMPIC_SplineClass.H"
#include "GEMPIC_Splitting.H"

using namespace Gempic;
using namespace Forms;
using namespace Particle;
using namespace ParticleMeshCoupling;
using namespace FieldSolvers;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    constexpr int vdim{3};

    // Node spline degrees (smoothing spline degree is one less in each direction)
    constexpr int degx{1};
    constexpr int degy{1};
    constexpr int degz{1};
    constexpr int maxSplineDegree{std::max(std::max(degx, degy), degz)};

    constexpr int hodgeDegree{2};

    {
        BL_PROFILE("VlasovMaxwellMain()");
        Io::Parameters parameters{};
        // Initialize computational_domain
        ComputationalDomain infra;

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, maxSplineDegree,
                                                        HodgeScheme::FDHodge);
        Gempic::Io::FieldRegistry fieldRegistry;
        // "HamiltonianSplittingExact", "HamiltonianSplittingNested" or
        // "HamiltonianSplittingNestedRelativistic"
        std::string simType{"HamiltonianSplittingExact"};
        parameters.get_or_set("simType", simType);

        if (simType == "HamiltonianSplittingExact")
        {
            amrex::Print() << "****************************************\n";
            amrex::Print() << "* HamiltonianSplittingExact simulation *\n";
            amrex::Print() << "****************************************\n";
        }
        else if (simType == "HamiltonianSplittingNested")
        {
            amrex::Print() << "*****************************************\n";
            amrex::Print() << "* HamiltonianSplittingNested simulation *\n";
            amrex::Print() << "*****************************************\n";
        }
        else if (simType == "HamiltonianSplittingNestedRelativistic")
        {
            amrex::Print() << "*****************************************************\n";
            amrex::Print() << "* HamiltonianSplittingNestedRelativistic simulation *\n";
            amrex::Print() << "*****************************************************\n";
        }
        else
        {
            GEMPIC_ERROR("Simulation type " + simType + " is not implemented");
        }

        auto [parseB, funcB] = Utils::parse_functions<3>({"Bx", "By", "Bz"});
        auto [parseE, funcE] = Utils::parse_functions<3>({"Ex", "Ey", "Ez"});
        auto E = Gempic::Io::registered_form<DeRhamField<Grid::primal, Space::edge>>(
            fieldRegistry, "E", deRham, funcE);
        DeRhamField<Grid::primal, Space::edge> eFiltered(deRham);
        auto D = Gempic::Io::registered_form<DeRhamField<Grid::dual, Space::face>>(
            fieldRegistry, "D", deRham, funcE);
        auto B = Gempic::Io::registered_form<DeRhamField<Grid::primal, Space::face>>(
            fieldRegistry, "B", deRham, funcB);
        auto H = Gempic::Io::registered_form<DeRhamField<Grid::dual, Space::edge>>(
            fieldRegistry, "H", deRham, funcB);
        auto J = Gempic::Io::registered_form<DeRhamField<Grid::dual, Space::face>>(fieldRegistry,
                                                                                   "J", deRham);
        DeRhamField<Grid::dual, Space::face> jFiltered(deRham); // Filtered J for diagnostics
        auto rho = Gempic::Io::registered_form<DeRhamField<Grid::dual, Space::cell>>(fieldRegistry,
                                                                                     "rho", deRham);
        DeRhamField<Grid::dual, Space::cell> rhoFiltered(deRham); // Filtered rho for diagnostics
        auto phi = Gempic::Io::registered_form<DeRhamField<Grid::primal, Space::node>>(
            fieldRegistry, "phi", deRham);
        DeRhamField<Grid::primal, Space::edge> ephi(deRham);
        auto divD = Gempic::Io::registered_form<DeRhamField<Grid::dual, Space::cell>>(
            fieldRegistry, "divD", deRham);

        // Speed of light in terms of the reference velocity
        amrex::Real c{1.0};
        parameters.get_or_set("sV", c);

        std::vector<std::shared_ptr<ParticleSpecies<vdim>>> particles;
        init_particles(particles, infra);

        auto poisson{make_poisson_solver(deRham, infra)};
        Gempic::TimeLoop::OperatorHamilton<vdim, degx, degy, degz> operatorHamilton;

        // Initializing filter
        std::unique_ptr<Filter::Filter> filter = Gempic::Filter::make_filter(infra);

        { //"Time Loop" scope

            Io::Parameters params("TimeLoop");
            amrex::Real dt;
            params.get("dt", dt);
            int nSteps;
            params.get("nSteps", nSteps);

            auto diagnostics =
                Io::make_diagnostics<degx, degy, degz>(infra, deRham, fieldRegistry, particles);

            // Deposit initial charge
            rho.m_data.setVal(0.0);
            deposit_particle_density<degx, degy, degx>(rho, particles, infra);

            // Add background charge (needs to be done after post_particle_loop_sync)
            amrex::Real rhoBackground =
                get_and_apply_neutralizing_background(rho, infra, parameters);

            // Apply filter and compute phi with filtered rho
            filter->apply(rhoFiltered, rho);

            // solve Poisson
            poisson->solve(phi, rhoFiltered);
            a_times_grad(ephi, phi, -1);
            E += ephi;
            hodge(D, E, deRham->scaling_eto_d());
            // compute div D for diagnostics
            div(divD, D);

            amrex::Real simTime{0.0};
            diagnostics.compute_and_write_to_file(0, simTime);

            for (int tStep = 0; tStep < nSteps; tStep++)
            {
                // Solve Ampere-Maxwell with no current
                // This step is equivalent to the apply_h_B step
                // in the Hamiltonian Splitting scheme
                hodge(H, B, deRham->scaling_bto_h());
                add_dt_curl(D, H, 0.5 * dt);

                // Solve the Faraday equation
                // This step is equivalent to the apply_h_e_field step
                // in the Hamiltonian Splitting scheme
                hodge(E, D, deRham->scaling_dto_e());
                add_dt_curl(B, E, -0.5 * dt);

                // Filtered E needed for pushing the particles (for symmetry with J filtering
                // needed for energy conservation)
                filter->apply(eFiltered, E);

                // Initialize rho and J to 0 for particle loop
                rho.m_data.setVal(0.0);
                for (int comp = 0; comp < vdim; ++comp)
                {
                    J.m_data[comp].setVal(0.0);
                }
                // Choose, which numerical method to use for the particle part of the splitting
                if (simType == "HamiltonianSplittingExact")
                {
                    operatorHamilton.template apply_h_p<operatorHamilton.HamiltonianSplittingExact>(
                        particles, J, eFiltered, B, infra, dt, c);
                }
                else if (simType == "HamiltonianSplittingNestedRelativistic")
                {
                    operatorHamilton.template apply_h_p<
                        operatorHamilton.HamiltonianSplittingNestedRelativistic>(
                        particles, J, eFiltered, B, infra, dt, c);
                }
                else if (simType == "HamiltonianSplittingNested")
                {
                    operatorHamilton
                        .template apply_h_p<operatorHamilton.HamiltonianSplittingNested>(
                            particles, J, eFiltered, B, infra, dt, c);
                }

                // Apply filter and compute D with filtered J
                filter->apply(jFiltered, J);
                // Update D (field part of H_p_i)
                D -= jFiltered;

                // Filtered E needed for pushing the particles
                hodge(E, D, deRham->scaling_dto_e());
                filter->apply(eFiltered, E);

                // Second particle loop for particle part from H_E
                for (auto const& particleSpecies : particles)
                {
                    amrex::Real charge = particleSpecies->get_charge();
                    amrex::Real chargeOverMass = charge / particleSpecies->get_mass();

                    for (auto& particleGrid : *particleSpecies)
                    {
                        long const np = particleGrid.numParticles();
                        auto const ptd = particleGrid.GetParticleTile().getParticleTileData();
                        auto const ii = particleSpecies->get_data_indices();

                        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;
                        amrex::Array4<amrex::Real> rhoarr;

                        rhoarr = rho.m_data[particleGrid].array();
                        for (int cc = 0; cc < vdim; cc++)
                        {
                            eA[cc] = (eFiltered.m_data[cc])[particleGrid].array();
                        }
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{
                            infra.geometry().ProbLoArray()};

                        // loop over particles: add contribution of old particle position to J and
                        // push
                        amrex::ParallelFor(
                            np,
                            [=] AMREX_GPU_DEVICE(long pp)
                            {
                                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> pos{AMREX_D_DECL(
                                    ptd.rdata(ii.m_iposx)[pp], ptd.rdata(ii.m_iposy)[pp],
                                    ptd.rdata(ii.m_iposz)[pp])};

                                amrex::GpuArray<amrex::Real, vdim> vel{ptd.rdata(ii.m_ivelx)[pp],
                                                                       ptd.rdata(ii.m_ively)[pp],
                                                                       ptd.rdata(ii.m_ivelz)[pp]};

                                SplineBase<degx, degy, degz> spline(pos, plo,
                                                                    infra.inv_cell_size_array());

                                operatorHamilton.apply_h_e_particle(vel, eA, spline, chargeOverMass,
                                                                    0.5 * dt);
                                ptd.rdata(ii.m_ivelx)[pp] = vel[xDir];
                                ptd.rdata(ii.m_ively)[pp] = vel[yDir];
                                ptd.rdata(ii.m_ivelz)[pp] = vel[zDir];
                                // compute rho for Gauss error diagnostics
                                // in principle this should be computed only
                                // if the Gauss error diagnostic is performed
                                deposit_rho(rhoarr, spline, charge * ptd.rdata(ii.m_iweight)[pp]);
                            });
                    }
                    particleSpecies->Redistribute();
                }
                // end treat particles
                rho.post_particle_loop_sync();
                // Add background charge (needs to be done after post_particle_loop_sync)
                rho += rhoBackground * infra.cell_volume();
                // Filtered rho used for diagnostics
                filter->apply(rhoFiltered, rho);

                // Solve the Faraday equation
                // This step is equivalent to the apply_h_e_field step
                // in the Hamiltonian Splitting scheme
                // The hodge does not have to be computed again
                // since it does not change from the last function
                // call
                add_dt_curl(B, E, -0.5 * dt);

                // Solve Ampere-Maxwell with no current
                // This step is equivalent to the apply_h_B step
                // in the Hamiltonian Splitting scheme
                hodge(H, B, deRham->scaling_bto_h());
                add_dt_curl(D, H, 0.5 * dt);

                // compute div D for diagnostics
                div(divD, D);
                simTime = dt * (tStep + 1);
                diagnostics.compute_and_write_to_file(tStep + 1, simTime);

                amrex::Print() << "finished time-step: " << tStep << std::endl;
            }
        }
    }
    amrex::Finalize();
}
