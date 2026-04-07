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
#include "GEMPIC_Filter.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_Particle.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_PoissonSolver.H"
#include "GEMPIC_RungeKutta.H"
#include "GEMPIC_Sampler.H"

using namespace Gempic;
using namespace Forms;
using namespace Particle;
using namespace ParticleMeshCoupling;
using namespace FieldSolvers;
using namespace TimeLoop;

void initialize_tensor (amrex::MFIter& mfi,
                        amrex::MultiFab& mf,
                        amrex::GpuArray<amrex::Real, 3> val)
{
    amrex::Box const& bx = mfi.tilebox();
    amrex::Array4<amrex::Real> tensorArray = mf[mfi].array();
    ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    tensorArray(i, j, k, xDir) = val[0];
                    tensorArray(i, j, k, yDir) = val[1];
                    tensorArray(i, j, k, zDir) = val[2];
                });
}

void get_d_from_e_drift_kinetic (DeRhamField<Grid::dual, Space::face>& D,
                                 DeRhamField<Grid::primal, Space::edge>& E,
                                 std::shared_ptr<Gempic::Forms::FDDeRhamComplex> deRham,
                                 amrex::Real vAlfven,
                                 amrex::Real /*somega*/)
{
    BL_PROFILE("Gempic::Forms::getDfromE_DriftKinetic()");
    // Compute the dielectric tensor to convert electric field E to D.
    // This tensor is used only once during the calculation of the initial D.
    DeRhamField<Grid::dual, Space::face> tensor(deRham, 3);
    amrex::GpuArray<amrex::Real, 3> val;

    // initialize tensor
    for (amrex::MFIter mfi(tensor.m_data[xDir], true); mfi.isValid(); ++mfi)
    {
        val = {1 + 1 / vAlfven / vAlfven, 0, 0};
        initialize_tensor(mfi, tensor.m_data[xDir], val);
    }
    for (amrex::MFIter mfi(tensor.m_data[yDir], true); mfi.isValid(); ++mfi)
    {
        val = {0, 1 + 1 / vAlfven / vAlfven, 0};
        initialize_tensor(mfi, tensor.m_data[yDir], val);
    }
    for (amrex::MFIter mfi(tensor.m_data[zDir], true); mfi.isValid(); ++mfi)
    {
        val = {0, 0, 1};
        initialize_tensor(mfi, tensor.m_data[zDir], val);
    }
    // Compute D
    hodge_dk(D, E, tensor, deRham->scaling_eto_d()); //get D from E
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    BL_PROFILE_VAR("main()", pmain);
    // Track git version
    if (git::IsPopulated())
    {
        if (git::AnyUncommittedChanges())
        {
            amrex::Warning("Warning: There were uncommitted changes at build-time");
        }
        amrex::Print() << "GEMPIC commit " << git::CommitSHA1() << " (" << git::Branch() << ")\n"
                       << "describe " << git::Describe() << "\n";
    }
    else
    {
        amrex::Warning("Warning: Failed to get the current git state. Is this a git repo?");
    }
    //
    // Tell the parameters class to print output
    Io::Parameters::set_print_output();

    // Linear splines is ok, and lower dimension Hodge is good enough
    constexpr int vdim{3};
    constexpr int ndata{1 + AMREX_SPACEDIM + vdim}; // Weight + auxilary variables in phase space
    // Spline degrees
    constexpr int degx{3};
    constexpr int degy{3};
    constexpr int degz{3};
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
        // Initialize simulation
        Io::Parameters paramsSim("Sim");
        std::string simType{"UNDEFINED"};
        paramsSim.get_or_set("Type", simType);

        auto poisson{make_poisson_solver(deRham, infra)};

        if (simType == "FullyKinetic")
        {
            amrex::Print() << "****************************\n";
            amrex::Print() << "* FullyKinetic simulation  *\n";
            amrex::Print() << "****************************\n";
            // paramsSim.get_or_set("Te", te);
        }
        else if (simType == "DriftKinetic")
        {
            amrex::Print() << "*****************************\n";
            amrex::Print() << "* DriftKinetic simulation   *\n";
            amrex::Print() << "*****************************\n";
        }
        else if (simType == "DeFi")
        {
            amrex::Print() << "*****************************\n";
            amrex::Print() << "* DriftKinetic electrons    *\n";
            amrex::Print() << "* Fully Kinetic ions        *\n";
            amrex::Print() << "*****************************\n";
        }
        else
        {
            GEMPIC_ERROR("Simulation type " + simType + " is not implemented");
        }

        // initialize LSRKsolver
        RungeKutta rkSolver(infra, deRham);

        amrex::Real scalingOmega;
        scalingOmega = deRham->get_s_omega();

        // initialize fields
        auto [parseB, funcB] = Utils::parse_functions<3>({"Bx", "By", "Bz"});
        auto [parseE, funcE] = Utils::parse_functions<3>({"Ex", "Ey", "Ez"});

        auto B = Gempic::Io::registered_form<DeRhamField<Grid::primal, Space::face>>(
            fieldRegistry, "B", deRham, funcB);
        auto H = Gempic::Io::registered_form<DeRhamField<Grid::dual, Space::edge>>(
            fieldRegistry, "H", deRham, funcB);
        auto E = Gempic::Io::registered_form<DeRhamField<Grid::primal, Space::edge>>(fieldRegistry,
                                                                                     "E", deRham);
        DeRhamField<Grid::primal, Space::edge> einit(deRham, funcE);
        auto D = Gempic::Io::registered_form<DeRhamField<Grid::dual, Space::face>>(
            fieldRegistry, "D", deRham, funcE);
        auto J = Gempic::Io::registered_form<DeRhamField<Grid::dual, Space::face>>(fieldRegistry,
                                                                                   "J", deRham);
        auto rho = Gempic::Io::registered_form<DeRhamField<Grid::dual, Space::cell>>(fieldRegistry,
                                                                                     "rho", deRham);
        auto divD = Gempic::Io::registered_form<DeRhamField<Grid::dual, Space::cell>>(
            fieldRegistry, "divD", deRham);
        auto divB = Gempic::Io::registered_form<DeRhamField<Grid::primal, Space::cell>>(
            fieldRegistry, "divB", deRham);
        DeRhamField<Grid::dual, Space::cell> rhoFiltered(deRham);
        auto phi = Gempic::Io::registered_form<DeRhamField<Grid::primal, Space::node>>(
            fieldRegistry, "phi", deRham);
        // tensor including polarization for DK
        DeRhamField<Grid::primal, Space::edge> tensor(deRham, 3);

        amrex::Real vAlfven;

        // Initialize particles
        std::vector<std::shared_ptr<ParticleSpecies<vdim, ndata>>> particles;
        init_particles(particles, infra);

        // Initializing filter
        std::unique_ptr<Filter::Filter> filter = Gempic::Filter::make_filter(infra);

        { // "Time Loop" scope. Should be a separate function

            // Initialize full diagnostics and write initial time step
            Io::Parameters params("TimeLoop");
            amrex::Real dt;
            params.get("dt", dt);
            int nSteps;
            params.get("nSteps", nSteps);

            // Initialize diagnostics and write initial time step
            auto diagnostics =
                Io::make_diagnostics<degx, degy, degz>(infra, deRham, fieldRegistry, particles);

            if (simType == "DriftKinetic" || simType == "DeFi")
            {
                //Calculate vAlfven electron
                vAlfven = 1.0 / scalingOmega;
                if (simType == "DriftKinetic")
                {
                    for (auto& particleSpecies : particles)
                    {
                        if (particleSpecies->get_name() == "ions")
                        {
                            vAlfven /= sqrt(particleSpecies->get_mass() + 1.0);
                        }
                    }
                }
                // Compute inverse of dielectric tensor going from D to E
                amrex::GpuArray<amrex::Real, 3> val;

                // initialize tensor
                for (amrex::MFIter mfi(tensor.m_data[xDir], true); mfi.isValid(); ++mfi)
                {
                    val = {vAlfven * vAlfven / (1 + vAlfven * vAlfven), 0, 0};
                    initialize_tensor(mfi, tensor.m_data[xDir], val);
                }
                for (amrex::MFIter mfi(tensor.m_data[yDir], true); mfi.isValid(); ++mfi)
                {
                    val = {0, vAlfven * vAlfven / (1 + vAlfven * vAlfven), 0};
                    initialize_tensor(mfi, tensor.m_data[yDir], val);
                }
                for (amrex::MFIter mfi(tensor.m_data[zDir], true); mfi.isValid(); ++mfi)
                {
                    val = {0, 0, 1};
                    initialize_tensor(mfi, tensor.m_data[zDir], val);
                }
            }

            // Deposit initial charge and current densities
            rho.m_data.setVal(0.0);
            for (int cc = 0; cc < 3; cc++)
            {
                J.m_data[cc].setVal(0.0);
            }
            for (auto& particleSpecies : particles)
            {
                amrex::Real charge = particleSpecies->get_charge();

                for (auto& particleGrid : *particleSpecies)
                {
                    long const np = particleGrid.numParticles();
                    auto const ptd = particleGrid.GetParticleTile().getParticleTileData();
                    auto const ii = particleSpecies->get_data_indices();

                    amrex::Array4<amrex::Real> const& rhoarr = rho.m_data[particleGrid].array();
                    amrex::GpuArray<amrex::Array4<amrex::Real>, 3> jarr;
                    for (int cc = 0; cc < 3; cc++)
                    {
                        jarr[cc] = (J.m_data[cc])[particleGrid].array();
                    }

                    auto plo{infra.geometry().ProbLoArray()};

                    if (simType == "FullyKinetic")
                    {
                        amrex::ParallelFor(
                            np,
                            [=] AMREX_GPU_DEVICE(long pp)
                            {
                                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle{
                                    AMREX_D_DECL(ptd.rdata(ii.m_iposx)[pp],
                                                 ptd.rdata(ii.m_iposy)[pp],
                                                 ptd.rdata(ii.m_iposz)[pp])};
                                SplineBase<degx, degy, degz> spline(positionParticle, plo,
                                                                    infra.inv_cell_size_array());
                                deposit_rho(rhoarr, spline, charge * ptd.rdata(ii.m_iweight)[pp]);
                                amrex::GpuArray<amrex::Real, 3> V{ptd.rdata(ii.m_ivelx)[pp],
                                                                  ptd.rdata(ii.m_ively)[pp],
                                                                  ptd.rdata(ii.m_ivelz)[pp]};

                                deposit_twoform(jarr, spline, V,
                                                charge * ptd.rdata(ii.m_iweight)[pp]);
                            });
                    }
                    else if (simType == "DriftKinetic")
                    {
                        amrex::ParallelFor(
                            np,
                            [=] AMREX_GPU_DEVICE(long pp)
                            {
                                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle{
                                    AMREX_D_DECL(ptd.rdata(ii.m_iposx)[pp],
                                                 ptd.rdata(ii.m_iposy)[pp],
                                                 ptd.rdata(ii.m_iposz)[pp])};
                                SplineBase<degx, degy, degz> spline(positionParticle, plo,
                                                                    infra.inv_cell_size_array());
                                deposit_rho(rhoarr, spline, charge * ptd.rdata(ii.m_iweight)[pp]);

                                // set Vx = Vy to 0, only Vz = Vpar is used
                                ptd.rdata(ii.m_ivelx)[pp] = 0;
                                ptd.rdata(ii.m_ively)[pp] = 0;

                                deposit_twoform(jarr, spline, {0.0, 0.0, ptd.rdata(ii.m_ivelz)[pp]},
                                                charge * ptd.rdata(ii.m_iweight)[pp]);
                            });
                    }
                    else if (simType == "DeFi")
                    {
                        if (particleSpecies->get_name() == "electrons")
                        {
                            amrex::ParallelFor(
                                np,
                                [=] AMREX_GPU_DEVICE(long pp)
                                {
                                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle{
                                        AMREX_D_DECL(ptd.rdata(ii.m_iposx)[pp],
                                                     ptd.rdata(ii.m_iposy)[pp],
                                                     ptd.rdata(ii.m_iposz)[pp])};
                                    SplineBase<degx, degy, degz> spline(
                                        positionParticle, plo, infra.inv_cell_size_array());
                                    deposit_rho(rhoarr, spline,
                                                charge * ptd.rdata(ii.m_iweight)[pp]);

                                    // set Vx = Vy to 0, only Vz = Vpar is used
                                    ptd.rdata(ii.m_ivelx)[pp] = 0;
                                    ptd.rdata(ii.m_ively)[pp] = 0;

                                    deposit_twoform(jarr, spline,
                                                    {0.0, 0.0, ptd.rdata(ii.m_ivelz)[pp]},
                                                    charge * ptd.rdata(ii.m_iweight)[pp]);
                                });
                        }
                        else
                        {
                            amrex::ParallelFor(
                                np,
                                [=] AMREX_GPU_DEVICE(long pp)
                                {
                                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle{
                                        AMREX_D_DECL(ptd.rdata(ii.m_iposx)[pp],
                                                     ptd.rdata(ii.m_iposy)[pp],
                                                     ptd.rdata(ii.m_iposz)[pp])};
                                    SplineBase<degx, degy, degz> spline(
                                        positionParticle, plo, infra.inv_cell_size_array());
                                    deposit_rho(rhoarr, spline,
                                                charge * ptd.rdata(ii.m_iweight)[pp]);
                                    amrex::GpuArray<amrex::Real, 3> V{ptd.rdata(ii.m_ivelx)[pp],
                                                                      ptd.rdata(ii.m_ively)[pp],
                                                                      ptd.rdata(ii.m_ivelz)[pp]};

                                    deposit_twoform(jarr, spline, V,
                                                    charge * ptd.rdata(ii.m_iweight)[pp]);
                                });
                        }
                    }
                }
            }

            rho.post_particle_loop_sync();
            J.post_particle_loop_sync();

            // Add background charge (needs to be done after post_particle_loop_sync)
            std::ignore = get_and_apply_neutralizing_background(rho, infra, parameters);

            // Apply filter and compute phi with filtered rho
            filter->apply(rhoFiltered, rho);
            poisson->solve(phi, rhoFiltered);
            grad(E, phi);
            E *= -1.0;
            E += einit; // add initial value of E (needs to be divergence free)

            // D is also needed to compute energy
            // get D from E
            if (simType == "FullyKinetic")
            {
                hodge(D, E); // get D from E
            }
            else if (simType == "DriftKinetic")
            {
                get_d_from_e_drift_kinetic(D, E, deRham, vAlfven, scalingOmega);
            }
            else if (simType == "DeFi")
            {
                get_d_from_e_drift_kinetic(D, E, deRham, vAlfven, scalingOmega);
            }

            // // Compute initial divB divD
            div(divD, D);

            // Write initial time step
            amrex::Real simTime{0.0};
            diagnostics.compute_and_write_to_file(0, simTime);

            for (int tStep = 0; tStep < nSteps; tStep++)
            {
                if (simType == "FullyKinetic")
                {
                    rkSolver.template lsrk_vlasov_maxwell<degx, degy, degz>(particles, E, D, B, H,
                                                                            J, dt);
                }
                else if (simType == "DriftKinetic")
                {
                    rkSolver.template lsrk_dk_vlasov_maxwell<degx, degy, degz>(particles, E, D, B,
                                                                               H, J, tensor, dt);
                }
                else if (simType == "DeFi")
                {
                    rkSolver.template lsrk_de_fi_vlasov_maxwell<degx, degy, degz>(
                        particles, E, D, B, H, J, tensor, dt);
                }

                // // Compute divB divD at each time step
                div(divD, D);

                // Write diagnostics
                simTime = dt * (tStep + 1);
                diagnostics.compute_and_write_to_file(tStep + 1, simTime);

                if (tStep % 10 == 0)
                {
                    amrex::Print() << "Time Step: " << tStep + 1 << std::endl;
                }
            }
        } // end of "time loop" scope
    }
    BL_PROFILE_VAR_STOP(pmain);
    amrex::Finalize();
}
