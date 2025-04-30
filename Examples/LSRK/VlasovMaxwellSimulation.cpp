#include <git.h>
#include <iostream>
#include <random>

#include <AMReX.H>
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
                                 amrex::Real somega)
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
    deRham->hodge_dk(D, E, tensor, deRham->scaling_eto_d()); //get D from E
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
            amrex::Warning("There were uncommitted changes at build-time.");
        }
        amrex::Print() << "GEMPIC commit " << git::CommitSHA1() << " (" << git::Branch() << ")\n"
                       << "describe " << git::Describe() << "\n";
    }
    else
    {
        amrex::Warning("Failed to get the current git state. Is this a git repo?");
    }
    //
    // Tell the parameters class to print output
    Io::Parameters::set_print_output();

    // Linear splines is ok, and lower dimension Hodge is good enough
    constexpr int vdim{3};
    constexpr int ndata{7}; // Weight + 6 auxilary variables
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
            amrex::AllPrint() << "Simulation type " << simType << " is not implemented\n";
            amrex::Abort();
        }

        // initialize LSRKsolver
        RungeKutta rkSolver(infra, deRham);

        amrex::Real scalingV;
        amrex::Real scalingOmega;
        scalingV = deRham->get_s_v();
        scalingOmega = deRham->get_s_omega();

        // initialize fields
        auto [parseB, funcB] = Utils::parse_functions<3>({"Bx", "By", "Bz"});
        auto [parseE, funcE] = Utils::parse_functions<3>({"Ex", "Ey", "Ez"});

        DeRhamField<Grid::primal, Space::face> B(deRham, funcB, "B");
        DeRhamField<Grid::dual, Space::edge> H(deRham, funcB, "H");
        DeRhamField<Grid::primal, Space::edge> E(deRham, "E");
        DeRhamField<Grid::primal, Space::edge> einit(deRham, funcE);
        DeRhamField<Grid::dual, Space::face> D(deRham, funcE, "D");
        DeRhamField<Grid::dual, Space::face> J(deRham, "J");
        DeRhamField<Grid::dual, Space::cell> rho(deRham, "rho");
        DeRhamField<Grid::dual, Space::cell> divD(deRham, "divD");
        DeRhamField<Grid::primal, Space::cell> divB(deRham, "divB");
        DeRhamField<Grid::dual, Space::cell> rhoFiltered(deRham);
        DeRhamField<Grid::primal, Space::node> phi(deRham, "phi");
        // tensor including polarization for DK
        DeRhamField<Grid::primal, Space::edge> tensor(deRham, 3);
        amrex::Real vAlfven;

        // Initialize particle groups
        std::vector<std::shared_ptr<ParticleGroups<vdim, ndata>>> particleGroup;
        init_particles(particleGroup, infra);

        // Initializing filter
        std::unique_ptr<Filter::Filter> biFilter = std::make_unique<Filter::BilinearFilter>();

        { // "Time Loop" scope. Should be a separate function

            // Initialize full diagnostics and write initial time step
            Io::Parameters params("TimeLoop");
            amrex::Real dt;
            params.get("dt", dt);
            int nSteps;
            params.get("nSteps", nSteps);

            // Initialize diagnostics and write initial time step
            auto diagnostics = Io::make_diagnostics<degx, degy, degz>(infra, deRham, particleGroup);

            if (simType == "DriftKinetic" || simType == "DeFi")
            {
                //Calculate vAlfven electron
                vAlfven = 1.0 / scalingOmega;
                if (simType == "DriftKinetic")
                {
                    for (auto& particleSpecies : particleGroup)
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
            for (auto& particleSpecies : particleGroup)
            {
                amrex::Real charge = particleSpecies->get_charge();

                for (auto& particleGrid : *particleSpecies)
                {
                    long const np = particleGrid.numParticles();
                    auto* const particles = particleGrid.GetArrayOfStructs()().data();
                    auto* const weight = particleGrid.GetStructOfArrays().GetRealData(vdim).data();
                    auto* const vx = particleGrid.GetStructOfArrays().GetRealData(0).data();
                    auto* const vy = particleGrid.GetStructOfArrays().GetRealData(1).data();
                    auto* const vz = particleGrid.GetStructOfArrays().GetRealData(2).data();

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
                                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle;
                                for (unsigned int d = 0; d < AMREX_SPACEDIM; ++d)
                                {
                                    positionParticle[d] = particles[pp].pos(d);
                                }
                                SplineBase<degx, degy, degz> spline(positionParticle, plo,
                                                                    infra.inv_cell_size_array());
                                deposit_rho(rhoarr, spline, charge * weight[pp]);
                                amrex::GpuArray<amrex::Real, 3> V{vx[pp], vy[pp], vz[pp]};

                                deposit_j(jarr, spline, V, charge * weight[pp]);
                            });
                    }
                    else if (simType == "DriftKinetic")
                    {
                        amrex::ParallelFor(
                            np,
                            [=] AMREX_GPU_DEVICE(long pp)
                            {
                                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle;
                                for (unsigned int d = 0; d < AMREX_SPACEDIM; ++d)
                                {
                                    positionParticle[d] = particles[pp].pos(d);
                                }
                                SplineBase<degx, degy, degz> spline(positionParticle, plo,
                                                                    infra.inv_cell_size_array());
                                deposit_rho(rhoarr, spline, charge * weight[pp]);

                                // set Vx = Vy to 0, only Vz = Vpar is used
                                vx[pp] = 0;
                                vy[pp] = 0;

                                deposit_j(jarr, spline, {0.0, 0.0, vz[pp]}, charge * weight[pp]);
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
                                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle;
                                    for (unsigned int d = 0; d < AMREX_SPACEDIM; ++d)
                                    {
                                        positionParticle[d] = particles[pp].pos(d);
                                    }
                                    SplineBase<degx, degy, degz> spline(
                                        positionParticle, plo, infra.inv_cell_size_array());
                                    deposit_rho(rhoarr, spline, charge * weight[pp]);

                                    // set Vx = Vy to 0, only Vz = Vpar is used
                                    vx[pp] = 0;
                                    vy[pp] = 0;

                                    deposit_j(jarr, spline, {0.0, 0.0, vz[pp]},
                                              charge * weight[pp]);
                                });
                        }
                        else
                        {
                            amrex::ParallelFor(
                                np,
                                [=] AMREX_GPU_DEVICE(long pp)
                                {
                                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle;
                                    for (unsigned int d = 0; d < AMREX_SPACEDIM; ++d)
                                    {
                                        positionParticle[d] = particles[pp].pos(d);
                                    }
                                    SplineBase<degx, degy, degz> spline(
                                        positionParticle, plo, infra.inv_cell_size_array());
                                    deposit_rho(rhoarr, spline, charge * weight[pp]);
                                    amrex::GpuArray<amrex::Real, 3> V{vx[pp], vy[pp], vz[pp]};

                                    deposit_j(jarr, spline, V, charge * weight[pp]);
                                });
                        }
                    }
                }
            }

            rho.post_particle_loop_sync();
            J.post_particle_loop_sync();

            // Apply filter and compute phi with filtered rho
            biFilter->apply_stencil(rhoFiltered, rho);
            poisson->solve(phi, rhoFiltered);
            deRham->grad(E, phi);
            E *= -1.0;
            E += einit; // add initial value of E (needs to be divergence free)

            // D is also needed to compute energy
            // get D from E
            if (simType == "FullyKinetic")
            {
                deRham->hodge(D, E); // get D from E
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
            deRham->div(divD, D);

            // Write initial time step
            amrex::Real simTime{0.0};
            diagnostics.compute_and_write_to_file(0, simTime);

            for (int tStep = 0; tStep < nSteps; tStep++)
            {
                if (simType == "FullyKinetic")
                {
                    rkSolver.template lsrk_vlasov_maxwell<degx, degy, degz>(particleGroup, E, D, B,
                                                                            H, J, dt);
                }
                else if (simType == "DriftKinetic")
                {
                    rkSolver.template lsrk_dk_vlasov_maxwell<degx, degy, degz>(particleGroup, E, D,
                                                                               B, H, J, tensor, dt);
                }
                else if (simType == "DeFi")
                {
                    rkSolver.template lsrk_de_fi_vlasov_maxwell<degx, degy, degz>(
                        particleGroup, E, D, B, H, J, tensor, dt);
                }

                // // Compute divB divD at each time step
                deRham->div(divD, D);

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
