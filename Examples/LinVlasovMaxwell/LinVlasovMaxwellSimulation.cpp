#include <git.h>
#include <iostream>
#include <random>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include "GEMPIC_AmrexInit.H"
#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_Diagnostics.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_ParticleGroups.H"
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

int main (int argc, char *argv[])
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

        auto [parseBBackground, funcBBackground] =
            Utils::parse_functions<3>({"BBackgroundX", "BBackgroundY", "BBackgroundZ"});

        DeRhamField<Grid::primal, Space::edge> E(deRham, "E");
        DeRhamField<Grid::dual, Space::face> D(deRham, "D");
        DeRhamField<Grid::primal, Space::face> B(deRham, "B");
        DeRhamField<Grid::primal, Space::face> bBackground(deRham, funcBBackground);
        DeRhamField<Grid::dual, Space::edge> H(deRham, "H");
        DeRhamField<Grid::dual, Space::face> J(deRham, "J");
        DeRhamField<Grid::dual, Space::cell> rho(deRham, "rho");
        DeRhamField<Grid::primal, Space::node> phi(deRham, "phi");

        // Initialize needed propagators/solvers
        Gempic::FieldSolvers::PoissonSolver poisson(deRham, infra);
        TimeLoop::OperatorHamilton<vDim, degx, degy, degz> operatorHamilton;

        // Initialize particle groups
        Io::Parameters params("Particle");
        amrex::Vector<std::string> speciesNames;
        params.get("speciesNames", speciesNames);

        std::vector<std::shared_ptr<ParticleGroupsLinVlasov<vDim, nData>>> partGrLinVlasov;
        std::vector<std::shared_ptr<ParticleGroups<vDim, nData>>> partGr;
        init_particles(partGrLinVlasov, partGr, infra);

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
            auto diagnostics = Io::make_diagnostics<degx, degy, degz>(infra, deRham, partGr);

            // Deposit initial charge and compute s0
            for (auto &particleSpecies : partGrLinVlasov)
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
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                    numGaussians == 1,
                    "Number of gaussians must be 1 for linearized Vlasov Model!");
                // Multiple Maxwellians not possible since for the evaluation of s0 it cannot be
                // reconstructed from which Maxwellian a particle was drawn
                amrex::Vector<amrex::Real> vThermal;
                amrex::Vector<amrex::Real> vMean;
                params.get("G0.vMean", vMean);
                params.get("G0.vThermal", vThermal);
                amrex::GpuArray<amrex::Real, vDim> vThermalGPU{vThermal[xDir], vThermal[yDir],
                                                               vThermal[zDir]};
                amrex::GpuArray<amrex::Real, vDim> vMeanGPU{vMean[xDir], vMean[yDir], vMean[zDir]};

                amrex::Real vThermalBackground = particleSpecies->get_v_thermal_background();

                for (amrex::ParIter<0, 0, vDim + nData, 0> pti(*particleSpecies, 0); pti.isValid();
                     ++pti)
                {
                    const long np = pti.numParticles();
                    auto *const particles = pti.GetArrayOfStructs()().data();
                    auto *const velx = pti.GetStructOfArrays().GetRealData(0).data();
                    auto *const vely = pti.GetStructOfArrays().GetRealData(1).data();
                    auto *const velz = pti.GetStructOfArrays().GetRealData(2).data();
                    auto *const weight = pti.GetStructOfArrays().GetRealData(3).data();
                    auto *const s0 = pti.GetStructOfArrays().GetRealData(4).data();
                    auto *const sqrtf0 = pti.GetStructOfArrays().GetRealData(5).data();

                    amrex::Array4<amrex::Real> const &rhoarr = rho.m_data[pti].array();
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{
                        infra.geometry().ProbLoArray()};

                    //compile functions on device
                    auto funcDensityBackground =
                        Utils::compile_function(particleSpecies->get_density_background());

                    amrex::ParallelFor(
                        np,
                        [=] AMREX_GPU_DEVICE(long pp)
                        {
                            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> pos;
                            amrex::GpuArray<amrex::Real, vDim> vel{velx[pp], vely[pp], velz[pp]};

                            for (unsigned int d = 0; d < AMREX_SPACEDIM; ++d)
                            {
                                pos[d] = particles[pp].pos(d);
                            }

                            sqrtf0[pp] = eval_sqrt_maxwellian(
                                vel,
                                funcDensityBackground(AMREX_D_DECL(pos[xDir], pos[yDir], pos[zDir]),
                                                      0.),
                                vThermalBackground);

                            // Rescale weights to remove f0 from energy computation
                            weight[pp] /= sqrtf0[pp];

                            SplineBase<degx, degy, degz> spline(pos, plo,
                                                                infra.inv_cell_size_array());

                            deposit_rho(rhoarr, spline, sqrtf0[pp] * charge * weight[pp]);

                            // Compute s0 multiplied by number of particles as needed for electric
                            // field update and particle energy
                            s0[pp] = nPart / domainVolume *
                                     eval_maxwellian(vel, 1.0, vThermalGPU, vMeanGPU);
                        });
                }
            }

            rho.post_particle_loop_sync();

            poisson.solve_amrex(phi, rho);
            deRham->grad(E, phi);
            E *= -1.0;
            deRham->hodge(D, E);

            // Write initial time step
            amrex::Real simTime{0.0};
            diagnostics.compute_and_write_to_file(0, simTime);

            for (int tStep = 0; tStep < nSteps; tStep++)
            {
                // Hb
                operatorHamilton.apply_h_b(D, deRham, B, H, 0.5 * dt);
                // He,field (also computes E from D, needed in He,particle)
                operatorHamilton.apply_h_e_field(B, deRham, E, D, 0.5 * dt);

                // Deposit particles in J and push particles: H_p = H_p1 + H_p2 + H_p3
                for (int comp = 0; comp < 3; ++comp)
                {
                    (J.m_data[comp]).setVal(0.0, 0);
                }
                J.fill_boundary();

                for (auto &particleSpecies : partGrLinVlasov)
                {
                    amrex::Real charge = particleSpecies->get_charge();
                    amrex::Real chargeMass = charge / particleSpecies->get_mass();
                    amrex::Real vThermalBackground = particleSpecies->get_v_thermal_background();
                    amrex::Real vThermalBackground2 = vThermalBackground * vThermalBackground;

                    for (amrex::ParIter<0, 0, vDim + nData, 0> pti(*particleSpecies, 0);
                         pti.isValid(); ++pti)
                    {
                        const long np = pti.numParticles();
                        const auto &particles = pti.GetArrayOfStructs()().data();
                        auto *const velx = pti.GetStructOfArrays().GetRealData(0).data();
                        auto *const vely = pti.GetStructOfArrays().GetRealData(1).data();
                        auto *const velz = pti.GetStructOfArrays().GetRealData(2).data();
                        auto *const weight = pti.GetStructOfArrays().GetRealData(3).data();
                        auto *const s0 = pti.GetStructOfArrays().GetRealData(4).data();
                        auto *const sqrtf0 = pti.GetStructOfArrays().GetRealData(5).data();

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
                                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> pos;
                                amrex::GpuArray<amrex::Real, vDim> vel{velx[pp], vely[pp],
                                                                       velz[pp]};

                                for (unsigned int d = 0; d < AMREX_SPACEDIM; ++d)
                                {
                                    pos[d] = particles[pp].pos(d);
                                }

                                ParticleMeshCoupling::SplineWithPrimitive<degx, degy, degz> spline(
                                    pos, plo, infra.inv_cell_size_array());

                                // He,particle
                                amrex::GpuArray<amrex::Real, 3> efield =
                                    spline.template eval_spline_field<Field::PrimalOneForm>(eA);
                                weight[pp] += 0.5 * dt * chargeMass / vThermalBackground2 *
                                              sqrtf0[pp] *
                                              (efield[xDir] * vel[xDir] + efield[yDir] * vel[yDir] +
                                               efield[zDir] * vel[zDir]) /
                                              s0[pp];

                                // Push particle and integrate current
                                operatorHamilton.apply_h_p(
                                    pos, vel, infra, spline, infra.cell_size_array(), jA, bA,
                                    chargeMass, sqrtf0[pp] * charge * weight[pp], dt);

                                // Write position and velocities
                                for (unsigned int d = 0; d < AMREX_SPACEDIM; ++d)
                                {
                                    particles[pp].pos(d) = pos[d];
                                }
                                velx[pp] = vel[xDir];
                                vely[pp] = vel[yDir];
                                velz[pp] = vel[zDir];
                            });
                    }
                    particleSpecies->Redistribute();
                }

                J.post_particle_loop_sync();
                D -= J;
                deRham->hodge(E, D);

                // He,particle
                for (auto &particleSpecies : partGrLinVlasov)
                {
                    amrex::Real charge = particleSpecies->get_charge();
                    amrex::Real chargeMass = charge / particleSpecies->get_mass();
                    amrex::Real vThermalBackground = particleSpecies->get_v_thermal_background();
                    amrex::Real vThermalBackground2 = vThermalBackground * vThermalBackground;

                    for (amrex::ParIter<0, 0, vDim + nData, 0> pti(*particleSpecies, 0);
                         pti.isValid(); ++pti)
                    {
                        const long np = pti.numParticles();
                        const auto &particles = pti.GetArrayOfStructs()().data();
                        auto *const velx = pti.GetStructOfArrays().GetRealData(0).data();
                        auto *const vely = pti.GetStructOfArrays().GetRealData(1).data();
                        auto *const velz = pti.GetStructOfArrays().GetRealData(2).data();
                        auto *const weight = pti.GetStructOfArrays().GetRealData(3).data();
                        auto *const s0 = pti.GetStructOfArrays().GetRealData(4).data();
                        auto *const sqrtf0 = pti.GetStructOfArrays().GetRealData(5).data();

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
                                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> pos;
                                amrex::GpuArray<amrex::Real, vDim> vel{velx[pp], vely[pp],
                                                                       velz[pp]};

                                for (unsigned int d = 0; d < AMREX_SPACEDIM; ++d)
                                {
                                    pos[d] = particles[pp].pos(d);
                                }

                                sqrtf0[pp] = eval_sqrt_maxwellian(
                                    vel,
                                    funcDensityBackground(
                                        AMREX_D_DECL(pos[xDir], pos[yDir], pos[zDir]), 0.),
                                    vThermalBackground);

                                ParticleMeshCoupling::SplineBase<degx, degy, degz> spline(
                                    pos, plo, infra.inv_cell_size_array());

                                amrex::GpuArray<amrex::Real, 3> efield =
                                    spline.template eval_spline_field<Field::PrimalOneForm>(eA);

                                weight[pp] += 0.5 * dt * chargeMass / vThermalBackground2 *
                                              sqrtf0[pp] *
                                              (efield[xDir] * vel[xDir] + efield[yDir] * vel[yDir] +
                                               efield[zDir] * vel[zDir]) /
                                              s0[pp];
                            });
                    }
                    // no redistribute since particle positions do not change
                }

                //He,field
                operatorHamilton.apply_h_e_field(B, deRham, E, D, 0.5 * dt);
                //Hb
                operatorHamilton.apply_h_b(D, deRham, B, H, 0.5 * dt);

                // Update primal electric field for energy computation
                deRham->hodge(E, D);

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
