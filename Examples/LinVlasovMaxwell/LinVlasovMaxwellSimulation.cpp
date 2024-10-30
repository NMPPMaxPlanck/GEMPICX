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
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_MultiFullDiagnostics.H"
#include "GEMPIC_MultiReducedDiagnostics.H"
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

    constexpr int vdim{3};
    constexpr int ndata{2};

    // Spline degrees
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

        auto [parseBPrimX, funcBPrimX] =
            Utils::parse_functions<3>({"BPrimXx", "BPrimXy", "BPrimXz"});
        auto [parseBPrimY, funcBPrimY] =
            Utils::parse_functions<3>({"BPrimYx", "BPrimYy", "BPrimYz"});
        auto [parseBPrimZ, funcBPrimZ] =
            Utils::parse_functions<3>({"BPrimZx", "BPrimZy", "BPrimZz"});

        DeRhamField<Grid::primal, Space::edge> E(deRham, "E");
        DeRhamField<Grid::dual, Space::face> D(deRham, "D");
        DeRhamField<Grid::primal, Space::face> B(deRham, "B");
        DeRhamField<Grid::dual, Space::edge> H(deRham, "H");
        DeRhamField<Grid::dual, Space::face> J(deRham, "J");
        DeRhamField<Grid::dual, Space::cell> rho(deRham, "rho");
        DeRhamField<Grid::primal, Space::node> phi(deRham, "phi");
        // temporary fields
        DeRhamField<Grid::primal, Space::face> auxPrimalF2(deRham);
        DeRhamField<Grid::dual, Space::face> auxDualF2(deRham);

        // Initialize needed propagators/solvers
        Gempic::FieldSolvers::PoissonSolver poisson(deRham, infra);
        TimeLoop::OperatorHamilton<vdim, degx, degy, degz, hodgeDegree> operatorHamilton;

        // Initialize particle groups
        Io::Parameters params("Particle");
        amrex::Vector<std::string> speciesNames;
        params.get("speciesNames", speciesNames);

        std::vector<std::shared_ptr<ParticleGroupsLinVlasov<vdim, ndata>>> partGrLinVlasov;
        std::vector<std::shared_ptr<ParticleGroups<vdim, ndata>>> partGr;
        init_particles(infra, partGrLinVlasov, partGr);

        // Domain volume and number of cells to normalize s0
        amrex::Real domainVolume =
            GEMPIC_D_MULT(infra.m_length[xDir], infra.m_length[yDir], infra.m_length[zDir]);
        amrex::Vector<int> nCellVector;
        parameters.get("nCellVector", nCellVector);
        int nCells = GEMPIC_D_MULT(nCellVector[xDir], nCellVector[yDir], nCellVector[zDir]);

        {
            // Initialize full diagnostics and write initial time step
            Io::Parameters params("TimeLoop");
            amrex::Real dt;
            params.get("dt", dt);
            int nSteps;
            params.get("nSteps", nSteps);
            Io::Parameters paramsSim("Sim");
            auto nGhost = deRham->get_n_ghost();
            Io::MultiDiagnostics<vdim, ndata> fullDiagn(dt);
            fullDiagn.init_data(infra, deRham->m_fieldsDiagnostics, deRham->m_fieldsScaling, partGr,
                                nGhost);

            // Initialize reduced diagnostics and write initial time step
            Io::MultiReducedDiagnostics<vdim, degx, degy, degz, hodgeDegree, ndata> redDiagn(
                deRham);

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
                amrex::GpuArray<amrex::Real, vdim> vThermalGPU{vThermal[xDir], vThermal[yDir],
                                                               vThermal[zDir]};
                amrex::GpuArray<amrex::Real, vdim> vMeanGPU{vMean[xDir], vMean[yDir], vMean[zDir]};

                amrex::Real vThermalBackground = particleSpecies->get_v_thermal_background();

                for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*particleSpecies, 0); pti.isValid();
                     ++pti)
                {
                    const long np = pti.numParticles();
                    auto *const particles = pti.GetArrayOfStructs()().data();
                    auto *const velx = pti.GetStructOfArrays().GetRealData(0).data();
                    auto *const vely = pti.GetStructOfArrays().GetRealData(1).data();
                    auto *const velz = pti.GetStructOfArrays().GetRealData(2).data();
                    auto *const weight = pti.GetStructOfArrays().GetRealData(vdim).data();
                    auto *const s0 = pti.GetStructOfArrays().GetRealData(vdim + 1).data();

                    amrex::Array4<amrex::Real> const &rhoarr = rho.m_data[pti].array();

                    //compile functions on device
                    auto funcDensityBackground =
                        Utils::compile_function(particleSpecies->get_density_background());

                    amrex::ParallelFor(
                        np,
                        [=] AMREX_GPU_DEVICE(long pp)
                        {
                            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> pos;
                            amrex::GpuArray<amrex::Real, vdim> vel{velx[pp], vely[pp], velz[pp]};

                            for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                            {
                                pos[d] = particles[pp].pos(d);
                            }

                            amrex::Real f0 = eval_sqrt_maxwellian(
                                vel,
                                funcDensityBackground(AMREX_D_DECL(pos[xDir], pos[yDir], pos[zDir]),
                                                      0.),
                                vThermalBackground);

                            // Rescale weights to remove f0 from energy computation
                            weight[pp] /= f0;

                            SplineBase<degx, degy, degz> spline(pos, infra.m_plo, infra.m_dxi);

                            gempic_deposit_rho(rhoarr, spline, f0 * charge * weight[pp]);

                            // Compute s0 multiplied by number of particles as needed for electric
                            // field update and particle energy
                            s0[pp] = nPart / domainVolume *
                                     eval_maxwellian(vel, 1.0, vThermalGPU, vMeanGPU);
                        });
                }
            }

            rho.post_particle_loop_sync();

            poisson.solve_amrex(rho, phi);
            deRham->grad(phi, E);
            E *= -1.0;
            deRham->hodge(E, D);

            // Write initial time step
            redDiagn.compute_diags(infra, deRham->m_fieldsDiagnostics, partGr);
            redDiagn.write_to_file(0, dt);
            fullDiagn.filter_compute_pack_flush(0);

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

                    for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*particleSpecies, 0);
                         pti.isValid(); ++pti)
                    {
                        const long np = pti.numParticles();
                        const auto &particles = pti.GetArrayOfStructs()().data();
                        auto *const velx = pti.GetStructOfArrays().GetRealData(0).data();
                        auto *const vely = pti.GetStructOfArrays().GetRealData(1).data();
                        auto *const velz = pti.GetStructOfArrays().GetRealData(2).data();
                        auto *const weight = pti.GetStructOfArrays().GetRealData(vdim).data();
                        auto *const s0 = pti.GetStructOfArrays().GetRealData(vdim + 1).data();

                        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;
                        for (int cc = 0; cc < vdim; cc++)
                        {
                            eA[cc] = (E.m_data[cc])[pti].array();
                        }

                        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> jA;
                        for (int cc = 0; cc < vdim; cc++)
                        {
                            jA[cc] = (J.m_data[cc])[pti].array();
                        }

                        // Compile functions on device
                        auto funcBPrimXDevice = Utils::compile_functions<3>(parseBPrimX);
                        auto funcBPrimYDevice = Utils::compile_functions<3>(parseBPrimY);
                        auto funcBPrimZDevice = Utils::compile_functions<3>(parseBPrimZ);
                        auto funcDensityBackground =
                            Utils::compile_function(particleSpecies->get_density_background());

                        amrex::ParallelFor(
                            np,
                            [=] AMREX_GPU_DEVICE(long pp)
                            {
                                // Local arrays for particle position and velocities
                                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> pos;
                                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> posOld;
                                amrex::GpuArray<amrex::Real, vdim> vel{velx[pp], vely[pp],
                                                                       velz[pp]};

                                for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                                {
                                    pos[d] = particles[pp].pos(d);
                                    posOld[d] = particles[pp].pos(d);
                                }

                                amrex::Real f0 = eval_sqrt_maxwellian(
                                    vel,
                                    funcDensityBackground(
                                        AMREX_D_DECL(pos[xDir], pos[yDir], pos[zDir]), 0.),
                                    vThermalBackground);

                                ParticleMeshCoupling::SplineBase<degx, degy, degz> spline(
                                    pos, infra.m_plo, infra.m_dxi);

                                // He,particle
                                amrex::GpuArray<amrex::Real, vdim> efield =
                                    spline.template eval_spline_field<Field::PrimalOneForm>(eA);
                                weight[pp] += 0.5 * dt * chargeMass / vThermalBackground2 * f0 *
                                              (efield[xDir] * vel[xDir] + efield[yDir] * vel[yDir] +
                                               efield[zDir] * vel[zDir]) /
                                              s0[pp];

                                // Add particle contribution to J
                                gempic_deposit_j(spline, vel, f0 * charge * weight[pp], jA);

                                // Particle push with Strang splitting in spatial components
                                push_particle_constant_b(pos, posOld, vel, funcBPrimXDevice,
                                                         funcBPrimYDevice, funcBPrimZDevice,
                                                         chargeMass, dt);

                                // Write position and velocities
                                for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                                {
                                    particles[pp].pos(d) = pos[d];
                                }
                                velx[pp] = vel[xDir];
                                vely[pp] = vel[yDir];
                                velz[pp] = vel[zDir];

                                //Update splines with new particle position
                                AMREX_D_TERM(
                                    (spline.update_1d_splines<Direction::xDir>(
                                        pos[xDir], infra.m_plo[xDir], infra.m_dxi[xDir]));
                                    , (spline.update_1d_splines<Direction::yDir>(
                                          pos[yDir], infra.m_plo[yDir], infra.m_dxi[yDir]));
                                    , (spline.update_1d_splines<Direction::zDir>(
                                          pos[zDir], infra.m_plo[zDir], infra.m_dxi[zDir])););

                                f0 = eval_sqrt_maxwellian(
                                    vel,
                                    funcDensityBackground(
                                        AMREX_D_DECL(pos[xDir], pos[yDir], pos[zDir]), 0.),
                                    vThermalBackground);

                                // Add particle contribution to J
                                gempic_deposit_j(spline, vel, f0 * charge * weight[pp], jA);
                            });
                    }
                    particleSpecies->Redistribute();
                }

                J.post_particle_loop_sync();
                J *= dt / 2;
                D -= J;

                deRham->hodge(D, E);

                // He,particle
                for (auto &particleSpecies : partGrLinVlasov)
                {
                    amrex::Real charge = particleSpecies->get_charge();
                    amrex::Real chargeMass = charge / particleSpecies->get_mass();
                    amrex::Real vThermalBackground = particleSpecies->get_v_thermal_background();
                    amrex::Real vThermalBackground2 = vThermalBackground * vThermalBackground;

                    for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*particleSpecies, 0);
                         pti.isValid(); ++pti)
                    {
                        const long np = pti.numParticles();
                        const auto &particles = pti.GetArrayOfStructs()().data();
                        auto *const velx = pti.GetStructOfArrays().GetRealData(0).data();
                        auto *const vely = pti.GetStructOfArrays().GetRealData(1).data();
                        auto *const velz = pti.GetStructOfArrays().GetRealData(2).data();
                        auto *const weight = pti.GetStructOfArrays().GetRealData(vdim).data();
                        auto *const s0 = pti.GetStructOfArrays().GetRealData(vdim + 1).data();

                        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;
                        for (int cc = 0; cc < vdim; cc++)
                        {
                            eA[cc] = (E.m_data[cc])[pti].array();
                        }

                        //compile functions on device
                        auto funcDensityBackground =
                            Utils::compile_function(particleSpecies->get_density_background());

                        amrex::ParallelFor(
                            np,
                            [=] AMREX_GPU_DEVICE(long pp)
                            {
                                // Local arrays for particle position and velocities
                                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> pos;
                                amrex::GpuArray<amrex::Real, vdim> vel{velx[pp], vely[pp],
                                                                       velz[pp]};

                                for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                                {
                                    pos[d] = particles[pp].pos(d);
                                }

                                amrex::Real f0 = eval_sqrt_maxwellian(
                                    vel,
                                    funcDensityBackground(
                                        AMREX_D_DECL(pos[xDir], pos[yDir], pos[zDir]), 0.),
                                    vThermalBackground);

                                ParticleMeshCoupling::SplineBase<degx, degy, degz> spline(
                                    pos, infra.m_plo, infra.m_dxi);

                                amrex::GpuArray<amrex::Real, vdim> efield =
                                    spline.template eval_spline_field<Field::PrimalOneForm>(eA);

                                weight[pp] += 0.5 * dt * chargeMass / vThermalBackground2 * f0 *
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

                //write outputs
                redDiagn.compute_diags(infra, deRham->m_fieldsDiagnostics, partGr);
                redDiagn.write_to_file(tStep + 1, dt);
                fullDiagn.filter_compute_pack_flush(tStep + 1);

                if (tStep % 10 == 0)
                {
                    std::cout << "Time Step: " << tStep + 1 << std::endl;
                }
            }
        }  // end of "time loop" scope
    }
    amrex::Finalize();
}
