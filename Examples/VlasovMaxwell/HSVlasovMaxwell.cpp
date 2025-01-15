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
#include "GEMPIC_SplineClass.H"
#include "GEMPIC_Splitting.H"

using namespace Gempic;
using namespace Forms;
using namespace Particle;
using namespace ParticleMeshCoupling;
using namespace FieldSolvers;

int main (int argc, char *argv[])
{
    amrex::Initialize(argc, argv);

    constexpr int vdim{3};
    constexpr int numspec{1};
    constexpr int ndata{1};

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

        auto [parseB, funcB] = Utils::parse_functions<3>({"Bx", "By", "Bz"});
        auto [parseE, funcE] = Utils::parse_functions<3>({"Ex", "Ey", "Ez"});

        DeRhamField<Grid::primal, Space::edge> E(deRham, funcE, "E");
        DeRhamField<Grid::dual, Space::face> D(deRham, funcE, "D");
        DeRhamField<Grid::primal, Space::face> B(deRham, funcB, "B");
        DeRhamField<Grid::dual, Space::edge> H(deRham, funcB, "H");
        DeRhamField<Grid::dual, Space::face> J(deRham, "J");
        DeRhamField<Grid::dual, Space::cell> rho(deRham, "rho");
        DeRhamField<Grid::primal, Space::node> phi(deRham, "phi");
        DeRhamField<Grid::primal, Space::edge> ephi(deRham);
        DeRhamField<Grid::dual, Space::cell> divD(deRham, "divD");

        std::vector<std::shared_ptr<ParticleGroups<vdim>>> partGr;
        init_particles(infra, partGr);
        amrex::Real rhoBackground{1.0};
        parameters.get_or_set("rhoBackground", rhoBackground);

        auto poisson = std::make_shared<PoissonSolver>(deRham, infra);
        ConjugateGradient<DeRhamField<Grid::dual, Space::cell>,
                          DeRhamField<Grid::primal, Space::node>, Operator::poisson>
            cgPoisson(deRham, poisson);
        Gempic::TimeLoop::OperatorHamilton<vdim, degx, degy, degz> operatorHamilton;

        // Initializing filter
        std::unique_ptr<Filter::Filter> biFilter = std::make_unique<Filter::BilinearFilter>();

        {  //"Time Loop" scope

            Io::Parameters params("TimeLoop");
            amrex::Real dt;
            params.get("dt", dt);
            int nSteps;
            params.get("nSteps", nSteps);

            auto diagnostics = Io::make_diagnostics<degx, degy, degz>(infra, deRham, partGr);

            // Deposit initial charge
            rho.m_data.setVal(0.0);

            for (int spec = 0; spec < numspec; spec++)
            {
                amrex::Real charge = partGr[spec]->get_charge();
                // reset total weight of species to domain volume to correct for inaccuracy of
                // sampling
                amrex::Print() << "total weight "
                               << GEMPIC_D_MULT(infra.m_length[0], infra.m_length[1],
                                                infra.m_length[2])
                               << std::endl;
                partGr[spec]->reset_total_weight(
                    GEMPIC_D_MULT(infra.m_length[0], infra.m_length[1], infra.m_length[2]));
                for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*partGr[spec], 0); pti.isValid();
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
                            gempic_deposit_rho(rhoarr, spline, charge * weight[pp]);
                        });
                }
            }

            rho.post_particle_loop_sync();

            // Add background charge (needs to be done after post_particle_loop_sync)
            rho +=
                rhoBackground * GEMPIC_D_MULT(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir]);

            // solve Poisson
            cgPoisson.solve(rho, phi);
            deRham->a_times_grad(phi, ephi, -1);
            E += ephi;
            deRham->hodge(E, D);
            // compute div D for diagnostics
            deRham->div(D, divD);

            amrex::Real simTime{0.0};
            diagnostics.compute_and_write_to_file(0, simTime);

            for (int tStep = 0; tStep < nSteps; tStep++)
            {
                operatorHamilton.apply_h_b(D, deRham, B, H, 0.5 * dt);

                operatorHamilton.apply_h_e_field(B, deRham, E, D, 0.5 * dt);

                // initialize rho and J to 0 for particle loop
                rho.m_data.setVal(0.0);
                for (int comp = 0; comp < vdim; ++comp)
                {
                    J.m_data[comp].setVal(0.0);
                }

                for (int spec = 0; spec < numspec; spec++)
                {
                    amrex::Real charge = partGr[spec]->get_charge();
                    amrex::Real chargeOverMass = charge / partGr[spec]->get_mass();

                    for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*partGr[spec], 0); pti.isValid();
                         ++pti)
                    {
                        const long np = pti.numParticles();
                        const auto &particles = pti.GetArrayOfStructs()().data();
                        auto *const velx = pti.GetStructOfArrays().GetRealData(0).data();
                        auto *const vely = pti.GetStructOfArrays().GetRealData(1).data();
                        auto *const velz = pti.GetStructOfArrays().GetRealData(2).data();
                        auto *const weight = pti.GetStructOfArrays().GetRealData(vdim).data();

                        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> jA;
                        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;
                        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> bA;

                        for (int cc = 0; cc < vdim; cc++)
                        {
                            jA[cc] = (J.m_data[cc])[pti].array();
                            eA[cc] = (E.m_data[cc])[pti].array();
                            bA[cc] = (B.m_data[cc])[pti].array();
                        }

                        amrex::ParallelFor(np,
                                           [=] AMREX_GPU_DEVICE(long pp)
                                           {
                                               // Read out particle position and compute according
                                               // splines
                                               amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> pos;
                                               for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                                               {
                                                   pos[d] = particles[pp].pos(d);
                                               }
                                               SplineWithPrimitive<degx, degy, degz> spline(
                                                   pos, infra.m_plo, infra.m_dxi);

                                               // Read out particle velocity
                                               amrex::GpuArray<amrex::Real, vdim> vel{
                                                   velx[pp], vely[pp], velz[pp]};

                                               operatorHamilton.apply_h_e_particle(
                                                   vel, eA, spline, chargeOverMass, 0.5 * dt);

                                               amrex::Real chargeWeight = charge * weight[pp];

                                               operatorHamilton.apply_h_p(
                                                   pos, vel, infra, spline, infra.m_dx, jA, bA,
                                                   chargeOverMass, chargeWeight, dt);

                                               for (int xd = 0; xd < GEMPIC_SPACEDIM; xd++)
                                               {
                                                   particles[pp].pos(xd) = pos[xd];
                                               }

                                               velx[pp] = vel[xDir];
                                               vely[pp] = vel[yDir];
                                               velz[pp] = vel[zDir];
                                           });
                    }
                    partGr[spec]->Redistribute();
                }
                J.post_particle_loop_sync();

                // Update D (field part of H_p_i)
                D -= J;

                // E needed for pushing the particles
                deRham->hodge(D, E);

                // Second particle loop for particle part from H_E
                for (int spec = 0; spec < numspec; spec++)
                {
                    amrex::Real charge = partGr[spec]->get_charge();
                    amrex::Real chargeOverMass = charge / partGr[spec]->get_mass();

                    for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*partGr[spec], 0); pti.isValid();
                         ++pti)
                    {
                        const long np = pti.numParticles();
                        const auto &particles = pti.GetArrayOfStructs()().data();
                        auto *const velx = pti.GetStructOfArrays().GetRealData(0).data();
                        auto *const vely = pti.GetStructOfArrays().GetRealData(1).data();
                        auto *const velz = pti.GetStructOfArrays().GetRealData(2).data();
                        auto *const weight = pti.GetStructOfArrays().GetRealData(vdim).data();

                        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;
                        amrex::Array4<amrex::Real> rhoarr;

                        rhoarr = rho.m_data[pti].array();
                        for (int cc = 0; cc < vdim; cc++)
                        {
                            eA[cc] = (E.m_data[cc])[pti].array();
                        }

                        // loop over particles: add contribution of old particle position to J and
                        // push
                        amrex::ParallelFor(
                            np,
                            [=] AMREX_GPU_DEVICE(long pp)
                            {
                                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> pos;
                                for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                                {
                                    pos[d] = particles[pp].pos(d);
                                }

                                amrex::GpuArray<amrex::Real, vdim> vel{velx[pp], vely[pp],
                                                                       velz[pp]};

                                SplineBase<degx, degy, degz> spline(pos, infra.m_plo, infra.m_dxi);

                                operatorHamilton.apply_h_e_particle(vel, eA, spline, chargeOverMass,
                                                                    0.5 * dt);
                                velx[pp] = vel[xDir];
                                vely[pp] = vel[yDir];
                                velz[pp] = vel[zDir];
                                // compute rho for Gauss error diagnostics
                                // in principle this should be computed only
                                // if the Gauss error diagnostic is performed
                                gempic_deposit_rho(rhoarr, spline, charge * weight[pp]);
                            });
                    }
                    partGr[spec]->Redistribute();
                }
                // end treat particles
                rho.post_particle_loop_sync();
                // Add background charge (needs to be done after post_particle_loop_sync)
                rho += rhoBackground *
                       GEMPIC_D_MULT(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir]);

                operatorHamilton.apply_h_e_field(B, deRham, E, D, 0.5 * dt);

                operatorHamilton.apply_h_b(D, deRham, B, H, 0.5 * dt);

                // compute div D for diagnostics
                deRham->div(D, divD);
                simTime = dt * (tStep + 1);
                diagnostics.compute_and_write_to_file(tStep + 1, simTime);

                amrex::Print() << "finished time-step: " << tStep << std::endl;
            }
        }
    }
    amrex::Finalize();
}