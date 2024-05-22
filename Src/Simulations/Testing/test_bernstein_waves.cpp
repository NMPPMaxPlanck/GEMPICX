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
#include "GEMPIC_Parameters.H"
#include "GEMPIC_ParticleGroups.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_Sampler.H"
//#include "GEMPIC_PoissonSolver.H"
#include "GEMPIC_BilinearFilter.H"
#include "GEMPIC_MultiFullDiagnostics.H"
#include "GEMPIC_MultiReducedDiagnostics.H"

using namespace Gempic;
using namespace Forms;
using namespace Particle;
using namespace ParticleMeshCoupling;

// apply npass times a bilinear filter to rho
void filter (DeRhamField<Grid::dual, Space::cell> &rho,
             DeRhamField<Grid::dual, Space::cell> &rhoTemp,
             int npass)
{
    BL_PROFILE("Gempic::Forms::Particle::ParticleMeshCoupling::filter()");
    auto nghost = rho.m_deRham->get_n_ghost();
    for (int pass = 0; pass < npass - 1; pass++)
    {
        // amrex::Print() << "filt pass " << pass << std::endl;
        for (int direction = 0; direction < GEMPIC_SPACEDIM; direction++)
        {
            for (amrex::MFIter mfi(rho.m_data); mfi.isValid(); ++mfi)  // Loop over grids
            {
                const amrex::Box &bx = mfi.validbox();
                amrex::Array4<amrex::Real> const &rhoArr = rho.m_data[mfi].array();
                amrex::Array4<amrex::Real> const &rhoTempArr = rhoTemp.m_data[mfi].array();

                ParallelFor(bx,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k)
                            {
                                //amrex::GpuArray<amrex::Real, 3> coef{1. / 6., 2. / 3., 1. / 6.};
                                amrex::GpuArray<amrex::Real, 3> coef{0.25, 0.5, 0.25};
                                amrex::Real val = 0.0;
                                for (int d = 0; d < 3; d++)
                                {
                                    val += coef[d] * rhoArr(i + (direction == xDir ? d - 1 : 0),
                                                            j + (direction == yDir ? d - 1 : 0),
                                                            k + (direction == zDir ? d - 1 : 0));
                                }
                                rhoTempArr(i, j, k) = val;
                            });
            }
            // Copy into rho
            rhoTemp.fill_boundary();
            amrex::Copy(rho.m_data, rhoTemp.m_data, 0, 0, rho.m_data.nComp(), nghost);
        }
    }
    // Compensation step
    for (int direction = 0; direction < GEMPIC_SPACEDIM; direction++)
    {
        for (amrex::MFIter mfi(rho.m_data); mfi.isValid(); ++mfi)  // Loop over grids
        {
            const amrex::Box &bx = mfi.validbox();
            amrex::Array4<amrex::Real> const &rhoArr = rho.m_data[mfi].array();
            amrex::Array4<amrex::Real> const &rhoTempArr = rhoTemp.m_data[mfi].array();

            ParallelFor(bx,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                            //amrex::GpuArray<amrex::Real, 3> coef{1. / 6., 2. / 3., 1. / 6.};
                            amrex::GpuArray<amrex::Real, 3> coef{
                                0.25 * (1 - npass), 0.5 * (1 + npass), 0.25 * (1 - npass)};
                            amrex::Real val = 0.0;
                            for (int d = 0; d < 3; d++)
                            {
                                val += coef[d] * rhoArr(i + (direction == xDir ? d - 1 : 0),
                                                        j + (direction == yDir ? d - 1 : 0),
                                                        k + (direction == zDir ? d - 1 : 0));
                            }
                            rhoTempArr(i, j, k) = val;
                        });
        }
        // Copy into rho
        rhoTemp.fill_boundary();
        amrex::Copy(rho.m_data, rhoTemp.m_data, 0, 0, rho.m_data.nComp(), nghost);
    }
}

int main (int argc, char *argv[])
{
    amrex::Initialize(argc, argv);
    BL_PROFILE_VAR("main()", pmain);

    // Linear splines is ok, and lower dimension Hodge is good enough
    constexpr int vdim{3};
    constexpr int numspec{1};
    constexpr int ndata{1};  // Needs to be 1 so that the correct ParIter type is defined. Putting 4
                             // gets a non-defined type
    // Spline degrees
    constexpr int degx{3};
    constexpr int degy{3};
    constexpr int degz{3};
    constexpr int maxSplineDegree{std::max(std::max(degx, degy), degz)};
    //
    constexpr int hodgeDegree{2};

    {
        // Parameters::setPrintOutput();  // uncomment to print an output file
        Io::Parameters parameters{};

        // Initialize computational_domain
        ComputationalDomain infra;

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, maxSplineDegree,
                                                        HodgeScheme::FDHodge);

        auto [parseB, funcB] = Utils::parse_functions<3>({"Bx", "By", "Bz"});

        DeRhamField<Grid::primal, Space::face> B(deRham, funcB, "B");
        DeRhamField<Grid::dual, Space::edge> H(deRham, funcB, "H");
        DeRhamField<Grid::primal, Space::edge> E(deRham, "E");
        DeRhamField<Grid::dual, Space::face> D(deRham, "D");
        DeRhamField<Grid::dual, Space::cell> rho(deRham, "rho");
        DeRhamField<Grid::dual, Space::cell> divD(deRham, "divD");
        DeRhamField<Grid::dual, Space::cell> rhoTemp(deRham);
        DeRhamField<Grid::primal, Space::node> phi(deRham, "phi");
        DeRhamField<Grid::dual, Space::face> currentDensity(deRham, "J");

        // Initialize particle groups
        amrex::GpuArray<std::shared_ptr<ParticleGroups<vdim>>, numspec> ions;
        init_particles(infra, ions, InitMethod::fullDomainCpu);

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
            Io::Parameters paramsSim("Sim");
            amrex::Real te{1.0};  // electron temperature (default 1.0)
            paramsSim.get_or_set("Te", te);
            auto nGhost = deRham->get_n_ghost();
            Io::MultiDiagnostics<vdim, numspec, ndata> fullDiagn(dt);
            fullDiagn.init_data(infra, deRham->m_fieldsDiagnostics, deRham->m_fieldsScaling, ions,
                                nGhost);

            // Initialize reduced diagnostics and write initial time step
            Io::MultiReducedDiagnostics<vdim, numspec, degx, degy, degz, hodgeDegree, 1> redDiagn(
                deRham);

            // Deposit initial charge
            for (int spec = 0; spec < numspec; spec++)
            {
                amrex::Real charge = ions[spec]->get_charge();

                for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*ions[spec], 0); pti.isValid();
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
                            gempic_deposit_rho(spline, charge * weight[pp], rhoarr);
                        });
                }
            }

            rho.post_particle_loop_sync();

            for (int component = 0; component < vdim; component++)
            {
                currentDensity.m_data[component].SumBoundary(
                    0, currentDensity.m_data[component].nComp(), nGhost,
                    amrex::IntVect(AMREX_D_DECL(0, 0, 0)), infra.m_geom.periodicity());
            }
            currentDensity.average_sync();
            currentDensity.fill_boundary();

            // Filter rho and compute phi with filtered array
            biFilter->apply_stencil(rhoTemp.m_data, rho.m_data, 0, 0, rho.m_data.nComp());
            deRham->hodge(rhoTemp, phi);
            deRham->grad(phi, E);
            // E *= -1.0;
            E *= -te;

            // D is also needed to compute energy
            deRham->hodge(E, D);

            // computing rho from D to check Gauss law // Makes no sense for quasi-neutral model
            // deRham->div(D,rho_gauss_law); //must be done before writing reduced diagnostics

            // Write initial time step
            redDiagn.compute_diags(infra, deRham->m_fieldsDiagnostics, ions);
            redDiagn.write_to_file(0, dt);
            fullDiagn.filter_compute_pack_flush(0);

            for (int tStep = 0; tStep < nSteps; tStep++)
            {
                for (int spec = 0; spec < numspec; spec++)
                {
                    amrex::Real charge = ions[spec]->get_charge();
                    amrex::Real chargemass = charge / ions[spec]->get_mass();
                    amrex::Real a = 0.5 * chargemass * dt * Bz;

                    rho.m_data.setVal(0.0);

                    for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*ions[spec], 0); pti.isValid();
                         ++pti)
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

                                gempic_deposit_rho(spline, charge * weight[pp], rhoarr);
                            });
                    }

                    ions[spec]->Redistribute();

                    rho.post_particle_loop_sync();
                    // Filter rho and compute phi with filtered array
                    biFilter->apply_stencil(rhoTemp.m_data, rho.m_data, 0, 0, rho.m_data.nComp());
                    deRham->hodge(rhoTemp, phi);
                    deRham->grad(phi, E);
                    // E *= -1.0;
                    E *= -te;
                    // D is also needed to compute energy
                    deRham->hodge(E, D);

                    rho.m_data.setVal(0.0);

                    for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*ions[spec], 0); pti.isValid();
                         ++pti)
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
                                push_v_efield(vel, dt * 0.5, chargemass, efield);

                                // rotate v with magnetic field over dt
                                amrex::Real vx = vel[xDir];
                                amrex::Real vy = vel[yDir];

                                // Crank Nicolson update
                                vel[xDir] = (vx * (1. - a * a) + 2. * a * vy) / (1. + a * a);
                                vel[yDir] = (vy * (1. - a * a) - 2. * a * vx) / (1. + a * a);

                                // push v with the electric field over dt/2
                                push_v_efield(vel, dt * 0.5, chargemass, efield);

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

                                gempic_deposit_rho(splineNew, charge * weight[pp], rhoarr);
                            });
                    }
                    ions[spec]->Redistribute();
                }

                rho.post_particle_loop_sync();

                //filter(rho, rhoTemp, npass);
                // biFilter->apply_stencil(rhoTemp.m_data, rho.m_data, 0, 0, rho.m_data.nComp());
                // // Copy into rho
                // rhoTemp.fill_boundary();
                // amrex::Copy(rho.m_data, rhoTemp.m_data, 0, 0, rho.m_data.nComp(), nGhost);

                redDiagn.compute_diags(infra, deRham->m_fieldsDiagnostics, ions);
                redDiagn.write_to_file(tStep + 1, dt);
                fullDiagn.filter_compute_pack_flush(tStep + 1);

                if (tStep % 10 == 0)
                {
                    std::cout << "Time Step: " << tStep + 1 << std::endl;
                }
            }
        }  // end of "time loop" scope
    }
    BL_PROFILE_VAR_STOP(pmain);
    amrex::Finalize();
}
