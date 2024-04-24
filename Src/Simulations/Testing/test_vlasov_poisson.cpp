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
#include "GEMPIC_Parameters.H"
#include "GEMPIC_ParticleGroups.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_Sampler.H"
//#include "GEMPIC_HsZigzag.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_PoissonSolver.H"

using namespace Gempic;
using namespace Forms;
using namespace Particle;
using namespace ParticleMeshCoupling;

void write_rho (DeRhamField<Grid::dual, Space::cell> &rho,
                ComputationalDomain &infra,
                double time,
                int step)
{
    BL_PROFILE("Gempic::Forms::Particle::ParticleMeshCoupling::write_rho()");
    // save fields
    // MultiFab Info -------------------------------------------------------------
    std::string plotfilename{"Plotfiles/" + amrex::Concatenate("rho", step)};

    amrex::Vector<std::string> varnames{{"rho"}};

    amrex::WriteSingleLevelPlotfile(plotfilename, rho.m_data, varnames, infra.m_geom, time, 0);
}

void write_phi (DeRhamField<Grid::primal, Space::node> &phi,
                ComputationalDomain &infra,
                double time,
                int step)
{
    BL_PROFILE("Gempic::Forms::Particle::ParticleMeshCoupling::write_phi()");
    // save fields
    // MultiFab Info -------------------------------------------------------------
    std::string plotfilename{"Plotfiles/" + amrex::Concatenate("phi", step)};

    amrex::Vector<std::string> varnames{"phi"};

    amrex::WriteSingleLevelPlotfile(plotfilename, phi.m_data, varnames, infra.m_geom, time, 0);
}

void write_ex (DeRhamField<Grid::primal, Space::edge> &E,
               ComputationalDomain &infra,
               double time,
               int step)
{
    BL_PROFILE("Gempic::Forms::Particle::ParticleMeshCoupling::write_ex()");
    // save fields
    // MultiFab Info -------------------------------------------------------------
    std::string plotfilename{"Plotfiles/" + amrex::Concatenate("E", step)};

    amrex::Vector<std::string> varnames{{"Ex"}};

    amrex::WriteSingleLevelPlotfile(plotfilename, E.m_data[xDir], varnames, infra.m_geom, time, 0);
}

// apply npass times a cubic spline filter to rho
void filter (DeRhamField<Grid::dual, Space::cell> &rho,
             DeRhamField<Grid::dual, Space::cell> &rhoTemp,
             int npass)
{
    BL_PROFILE("Gempic::Forms::Particle::ParticleMeshCoupling::filter()");
    auto nghost = rho.m_deRham->get_n_ghost();
    for (int pass = 0; pass < npass; pass++)
    {
        // amrex::Print() << "filt pass " << pass << std::endl;
        for (int direction = 0; direction < 3; direction++)
        {
            for (amrex::MFIter mfi(rho.m_data); mfi.isValid(); ++mfi)  // Loop over grids
            {
                const amrex::Box &bx = mfi.validbox();
                amrex::Array4<amrex::Real> const &rhoArr = rho.m_data[mfi].array();
                amrex::Array4<amrex::Real> const &rhoTempArr = rhoTemp.m_data[mfi].array();

                ParallelFor(bx,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k)
                            {
                                amrex::GpuArray<amrex::Real, 3> coef{1. / 6., 2. / 3., 1. / 6.};
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
            amrex::Copy(rho.m_data, rhoTemp.m_data, 0, 0, 1, nghost);
        }
    }
}

int main (int argc, char *argv[])
{
    amrex::Initialize(argc, argv);
    // Linear splines is ok, and lower dimension Hodge is good enough
    constexpr unsigned int vdim{3};
    constexpr unsigned int numspec{1};
    // Spline degrees
    constexpr int degx{1};
    constexpr int degy{1};
    constexpr int degz{1};
    constexpr int maxSplineDegree{std::max(std::max(degx, degy), degz)};
    //
    constexpr int hodgeDegree{2};

    {
        BL_PROFILE("main()");
        Io::Parameters parameters{};

        // Initialize computational_domain
        ComputationalDomain infra;

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, maxSplineDegree,
                                                        HodgeScheme::FDHodge);

        auto [parseB, funcB] = Utils::parse_functions<3>({"Bx", "By", "Bz"});

        DeRhamField<Grid::primal, Space::face> B(deRham, funcB);
        DeRhamField<Grid::primal, Space::edge> E(deRham);
        DeRhamField<Grid::dual, Space::cell> rho(deRham);
        DeRhamField<Grid::dual, Space::cell> rhoTemp(deRham);
        DeRhamField<Grid::primal, Space::node> phi(deRham);
        DeRhamField<Grid::primal, Space::node> phiTemp(deRham);

        // For the moment we consider only a constant background field.
        amrex::Real Bz = funcB[xDir](AMREX_D_DECL(0., 0., 0.), 0.);
        amrex::Print() << "Bz " << Bz << std::endl;

        // Initialize Poisson solver
        FieldSolvers::PoissonSolver poisson(deRham);
        // Initialize particle groups
        amrex::GpuArray<std::shared_ptr<ParticleGroups<vdim>>, numspec> electrons;
        init_particles(infra, electrons, InitMethod::fullDomainCpu);

        {                         // "Time Loop" scope. Should be a separate function
            const int ndata = 1;  // Needs to be 1 so that the correct ParIter type is defined.
                                  // Putting 4 gets a non-defined type
            const int npass = 3;  // Number of filter passes

            Io::Parameters params("TimeLoop");
            // Deposit initial charge
            // start with background charge

            rho.m_data.setVal(0.0);
            for (int spec = 0; spec < numspec; spec++)
            {
                amrex::Real charge = electrons[spec]->get_charge();

                for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*electrons[spec], 0); pti.isValid();
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
                            // spline, charge * infra.dxi[GEMPIC_SPACEDIM] * weight[pp],
                            // rhoarr);
                        });
                }
            }

            rho.post_particle_loop_sync();

            filter(rho, rhoTemp, npass);
            write_rho(rho, infra, 0, 0);
            // solve Poisson. AMReX nodal based Poisson solver is used
            poisson.solve(infra, rho, phi);
            deRham->grad(phi, E);
            E *= -1.0;
            write_ex(E, infra, 0, 0);
            amrex::Real dt;
            params.get("dt", dt);
            int nSteps;
            params.get("nSteps", nSteps);

            int saveFields = 0;
            params.get_or_set("saveFields", saveFields);

            for (int tStep = 0; tStep < nSteps; tStep++)
            {
                for (int spec = 0; spec < numspec; spec++)
                {
                    amrex::Real charge = electrons[spec]->get_charge();
                    amrex::Real chargemass = charge / electrons[spec]->get_mass();
                    amrex::Real a = 0.5 * chargemass * dt * Bz;

                    // start with background charge
                    rho.m_data.setVal(0.0);

                    for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*electrons[spec], 0);
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

                                gempic_deposit_rho(spline, charge * weight[pp], rhoarr);
                                // spline, charge * infra.dxi[GEMPIC_SPACEDIM] * weight[pp],
                                // rhoarr);
                            });
                    }

                    electrons[spec]->Redistribute();

                    rho.post_particle_loop_sync();
                    filter(rho, rhoTemp, npass);

                    // solve Poisson. AMReX nodal based Poisson solver is used (Hodge needs to be
                    // applied first)
                    // deRham->hodgeFD<hodgeDegree>(rho, phiTemp);
                    poisson.solve(infra, rho, phi);
                    deRham->grad(phi, E);
                    E *= -1.0;

                    // start with background charge
                    rho.m_data.setVal(0.0);

                    for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*electrons[spec], 0);
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
                                // splineNew, charge * infra.dxi[GEMPIC_SPACEDIM] * weight[pp],
                                // rhoarr);
                            });
                    }
                    electrons[spec]->Redistribute();
                }

                rho.post_particle_loop_sync();

                filter(rho, rhoTemp, npass);

                if (tStep % saveFields == 0)
                {
                    write_rho(rho, infra, (tStep + 1) * dt, tStep + 1);
                    write_ex(E, infra, (tStep + 1) * dt, tStep + 1);
                }
                if (tStep % 1 == 0)
                {
                    std::cout << "Time Step: " << tStep + 1 << std::endl;
                }
            }
        }  // end of "time loop" scope
    }
    amrex::Finalize();
}
