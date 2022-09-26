#ifndef TIME_LOOP_LS_RUNGEKUTTA
#define TIME_LOOP_LS_RUNGEKUTTA

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_CoordSys.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>
#include <GEMPIC_SumBoundary.H>
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_diagnostics.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_mesh_coupling_C2.H>
#include <GEMPIC_particle_positions.H>
#include <GEMPIC_profiling.H>

using namespace Diagnostics_Output;
using namespace Field_solvers;
using namespace Particles;

namespace Gempic
{
namespace Time_Loop
{

template <int stages, int vdim, int numspec>
class ls_rungekutta
{
public:
    amrex::GpuArray<amrex::Real, stages - 1> m_coeffs_a;
    amrex::GpuArray<amrex::Real, stages> m_coeffs_b;

    amrex::GpuArray<std::unique_ptr<amrex::MultiFab>, 3> m_E_stage;
    amrex::GpuArray<std::unique_ptr<amrex::MultiFab>, 3> m_B_stage;

public:
    // The default constructor.
    /**
     * Default Constructor.
     */
    ls_rungekutta()
    {
        switch (stages)
        {
            case 3:
                m_coeffs_b = {{0.245170287303492, 0.184896052186740, 0.569933660509768}};
                m_coeffs_a = {{0.755726351946097, 0.386954477304099}};
                break;
            default:
                amrex::Print() << "Order of Runge-Kutta method not implemented.";
                break;
        }
    }
    ls_rungekutta(computational_domain domain,
                  maxwell_yee<vdim> &mw_yee)  //:
    // m_part_gr_stage(std::make_unique<particle_groups<vdim>(charge[0], mass[0], domain))  // TODO:
    // Solution for more than one species
    {
        switch (stages)
        {
            case 1:
                m_coeffs_b = {1.0};
                m_coeffs_a = {};
                break;
            case 3:
                m_coeffs_b = {{0.245170287303492, 0.184896052186740, 0.569933660509768}};
                m_coeffs_a = {{0.755726351946097, 0.386954477304099}};
                break;
            default:
                amrex::Print() << "Order of Runge-Kutta method not implemented.";
                break;
        }

        // for (int spec = 0; spec < numspec; ++spec)
        //   m_part_gr_stage[spec](charge[spec], mass[spec], domain);
        for (int i = 0; i < 3; i++)
        {
            m_E_stage[i].reset(new amrex::MultiFab(convert(domain.grid, *mw_yee.E_Index[i]),
                                                   domain.distriMap, 1, mw_yee.Nghost));
            m_B_stage[i].reset(new amrex::MultiFab(convert(domain.grid, *mw_yee.B_Index[i]),
                                                   domain.distriMap, 1, mw_yee.Nghost));
        }
    }

    template <int degx, int degy, int degz, int degmw, int ndata, bool electromagnetic = true>
    void time_loop_step(
        computational_domain infra, maxwell_yee<vdim> &mw_yee,
        amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec> &part_gr)
    {
        amrex::Real dt = mw_yee.dt;
        amrex::Real inv_vol = infra.dxi[GEMPIC_SPACEDIM];
        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> plo = infra.plo;
        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM + 1> dxi = infra.dxi; 
        // Copy the E and B solutions
        for (int cc = 0; cc < 3; cc++)
        {
            amrex::MultiFab::Copy(*m_E_stage[cc], (*mw_yee.E_Array[cc]), 0, 0, 1, mw_yee.Nghost);
            amrex::MultiFab::Copy(*m_B_stage[cc], (*mw_yee.B_Array[cc]), 0, 0, 1, mw_yee.Nghost);
        }

        // int old_flag = amrex::MFIter::allowMultipleMFIters(true);
        for (int s = 0; s < stages; ++s)
        {
            // set J to 0 before we start depositing values from current particles
            for (int i = 0; i < vdim; i++)
            {
                mw_yee.J_Array[i]->setVal(0.0);
            }
            // particle loop
            for (int spec = 0; spec < numspec; spec++)
            {
                amrex::Real charge = part_gr[spec]->getCharge();
                amrex::Real chargemass = charge / part_gr[spec]->getMass();

                for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*part_gr[spec], 0); pti.isValid();
                     ++pti)
                {
                    const long np = pti.numParticles();

                    auto &particles = pti.GetArrayOfStructs();
                    auto partData = particles().data();
                    auto &particleAttributes = pti.GetStructOfArrays();
                    auto weight = particleAttributes.GetRealData(vdim).data();
                    auto velx = particleAttributes.GetRealData(0).data();
                    auto vely = particleAttributes.GetRealData(1).data();
                    auto velz = particleAttributes.GetRealData(2).data();

                    auto possx = particleAttributes.GetRealData(4).data();
                    auto possy = particleAttributes.GetRealData(5).data();
                    auto possz = particleAttributes.GetRealData(6).data();
                    auto velsx = particleAttributes.GetRealData(7).data();
                    auto velsy = particleAttributes.GetRealData(8).data();
                    auto velsz = particleAttributes.GetRealData(9).data();

                    amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> jA;
                    amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;
                    amrex::GpuArray<amrex::Array4<amrex::Real>, int(vdim / 2.5) * 2 + 1> bA;

                    for (int cc = 0; cc < vdim; cc++)
                    {
                        jA[cc] = (*mw_yee.J_Array[cc])[pti].array();
                        eA[cc] = (*mw_yee.HE_Array[cc])[pti].array();
                    }
                    for (int cc = 0; cc < int(vdim / 2.5) * 2 + 1; cc++)
                    {
                        bA[cc] = (*m_B_stage[cc])[pti].array();
                    }

                    // loop over particles: add contribution of old particle position to J
                    // and push
                    amrex::ParallelFor(
                        np,
                        [=] AMREX_GPU_DEVICE(long pp)
                        {
                            splines_at_particles<degx, degy, degz> spline;
                            // Read out particle position
                            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> pos;
                            amrex::GpuArray<amrex::Real, vdim> vel{velx[pp], vely[pp], velz[pp]};
                            amrex::GpuArray<amrex::Real, vdim> vels;
                            if (s == 0)
                            {
                                for (int xd = 0; xd < GEMPIC_SPACEDIM; xd++)
                                    pos[xd] = partData[pp].pos(xd);
                                vels[0] = velx[pp];
                                vels[1] = vely[pp];
                                vels[2] = velz[pp];
                            }
                            else
                            {
                                pos[0] = possx[pp];
                                pos[1] = possy[pp];
                                pos[2] = possz[pp];
                                vels[0] = velsx[pp];
                                vels[1] = velsy[pp];
                                vels[2] = velsz[pp];
                            }

                            spline.init_particles(pos, plo, dxi);
                            // evaluate the electric field
                            amrex::GpuArray<amrex::Real, vdim> efield =
                                evaluate_efield<vdim, degx, degy, degz>(spline, eA);
                            amrex::GpuArray<amrex::Real, vdim> bfield =
                                evaluate_efield<vdim, degx, degy, degz>(spline, bA);
                            // push v with the electric field

                            //   Accumulation of j
                            gempic_accumulate_j<vdim, degx, degy, degz>(charge, vels, spline, jA,
                                                                        inv_vol * weight[pp]);
                            //   update the solution
                            amrex::GpuArray<amrex::Real, vdim> newvel;
                            newvel[0] = vel[0] +
                                        chargemass * dt * m_coeffs_b[s] *
                                            (efield[0] + vels[1] * bfield[2] - vels[2] * bfield[1]);
                            newvel[1] = vel[1] +
                                        chargemass * dt * m_coeffs_b[s] *
                                            (efield[1] + vels[2] * bfield[0] - vels[0] * bfield[2]);
                            newvel[2] = vel[2] +
                                        chargemass * dt * m_coeffs_b[s] *
                                            (efield[2] + vels[0] * bfield[1] - vels[1] * bfield[0]);

                            for (int xd = 0; xd < GEMPIC_SPACEDIM; xd++)
                            {
                                pos[xd] = partData[pp].pos(xd);
                                partData[pp].pos(xd) = pos[xd] + dt * m_coeffs_b[s] * vels[xd];
                            }
                            velx[pp] = newvel[0];
                            vely[pp] = newvel[1];
                            velz[pp] = newvel[2];
                            if (s < stages - 1)
                            {
                                // update where to evaluate in next stage
                                newvel[0] = vel[0] + chargemass * dt * m_coeffs_a[s] *
                                                         (efield[0] + vels[1] * bfield[2] -
                                                          vels[2] * bfield[1]);
                                newvel[1] = vel[1] + chargemass * dt * m_coeffs_a[s] *
                                                         (efield[1] + vels[2] * bfield[0] -
                                                          vels[0] * bfield[2]);
                                newvel[2] = vel[2] + chargemass * dt * m_coeffs_a[s] *
                                                         (efield[2] + vels[0] * bfield[1] -
                                                          vels[1] * bfield[0]);

                                possx[pp] = pos[0] + dt * m_coeffs_a[s] * vels[0];
                                possy[pp] = pos[1] + dt * m_coeffs_a[s] * vels[1];
                                possz[pp] = pos[2] + dt * m_coeffs_a[s] * vels[2];
                                velsx[pp] = newvel[0];
                                velsy[pp] = newvel[1];
                                velsz[pp] = newvel[2];
                            }
                        });
                }
                part_gr[spec]->Redistribute();
            }
            // Update the fields
            Gempic_SumBoundary(&mw_yee,
                               &infra);  // Sums Boundary of J and Overwrites overlap
            for (int i = 0; i < vdim; i++)
            {
                mw_yee.J_Array[i]->FillBoundary(infra.geom.periodicity());
            }
            // Copy the E and B solutions
            if (s < stages - 1)
            {
                for (int cc = 0; cc < 3; cc++)
                {
                    amrex::MultiFab::Copy(*m_E_stage[cc], (*mw_yee.E_Array[cc]), 0, 0, 1,
                                          mw_yee.Nghost);
                    amrex::MultiFab::Copy(*m_B_stage[cc], (*mw_yee.B_Array[cc]), 0, 0, 1,
                                          mw_yee.Nghost);
                }
            }
            mw_yee.advance_E(infra, dt * m_coeffs_b[s], electromagnetic, true, mw_yee.HB_Array,
                             mw_yee.E_Array);
            if (s < stages - 1)
            {
                mw_yee.advance_E(infra, dt * m_coeffs_a[s], electromagnetic, true, mw_yee.HB_Array,
                                 m_E_stage);
            }

            if (electromagnetic)
            {
                mw_yee.advance_B(infra, dt * m_coeffs_b[s], mw_yee.HE_Array, mw_yee.B_Array);
                if (s < stages - 1)
                {
                    mw_yee.advance_B(infra, dt * m_coeffs_a[s], mw_yee.HE_Array, m_B_stage);

                    // Operator H_B for dt/2
                    mw_yee.template hodge_full<degmw>(infra, m_B_stage, mw_yee.HB_Array, false);
                }
                else
                {
                    // Operator H_B for dt/2
                    mw_yee.template hodge_full<degmw>(infra, mw_yee.B_Array, mw_yee.HB_Array,
                                                      false);
                }
            }
            if (s < stages - 1)
            {
                mw_yee.template hodge_full<degmw>(infra, m_E_stage, mw_yee.HE_Array, true);
            }
            else
            {
                mw_yee.template hodge_full<degmw>(infra, mw_yee.E_Array, mw_yee.HE_Array, true);
            }
        }
    }

    template <int degx, int degy, int degz, int degmw, int ndata = 1 + 2 * GEMPIC_SPACEDIM,
              bool electromagnetic = true>
    void time_loop(computational_domain infra, maxwell_yee<vdim> &mw_yee,
                   amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec> &part_gr,
                   diagnostics<vdim, numspec, degx, degy, degz, degmw> *diagn, bool ctest,
                   std::string simulation_name, int strang_order)
    {
        timers profiling_timers(false);

        for (int t_step = 0; t_step < mw_yee.nsteps; t_step++)
        {
            amrex::Print() << "time step: " << t_step << std::endl;
            time_loop_step<degx, degy, degz, degmw, ndata, electromagnetic>(infra, mw_yee, part_gr);
            diagn->template end_of_timestep<ndata>(&profiling_timers, t_step, infra, &mw_yee,
                                                   part_gr, simulation_name, ctest);
        }
    }
};
}  // namespace Time_Loop

}  // namespace Gempic

#endif
