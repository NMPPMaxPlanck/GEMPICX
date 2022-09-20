#ifndef LOOP_PREPARATION_H_
#define LOOP_PREPARATION_H_

#include <AMReX.H>
#include <AMReX_BLProfiler.H>
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
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_diagnostics.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_parameters.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_mesh_coupling.H>
#include <GEMPIC_particle_mesh_coupling_C2.H>

using namespace Diagnostics_Output;
using namespace Field_solvers;
using namespace CompDom;
using namespace Particles;

namespace Gempic
{
namespace Time_Loop
{
template <int vdim, int numspec, int degx, int degy, int degz, int degmw, int ndata, bool output>
void loop_preparation_init(
    gempic_parameters<vdim, numspec> meanVelocity, computational_domain infra,
    maxwell_yee<vdim> *mw_yee,
    amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec> &part_gr)
{
    BL_PROFILE("Gempic::Time_Loop::loop_preparation_init");

    // initial rho by: deposition of particle charge
    // Set rho to 0
    // mw_yee->rho.setVal(0.0);
    // Not needed setVal(0.0) set all values including ghost cells to zero
    mw_yee->rho.setVal(0.0);  // value and component

    // Set J and E to 0: This is important in order not to have uninitialized arrays when calling
    // diagnostics
    for (int dim = 0; dim < vdim; dim++)
    {
        mw_yee->J_Array[dim]->setVal(0.0);

        mw_yee->E_Array[dim]->setVal(0.0);
    }

    (mw_yee->rho).setVal(0.0);

    // Deposit charges:
    for (int spec = 0; spec < numspec; spec++)
    {
        amrex::Real charge = part_gr[spec]->getCharge();

        for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*part_gr[spec], 0); pti.isValid(); ++pti)
        {
            const long np = pti.numParticles();
            const auto &particles = pti.GetArrayOfStructs();
            const auto partData = particles().data();
            const auto weight = pti.GetStructOfArrays().GetRealData(vdim).data();

            amrex::Array4<amrex::Real> const &rhoarr = mw_yee->rho[pti].array();
            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp) {
                // for (int pp=0;pp<np;pp++) {
                // gempic_deposit_rho<amrex::Particle<vdim+1>,vdim,degx,degy,degz>(particles[pp],
                // (*part_gr).charge[spec], rhoarr, infra.plo, infra.dxi);
                splines_at_particles<degx, degy, degz> spline;
                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
                for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                    position[d] = partData[pp].pos(d);
                spline.init_particles(position, infra.plo, infra.dxi);
                gempic_deposit_rho_C3<degx, degy, degz>(
                    spline, charge * infra.dxi[GEMPIC_SPACEDIM] * weight[pp], rhoarr);
            });
        }
    }

    // add values from ghost cells and periodic boundary
    mw_yee->rho.SumBoundary(0, 1, {mw_yee->Nghost, mw_yee->Nghost, mw_yee->Nghost}, {0, 0, 0},
                            infra.geom.periodicity());
    mw_yee->rho.OverrideSync(*(mw_yee->nodal_Mask), infra.geom.periodicity());
    mw_yee->rho.FillBoundary(infra.geom.periodicity());

    // initial E by rho
    amrex::MultiFab kx(convert(infra.grid, *mw_yee->E_Index[0]), infra.distriMap, 1,
                       mw_yee->Nghost);
    kx.setVal(1.0);
    amrex::MultiFab ky(convert(infra.grid, *mw_yee->E_Index[1]), infra.distriMap, 1,
                       mw_yee->Nghost);
    ky.setVal(1.0);
    amrex::MultiFab kz(convert(infra.grid, *mw_yee->E_Index[2]), infra.distriMap, 1,
                       mw_yee->Nghost);
    kz.setVal(1.0);

    mw_yee->solve_poisson(infra, 1e-11, meanVelocity.max_coarsening_level);
    if (degmw > 2)
    {
        mw_yee->template solve_poisson_CG<degmw>(mw_yee->rho, mw_yee->phi, kx, ky, kz, infra, 2,
                                                 1e-11, 100);
        mw_yee->template D_from_phi<degmw>(mw_yee->phi, mw_yee->E_Array, kx, ky, kz, infra);
    }

    mw_yee->template hodge_full<degmw>(infra, mw_yee->B_Array, mw_yee->HB_Array, false);
    mw_yee->template hodge_full<degmw>(infra, (mw_yee->E_Array), mw_yee->HE_Array, true);
}

template <int vdim, int numspec, int degx, int degy, int degz, int degmw, int ndata, bool output>
void loop_staggering(
    gempic_parameters<vdim, numspec> meanVelocity, computational_domain infra,
    maxwell_yee<vdim> *mw_yee,
    amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec> &part_gr,
    diagnostics<vdim, numspec, degx, degy, degz, degmw> *diagn, bool time_staggered)
{
    if (time_staggered)
    {
        amrex::Print() << "ATTENTION: method is time staggered" << std::endl;
        //------------------------------------------------------------------------------
        // approximate e^1/2
        mw_yee->template hodge_full<degmw>(infra, mw_yee->B_Array, mw_yee->HB_Array, false);
        mw_yee->advance_E(infra, 0.5 * mw_yee->dt, true, true, mw_yee->HB_Array, mw_yee->E_Array);
        //------------------------------------------------------------------------------
        // approximate x^1/2
        for (int spec = 0; spec < numspec; spec++)
        {
            for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*part_gr[spec], 0); pti.isValid(); ++pti)
            {
                const long np = pti.numParticles();
                auto &particles = pti.GetArrayOfStructs();
                auto partData = particles().data();
                auto &particleAttributes = pti.GetStructOfArrays();
                auto velx = particleAttributes.GetRealData(0).data();
                auto vely = particleAttributes.GetRealData(1).data();
                auto velz = particleAttributes.GetRealData(2).data();

                const amrex::Real dt = mw_yee->dt;
                amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp) {
                    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> pos;
                    for (int comp = 0; comp < GEMPIC_SPACEDIM; comp++)
                    {
                        pos[comp] = partData[pp].pos(comp);
                    }
                    amrex::GpuArray<amrex::Real, vdim> vel{velx[pp], vely[pp], velz[pp]};

                    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM + vdim> newPos =
                        push_particle_pos<vdim>(infra.plo, infra.phi, pos, vel, 0.5 * dt);
                    for (int xd = 0; xd < GEMPIC_SPACEDIM; xd++) partData[pp].pos(xd) = newPos[xd];
                });
            }
            part_gr[spec]->Redistribute();  // assign particles to the tile they are in
        }
    }

    //------------------------------------------------------------------------------
    // Save "step 0" data
    if (output)
    {
        int t_step = 0;
        diagn->ComputeReducedDiag(t_step, infra, mw_yee, part_gr);
        // save fields and particles at initial time
        int save_part = true;
        Gempic_WritePlotFile<vdim, numspec>(part_gr, mw_yee, &infra, diagn->sim_name, t_step,
                                            save_part);
    }
}

template <int vdim, int numspec, int degx, int degy, int degz, int degmw, int ndata, bool output>
void loop_preparation(
    gempic_parameters<vdim, numspec> meanVelocity, computational_domain infra,
    maxwell_yee<vdim> *mw_yee,
    amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec> part_gr,
    diagnostics<vdim, numspec, degx, degy, degz, degmw> *diagn, bool time_staggered,
    amrex::GpuArray<std::string, int(vdim / 2.5) * 2 + 1> fields_B)
{
    loop_preparation_init<vdim, numspec, degx, degy, degz, degmw, ndata, output>(
        meanVelocity, infra, mw_yee, part_gr);

    // initial B_field
    mw_yee->template initB<degmw>(fields_B, meanVelocity.k, infra);

    loop_staggering<vdim, numspec, degx, degy, degz, degmw, ndata, output>(
        meanVelocity, infra, mw_yee, part_gr, diagn, time_staggered);
}

template <int vdim, int numspec, int degx, int degy, int degz, int degmw, int ndata, bool output>
void loop_preparation(
    gempic_parameters<vdim, numspec> meanVelocity, computational_domain infra,
    maxwell_yee<vdim> *mw_yee,
    amrex::GpuArray<std::unique_ptr<particle_groups<vdim>>, numspec> &part_gr,
    diagnostics<vdim, numspec, degx, degy, degz, degmw> *diagn, bool time_staggered,
    amrex::Real (*function_to_project_0)(amrex::Real, amrex::Real, amrex::Real, amrex::Real),
    amrex::Real (*function_to_project_1)(amrex::Real, amrex::Real, amrex::Real, amrex::Real),
    amrex::Real (*function_to_project_2)(amrex::Real, amrex::Real, amrex::Real, amrex::Real))
{
    loop_preparation_init<vdim, numspec, degx, degy, degz, degmw, ndata, output>(
        meanVelocity, infra, mw_yee, part_gr);

    // initial B_field
    mw_yee->template initB<degmw>(function_to_project_0, function_to_project_1,
                                  function_to_project_2, infra);

    loop_staggering<vdim, numspec, degx, degy, degz, degmw, ndata, output>(
        meanVelocity, infra, mw_yee, part_gr, diagn, time_staggered);
}

template <int vdim, int numspec, int degx, int degy, int degz, int degmw, int ndata, bool output>
void loop_preparation(
    gempic_parameters<vdim, numspec> meanVelocity, computational_domain infra,
    maxwell_yee<vdim> *mw_yee,
    amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec> &part_gr,
    diagnostics<vdim, numspec, degx, degy, degz, degmw> *diagn, bool time_staggered,
    amrex::GpuArray<int, int(vdim / 2.5) * 2 + 1> funcSelect)
{
    loop_preparation_init<vdim, numspec, degx, degy, degz, degmw, ndata, output>(
        meanVelocity, infra, mw_yee, part_gr);

    // initial B_field
    mw_yee->template initB<degmw>(infra, funcSelect);

    loop_staggering<vdim, numspec, degx, degy, degz, degmw, ndata, output>(
        meanVelocity, infra, mw_yee, part_gr, diagn, time_staggered);
}

template <int vdim, int numspec, int degx, int degy, int degz, int degmw, int ndata, bool output>
void loop_preparation(
    gempic_parameters<vdim, numspec> meanVelocity, computational_domain infra,
    maxwell_yee<vdim> *mw_yee,
    amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec> &part_gr,
    diagnostics<vdim, numspec, degx, degy, degz, degmw> *diagn, bool time_staggered,
    amrex::ParserExecutor<4> BxEval, amrex::ParserExecutor<4> ByEval,
    amrex::ParserExecutor<4> BzEval)
{
    BL_PROFILE("loop_preparation  (...,parserExec)");
    loop_preparation_init<vdim, numspec, degx, degy, degz, degmw, ndata, output>(
        meanVelocity, infra, mw_yee, part_gr);

    // initial B_field
    mw_yee->template initB<degmw>(BxEval, ByEval, BzEval, infra);

    loop_staggering<vdim, numspec, degx, degy, degz, degmw, ndata, output>(
        meanVelocity, infra, mw_yee, part_gr, diagn, time_staggered);
}
}  // namespace Time_Loop
}  // namespace Gempic

#endif
