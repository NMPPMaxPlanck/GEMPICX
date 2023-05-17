#ifndef TIME_LOOP_HS_ZIGZAG_
#define TIME_LOOP_HS_ZIGZAG_

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
#include <GEMPIC_SumBoundary.H>
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_Params.H>
#include <GEMPIC_diagnostics.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_mesh_coupling_C2.H>
#include <GEMPIC_PlotFile.H>
#include <GEMPIC_profiling.H>
#include <GEMPIC_Fields.H>
#include <GEMPIC_FDDeRhamComplex.H>
#include <GEMPIC_PoissonSolver.H>

using namespace Diagnostics_Output;
using namespace Field_solvers;
using namespace CompDom;
using namespace Particles;
using namespace Profiling;
using namespace GEMPIC_Fields;
using namespace GEMPIC_FDDeRhamComplex;
using namespace GEMPIC_PoissonSolver;
namespace Gempic
{
namespace Time_Loop
{
template <int vdim, int numspec, int degx, int degy, int degz, int degmw, int ndata,
          bool electromagnetic = true, bool profiling = true>
void time_loop_hs_zigzag_step(std::shared_ptr<FDDeRhamComplex> deRham,
                              DeRhamField<Grid::dual, Space::cell>& rho,
                              DeRhamField<Grid::primal, Space::edge>& E,
                              DeRhamField<Grid::primal, Space::face>& B,
                              DeRhamField<Grid::dual, Space::face>& D,
                              DeRhamField<Grid::dual, Space::edge>& H,
                              DeRhamField<Grid::dual, Space::face>& J,
                              DeRhamField<Grid::primal, Space::face>& auxPrimalF2,
                              DeRhamField<Grid::primal, Space::face>& auxPrimalF2_2,
                              DeRhamField<Grid::dual, Space::face>& auxDualF2,
                              DeRhamField<Grid::dual, Space::face>& auxDualF2_2,
                              computational_domain infra,
                              const amrex::Real dt,
                              amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec>& partGr
                             )
{

    // for (amrex::MFIter mfi(E.data[0]); mfi.isValid(); ++mfi)
    // {
    //     const amrex::Box &bx = mfi.validbox();
    //     amrex::Array4<amrex::Real> const &EMF = (E.data[0])[mfi].array();

    //     ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) { amrex::Print() << "("<< i << "," << j << "," << k <<
    //              ") E: " << EMF(i, j, k) << std::endl; });
    // }
    
    // Operator H_B for dt/2
    // Solve Ampere-Maxwell
    deRham -> hodgeFD<degmw>(B, H);
    deRham -> curl(H, auxDualF2);
    auxDualF2 *= 0.5*dt;
    D += auxDualF2;

    // H_E from t_0 to t_0 + dt/2)
    // Solve Faraday's Law
    deRham -> hodgeFD<degmw>(D, E);
    deRham -> curl(E, auxPrimalF2);
    auxPrimalF2 *= 0.5*dt;
    B -= auxPrimalF2;

    
    // Deposit particles in J and push particles: H_p = H_p1 + H_p2 + H_p3
    for (int comp = 0; comp < GEMPIC_SPACEDIM; ++comp)
                (J.data[comp]).setVal(0.0, 0);

    J.fillBoundary();


    for (int spec = 0; spec < numspec; spec++)
    {
        amrex::Real chargemass = partGr[spec]->getCharge() / partGr[spec]->getMass();
        amrex::Real charge = partGr[spec]->getCharge();

        for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*partGr[spec], 0); pti.isValid(); ++pti)
        {
            amrex::Particle<0, 0>* AMREX_RESTRICT particles =
                &(pti.GetArrayOfStructs()[0]);  // get particles
            const long np = pti.numParticles();

            auto particle_attributes = &pti.GetStructOfArrays();
            amrex::ParticleReal* const AMREX_RESTRICT part_weight =
                particle_attributes->GetRealData(vdim).data();
            amrex::ParticleReal* const AMREX_RESTRICT velx =
                particle_attributes->GetRealData(0).data();
            amrex::ParticleReal* const AMREX_RESTRICT vely =
                particle_attributes->GetRealData(1).data();
            amrex::ParticleReal* const AMREX_RESTRICT velz =
                particle_attributes->GetRealData(2).data();

            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> jA;
            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;
            amrex::GpuArray<amrex::Array4<amrex::Real>, int(vdim / 2.5) * 2 + 1> bA;

            //amrex::Print() << "np: " << np << std::endl;

            // Fill boundary of J or it will not add the contributions of the ghost cells in accumulate_J.
            
            for (int cc = 0; cc < vdim; cc++)
            {
                jA[cc] = (J.data[cc])[pti].array();
                eA[cc] = (E.data[cc])[pti].array();
            }

            //amrex::Print() << "eA[0] = " << eA[0](0, 0, 0) << std::endl;
            
            // Fill boundary of B to calculate the B content of the primitive in accumulate_J
            //B.fillBoundary();
            for (int cc = 0; cc < int(vdim / 2.5) * 2 + 1; cc++)
            {
                bA[cc] = (B.data[cc])[pti].array();
            }

            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const dx = {AMREX_D_DECL(infra.geom.CellSize()[0], infra.geom.CellSize()[1], infra.geom.CellSize()[2])};
            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp)
            {

                    splines_at_particles<degx, degy, degz> spline;
                    spline1d_at_particles<degx> spline_new_x;
                    spline1d_at_particles<degx> spline_old_x;
                    spline1d_at_particles<degy> spline_new_y;
                    spline1d_at_particles<degy> spline_old_y;
                    spline1d_at_particles<degz> spline_new_z;
                    spline1d_at_particles<degz> spline_old_z;
                    // Read out particle position
                    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> pos;
                    for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                        pos[d] = particles[pp].pos(d);

                    // evaluate the spline
                    spline.init_particles(pos, infra.plo, infra.dxi);
                    // evaluate the electric field

                    amrex::GpuArray<amrex::Real, vdim> efield =
                        evaluate_efield<vdim, degx, degy, degz>(spline, eA);

                    // push v with the electric field
                    amrex::GpuArray<amrex::Real, vdim> vel{velx[pp], vely[pp], velz[pp]};
                    amrex::GpuArray<amrex::Real, vdim> newPosE =
                        push_v_efield<vdim>(vel, dt * 0.5, chargemass, efield);

                    velx[pp] = newPosE[0];
                    vely[pp] = newPosE[1];
                    velz[pp] = newPosE[2];

                    // amrex::Real weight =
                    // charge*particles[pp].rdata(vdim)*infra.dxi[GEMPIC_SPACEDIM];
                    amrex::Real weight = charge * part_weight[pp] * infra.dxi[GEMPIC_SPACEDIM];

                    // for (unsigned int d=0;d<vdim;++d)
                    // vel[d] = particles[pp].rdata(d);
                    //  vel[d] = particle_attributes->GetRealData(d)[pp];
                    vel[0] = velx[pp];
                    vel[1] = vely[pp];
                    vel[2] = velz[pp];

                    amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};
                    // push x1
                    amrex::Real x1_new = pos[0] + dt * 0.5 * vel[0];

                    spline_new_x.init_position(x1_new, infra.plo[0], infra.dxi[0]);
                    spline_old_x.init_position(pos[0], infra.plo[0], infra.dxi[0]);
                    spline.template init_position<0, degx>(x1_new, infra.plo[0], infra.dxi[0]);

                    amrex::GpuArray<amrex::Real, std::max(degx, std::max(degy, degz)) + 4>
                        primitive;

                    accumulate_j_update_v_C2<splines_at_particles<degx, degy, degz>, vdim, degx,
                                             degy, degz, 0,
                                             std::max(degx, std::max(degy, degz)) + 4>(
                        spline, spline_new_x, spline_old_x, weight, dx, bA, jA, bfields, primitive);

                    // Push v2 and v3
                    if (vdim > 1) vel[1] = vel[1] - chargemass * bfields[0];
                    if (vdim > 2) vel[2] = vel[2] + chargemass * bfields[1];
                    pos[0] = x1_new;

                    if (vdim > 1)
                    {
#if GEMPIC_SPACEDIM > 1
                        // push x2
                        amrex::Real x2_new = pos[1] + dt * 0.5 * vel[1];

                        spline_new_y.init_position(x2_new, infra.plo[1], infra.dxi[1]);
                        spline_old_y.init_position(pos[1], infra.plo[1], infra.dxi[1]);
                        spline.template init_position<1, degy>(x2_new, infra.plo[1], infra.dxi[1]);

                        accumulate_j_update_v_C2<splines_at_particles<degx, degy, degz>, vdim, degy,
                                                 degz, degx, 1,
                                                 std::max(degx, std::max(degy, degz)) + 4>(
                            spline, spline_new_y, spline_old_y, weight, dx, bA, jA, bfields,
                            primitive);

                        // Push v1 and v3
                        vel[0] = vel[0] + chargemass * bfields[1];
                        if (vdim > 2) vel[2] = vel[2] - chargemass * bfields[0];
                        pos[1] = x2_new;
#endif
                    }
                    if (vdim > 2)
                    {
#if GEMPIC_SPACEDIM == 3
                        // push x3
                        amrex::Real x3_new = pos[2] + dt * vel[2];

                        spline_new_z.init_position(x3_new, infra.plo[2], infra.dxi[2]);
                        spline_old_z.init_position(pos[2], infra.plo[2], infra.dxi[2]);
                        spline.template init_position<2, degz>(x3_new, infra.plo[2], infra.dxi[2]);

                        accumulate_j_update_v_C2<splines_at_particles<degx, degy, degz>, vdim, degz,
                                                 degx, degy, 2,
                                                 std::max(degx, std::max(degy, degz)) + 4>(
                            spline, spline_new_z, spline_old_z, weight, dx, bA, jA, bfields,
                            primitive);

                        // Push v1 and v2
                        vel[0] = vel[0] - chargemass * bfields[0];
                        vel[1] = vel[1] + chargemass * bfields[1];
                        pos[2] = x3_new;
#endif
                    }

                    if (vdim > 1)
                    {
#if GEMPIC_SPACEDIM > 1
                        // push x2
                        amrex::Real x2_new = pos[1] + dt * 0.5 * vel[1];

                        spline_new_y.init_position(x2_new, infra.plo[1], infra.dxi[1]);
                        spline_old_y.init_position(pos[1], infra.plo[1], infra.dxi[1]);
                        spline.template init_position<1, degy>(x2_new, infra.plo[1], infra.dxi[1]);

                        accumulate_j_update_v_C2<splines_at_particles<degx, degy, degz>, vdim, degy,
                                                 degz, degx, 1,
                                                 std::max(degx, std::max(degy, degz)) + 4>(
                            spline, spline_new_y, spline_old_y, weight, dx, bA, jA, bfields,
                            primitive);

                        // Push v1 and v3
                        vel[0] = vel[0] + chargemass * bfields[1];
                        if (vdim > 2) vel[2] = vel[2] - chargemass * bfields[0];
                        pos[1] = x2_new;
#endif
                    }

                    // second half step of Hp1
                    x1_new = pos[0] + dt * 0.5 * vel[0];
                    spline_new_x.init_position(x1_new, infra.plo[0], infra.dxi[0]);
                    spline_old_x.init_position(pos[0], infra.plo[0], infra.dxi[0]);
                    spline.template init_position<0, degx>(x1_new, infra.plo[0], infra.dxi[0]);

                    accumulate_j_update_v_C2<splines_at_particles<degx, degy, degz>, vdim, degx,
                                             degy, degz, 0,
                                             std::max(degx, std::max(degy, degz)) + 4>(
                        spline, spline_new_x, spline_old_x, weight, dx, bA, jA, bfields, primitive);

                    // Push v2 and v3
                    if (vdim > 1) vel[1] = vel[1] - chargemass * bfields[0];
                    if (vdim > 2) vel[2] = vel[2] + chargemass * bfields[1];
                    pos[0] = x1_new;

                    for (int xd = 0; xd < GEMPIC_SPACEDIM; xd++) particles[pp].pos(xd) = pos[xd];
                    // std::copy(pos.begin(),pos.end(),&particles.data()[pp*(GEMPIC_SPACEDIM+vdim+2)]);
                    // for (int vd=0;vd<vdim; vd++)
                    //    particle_attributes->GetRealData(vd)[pp] = vel[vd];
                    velx[pp] = vel[0];
                    vely[pp] = vel[1];
                    velz[pp] = vel[2];
                    // exit(0);

                });

        }
        partGr[spec] -> Redistribute();
    }

    // Sum boundary of J and overwrite overlap with averageSync
    // Then Fill Boundary of J
    auto nGhost = deRham -> getNGhost();
    for (int comp = 0; comp < 3; ++comp)
#if (GEMPIC_SPACEDIM == 3)
        (J.data[comp]).SumBoundary(0, 1, {nGhost[0], nGhost[1], nGhost[2]}, {0, 0, 0}, infra.geom.periodicity());
#endif


    J.averageSync();
    J.fillBoundary();

    // Advance E from j (field part of H_p). DO NOT MULTIPLY BY DT, ACCUMULATE_J ALREADY INTEGRATES J
    
    //J *= dt;
    D -= J;

    // Second particle loop for particle part from H_E
    for (int spec = 0; spec < numspec; spec++)
    {
        amrex::Real chargemass = partGr[spec]->getCharge() / partGr[spec]->getMass();

        for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*partGr[spec], 0); pti.isValid(); ++pti)
        {
            amrex::Particle<0, 0>* AMREX_RESTRICT particles = &(pti.GetArrayOfStructs()[0]);
            // amrex::ArrayOfStructs<4, 0, amrex::DefaultAllocator>*  AMREX_RESTRICT particles =
            // pti.GetArrayOfStructs(); // get particles
            const long np = pti.numParticles();
            auto particle_attributes = &pti.GetStructOfArrays();
            amrex::ParticleReal* const AMREX_RESTRICT velx =
                particle_attributes->GetRealData(0).data();
            amrex::ParticleReal* const AMREX_RESTRICT vely =
                particle_attributes->GetRealData(1).data();
            amrex::ParticleReal* const AMREX_RESTRICT velz =
                particle_attributes->GetRealData(2).data();

            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;

            // Extract E, not D ?
            for (int cc = 0; cc < vdim; cc++)
            {
                eA[cc] = (E.data[cc])[pti].array();
            }

            // loop over particles: add contribution of old particle position to J and push
            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp)
            {
                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> pos;
                for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                    pos[d] = particles[pp].pos(d);

                splines_at_particles<degx, degy, degz> splines;
                // evaluate the spline
                splines.init_particles(pos, infra.plo, infra.dxi);
                // evaluate the electric field
                amrex::GpuArray<amrex::Real, vdim> efield =
                    evaluate_efield<vdim, degx, degy, degz>(splines, eA);
                // push v with the electric field
                amrex::GpuArray<amrex::Real, vdim> vel{velx[pp], vely[pp], velz[pp]};
                amrex::GpuArray<amrex::Real, vdim> newPosE =
                    push_v_efield<vdim>(vel, dt * 0.5, chargemass, efield);

                velx[pp] = newPosE[0];
                vely[pp] = newPosE[1];
                velz[pp] = newPosE[2];

            });
        }
        partGr[spec] -> Redistribute();
    }


    deRham -> hodgeFD<degmw>(D, E);
    // Operator H_E for dt/2
    deRham -> curl(E, auxPrimalF2_2);
    auxPrimalF2_2 *= 0.5*dt;
    B -= auxPrimalF2_2;

    // Operator H_B for dt/2
    deRham -> hodgeFD<degmw>(B, H);
    deRham -> curl(H, auxDualF2_2);
    auxDualF2_2 *= 0.5*dt;
    D += auxDualF2_2;

}


template <int vdim, int numspec, int degx, int degy, int degz, int degmw, int ndata,
          bool electromagnetic = true, bool profiling = true, bool output = true>
void time_loop_hs_zigzag(std::shared_ptr<FDDeRhamComplex> deRham,
                         DeRhamField<Grid::dual, Space::cell>& rho,
                         DeRhamField<Grid::primal, Space::edge>& E,
                         DeRhamField<Grid::primal, Space::face>& B,
                         DeRhamField<Grid::dual, Space::face>& D,
                         DeRhamField<Grid::dual, Space::edge>& H,
                         DeRhamField<Grid::dual, Space::face>& J,
                         computational_domain infra,
                         const amrex::Real dt,
                         amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec>& partGr,
                         const int nSteps,
                         int strang_order)
{
    BL_PROFILE("time_loop_hs_zigzag_C2");

    //bool profiling_count = false;
    //timers profiling_timers(true);
    DeRhamField<Grid::primal, Space::face> auxPrimalF2(deRham);
    DeRhamField<Grid::primal, Space::face> auxPrimalF2_2(deRham);
    DeRhamField<Grid::dual, Space::face> auxDualF2(deRham);
    DeRhamField<Grid::dual, Space::face> auxDualF2_2(deRham);

    /*
    for (int comp = 0; comp < 3; ++comp)
    {
        amrex::Print() << "infra_plo_mein: " << infra.plo[comp] << std::endl;
        amrex::Print() << "infra_dxi_mein: " << infra.dxi[comp] << std::endl;
    }
    */

    // if (profiling_timers.profiling) profiling_timers.counter_all -= MPI_Wtime();
    for (int t_step = 0; t_step < nSteps; t_step++)
    {
        BL_PROFILE("time_loop_hs_zigzag_C2::step");

        // Rescale the fields
        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const dr = {AMREX_D_DECL(infra.geom.CellSize()[0], infra.geom.CellSize()[1], infra.geom.CellSize()[2])};
        //(rho.data).mult((dr[0] * dr[1] * dr[2]));

        (E.data[0]).mult((1/dr[0]));
        (E.data[1]).mult((1/dr[1]));
        (E.data[2]).mult((1/dr[2]));

        (D.data[0]).mult(1/(dr[1] * dr[2]));
        (D.data[1]).mult(1/(dr[0] * dr[2]));
        (D.data[2]).mult(1/(dr[0] * dr[1]));

        //(J.data[0]).mult(1/(dr[1] * dr[2]));
        //(J.data[1]).mult(1/(dr[0] * dr[2]));
        //(J.data[2]).mult(1/(dr[0] * dr[1]));

        (B.data[0]).mult(1/(dr[1] * dr[2]));
        (B.data[1]).mult(1/(dr[0] * dr[2]));
        (B.data[2]).mult(1/(dr[0] * dr[1]));

        (H.data[0]).mult(1/(dr[0]));
        (H.data[1]).mult(1/(dr[1]));
        (H.data[2]).mult(1/(dr[2]));


        //if (profiling_timers.profiling && t_step == 0) profiling_timers.counter_all -= MPI_Wtime();
        time_loop_hs_zigzag_step<vdim, numspec, degx, degy, degz, degmw, ndata,
                                    electromagnetic, profiling>(deRham, rho, E, B, D, H, J, auxPrimalF2, auxPrimalF2_2, auxDualF2, auxDualF2_2, infra, dt, partGr);

        amrex::Real D_rms = 0.0;
        for (amrex::MFIter mfi(D.data[0]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            amrex::Array4<amrex::Real> const &DMF = (D.data[0])[mfi].array();

            for (int i = lo.x; i <= hi.x; ++i) 
                D_rms = DMF(i, 0, 0) * DMF(i, 0, 0);
        }
        D_rms = std::sqrt((1./16.)*D_rms);

        

        amrex::Print() << "finished time-step: " << t_step << " D_rms: " << D_rms << std::endl;
    }

    
}


}
}
#endif
