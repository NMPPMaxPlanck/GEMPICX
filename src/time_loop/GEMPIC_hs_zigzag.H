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
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_Params.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_profiling.H>
#include <GEMPIC_Fields.H>
#include <GEMPIC_FDDeRhamComplex.H>
#include <GEMPIC_PoissonSolver.H>
#include <GEMPIC_Splitting.H>
#include <GEMPIC_Spline_Class.H>

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

template<int vdim, int numspec, int degx, int degy, int degz, int degmw, int ndata,
         bool electromagnetic = true, bool profiling = true, bool output = true>
class HSZigZagC2: public Splitting<vdim, numspec, degx, degy, degz, degmw, ndata, electromagnetic, profiling>
{
    public:
    virtual void time_loop(std::shared_ptr<FDDeRhamComplex> deRham,
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
                            int strang_order,
                            OperatorHamilton<vdim, degx, degy, degz, degmw> operatorHamilton) override
    {
        BL_PROFILE("time_loop_hs_zigzag_C2");

        //bool profiling_count = false;
        //timers profiling_timers(true);
        DeRhamField<Grid::primal, Space::face> auxPrimalF2(deRham);
        DeRhamField<Grid::primal, Space::face> auxPrimalF2_2(deRham);
        DeRhamField<Grid::dual, Space::face> auxDualF2(deRham);
        DeRhamField<Grid::dual, Space::face> auxDualF2_2(deRham);

        // if (profiling_timers.profiling) profiling_timers.counter_all -= MPI_Wtime();
        for (int t_step = 0; t_step < nSteps; t_step++)
        {
            BL_PROFILE("time_loop_hs_zigzag_C2::step");

            // Rescale the fields
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const dr = {infra.geom.CellSize()[0], infra.geom.CellSize()[1], infra.geom.CellSize()[2]};
            //(rho.data).mult((dr[0] * dr[1] * dr[2]));

            (E.data[0]).mult((1/dr[0]));
            (E.data[1]).mult((1/dr[1]));
            (E.data[2]).mult((1/dr[2]));

            (D.data[0]).mult(1/(dr[1] * dr[2]));
            (D.data[1]).mult(1/(dr[0] * dr[2]));
            (D.data[2]).mult(1/(dr[0] * dr[1]));

            (B.data[0]).mult(1/(dr[1] * dr[2]));
            (B.data[1]).mult(1/(dr[0] * dr[2]));
            (B.data[2]).mult(1/(dr[0] * dr[1]));

            (H.data[0]).mult(1/(dr[0]));
            (H.data[1]).mult(1/(dr[1]));
            (H.data[2]).mult(1/(dr[2]));

            operatorHamilton.apply_H_b(deRham, B, D, H, auxDualF2, 0.5*dt);

            operatorHamilton.apply_H_e_field(deRham, E, B, D, auxPrimalF2, 0.5*dt);

            // start treat particles
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

                    for (int cc = 0; cc < vdim; cc++)
                    {
                        jA[cc] = (J.data[cc])[pti].array();
                        eA[cc] = (E.data[cc])[pti].array();
                    }

                    //amrex::Print() << "eA[0] = " << eA[0](0, 0, 0) << std::endl;

                    //B.fillBoundary();
                    for (int cc = 0; cc < int(vdim / 2.5) * 2 + 1; cc++)
                    {
                        bA[cc] = (B.data[cc])[pti].array();
                    }

                    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const dx = {infra.geom.CellSize()[0], infra.geom.CellSize()[1], infra.geom.CellSize()[2]};
                    amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp)
                    {
                        amrex::GpuArray<amrex::Real, vdim> vel{velx[pp], vely[pp], velz[pp]};

                        // Read out particle position
                        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> pos;
                        for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                            pos[d] = particles[pp].pos(d);

                        Spline::SplineWithPrimitive<degx, degy, degz> spline(pos, infra.plo, infra.dxi);

                        operatorHamilton.apply_H_e_particle(pos, particles, infra, eA, spline, vel, velx, vely, velz, chargemass, 0.5*dt, pp);

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
                            operatorHamilton.template apply_H_p_i<0>(pos, vel, infra, spline, bfields, dx, jA, bA, chargemass, weight, 0.5*dt);

                            if (vdim > 1)
                            {
        #if GEMPIC_SPACEDIM > 1
                                // push x2
                                operatorHamilton.template apply_H_p_i<1>(pos, vel, infra, spline, bfields, dx, jA, bA, chargemass, weight, 0.5*dt);
        #endif
                            }
                            if (vdim > 2)
                            {
        #if GEMPIC_SPACEDIM == 3
                                // push x3
                                operatorHamilton.template apply_H_p_i<2>(pos, vel, infra, spline, bfields, dx, jA, bA, chargemass, weight, dt);
        #endif
                            }

                            if (vdim > 1)
                            {
        #if GEMPIC_SPACEDIM > 1
                                // push x2
                                operatorHamilton.template apply_H_p_i<1>(pos, vel, infra, spline, bfields, dx, jA, bA, chargemass, weight, 0.5*dt);
        #endif
                            }

                            // second half step of Hp1
                        operatorHamilton.template apply_H_p_i<0>(pos, vel, infra, spline, bfields, dx, jA, bA, chargemass, weight, 0.5*dt);

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
            // evaluate the spline

            // Then Fill Boundary of J
            auto nGhost = deRham -> getNGhost();
            for (int comp = 0; comp < 3; ++comp)
                (J.data[comp]).SumBoundary(0, 1, {nGhost[0], nGhost[1], nGhost[2]}, {0, 0, 0}, infra.geom.periodicity());

            J.averageSync();
            J.fillBoundary();

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

                        amrex::GpuArray<amrex::Real, vdim> vel{velx[pp], vely[pp], velz[pp]};
                        
                        Spline::SplineBase<degx, degy, degz> spline(pos, infra.plo, infra.dxi);

                        operatorHamilton.apply_H_e_particle(pos, particles, infra, eA, spline, vel, velx, vely, velz, chargemass, 0.5*dt, pp);

                    });
                }
                partGr[spec] -> Redistribute();
            }
            // end treat particles

            operatorHamilton.apply_H_e_field(deRham, E, B, D, auxPrimalF2_2, 0.5*dt);

            operatorHamilton.apply_H_b(deRham, B, D, H, auxDualF2_2, 0.5*dt);

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
};

} // namespace Time_Loop
} // namespace Gempic
#endif