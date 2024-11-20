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

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_ParticleGroups.H"
#include "GEMPIC_PoissonSolver.H"
#include "GEMPIC_Profiling.H"
#include "GEMPIC_SplineClass.H"
#include "GEMPIC_Splitting.H"

namespace Gempic::TimeLoop
{

template <unsigned int vdim,
          int degx,
          int degy,
          int degz,
          unsigned int ndata,
          bool electromagnetic = true,
          bool profiling = true,
          bool output = true>
class HSZigZagC2 : public Splitting<vdim, degx, degy, degz, ndata, electromagnetic, profiling>
{
public:
    void time_loop (std::shared_ptr<FDDeRhamComplex> deRham,
                    DeRhamField<Grid::dual, Space::cell>& rho,
                    DeRhamField<Grid::primal, Space::edge>& E,
                    DeRhamField<Grid::primal, Space::face>& B,
                    DeRhamField<Grid::dual, Space::face>& D,
                    DeRhamField<Grid::dual, Space::edge>& H,
                    DeRhamField<Grid::dual, Space::face>& J,
                    ComputationalDomain infra,
                    const amrex::Real dt,
                    std::vector<std::unique_ptr<Particle::ParticleGroups<vdim, ndata>>>& partGr,
                    const int nSteps,
                    int strangOrder,
                    OperatorHamilton<vdim, degx, degy, degz> operatorHamilton) override
    {
        BL_PROFILE("Gempic::TimeLoop::HSZigZagC2::time_loop()");

        // bool profiling_count = false;
        // timers profiling_timers(true);
        DeRhamField<Grid::primal, Space::face> auxPrimalF2(deRham);
        DeRhamField<Grid::primal, Space::face> auxPrimalF22(deRham);
        DeRhamField<Grid::dual, Space::face> auxDualF2(deRham);
        DeRhamField<Grid::dual, Space::face> auxDualF22(deRham);

        // if (profiling_timers.profiling) profiling_timers.counter_all -= MPI_Wtime();
        for (int tStep = 0; tStep < nSteps; tStep++)
        {
            BL_PROFILE("time_loop_hs_zigzag_C2::step");

            // Rescale the fields
            amrex::GpuArray<amrex::Real, 3> const dr = {infra.m_geom.CellSize()[xDir],
                                                        infra.m_geom.CellSize()[yDir],
                                                        infra.m_geom.CellSize()[zDir]};
            //(rho.data).mult((dr[0] * dr[1] * dr[2]));

            (E.m_data[xDir]).mult((1 / dr[xDir]));
            (E.m_data[yDir]).mult((1 / dr[yDir]));
            (E.m_data[zDir]).mult((1 / dr[zDir]));

            (D.m_data[xDir]).mult(1 / (dr[yDir] * dr[zDir]));
            (D.m_data[yDir]).mult(1 / (dr[xDir] * dr[zDir]));
            (D.m_data[zDir]).mult(1 / (dr[xDir] * dr[yDir]));

            (B.m_data[xDir]).mult(1 / (dr[yDir] * dr[zDir]));
            (B.m_data[yDir]).mult(1 / (dr[xDir] * dr[zDir]));
            (B.m_data[zDir]).mult(1 / (dr[xDir] * dr[yDir]));

            (H.m_data[xDir]).mult(1 / (dr[xDir]));
            (H.m_data[yDir]).mult(1 / (dr[yDir]));
            (H.m_data[zDir]).mult(1 / (dr[zDir]));

            operatorHamilton.apply_H_b(deRham, B, D, H, auxDualF2, 0.5 * dt);

            operatorHamilton.apply_H_e_field(deRham, E, B, D, auxPrimalF2, 0.5 * dt);

            // start treat particles
            // Deposit particles in J and push particles: H_p = H_p1 + H_p2 + H_p3
            for (int comp = 0; comp < GEMPIC_SPACEDIM; ++comp)
            {
                (J.m_data[comp]).setVal(0.0, 0);
            }

            J.fill_boundary();

            for (auto& particleSpecies : partGr)
            {
                amrex::Real chargemass = particleSpecies->getCharge() / particleSpecies->getMass();
                amrex::Real charge = particleSpecies->getCharge();

                for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*particleSpecies, 0); pti.isValid();
                     ++pti)
                {
                    amrex::Particle<0, 0>* AMREX_RESTRICT particles =
                        &(pti.GetArrayOfStructs()[0]);  // get particles
                    const long np = pti.numParticles();

                    auto particleAttributes = &pti.GetStructOfArrays();
                    amrex::ParticleReal* const AMREX_RESTRICT partWeight =
                        particleAttributes->GetRealData(vdim).data();
                    amrex::ParticleReal* const AMREX_RESTRICT velx =
                        particleAttributes->GetRealData(0).data();
                    amrex::ParticleReal* const AMREX_RESTRICT vely =
                        particleAttributes->GetRealData(1).data();
                    amrex::ParticleReal* const AMREX_RESTRICT velz =
                        particleAttributes->GetRealData(2).data();

                    amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> jA;
                    amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;
                    amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> bA;

                    // amrex::Print() << "np: " << np << std::endl;

                    for (int cc = 0; cc < vdim; cc++)
                    {
                        jA[cc] = (J.m_data[cc])[pti].array();
                        eA[cc] = (E.m_data[cc])[pti].array();
                    }

                    // amrex::Print() << "eA[0] = " << eA[0](0, 0, 0) << std::endl;

                    // B.fillBoundary();
                    for (int cc = 0; cc < vdim; cc++)
                    {
                        bA[cc] = (B.m_data[cc])[pti].array();
                    }

                    amrex::ParallelFor(
                        np,
                        [=] AMREX_GPU_DEVICE(long pp)
                        {
                            amrex::GpuArray<amrex::Real, vdim> vel{velx[pp], vely[pp], velz[pp]};

                            // Read out particle position
                            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> pos;
                            for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                            {
                                pos[d] = particles[pp].pos(d);
                            }

                            ParticleMeshCoupling::SplineWithPrimitive<degx, degy, degz> spline(
                                pos, infra.m_plo, infra.m_dxi);

                            operatorHamilton.apply_H_e_particle(eA, spline, vel, velx, vely, velz,
                                                                chargemass, 0.5 * dt, pp);

                            // amrex::Real weight =
                            // charge*particles[pp].rdata(vdim)*infra.dxi[GEMPIC_SPACEDIM];
                            amrex::Real chargeWeight = charge * partWeight[pp];

                            // for (unsigned int d=0;d<vdim;++d)
                            // vel[d] = particles[pp].rdata(d);
                            //  vel[d] = particle_attributes->GetRealData(d)[pp];
                            vel[xDir] = velx[pp];
                            vel[yDir] = vely[pp];
                            vel[zDir] = velz[pp];

                            amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

                            // push x1
                            operatorHamilton.template apply_H_p_i<xDir>(
                                pos, vel, infra, spline, bfields, infra.m_dx, jA, bA, chargemass,
                                chargeWeight, 0.5 * dt);

#if GEMPIC_SPACEDIM > 1
                            if constexpr (vdim > 1)
                            {
                                // push x2
                                operatorHamilton.template apply_H_p_i<yDir>(
                                    pos, vel, infra, spline, bfields, infra.m_dx, jA, bA,
                                    chargemass, chargeWeight, 0.5 * dt);
                            }
#endif
#if GEMPIC_SPACEDIM == 3
                            if constexpr (vdim > 2)
                            {
                                // push x3
                                operatorHamilton.template apply_H_p_i<zDir>(
                                    pos, vel, infra, spline, bfields, infra.m_dx, jA, bA,
                                    chargemass, chargeWeight, dt);
                            }
#endif

#if GEMPIC_SPACEDIM > 1
                            if constexpr (vdim > 1)
                            {
                                // push x2
                                operatorHamilton.template apply_H_p_i<yDir>(
                                    pos, vel, infra, spline, bfields, infra.m_dx, jA, bA,
                                    chargemass, chargeWeight, 0.5 * dt);
                            }
#endif

                            // second half step of Hp1
                            operatorHamilton.template apply_H_p_i<xDir>(
                                pos, vel, infra, spline, bfields, infra.m_dx, jA, bA, chargemass,
                                chargeWeight, 0.5 * dt);

                            for (int xd = 0; xd < GEMPIC_SPACEDIM; xd++)
                            {
                                particles[pp].pos(xd) = pos[xd];
                            }
                            // std::copy(pos.begin(),pos.end(),&particles.data()[pp*(GEMPIC_SPACEDIM+vdim+2)]);
                            // for (int vd=0;vd<vdim; vd++)
                            //    particle_attributes->GetRealData(vd)[pp] = vel[vd];
                            velx[pp] = vel[xDir];
                            vely[pp] = vel[yDir];
                            velz[pp] = vel[zDir];
                            // exit(0);
                        });
                }
                particleSpecies->Redistribute();
            }

            // Sum boundary of J and overwrite overlap with averageSync, then Fill Boundary of J
            J.post_particle_loop_sync();

            // J *= dt;
            D -= J;

            // Second particle loop for particle part from H_E
            for (auto& particleSpecies : partGr)
            {
                amrex::Real chargemass = particleSpecies->getCharge() / particleSpecies->getMass();

                for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*particleSpecies, 0); pti.isValid();
                     ++pti)
                {
                    amrex::Particle<0, 0>* AMREX_RESTRICT particles = &(pti.GetArrayOfStructs()[0]);
                    // amrex::ArrayOfStructs<4, 0, amrex::DefaultAllocator>*  AMREX_RESTRICT
                    // particles = pti.GetArrayOfStructs(); // get particles
                    const long np = pti.numParticles();
                    auto particleAttributes = &pti.GetStructOfArrays();
                    amrex::ParticleReal* const AMREX_RESTRICT velx =
                        particleAttributes->GetRealData(0).data();
                    amrex::ParticleReal* const AMREX_RESTRICT vely =
                        particleAttributes->GetRealData(1).data();
                    amrex::ParticleReal* const AMREX_RESTRICT velz =
                        particleAttributes->GetRealData(2).data();

                    amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;

                    // Extract E, not D ?
                    for (int cc = 0; cc < vdim; cc++)
                    {
                        eA[cc] = (E.m_data[cc])[pti].array();
                    }

                    // loop over particles: add contribution of old particle position to J and push
                    amrex::ParallelFor(
                        np,
                        [=] AMREX_GPU_DEVICE(long pp)
                        {
                            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> pos;
                            for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                            {
                                pos[d] = particles[pp].pos(d);
                            }

                            amrex::GpuArray<amrex::Real, vdim> vel{velx[pp], vely[pp], velz[pp]};

                            ParticleMeshCoupling::SplineBase<degx, degy, degz> spline(
                                pos, infra.m_plo, infra.m_dxi);

                            operatorHamilton.apply_H_e_particle(eA, spline, vel, velx, vely, velz,
                                                                chargemass, 0.5 * dt, pp);
                        });
                }
                particleSpecies->Redistribute();
            }
            // end treat particles

            operatorHamilton.apply_H_e_field(deRham, E, B, D, auxPrimalF22, 0.5 * dt);

            operatorHamilton.apply_H_b(deRham, B, D, H, auxDualF22, 0.5 * dt);

            amrex::Real dRms = 0.0;
            for (amrex::MFIter mfi(D.m_data[xDir]); mfi.isValid(); ++mfi)
            {
                const amrex::Box& bx = mfi.validbox();
                const auto lo = lbound(bx);
                const auto hi = ubound(bx);

                amrex::Array4<amrex::Real> const& dmf = (D.m_data[xDir])[mfi].array();

                for (int i = lo.x; i <= hi.x; ++i)
                {
                    dRms = dmf(i, 0, 0) * dmf(i, 0, 0);
                }
            }
            dRms = std::sqrt((1. / 16.) * dRms);

            amrex::Print() << "finished time-step: " << tStep << " D_rms: " << dRms << '\n';
        }
    }
};

}  // namespace Gempic::TimeLoop

#endif