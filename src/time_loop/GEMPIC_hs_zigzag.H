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
#include <GEMPIC_Splitting.H>

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
template<int vdim, int numspec, int degx, int degy, int degz, int degmw, int ndata,
         bool electromagnetic = true, bool profiling = true, bool output = true>
class HSZigZagC2: public HamiltonianSplitting<vdim, numspec, degx, degy, degz, degmw, ndata, electromagnetic, profiling>
{
    public:
    inline void treat_particles(std::shared_ptr<FDDeRhamComplex> deRham,
                                DeRhamField<Grid::primal, Space::edge>& E,
                                DeRhamField<Grid::primal, Space::face>& B,
                                DeRhamField<Grid::dual, Space::face>& D,
                                DeRhamField<Grid::dual, Space::face>& J,
                                computational_domain infra,
                                const amrex::Real dt,
                                amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec>& partGr
    ) override final
    {
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

                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const dx = {infra.geom.CellSize()[0], infra.geom.CellSize()[1], infra.geom.CellSize()[2]};
                amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp)
                {
                    amrex::GpuArray<amrex::Real, vdim> vel{velx[pp], vely[pp], velz[pp]};

                    splines_at_particles<degx, degy, degz> spline;
                    // Read out particle position
                    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> pos;
                    for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                        pos[d] = particles[pp].pos(d);

                    apply_H_e_particle(pos, particles, infra, eA, spline, vel, velx, vely, velz, chargemass, dt, pp);

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

                        amrex::GpuArray<amrex::Real, std::max(degx, std::max(degy, degz)) + 4>
                            primitive;
                        // push x1
                        apply_H_p_i<0>(pos, vel, infra, primitive, spline, bfields, dx, jA, bA, chargemass, weight, dt);

                        if (vdim > 1)
                        {
    #if GEMPIC_SPACEDIM > 1
                            // push x2
                            apply_H_p_i<1>(pos, vel, infra, primitive, spline, bfields, dx, jA, bA, chargemass, weight, dt);
    #endif
                        }
                        if (vdim > 2)
                        {
    #if GEMPIC_SPACEDIM == 3
                            // push x3
                            apply_H_p_i<2>(pos, vel, infra, primitive, spline, bfields, dx, jA, bA, chargemass, weight, dt);
    #endif
                        }

                        if (vdim > 1)
                        {
    #if GEMPIC_SPACEDIM > 1
                            // push x2
                            apply_H_p_i<1>(pos, vel, infra, primitive, spline, bfields, dx, jA, bA, chargemass, weight, dt);
    #endif
                        }

                        // second half step of Hp1
                        apply_H_p_i<0>(pos, vel, infra, primitive, spline, bfields, dx, jA, bA, chargemass, weight, dt);

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

                    amrex::GpuArray<amrex::Real, vdim> vel{velx[pp], vely[pp], velz[pp]};

                    splines_at_particles<degx, degy, degz> spline;

                    apply_H_e_particle(pos, particles, infra, eA, spline, vel, velx, vely, velz, chargemass, dt, pp);

                });
            }
            partGr[spec] -> Redistribute();
        }
    }

    inline void apply_H_e_particle(amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> &pos,
                                   amrex::Particle<0, 0>* AMREX_RESTRICT particles,
                                   computational_domain const &infra,
                                   amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const &eA,
                                   splines_at_particles<degx, degy, degz> &spline,
                                   amrex::GpuArray<amrex::Real, vdim> vel,
                                   amrex::ParticleReal* const AMREX_RESTRICT velx,
                                   amrex::ParticleReal* const AMREX_RESTRICT vely,
                                   amrex::ParticleReal* const AMREX_RESTRICT velz,
                                   amrex::Real const chargemass,
                                   amrex::Real const dt,
                                   const long pp
    )
    {
        // evaluate the spline
        spline.init_particles(pos, infra.plo, infra.dxi);
        // evaluate the electric field

        amrex::GpuArray<amrex::Real, vdim> efield =
            evaluate_efield<vdim, degx, degy, degz>(spline, eA);

        // push v with the electric field
        amrex::GpuArray<amrex::Real, vdim> newPosE =
            push_v_efield<vdim>(vel, dt * 0.5, chargemass, efield);

        velx[pp] = newPosE[0];
        vely[pp] = newPosE[1];
        velz[pp] = newPosE[2];
    }

    template<int xidx>
    inline void apply_H_p_i(amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> &pos,
                            amrex::GpuArray<amrex::Real, vdim> &vel,
                            computational_domain const &infra,
                            amrex::GpuArray<amrex::Real, std::max(degx, std::max(degy, degz)) + 4> &primitive,
                            splines_at_particles<degx, degy, degz> &spline,amrex::GpuArray<amrex::Real, 2> bfields,
                            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const dx, 
                            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const &jA,
                            amrex::GpuArray<amrex::Array4<amrex::Real>, int(vdim / 2.5) * 2 + 1> const &bA,
                            amrex::Real const chargemass,
                            amrex::Real const weight, 
                            amrex::Real const dt
    )
    {
        constexpr int degrees[3] = {degx, degy, degz};

        amrex::Real x_new = pos[xidx] + dt * 0.5 * vel[xidx];

        spline1d_at_particles<degrees[xidx]> spline_new;
        spline1d_at_particles<degrees[xidx]> spline_old;

        spline_new.init_position(x_new, infra.plo[xidx], infra.dxi[xidx]);
        spline_old.init_position(pos[xidx], infra.plo[xidx], infra.dxi[xidx]);
        spline.template init_position<xidx, degrees[xidx]>(x_new, infra.plo[xidx], infra.dxi[xidx]);

        accumulate_j_update_v_C2<splines_at_particles<degx, degy, degz>,
                                    vdim, degrees[xidx], degrees[(xidx+1)%3],
                                    degrees[(xidx+2)%3], xidx, std::max(degx, std::max(degy, degz)) + 4>
                                    (spline, spline_new, spline_old, weight, dx, bA, jA, bfields, primitive);


        if (vdim > 1) vel[(xidx+1)%3] = vel[(xidx+1)%3] - chargemass * bfields[0];
        if (vdim > 2) vel[(xidx+2)%3] = vel[(xidx+2)%3] + chargemass * bfields[1];
        pos[xidx] = x_new;
    }

    private:
    inline void apply_H_b(std::shared_ptr<FDDeRhamComplex> deRham,
                          DeRhamField<Grid::primal, Space::face>& B,
                          DeRhamField<Grid::dual, Space::face>& D,
                          DeRhamField<Grid::dual, Space::edge>& H,
                          DeRhamField<Grid::dual, Space::face>& auxDualF2,
                          const amrex::Real dt
    )
    {
        // Operator H_B for dt/2
        // Solve Ampere-Maxwell
        deRham -> hodgeFD<degmw>(B, H);
        deRham -> curl(H, auxDualF2);
        auxDualF2 *= 0.5*dt;
        D += auxDualF2;
    }

    inline void apply_H_e_field(std::shared_ptr<FDDeRhamComplex> deRham,
                                DeRhamField<Grid::primal, Space::edge>& E,
                                DeRhamField<Grid::primal, Space::face>& B,
                                DeRhamField<Grid::dual, Space::face>& D,
                                DeRhamField<Grid::primal, Space::face>& auxPrimalF2,
                                const amrex::Real dt
    )
    {
        deRham -> hodgeFD<degmw>(D, E);
        // Operator H_E for dt/2
        deRham -> curl(E, auxPrimalF2);
        auxPrimalF2 *= 0.5*dt;
        B -= auxPrimalF2;
    }

    virtual void time_step(std::shared_ptr<FDDeRhamComplex> deRham,
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
    ) override
    {

        apply_H_b(deRham, B, D, H, auxDualF2, dt);

        apply_H_e_field(deRham, E, B, D, auxPrimalF2, dt);

        treat_particles(deRham, E, B, D, J, infra, dt, partGr);

        apply_H_e_field(deRham, E, B, D, auxPrimalF2_2, dt);

        apply_H_b(deRham, B, D, H, auxDualF2_2, dt);

    }
};

} // namespace Time_Loop
} // namespace Gempic
#endif