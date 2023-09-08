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
#include <GEMPIC_particle_mesh_coupling.H>
#include <GEMPIC_profiling.H>
#include <GEMPIC_Fields.H>
#include <GEMPIC_FDDeRhamComplex.H>
#include <GEMPIC_PoissonSolver.H>
#include <GEMPIC_Splitting.H>

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


    /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    *
    * ampere_step
    * 
    * (solve Ampere-Maxwell)
    * modified from apply_H_b (/!\ dt)
    * mcp: should be moved to an upper class
    * 
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
    inline void ampere_step(std::shared_ptr<FDDeRhamComplex> deRham,
                          DeRhamField<Grid::primal, Space::face>& B,
                          DeRhamField<Grid::dual, Space::face>& D,
                          DeRhamField<Grid::dual, Space::face>& J,
                          DeRhamField<Grid::dual, Space::edge>& H,          // mcp: implicit variable in deRham ?
                          DeRhamField<Grid::dual, Space::face>& auxDualF2,  // mcp: implicit variable in deRham ?
                          const amrex::Real dt
    )
    {
        // Operator H_B for dt/2   MCP: here dt not dt/2
        deRham -> hodgeFD<degmw>(B, H);
        deRham -> curl(H, auxDualF2);
        auxDualF2 *= dt;
        D += auxDualF2;   // mcp: why not   D += dt*auxDualF2; ?
        D -= dt*J;     // mcp: clearer if J = J and not dt*J ...
    }



    /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    *
    * faraday_step
    * 
    * (solve Faraday-Maxwell)
    * modified from apply_H_e_field (/!\ dt)
    * mcp: should be moved to an upper class
    * 
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

    inline void faraday_step(std::shared_ptr<FDDeRhamComplex> deRham,
                                DeRhamField<Grid::primal, Space::face>& B,
                                DeRhamField<Grid::dual, Space::face>& D,    
                                DeRhamField<Grid::primal, Space::edge>& E,              // mcp: implicit variable in deRham ?
                                DeRhamField<Grid::primal, Space::face>& auxPrimalF2,    // mcp: implicit variable in deRham ?
                                const amrex::Real dt
    )
    {
        deRham -> hodgeFD<degmw>(D, E);
        deRham -> curl(E, auxPrimalF2);
        auxPrimalF2 *= dt;
        B -= auxPrimalF2;
    }



    /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    *
    * v_boris_kick 
    * 
    * update single particle velocity with Boris method
    * 
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

    private:
        // mcp: need to return an array ?
        AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM + vdim> v_boris_kick(
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> pos,
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> plo,
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> phi, amrex::GpuArray<amrex::Real, vdim> vel,
            amrex::Real dt,
            amrex::Real chargemass,  // charge/mass
            amrex::GpuArray<amrex::Real, vdim> &Ep,
            amrex::GpuArray<amrex::Real, int(vdim / 2.5) * 2 + 1> &Bp
        )
        {
            amrex::GpuArray<amrex::Real, vdim> vm;
            amrex::GpuArray<amrex::Real, vdim> vt;
            amrex::GpuArray<amrex::Real, vdim> vp;
            amrex::Real S;

            // compute the E and B fields at the particle position

            // evaluate the spline: 
            // (mcp: (?) name not very clear)
            spline.init_particles(pos, infra.plo, infra.dxi);
            
            // evaluate the electric field
            amrex::GpuArray<amrex::Real, vdim> Ep =
                evaluate_efield<vdim, degx, degy, degz>(spline, eA);

            // evaluate the magnetic field
            amrex::GpuArray<amrex::Real, int(vdim / 2.5)* 2 + 1> Bp =
                evaluate_bfield<vdim, degx, degy, degz>(spline, bA);

            // v-
            for (int vd = 0; vd < vdim; vd++)
            {
                vm[vd] = vel[vd] + 0.5 * dt * chargemass * Ep[vd];
            }

            // rotate single particle velocity with B field:   v-  ->  v+
            if (vdim == 1)
            {
                vp[0] = vm[0];
            }
            else if (vdim == 2)
            {
                amrex::Real alpha = chargemass * dt / 2 * Bp[0];
                vp[0] = (1 - pow(alpha, 2)) / (1 + pow(alpha, 2)) * vm[0] +
                        2 * alpha / (1 + pow(alpha, 2)) * vm[1];
                vp[1] = -2 * alpha / (1 + pow(alpha, 2)) * vm[0] +
                        (1 - pow(alpha, 2)) / (1 + pow(alpha, 2)) * vm[1];
            }
            else if (vdim == 3)
            {
                // v'
                amrex::Real coeff_t = dt * chargemass / 2;
                vt[0] = vm[0] + coeff_t * (vm[1] * Bp[2] - vm[2] * Bp[1]);
                vt[1] = vm[1] + coeff_t * (vm[2] * Bp[0] - vm[0] * Bp[2]);
                vt[2] = vm[2] + coeff_t * (vm[0] * Bp[1] - vm[1] * Bp[0]);

                S = pow(dt * chargemass / 2 * Bp[0], 2.) + pow(dt * chargemass / 2 * Bp[1], 2.) +
                    pow(dt * chargemass / 2 * Bp[2], 2.);

                // v+
                amrex::Real coeff = dt * chargemass / (1. + S);
                vp[0] = vm[0] + coeff * (vt[1] * Bp[2] - vt[2] * Bp[1]);
                vp[1] = vm[1] + coeff * (vt[2] * Bp[0] - vt[0] * Bp[2]);
                vp[2] = vm[2] + coeff * (vt[0] * Bp[1] - vt[1] * Bp[0]);
            }


            // v
            for (int vd = 0; vd < vdim; vd++)
            {
                vel[vd] = vp[vd] + 0.5 * dt * chargemass * Ep[vd];
            }

            // need to return an array ?            
        }


    /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    *
    * x_push_and_J_deposit_zigzag 
    * 
    * update single particle position with zig-zag method 
    * and deposit charge-conserving current
    * 
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

    private:
        AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM + vdim> x_push_and_J_deposit_zigzag(
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> pos,
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> plo,
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> phi, amrex::GpuArray<amrex::Real, vdim> vel,
            amrex::Real dt,
            amrex::Real chargemass,  // charge/mass
            amrex::GpuArray<amrex::Real, vdim> &Ep,
            amrex::GpuArray<amrex::Real, int(vdim / 2.5) * 2 + 1> &Bp
        )
        {
        
            amrex::GpuArray<amrex::Real, std::max(degx, std::max(degy, degz)) + 4> primitive;
            // push x1
            x_push_and_J_deposit_1di<0>(pos, vel, infra, primitive, spline, dx, jA, chargemass, weight, dt);

            if (vdim > 1)
            {
#if GEMPIC_SPACEDIM > 1
            // push x2
            x_push_and_J_deposit_1di<1>(pos, vel, infra, primitive, spline, bfields, dx, jA, bA, chargemass, weight, dt);
#endif
            }
            if (vdim > 2)
            {
#if GEMPIC_SPACEDIM == 3
            // push x3
            x_push_and_J_deposit_1di<2>(pos, vel, infra, primitive, spline, bfields, dx, jA, bA, chargemass, weight, dt);
#endif
            }
        }


    /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    *
    * x_push_and_J_deposit_1di
    * 
    * update single particle position along 1 arbitrary direction 
    * and deposit charge-conserving current
    * 
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */


    // mcp: modified from apply_H_p_i
    template<int xidx>
    AMREX_GPU_HOST_DEVICE inline void x_push_and_J_deposit_1di(amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> &pos,
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

        // mcp: here we should call a modified version of 'accumulate_j_update_v_C2' below,
        // with no velocity update (and no B field needed)
        accumulate_j_update_v_C2<splines_at_particles<degx, degy, degz>,
                                    vdim, degrees[xidx], degrees[(xidx+1)%3],
                                    degrees[(xidx+2)%3], xidx, std::max(degx, std::max(degy, degz)) + 4>
                                    (spline, spline_new, spline_old, weight, dx, bA, jA, bfields, primitive);

        // if (vdim > 1) vel[(xidx+1)%3] = vel[(xidx+1)%3] - chargemass * bfields[0];
        // if (vdim > 2) vel[(xidx+2)%3] = vel[(xidx+2)%3] + chargemass * bfields[1];
        pos[xidx] = x_new;
    }






    /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    *
    * global_boris_push_and_deposit 
    * 
    * update all the particles position and velocities
    * and deposit charge-conserving current
    * 
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */


    // mcp: modified from "treat_particles"   > 2 different functions (eg global_boris_v_kick and global_x_push_and_J_deposit ?)
    public:
    inline void global_boris_push_and_deposit(std::shared_ptr<FDDeRhamComplex> deRham,
                            DeRhamField<Grid::primal, Space::edge>& E,
                            DeRhamField<Grid::primal, Space::face>& B,
                            DeRhamField<Grid::dual, Space::face>& D,
                            computational_domain infra,
                            const amrex::Real dt,
                            amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec>& partGr
    ) override final
    {
        // push particles following Boris' method, and deposit the current in a charge-conserving way
        // here the same deposition as in the HS_zigzag scheme is used (we should add more details) 

        for (int spec = 0; spec < numspec; spec++)
        {
            amrex::Real chargemass = partGr[spec]->getCharge() / partGr[spec]->getMass();
            amrex::Real charge = partGr[spec]->getCharge();

            for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*partGr[spec], 0); pti.isValid(); ++pti)
            {

                // mcp: what is this loop ?

                // 0. declaration and initialization of some variables before the loop on particle index pp
                //    (not exactly sure what is needed here)

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

                // amrex::Print() << "np: " << np << std::endl;

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

                    // mcp: discuss the proper arguments for these functions

                    v_boris_kick(pos, particles, infra, bfields ??, bA ??, spline, vel, velx, vely, velz, chargemass, dt, pp);
                    
                    x_push_and_J_deposit_zigzag(pos, vel, infra, primitive, spline, dx, jA, chargemass, weight, dt);


                });

            }
            partGr[spec] -> Redistribute();   // mcp: needed ?
        }
    }







    /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    *
    * time_step 
    * 
    * apply Boris method to push (update) all particles
    * (includes charge-conserving current deposition)
    * and solve Maxwell with leap-frog scheme
    * 
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */




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

        // 1. Faraday update (Dt/2)
        // B^{n} -> B^{n+1/2}
    
        faraday_step(deRham, B, D, 0.5*dt); 

        // 2. Boris particle push and J deposition
        // (x^n, v^n) -> (x^{n+1}, v^{n+1}, J^{n+1/2}) 
    
        // mcp: use a common terminology for 'global' and 'single-particle' functions ?
        global_boris_push_and_deposit(deRham, B, D, J, partGr, dt)      
        
        // 3. Ampere split update (Dt)
        // D^{n} -> D^{n+1}
    
        ampere_step(deRham, B, D, J, dt); 

        // 4. Faraday update (Dt/2)
        // B^{n+1/2} -> B^{n+1}
    
        faraday_step(deRham, B, D, 0.5*dt); 
        
    }
};

} // namespace Time_Loop
} // namespace Gempic
#endif