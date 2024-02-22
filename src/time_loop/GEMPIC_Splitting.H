#ifndef GEMPIC_SPLITTING_
#define GEMPIC_SPLITTING_

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
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_mesh_coupling.H>
#include <GEMPIC_profiling.H>
#include <GEMPIC_Fields.H>
#include <GEMPIC_FDDeRhamComplex.H>
#include <GEMPIC_PoissonSolver.H>


namespace Gempic {

namespace Time_Loop {
using namespace CompDom;
using namespace Particles;
using namespace Profiling;
using namespace GEMPIC_Fields;
using namespace GEMPIC_FDDeRhamComplex;
using namespace GEMPIC_PoissonSolver;

/**
 *  Algorithms to solve the ODE systems associated with a splitting of the semi-discrete Vlasov-Maxwell Hamiltonian
 * 
 *  H(X,V,D,B) = 1/2 V^T W_m V + 1/2 D^T tHH_2 D + 1/2 B^T HH_2 B
 *  
 *  where:
 * 
 *   X contains the particle positions x_p 
 *   V contains the particle velocities v_p 
 *   D contains the dual face dofs of the electric field 
 *   B contains the primal face dofs of the magnetic field 
 * 
 *   (X, V, D, B are stacked as column vectors, ^T denotes their transpose)
 * 
 *   W_m   is the diagonal matrix with the particle masses m_p
 *   tHH_2 is the dual Hodge_2 matrix (mapping dual face dofs to primal edge dofs)
 *   HH_2  is the primal Hodge_2 matrix (mapping primal face dofs to dual edge dofs)
 * 
 *  The Hamiltonain is split in three main parts: H = H_kin + H_e + H_b 
 *  
 *  - the kinetic part is further split in directional parts:
 *      H_kin = 1/2 V^T W_m V = sum_{i = 1 .. vdim} H_kin_i
 *    where each term
 *      H_kin_i = sum_{p} m_p/2 (v_p_i)**2
 *    leads to the ODE system
 * 
 *      (d/dt)x_p_i     = v_p_i                             (H_kin_i_x)
 *      (d/dt)v_p_{i-1} =  (q/m)_p v_p_i B_{i+1}(x_p)       (H_kin_i_viminus)
 *      (d/dt)v_p_i     = 0                                 (H_kin_i_vi)
 *      (d/dt)v_p_{i+1} = -(q/m)_p v_p_i B_{i-1}(x_p)       (H_kin_i_viplus)
 *      (d/dt) D_i  = - J_i                                 (H_kin_i_D)
 *      (d/dt) B  = 0                                       (H_kin_i_B)
 *      
 *    where D_i (and J_i) contain the dofs of D (and J) on dual faces normal to direction i 
 * 
 *  - the electric part 
 *      H_e = 1/2 D^T tHH_2 D
 *    leads to the ODE system
 * 
 *      (d/dt)x_p = 0                                       (He_x)
 *      (d/dt)v_p = (q/m)_p E(x_p)                          (He_v)
 *      (d/dt) D  = 0                                       (He_D)
 *      (d/dt) B  = -CC E                                   (He_B)
 *  
 *    where CC is the matrix of the primal curl operator (mapping primal edge dofs to primal face dofs)
 * 
 *  - the magnetic part 
 *      H_b = 1/2 B^T HH_2 B
 *    leads to the ODE system
 * 
 *      (d/dt)x_p = 0                                       (Hb_x)
 *      (d/dt)v_p = 0                                       (Hb_v)
 *      (d/dt) D  = tCC B                                   (Hb_D)
 *      (d/dt) B  = 0                                       (Hb_B)
 *  
 *    where tCC is the matrix of the dual curl operator (mapping dual edge dofs to dual face dofs)
*/
template<unsigned int vdim, int degx, int degy, int degz, int degmw>
class OperatorHamilton {
    public:

/**
* @brief solves equation (He_vp) for one given particle
*  
* @tparam SplineType : spline class, for unit testing purposes
* 
* @param eA : array of E field coefficients (primal edge dofs)
* @param spline : object containing the values of non-zero grid splines at the particle position
* @param vel : velocity of the given particle
* @param velx, vely, velz : arrays with velocities of all the particles
* @param chargemass : (q/m)_p ratio
* @param dt : time step
* @param pp : index of the given particle
* 
*/
    template<class SplineType = Spline::SplineBase<degx, degy, degz>>
    AMREX_GPU_HOST_DEVICE inline void apply_H_e_particle (
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const &eA,
        SplineType &spline,
        amrex::GpuArray<amrex::Real, vdim> vel,
        amrex::ParticleReal* const AMREX_RESTRICT velx,
        amrex::ParticleReal* const AMREX_RESTRICT vely,
        amrex::ParticleReal* const AMREX_RESTRICT velz,
        amrex::Real const chargemass,
        amrex::Real const dt,
        const long pp
    ) const
    {
        // evaluate the electric field
        amrex::GpuArray<amrex::Real, vdim> efield =
            spline.template evalSplineField<Field::PrimalOneForm>(eA);

        // push v with the electric field
        push_v_efield(vel, dt, chargemass, efield);

        velx[pp] = vel[xDir];
        vely[pp] = vel[yDir];
        velz[pp] = vel[zDir];
    }

/**
 * @brief computes different fields in order to solve the ODE system (H_kin_i) for one given particle and direction i
 * 
 * Specifically, this function:
 *  - (1) pushes x_p with constant velocity along direction i
 *  - (2) deposits the current associated with this trajectory
 *  - (3) computes the integrated B field along this trajectory \<B\>_p = int_{0}^dt B(x_p(t)) dt 
 *  - (4) pushes v_p on directions i-1 and i+1 using \<B\>_p
 *   
 * Equation (H_kin_i_D) is only solved after this function has been called for all particles:
 * then the full current density has been computed and the full electric field can be updated.
 * 
 * Also, the new particle position and velocities are not updated in the global particle arrays.
 * This must be done outside this function.
 * 
 * @tparam dir : direction of the splitting (starting at dir = 0)
 * 
 * @param pos : position of the given particle
 * @param vel : velocity of the given particle
 * @param infra : computational domain
 * @param spline : object containing the values of non-zero grid splines at the particle position
 * @param bfields : temporary array to store the values of the integrated B field
 * @param dx : cell size in every direction
 * @param jA : array of J density coefficients (dual face dofs)
 * @param bA : array of B field coefficients (primal face dofs)
 * @param chargemass : (q/m)_p ratio
 * @param chargeWeight : weight * charge of the given particle
 * @param dt : time step
 * 
*/
    template<Direction dir, class SplineType = Spline::SplineBase<degx, degy, degz>>
    AMREX_GPU_HOST_DEVICE inline void apply_H_p_i (
                            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> &pos,
                            amrex::GpuArray<amrex::Real, vdim> &vel,
                            computational_domain const &infra,
                            SplineType &spline,
                            amrex::GpuArray<amrex::Real, 2> bfields,
                            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const dx, 
                            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const &jA,
                            amrex::GpuArray<amrex::Array4<amrex::Real>, int(vdim / 2.5) * 2 + 1> const &bA,
                            amrex::Real const chargemass,
                            amrex::Real const chargeWeight, 
                            amrex::Real const dt
    ) const
    {
        amrex::Real x_new = pos[dir] + dt * vel[dir];

        spline.template update1DSplines<dir>(x_new, infra.plo[dir], infra.dxi[dir]);
        spline.template update1DPrimitive<dir>(x_new, infra.plo[dir], infra.dxi[dir]);

        accumulate_J_integrate_B<dir>(spline, chargeWeight, dx, bA, jA, bfields);

        if (vdim > 1) vel[(dir+1)%3] = vel[(dir+1)%3] - chargemass * bfields[0];
        if (vdim > 2) vel[(dir+2)%3] = vel[(dir+2)%3] + chargemass * bfields[1];
        pos[dir] = x_new;
    }

/**
 * @brief solves the ODE (Hb_D) corresponding to H_b part of the Hamiltonian
 * 
 * @param deRham : object describing the discrete de Rham sequence
 * @param B : magnetic field as primal 2-form (input of the function)
 * @param D : electric field as dual 2-form (modified by the function)
 * @param auxDualH : magnetic field as dual 1-form (temporary field)
 * @param auxDualF2 : temporary field as dual 2-form
 * @param dt : time step
 * 
*/
    inline void apply_H_b (std::shared_ptr<FDDeRhamComplex> deRham,
                          const DeRhamField<Grid::primal, Space::face>& B,
                          DeRhamField<Grid::dual, Space::face>& D,
                          DeRhamField<Grid::dual, Space::edge>& auxDualH,
                          DeRhamField<Grid::dual, Space::face>& auxDualF2,
                          const amrex::Real dt
    )
    {
        // Operator H_B for dt/2
        // Solve Ampere-Maxwell
        deRham->hodge(B, auxDualH);
        deRham -> curl(auxDualH, auxDualF2);
        auxDualF2 *= dt;
        D += auxDualF2;
    }

/**
 * @brief solves equation (He_B), ie the Faraday equation
 * 
 * @param deRham : object describing the discrete de Rham sequence
 * @param auxE : electric field as primal 1-form (temporary field)
 * @param B : magnetic field as primal 2-form (modified by the function)
 * @param D : electric field as dual 2-form (input of the function)
 * @param auxPrimalF2 : temporary field as primal 2-form
 * @param dt : time step
 * 
*/
    inline void apply_H_e_field (std::shared_ptr<FDDeRhamComplex> deRham,
                                DeRhamField<Grid::primal, Space::edge>& auxE,
                                DeRhamField<Grid::primal, Space::face>& B,
                                DeRhamField<Grid::dual, Space::face>& D,
                                DeRhamField<Grid::primal, Space::face>& auxPrimalF2,
                                const amrex::Real dt
    )
    {
        deRham->hodge(D, auxE);
        // Operator H_E for dt/2
        deRham -> curl(auxE, auxPrimalF2);
        auxPrimalF2 *= dt;
        B -= auxPrimalF2;
    }
};

template<unsigned int vdim, unsigned int numspec, int degx, int degy, int degz, int degmw,
         unsigned int ndata, bool electromagnetic, bool profiling>
class Splitting {

    public:
    virtual void time_loop (std::shared_ptr<FDDeRhamComplex> deRham,
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
                   OperatorHamilton<vdim, degx, degy, degz, degmw> operatorHamilton) = 0;

};

}  // namespace Time_Loop

}  // namespace Gempic

#endif
