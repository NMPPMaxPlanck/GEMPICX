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

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_ParticleGroups.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_PoissonSolver.H"
#include "GEMPIC_Profiling.H"

namespace Gempic::TimeLoop
{
using namespace Forms;

/**
 *  Algorithms to solve the ODE systems associated with a splitting of the semi-discrete
 * Vlasov-Maxwell Hamiltonian
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
 *  The Hamiltonian is split in three main parts: H = H_kin + H_e + H_b
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
 *    where CC is the matrix of the primal curl operator (mapping primal edge dofs to primal face
 * dofs)
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
template <unsigned int vdim, int degx, int degy, int degz, int degmw>
class OperatorHamilton
{
public:
    /**
     * @brief solves equation (He_vp) for one given particle
     *
     * @tparam SplineType : spline class, for unit testing purposes
     *
     * @param vel[out] : velocity of the given particle
     * @param eA : array of E field coefficients (primal edge dofs)
     * @param spline : object containing the values of non-zero grid splines at the particle
     * position
     * @param chargeOverMass : (q/m)_p ratio
     * @param dt : time step
     *
     */
    template <class splineType = ParticleMeshCoupling::SplineBase<degx, degy, degz>>
    AMREX_GPU_HOST_DEVICE inline void apply_h_e_particle (
        amrex::GpuArray<amrex::Real, vdim>& vel,
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& eA,
        splineType const& spline,
        amrex::Real const chargeOverMass,
        amrex::Real const dt) const
    {
        BL_PROFILE("Gempic::TimeLoop::Forms::OperatorHamilton::apply_h_e_particle()");
        // evaluate the electric field
        amrex::GpuArray<amrex::Real, vdim> efield =
            spline.template eval_spline_field<Field::PrimalOneForm>(eA);

        // push v with the electric field
        for (int i = 0; i < vdim; i++)
        {
            vel[i] += dt * chargeOverMass * efield[i];
        }
    }

    /**
     * @brief computes different fields in order to solve the ODE system (H_kin_i) for one given
     * particle and direction i
     *
     * Specifically, this function:
     *  - (1) pushes x_p with constant velocity along direction i
     *  - (2) deposits the current associated with this trajectory
     *  - (3) computes the integrated B field along this trajectory \<B\>_p = int_{0}^dt B(x_p(t))
     * dt
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
     * @param pos[out] : position of the given particle
     * @param vel[out] : velocity of the given particle
     * @param infra : computational domain
     * @param spline : object containing the values of non-zero grid splines at the particle
     * position
     * @param bfields : temporary array to store the values of the integrated B field
     * @param dx : cell size in every direction
     * @param jA : array of J density coefficients (dual face dofs)
     * @param bA : array of B field coefficients (primal face dofs)
     * @param chargeOverMass : (q/m)_p ratio
     * @param chargeWeight : weight * charge of the given particle
     * @param dt : time step
     *
     */
    template <Direction dir, class splineType = ParticleMeshCoupling::SplineBase<degx, degy, degz>>
    AMREX_GPU_HOST_DEVICE inline void apply_h_p_i (
        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>& pos,
        amrex::GpuArray<amrex::Real, vdim>& vel,
        ComputationalDomain const& infra,
        splineType& spline,
        amrex::GpuArray<amrex::Real, 2>& bfields,
        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const dx,
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& jA,
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& bA,
        amrex::Real const chargeOverMass,
        amrex::Real const chargeWeight,
        amrex::Real const dt) const
    {
        BL_PROFILE("Gempic::TimeLoop::Forms::OperatorHamilton::apply_h_p_i()");

        amrex::Real posNew = pos[dir] + dt * vel[dir];
        spline.template update_1d_splines<dir>(posNew, infra.m_plo[dir], infra.m_dxi[dir]);
        ParticleMeshCoupling::accumulate_j_integrate_b<dir>(bfields, spline, chargeWeight, dx, bA,
                                                            jA);
        pos[dir] = posNew;

        // we assume vdim = 3
        vel[(dir + 1) % 3] -= chargeOverMass * bfields[0];
        vel[(dir + 2) % 3] += chargeOverMass * bfields[1];
    }

    /**
     * @brief computes different fields in order to solve the ODE system (H_kin_i) for one given
     * particle and direction i for lower spatial dimensions.
     *
     * If GEMPIC_SPACEDIM < 3, there are spatial directions, where the primitive of the univariate
     * spline S^{d_x-1} can not be computed. This happens, if the spatial dimension in this
     * direction does not exist. The current and velocities therefore have to be updated via an
     * explicit Euler step.
     *
     * Specifically, this function:
     *  - (1) deposits the current associated with this trajectory
     *  - (2) computes the integrated B field along this trajectory \<B\>_p = int_{0}^dt B(x_p(t))
     * dt
     *  - (3) pushes v_p on directions i-1 and i+1 using \<B\>_p
     *
     * Equation (H_kin_i_D) is only solved after this function has been called for all particles:
     * then the full current density has been computed and the full electric field can be updated.
     *
     * Also, the new particle position and velocities are not updated in the global particle arrays.
     * This must be done outside this function.
     *
     * @tparam dir : direction of the splitting (starting at dir = 0)
     *
     * @param vel[out] : velocity of the given particle
     * @param infra : computational domain
     * @param spline : object containing the values of non-zero grid splines at the particle
     * position
     * @param bfields : temporary array to store the values of the integrated B field
     * @param dx : cell size in every direction
     * @param jA : array of J density coefficients (dual face dofs)
     * @param bA : array of B field coefficients (primal face dofs)
     * @param chargeOverMass : (q/m)_p ratio
     * @param chargeWeight : weight * charge of the given particle
     * @param dt : time step
     *
     */
    template <Direction pDir, class splineType = ParticleMeshCoupling::SplineBase<degx, degy, degz>>
    AMREX_GPU_HOST_DEVICE inline void apply_h_p_i_euler (
        amrex::GpuArray<amrex::Real, vdim>& vel,
        ComputationalDomain const& infra,
        splineType& spline,
        amrex::GpuArray<amrex::Real, 2>& bfields,
        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const dx,
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& jA,
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& bA,
        amrex::Real const chargeOverMass,
        amrex::Real const chargeWeight,
        amrex::Real const dt) const
    {
        BL_PROFILE("Gempic::TimeLoop::Forms::OperatorHamilton::apply_h_p_i_euler()");

        amrex::Real const dtv = dt * vel[pDir];

        if constexpr (pDir == yDir)
        {
            accumulate_j_integrate_b_euler_y(bfields, spline, dtv, chargeWeight, bA, jA);
        }
        else if constexpr (pDir == zDir)
        {
            accumulate_j_integrate_b_euler_z(bfields, spline, dtv, chargeWeight, bA, jA);
        }

        vel[(pDir + 1) % 3] -= dtv * chargeOverMass * bfields[0];
        vel[(pDir + 2) % 3] += dtv * chargeOverMass * bfields[1];
    }

    /**
     * @brief computes different fields in order to solve the ODE system (H_p) for one particle
     *
     * @param pos[out] : position of the particle
     * @param vel[out] : velocity of the article
     * @param infra : computational domain
     * @param spline : object containing the values of non-zero grid splines at the particle
     * position
     * @param dx : cell size in every direction
     * @param jA : array of J density coefficients (dual face dofs)
     * @param bA : array of B field coefficients (primal face dofs)
     * @param chargeOverMass : (q/m)_p ratio
     * @param chargeWeight : weight * charge of the given particle
     * @param dt : time step
     *
     */
    template <class splineType = ParticleMeshCoupling::SplineBase<degx, degy, degz>>
    AMREX_GPU_HOST_DEVICE inline void apply_h_p (
        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>& pos,
        amrex::GpuArray<amrex::Real, vdim>& vel,
        ComputationalDomain const& infra,
        splineType& spline,
        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const dx,
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& jA,
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& bA,
        amrex::Real const chargeOverMass,
        amrex::Real const chargeWeight,
        amrex::Real const dt) const
    {
        BL_PROFILE("Gempic::TimeLoop::Forms::OperatorHamilton::apply_h_p_x()");
        amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

        apply_h_p_i<xDir>(pos, vel, infra, spline, bfields, infra.m_dx, jA, bA, chargeOverMass,
                          chargeWeight, 0.5 * dt);
#if GEMPIC_SPACEDIM == 3
        apply_h_p_i<yDir>(pos, vel, infra, spline, bfields, infra.m_dx, jA, bA, chargeOverMass,
                          chargeWeight, 0.5 * dt);
        apply_h_p_i<zDir>(pos, vel, infra, spline, bfields, infra.m_dx, jA, bA, chargeOverMass,
                          chargeWeight, dt);
        apply_h_p_i<yDir>(pos, vel, infra, spline, bfields, infra.m_dx, jA, bA, chargeOverMass,
                          chargeWeight, 0.5 * dt);

#elif GEMPIC_SPACEDIM == 2
        apply_h_p_i<yDir>(pos, vel, infra, spline, bfields, infra.m_dx, jA, bA, chargeOverMass,
                          chargeWeight, 0.5 * dt);
        apply_h_p_i_euler<zDir>(vel, infra, spline, bfields, infra.m_dx, jA, bA, chargeOverMass,
                                chargeWeight, dt);
        apply_h_p_i<yDir>(pos, vel, infra, spline, bfields, infra.m_dx, jA, bA, chargeOverMass,
                          chargeWeight, 0.5 * dt);

#elif GEMPIC_SPACEDIM == 1
        apply_h_p_i_euler<yDir>(vel, infra, spline, bfields, infra.m_dx, jA, bA, chargeOverMass,
                                chargeWeight, 0.5 * dt);
        apply_h_p_i_euler<zDir>(vel, infra, spline, bfields, infra.m_dx, jA, bA, chargeOverMass,
                                chargeWeight, dt);
        apply_h_p_i_euler<yDir>(vel, infra, spline, bfields, infra.m_dx, jA, bA, chargeOverMass,
                                chargeWeight, 0.5 * dt);
#endif
        apply_h_p_i<xDir>(pos, vel, infra, spline, bfields, infra.m_dx, jA, bA, chargeOverMass,
                          chargeWeight, 0.5 * dt);
    }

    /**
     * @brief solves the ODE (Hb_D) corresponding to H_b part of the Hamiltonian
     * which is the ampere equation without current
     *
     * @param D[out] : electric field as dual 2-form (modified by the function)
     * @param deRham : object describing the discrete de Rham sequence
     * @param B : magnetic field as primal 2-form
     * @param H : magnetic field as dual 1-form
     * @param dt : time step
     *
     */
    inline void apply_h_b (DeRhamField<Grid::dual, Space::face>& D,
                           std::shared_ptr<FDDeRhamComplex> deRham,
                           const DeRhamField<Grid::primal, Space::face>& B,
                           DeRhamField<Grid::dual, Space::edge>& H,
                           const amrex::Real dt)
    {
        BL_PROFILE("Gempic::TimeLoop::Forms::OperatorHamilton::apply_h_b()");
        // Operator H_B for dt/2
        // Solve Ampere-Maxwell with no current
        deRham->hodge(B, H);
        deRham->add_dt_curl(H, D, dt);
    }

    /**
     * @brief solves equation (He_field), ie the Faraday equation
     *
     * @param B[out] : magnetic field as primal 2-form (modified by the function)
     * @param deRham : object describing the discrete de Rham sequence
     * @param E : electric field as primal 1-form (temporary field)
     * @param D : electric field as dual 2-form (input of the function)
     * @param dt : time step
     *
     */
    inline void apply_h_e_field (DeRhamField<Grid::primal, Space::face>& B,
                                 std::shared_ptr<FDDeRhamComplex> deRham,
                                 DeRhamField<Grid::primal, Space::edge>& E,
                                 DeRhamField<Grid::dual, Space::face>& D,
                                 const amrex::Real dt)
    {
        BL_PROFILE("Gempic::TimeLoop::Forms::OperatorHamilton::apply_h_e_field()");
        deRham->hodge(D, E);
        // Operator H_E for dt (-dt for the Faraday equation)
        deRham->add_dt_curl(E, B, -dt);
    }
};

template <unsigned int vdim,
          int degx,
          int degy,
          int degz,
          int degmw,
          unsigned int ndata,
          bool electromagnetic,
          bool profiling>
class Splitting
{
public:
    virtual void time_loop(
        std::shared_ptr<FDDeRhamComplex> deRham,
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
        OperatorHamilton<vdim, degx, degy, degz, degmw> operatorHamilton) = 0;
};

// Cold plasma step
void apply_h_j(std::shared_ptr<FDDeRhamComplex> deRham,
               DeRhamField<Grid::dual, Space::face>& D,
               DeRhamField<Grid::dual, Space::face>& J,
               amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> funcBEquilibrium,
               amrex::Real dt);

}  // namespace Gempic::TimeLoop

#endif
