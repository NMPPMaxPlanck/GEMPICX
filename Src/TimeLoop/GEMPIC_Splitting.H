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
#include "GEMPIC_Particle.H"
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
template <unsigned int vdim, int degx, int degy, int degz>
class OperatorHamilton
{
public:
    enum SplittingScheme
    {
        HamiltonianSplittingExact,
        HamiltonianSplittingNested,
        HamiltonianSplittingNestedRelativistic
    };
    /**
     * @brief solves equation (He_vp) for one given particle
     *
     * @tparam SplineType spline class, for unit testing purposes
     *
     * @param[out] vel velocity of the given particle
     * @param eA array of E field coefficients (primal edge dofs)
     * @param spline object containing the values of non-zero grid splines at the particle
     * position
     * @param chargeOverMass (q/m)_p ratio
     * @param dt time step
     *
     */
    template <class SplineType = ParticleMeshCoupling::SplineBase<degx, degy, degz>>
    AMREX_GPU_HOST_DEVICE inline void apply_h_e_particle (
        amrex::GpuArray<amrex::Real, vdim>& vel,
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& eA,
        SplineType const& spline,
        amrex::Real const chargeOverMass,
        amrex::Real const dt) const
    {
        BL_PROFILE_HOST_DEVICE("Gempic::TimeLoop::Forms::OperatorHamilton::apply_h_e_particle()");
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
     * @tparam dir direction of the splitting (starting at dir = 0)
     *
     * @param[out] pos position of the given particle
     * @param[out] vel velocity of the given particle
     * @param infra computational domain
     * @param spline object containing the values of non-zero grid splines at the particle
     * position
     * @param bfields temporary array to store the values of the integrated B field
     * @param dx cell size in every direction
     * @param jA array of J density coefficients (dual face dofs)
     * @param bA array of B field coefficients (primal face dofs)
     * @param chargeOverMass (q/m)_p ratio
     * @param chargeWeight weight * charge of the given particle
     * @param dt time step
     *
     */
    template <Direction dir, class SplineType = ParticleMeshCoupling::SplineBase<degx, degy, degz>>
    AMREX_GPU_HOST_DEVICE inline void apply_h_p_i (
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& pos,
        amrex::GpuArray<amrex::Real, vdim>& vel,
        ComputationalDomain const& infra,
        SplineType& spline,
        amrex::GpuArray<amrex::Real, 2>& bfields,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx,
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& jA,
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& bA,
        amrex::Real const chargeOverMass,
        amrex::Real const chargeWeight,
        amrex::Real const dt) const
    {
        BL_PROFILE_HOST_DEVICE("Gempic::TimeLoop::Forms::OperatorHamilton::apply_h_p_i()");

        amrex::Real posNew = pos[dir] + dt * vel[dir];
        spline.template update_1d_splines<dir>(posNew, infra.geometry_data().ProbLo(dir),
                                               1.0 / infra.geometry_data().CellSize(dir));
        ParticleMeshCoupling::accumulate_j_integrate_b<dir>(bfields, spline, chargeWeight, dx, bA,
                                                            jA);
        pos[dir] = posNew;

        // we assume vdim = 3
        vel[(dir + 1) % 3] -= chargeOverMass * bfields[0];
        vel[(dir + 2) % 3] += chargeOverMass * bfields[1];
    }

    /**
     * @brief computes the spatial update and the D field for the relativistic splitting
     * in order to solve part of the ODE system H_kin_i for one given particle and
     * direction i.
     *
     * Specifically, this function:
     *  - (1) pushes x_p with constant velocity along direction i
     *  - (2) deposits the current associated with this trajectory
     *
     * Equation (H_kin_i_D) is only solved after this function has been called for all particles:
     * then the full current density has been computed and the full electric field can be updated.
     *
     * NOTE: The new particle position and velocities are not updated in the global particle arrays.
     * This must be done outside this function.
     *
     * @tparam dir direction of the splitting (starting at dir = 0)
     *
     * @param[out] pos position of the given particle
     * @param vel velocity of the given particle
     * @param infra computational domain
     * @param spline object containing the values of non-zero grid splines at the particle
     * position
     * @param dx cell size in every direction
     * @param jA array of J density coefficients (dual face dofs)
     * @param chargeWeight weight * charge of the given particle
     * @param dt time step
     *
     */

    template <Direction dir, class SplineType = ParticleMeshCoupling::SplineBase<degx, degy, degz>>
    AMREX_GPU_HOST_DEVICE inline void apply_h_p_i_nested_d (
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& pos,
        amrex::GpuArray<amrex::Real, vdim>& vel,
        ComputationalDomain const& infra,
        SplineType& spline,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx,
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& jA,
        amrex::Real const chargeWeight,
        amrex::Real const dt) const
    {
        BL_PROFILE_HOST_DEVICE("Gempic::TimeLoop::Forms::OperatorHamilton::apply_h_p_i_nested_d()");

        amrex::Real posNew = pos[dir] + dt * vel[dir];

        spline.template update_1d_splines<dir>(posNew, infra.geometry_data().ProbLo(dir),
                                               1.0 / infra.geometry_data().CellSize(dir));
        ParticleMeshCoupling::accumulate_j<dir>(spline, chargeWeight, dx, jA);

        pos[dir] = posNew;
    }

    /**
     * @brief computes the velocity update for the ODE system H_kin_i for one given
     * particle in all directions
     *
     * Specifically, this function:
     *  - (1) computes the integrated B field along this trajectory \<B\>_p = int_{0}^dt B(x_p(t))
     * dt
     *  - (2) pushes v_p on directions i-1 and i+1 using \<B\>_p
     *
     * Equation (H_kin_i_D) is only solved after this function has been called for all particles:
     * then the full current density has been computed and the full electric field can be updated.
     *
     * NOTE: The new particle position and velocities are not updated in the global particle arrays.
     * This must be done outside this function.
     *
     * @param[out] u relativistic velocity of the given particle
     * @param spline object containing the values of non-zero grid splines at the particle
     * position
     * @param bA array of B field coefficients (primal face dofs)
     * @param chargeOverMass (q/m)_p ratio
     * @param dt time step
     * @param gamma Lorentz factor
     *
     */
    template <class SplineType = ParticleMeshCoupling::SplineBase<degx, degy, degz>>
    AMREX_GPU_HOST_DEVICE inline void apply_h_p_rel_u (
        amrex::GpuArray<amrex::Real, vdim>& u,
        SplineType& spline,
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& bA,
        amrex::Real const chargeOverMass,
        amrex::Real const dt,
        amrex::Real const gamma) const
    {
        BL_PROFILE("Gempic::TimeLoop::Forms::OperatorHamilton::apply_h_p_rel_u()");

        amrex::Real uX = u[xDir];
        amrex::Real uY = u[yDir];
        amrex::Real uZ = u[zDir];

        amrex::GpuArray<amrex::Real, 3> bValues = {0.0, 0.0, 0.0};
        amrex::Real normB = 0;

        ParticleMeshCoupling::compute_b(bValues, normB, bA, spline);

        amrex::Real alpha = -chargeOverMass * normB * dt * 1 / gamma;
        amrex::Real cosAlpha = cos(alpha);
        amrex::Real oneMinCos = 1 - cos(alpha);
        amrex::Real sinAlpha = sin(alpha);

        static_assert(vdim == 3, "apply_h_p_rel_u not implemented for vdim != 3.");
        u[xDir] = (bValues[0] * bValues[0] * oneMinCos + cosAlpha) * uX +
                  (bValues[0] * bValues[1] * oneMinCos - bValues[2] * sinAlpha) * uY +
                  (bValues[0] * bValues[2] * oneMinCos + bValues[1] * sinAlpha) * uZ;
        u[yDir] = (bValues[0] * bValues[1] * oneMinCos + bValues[2] * sinAlpha) * uX +
                  (bValues[1] * bValues[1] * oneMinCos + cosAlpha) * uY +
                  (bValues[1] * bValues[2] * oneMinCos - bValues[0] * sinAlpha) * uZ;
        u[zDir] = (bValues[0] * bValues[2] * oneMinCos - bValues[1] * sinAlpha) * uX +
                  (bValues[1] * bValues[2] * oneMinCos + bValues[0] * sinAlpha) * uY +
                  (bValues[2] * bValues[2] * oneMinCos + cosAlpha) * uZ;
    }

    /**
     * @brief computes the velocity update for the ODE system H_kin_i for one given
     * particle in all directions
     *
     * Specifically, this function:
     *  - (1) computes the integrated B field along this trajectory \<B\>_p = int_{0}^dt B(x_p(t))
     * dt
     *  - (2) pushes v_p on directions i-1 and i+1 using \<B\>_p
     *
     * Equation (H_kin_i_D) is only solved after this function has been called for all particles:
     * then the full current density has been computed and the full electric field can be updated.
     *
     * NOTE: The new particle position and velocities are not updated in the global particle arrays.
     * This must be done outside this function.
     *
     * @param[out] vel velocity of the given particle
     * @param spline object containing the values of non-zero grid splines at the particle
     * position
     * @param bA array of B field coefficients (primal face dofs)
     * @param chargeOverMass (q/m)_p ratio
     * @param dt time step
     *
     */
    template <class SplineType = ParticleMeshCoupling::SplineBase<degx, degy, degz>>
    AMREX_GPU_HOST_DEVICE inline void apply_h_p_nested_rot (
        amrex::GpuArray<amrex::Real, vdim>& vel,
        SplineType& spline,
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& bA,
        amrex::Real const chargeOverMass,
        amrex::Real const dt) const
    {
        BL_PROFILE("Gempic::TimeLoop::Forms::OperatorHamilton::apply_h_p_nested_rot()");

        amrex::Real uXtemp = vel[xDir];
        amrex::Real uYtemp = vel[yDir];
        amrex::Real uZtemp = vel[zDir];

        amrex::GpuArray<amrex::Real, 3> bValues = {0.0, 0.0, 0.0};
        amrex::Real normB = 0;

        ParticleMeshCoupling::compute_b(bValues, normB, bA, spline);

        amrex::Real alpha = -chargeOverMass * normB * dt;
        amrex::Real cosAlpha = cos(alpha);
        amrex::Real oneMinCos = 1 - cos(alpha);
        amrex::Real sinAlpha = sin(alpha);

        static_assert(vdim == 3, "apply_h_p_nested_rot not implemented for vdim != 3.");
        vel[xDir] = (bValues[0] * bValues[0] * oneMinCos + cosAlpha) * uXtemp +
                    (bValues[0] * bValues[1] * oneMinCos - bValues[2] * sinAlpha) * uYtemp +
                    (bValues[0] * bValues[2] * oneMinCos + bValues[1] * sinAlpha) * uZtemp;
        vel[yDir] = (bValues[0] * bValues[1] * oneMinCos + bValues[2] * sinAlpha) * uXtemp +
                    (bValues[1] * bValues[1] * oneMinCos + cosAlpha) * uYtemp +
                    (bValues[1] * bValues[2] * oneMinCos - bValues[0] * sinAlpha) * uZtemp;
        vel[zDir] = (bValues[0] * bValues[2] * oneMinCos - bValues[1] * sinAlpha) * uXtemp +
                    (bValues[1] * bValues[2] * oneMinCos + bValues[0] * sinAlpha) * uYtemp +
                    (bValues[2] * bValues[2] * oneMinCos + cosAlpha) * uZtemp;
    }

    /**
     * @brief computes different fields in order to solve the ODE system (H_kin_i) for one given
     * particle and direction i for lower spatial dimensions.
     *
     * If AMREX_SPACEDIM < 3, there are spatial directions, where the primitive of the univariate
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
     * @tparam dir direction of the splitting (starting at dir = 0)
     *
     * @param[out] vel velocity of the given particle
     * @param infra computational domain
     * @param spline object containing the values of non-zero grid splines at the particle
     * position
     * @param bfields temporary array to store the values of the integrated B field
     * @param dx cell size in every direction
     * @param jA array of J density coefficients (dual face dofs)
     * @param bA array of B field coefficients (primal face dofs)
     * @param chargeOverMass (q/m)_p ratio
     * @param chargeWeight weight * charge of the given particle
     * @param dt time step
     *
     */
    template <Direction pDir, class SplineType = ParticleMeshCoupling::SplineBase<degx, degy, degz>>
    AMREX_GPU_HOST_DEVICE inline void apply_h_p_i_euler (
        amrex::GpuArray<amrex::Real, vdim>& vel,
        ComputationalDomain const& infra,
        SplineType& spline,
        amrex::GpuArray<amrex::Real, 2>& bfields,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx,
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& jA,
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& bA,
        amrex::Real const chargeOverMass,
        amrex::Real const chargeWeight,
        amrex::Real const dt) const
    {
        BL_PROFILE_HOST_DEVICE("Gempic::TimeLoop::Forms::OperatorHamilton::apply_h_p_i_euler()");

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
     * @brief computes the spatial update and the D field for the nested splitting (relativistic or
     * not) in order to solve part of the ODE system H_kin_i for one given particle and direction i.
     *
     * If AMREX_SPACEDIM < 3, there are spatial directions where the primitive of the univariate
     * spline S^{d_x-1} cannot be computed. This happens if the spatial dimension in this
     * direction does not exist. The current and velocities therefore have to be updated via an
     * explicit Euler step.
     *
     * Specifically, this function:
     *  - (1) pushes x_p with constant velocity along direction i
     *  - (2) deposits the current associated with this trajectory
     *
     * Equation (H_kin_i_D) is only solved after this function has been called for all particles:
     * then the full current density has been computed and the full electric field can be updated.
     *
     * NOTE: The new particle position and velocities are not updated in the global particle arrays.
     * This must be done outside this function.
     *
     * @tparam pDir direction of the splitting (starting at dir = 0)
     *
     * @param[out] vel velocity of the given particle
     * @param spline object containing the values of non-zero grid splines at the particle
     * position
     * @param jA array of J density coefficients (dual face dofs)
     * @param chargeWeight weight * charge of the given particle
     * @param dt time step
     *
     */
    template <Direction pDir, class SplineType = ParticleMeshCoupling::SplineBase<degx, degy, degz>>
    AMREX_GPU_HOST_DEVICE inline void apply_h_p_i_nested_d_euler (
        amrex::GpuArray<amrex::Real, vdim>& vel,
        SplineType& spline,
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& jA,
        amrex::Real const chargeWeight,
        amrex::Real const dt) const
    {
        BL_PROFILE_HOST_DEVICE(
            "Gempic::TimeLoop::Forms::OperatorHamilton::apply_h_p_i_nested_d_euler()");
        static_assert(pDir != xDir,
                      "apply_h_p_i_nested_d_euler should not be called for the x-direction!");
        amrex::Real const dtv = dt * vel[pDir];

        if constexpr (pDir == yDir)
        {
            accumulate_j_euler_y(spline, dtv, chargeWeight, jA);
        }
        else if constexpr (pDir == zDir)
        {
            accumulate_j_euler_z(spline, dtv, chargeWeight, jA);
        }
    }

    /**
     * @brief computes different fields in order to solve the ODE system (H_p) for one particle
     *
     * @param[out] pos position of the particle
     * @param[out] vel velocity of the article
     * @param infra computational domain
     * @param spline object containing the values of non-zero grid splines at the particle
     * position
     * @param dx cell size in every direction
     * @param jA array of J density coefficients (dual face dofs)
     * @param bA array of B field coefficients (primal face dofs)
     * @param chargeOverMass (q/m)_p ratio
     * @param chargeWeight weight * charge of the given particle
     * @param dt time step
     *
     */
    template <class SplineType = ParticleMeshCoupling::SplineBase<degx, degy, degz>>
    AMREX_GPU_HOST_DEVICE inline void apply_h_p_exact (
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& pos,
        amrex::GpuArray<amrex::Real, vdim>& vel,
        ComputationalDomain const& infra,
        SplineType& spline,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx,
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& jA,
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& bA,
        amrex::Real const chargeOverMass,
        amrex::Real const chargeWeight,
        amrex::Real const dt) const
    {
        BL_PROFILE_HOST_DEVICE("Gempic::TimeLoop::Forms::OperatorHamilton::apply_h_p_exact_x()");

        amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

        apply_h_p_i<xDir>(pos, vel, infra, spline, bfields, dx, jA, bA, chargeOverMass,
                          chargeWeight, 0.5 * dt);
#if AMREX_SPACEDIM == 3
        apply_h_p_i<yDir>(pos, vel, infra, spline, bfields, dx, jA, bA, chargeOverMass,
                          chargeWeight, 0.5 * dt);
        apply_h_p_i<zDir>(pos, vel, infra, spline, bfields, dx, jA, bA, chargeOverMass,
                          chargeWeight, dt);
        apply_h_p_i<yDir>(pos, vel, infra, spline, bfields, dx, jA, bA, chargeOverMass,
                          chargeWeight, 0.5 * dt);

#elif AMREX_SPACEDIM == 2
        apply_h_p_i<yDir>(pos, vel, infra, spline, bfields, dx, jA, bA, chargeOverMass,
                          chargeWeight, 0.5 * dt);
        apply_h_p_i_euler<zDir>(vel, infra, spline, bfields, dx, jA, bA, chargeOverMass,
                                chargeWeight, dt);
        apply_h_p_i<yDir>(pos, vel, infra, spline, bfields, dx, jA, bA, chargeOverMass,
                          chargeWeight, 0.5 * dt);

#elif AMREX_SPACEDIM == 1
        apply_h_p_i_euler<yDir>(vel, infra, spline, bfields, dx, jA, bA, chargeOverMass,
                                chargeWeight, 0.5 * dt);
        apply_h_p_i_euler<zDir>(vel, infra, spline, bfields, dx, jA, bA, chargeOverMass,
                                chargeWeight, dt);
        apply_h_p_i_euler<yDir>(vel, infra, spline, bfields, dx, jA, bA, chargeOverMass,
                                chargeWeight, 0.5 * dt);
#endif
        apply_h_p_i<xDir>(pos, vel, infra, spline, bfields, dx, jA, bA, chargeOverMass,
                          chargeWeight, 0.5 * dt);
    }

    /**
     * @brief computes different fields in order to solve the ODE system (H_p) for one particle
     * for the relativistic implementation
     *
     * @param[out] pos position of the particle
     * @param[out] u relativistic velocity of the particle
     * @param infra computational domain
     * @param spline object containing the values of non-zero grid splines at the particle
     * position
     * @param dx cell size in every direction
     * @param jA array of J density coefficients (dual face dofs)
     * @param bA array of B field coefficients (primal face dofs)
     * @param chargeOverMass (q/m)_p ratio
     * @param chargeWeight weight * charge of the given particle
     * @param dt time step
     * @param c speed of light (if not defined otherwise in input file, this value is set to 1)
     *
     */
    template <class SplineType = ParticleMeshCoupling::SplineBase<degx, degy, degz>>
    AMREX_GPU_HOST_DEVICE inline void apply_h_p_nested_rel (
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& pos,
        amrex::GpuArray<amrex::Real, vdim>& u,
        ComputationalDomain const& infra,
        SplineType& spline,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx,
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& jA,
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& bA,
        amrex::Real const chargeOverMass,
        amrex::Real const chargeWeight,
        amrex::Real const dt,
        amrex::Real const c) const
    {
        BL_PROFILE("Gempic::TimeLoop::Forms::OperatorHamilton::apply_h_p_nested_rel()");

        amrex::GpuArray<amrex::Real, 3> vel{0., 0., 0.};
        amrex::Real gamma = sqrt(1 + (u[xDir] * u[xDir] + u[yDir] * u[yDir] + u[zDir] * u[zDir]) /
                                         (c * c)); // Lorentz factor: sqrt(1+u^2/c^2)

        vel[xDir] = u[xDir] / gamma;
        vel[yDir] = u[yDir] / gamma;
        vel[zDir] = u[zDir] / gamma;

#if AMREX_SPACEDIM == 3

        apply_h_p_i_nested_d<xDir>(pos, vel, infra, spline, dx, jA, chargeWeight, 0.5 * dt);
        apply_h_p_i_nested_d<yDir>(pos, vel, infra, spline, dx, jA, chargeWeight, 0.5 * dt);
        apply_h_p_i_nested_d<zDir>(pos, vel, infra, spline, dx, jA, chargeWeight, 0.5 * dt);

        apply_h_p_rel_u(u, spline, bA, chargeOverMass, dt, gamma);

        vel[xDir] = u[xDir] / gamma;
        vel[yDir] = u[yDir] / gamma;
        vel[zDir] = u[zDir] / gamma;

        apply_h_p_i_nested_d<zDir>(pos, vel, infra, spline, dx, jA, chargeWeight, 0.5 * dt);
        apply_h_p_i_nested_d<yDir>(pos, vel, infra, spline, dx, jA, chargeWeight, 0.5 * dt);
        apply_h_p_i_nested_d<xDir>(pos, vel, infra, spline, dx, jA, chargeWeight, 0.5 * dt);

#elif AMREX_SPACEDIM == 2

        apply_h_p_i_nested_d<xDir>(pos, vel, infra, spline, dx, jA, chargeWeight, 0.5 * dt);
        apply_h_p_i_nested_d<yDir>(pos, vel, infra, spline, dx, jA, chargeWeight, 0.5 * dt);
        apply_h_p_i_nested_d_euler<zDir>(vel, spline, jA, chargeWeight, 0.5 * dt);

        apply_h_p_rel_u(u, spline, bA, chargeOverMass, dt, gamma);

        vel[xDir] = u[xDir] / gamma;
        vel[yDir] = u[yDir] / gamma;
        vel[zDir] = u[zDir] / gamma;

        apply_h_p_i_nested_d_euler<zDir>(vel, spline, jA, chargeWeight, 0.5 * dt);
        apply_h_p_i_nested_d<yDir>(pos, vel, infra, spline, dx, jA, chargeWeight, 0.5 * dt);
        apply_h_p_i_nested_d<xDir>(pos, vel, infra, spline, dx, jA, chargeWeight, 0.5 * dt);

#elif AMREX_SPACEDIM == 1

        apply_h_p_i_nested_d<xDir>(pos, vel, infra, spline, dx, jA, chargeWeight, 0.5 * dt);
        apply_h_p_i_nested_d_euler<yDir>(vel, spline, jA, chargeWeight, 0.5 * dt);
        apply_h_p_i_nested_d_euler<zDir>(vel, spline, jA, chargeWeight, 0.5 * dt);

        apply_h_p_rel_u(u, spline, bA, chargeOverMass, dt, gamma);

        vel[xDir] = u[xDir] / gamma;
        vel[yDir] = u[yDir] / gamma;
        vel[zDir] = u[zDir] / gamma;

        apply_h_p_i_nested_d_euler<zDir>(vel, spline, jA, chargeWeight, 0.5 * dt);
        apply_h_p_i_nested_d_euler<yDir>(vel, spline, jA, chargeWeight, 0.5 * dt);
        apply_h_p_i_nested_d<xDir>(pos, vel, infra, spline, dx, jA, chargeWeight, 0.5 * dt);
#endif
    }

    /**
     * @brief computes different fields in order to solve the ODE system (H_p) for one particle
     * for the non-relativistic nested splitting implementation
     *
     * @param[out] pos position of the particle
     * @param[out] vel velocity of the particle
     * @param infra computational domain
     * @param spline object containing the values of non-zero grid splines at the particle
     * position
     * @param dx cell size in every direction
     * @param jA array of J density coefficients (dual face dofs)
     * @param bA array of B field coefficients (primal face dofs)
     * @param chargeOverMass (q/m)_p ratio
     * @param chargeWeight weight * charge of the given particle
     * @param dt time step
     *
     */
    template <class SplineType = ParticleMeshCoupling::SplineBase<degx, degy, degz>>
    AMREX_GPU_HOST_DEVICE inline void apply_h_p_nested (
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& pos,
        amrex::GpuArray<amrex::Real, vdim>& vel,
        ComputationalDomain const& infra,
        SplineType& spline,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx,
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& jA,
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& bA,
        amrex::Real const chargeOverMass,
        amrex::Real const chargeWeight,
        amrex::Real const dt) const
    {
        BL_PROFILE("Gempic::TimeLoop::Forms::OperatorHamilton::apply_h_p_nested()");

#if AMREX_SPACEDIM == 3

        apply_h_p_i_nested_d<xDir>(pos, vel, infra, spline, dx, jA, chargeWeight, 0.5 * dt);
        apply_h_p_i_nested_d<yDir>(pos, vel, infra, spline, dx, jA, chargeWeight, 0.5 * dt);
        apply_h_p_i_nested_d<zDir>(pos, vel, infra, spline, dx, jA, chargeWeight, 0.5 * dt);

        apply_h_p_nested_rot(vel, spline, bA, chargeOverMass, dt);

        apply_h_p_i_nested_d<zDir>(pos, vel, infra, spline, dx, jA, chargeWeight, 0.5 * dt);
        apply_h_p_i_nested_d<yDir>(pos, vel, infra, spline, dx, jA, chargeWeight, 0.5 * dt);
        apply_h_p_i_nested_d<xDir>(pos, vel, infra, spline, dx, jA, chargeWeight, 0.5 * dt);

#elif AMREX_SPACEDIM == 2

        apply_h_p_i_nested_d<xDir>(pos, vel, infra, spline, dx, jA, chargeWeight, 0.5 * dt);
        apply_h_p_i_nested_d<yDir>(pos, vel, infra, spline, dx, jA, chargeWeight, 0.5 * dt);
        apply_h_p_i_nested_d_euler<zDir>(vel, spline, jA, chargeWeight, 0.5 * dt);

        apply_h_p_nested_rot(vel, spline, bA, chargeOverMass, dt);

        apply_h_p_i_nested_d_euler<zDir>(vel, spline, jA, chargeWeight, 0.5 * dt);
        apply_h_p_i_nested_d<yDir>(pos, vel, infra, spline, dx, jA, chargeWeight, 0.5 * dt);
        apply_h_p_i_nested_d<xDir>(pos, vel, infra, spline, dx, jA, chargeWeight, 0.5 * dt);

#elif AMREX_SPACEDIM == 1

        apply_h_p_i_nested_d<xDir>(pos, vel, infra, spline, dx, jA, chargeWeight, 0.5 * dt);
        apply_h_p_i_nested_d_euler<yDir>(vel, spline, jA, chargeWeight, 0.5 * dt);
        apply_h_p_i_nested_d_euler<zDir>(vel, spline, jA, chargeWeight, 0.5 * dt);

        apply_h_p_nested_rot(vel, spline, bA, chargeOverMass, dt);

        apply_h_p_i_nested_d_euler<zDir>(vel, spline, jA, chargeWeight, 0.5 * dt);
        apply_h_p_i_nested_d_euler<yDir>(vel, spline, jA, chargeWeight, 0.5 * dt);
        apply_h_p_i_nested_d<xDir>(pos, vel, infra, spline, dx, jA, chargeWeight, 0.5 * dt);
#endif
    }

    /**
     * @brief solves the ODE (Hb_D) corresponding to H_b part of the Hamiltonian
     * which is the ampere equation without current
     *
     * @param[out] D electric field as dual 2-form (modified by the function)
     * @param deRham object describing the discrete de Rham sequence
     * @param B magnetic field as primal 2-form
     * @param H magnetic field as dual 1-form
     * @param dt time step
     *
     */
    inline void apply_h_b (DeRhamField<Grid::dual, Space::face>& D,
                           std::shared_ptr<FDDeRhamComplex> deRham,
                           DeRhamField<Grid::primal, Space::face> const& B,
                           DeRhamField<Grid::dual, Space::edge>& H,
                           amrex::Real const dt)
    {
        BL_PROFILE_HOST_DEVICE("Gempic::TimeLoop::Forms::OperatorHamilton::apply_h_b()");
        // Operator H_B for dt/2
        // Solve Ampere-Maxwell with no current
        hodge(H, B, deRham->scaling_bto_h());
        add_dt_curl(D, H, dt);
    }

    /**
     * @brief solves equation (He_field), ie the Faraday equation
     *
     * @param[out] B magnetic field as primal 2-form (modified by the function)
     * @param[out] E electric field as primal 1-form (temporary field)
     * @param deRham object describing the discrete de Rham sequence
     * @param D electric field as dual 2-form (input of the function)
     * @param dt time step
     *
     */
    inline void apply_h_e_field (DeRhamField<Grid::primal, Space::face>& B,
                                 DeRhamField<Grid::primal, Space::edge>& E,
                                 std::shared_ptr<FDDeRhamComplex> deRham,
                                 DeRhamField<Grid::dual, Space::face>& D,
                                 amrex::Real const dt)
    {
        BL_PROFILE_HOST_DEVICE("Gempic::TimeLoop::Forms::OperatorHamilton::apply_h_e_field()");
        hodge(E, D, deRham->scaling_dto_e());
        // Operator H_E for dt (-dt for the Faraday equation)
        add_dt_curl(B, E, -dt);
    }

    /**
     * @brief solves equation (Hp), corresponding to H_p part of the Hamiltonian
     * for each particle independently via a particle loop. This part of the
     * splitting is different for the exact, nested, and relativistic method.
     *
     */
    template <SplittingScheme splittingScheme>
    void apply_h_p (
        std::vector<std::shared_ptr<Gempic::Particle::ParticleSpecies<vdim>>>& particles,
        DeRhamField<Grid::dual, Space::face>& J,
        DeRhamField<Grid::primal, Space::edge>& E,
        DeRhamField<Grid::primal, Space::face>& B,
        ComputationalDomain const& infra,
        amrex::Real const dt,
        amrex::Real const c) const
    {
        for (auto const& particleSpecies : particles)
        {
            amrex::Real charge = particleSpecies->get_charge();
            amrex::Real chargeOverMass = charge / particleSpecies->get_mass();

            for (auto& particleGrid : *particleSpecies)
            {
                long const np = particleGrid.numParticles();
                auto const ptd = particleGrid.GetParticleTile().getParticleTileData();
                auto const ii = particleSpecies->get_data_indices();

                amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> jA;
                amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;
                amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> bA;

                for (int cc = 0; cc < vdim; cc++)
                {
                    jA[cc] = (J.m_data[cc])[particleGrid].array();
                    eA[cc] = (E.m_data[cc])[particleGrid].array(); // Filtered E for pushing
                    bA[cc] = (B.m_data[cc])[particleGrid].array();
                }
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const plo{
                    infra.geometry().ProbLoArray()};

                amrex::ParallelFor(
                    np,
                    [=, c = c, bA = bA, jA = jA, *this] AMREX_GPU_DEVICE(long pp)
                    {
                        // Read out particle position and compute according
                        // splines
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> pos;
                        AMREX_D_EXPR(pos[xDir] = ptd.rdata(ii.m_iposx)[pp],
                                     pos[yDir] = ptd.rdata(ii.m_iposy)[pp],
                                     pos[zDir] = ptd.rdata(ii.m_iposz)[pp]);

                        Gempic::ParticleMeshCoupling::SplineWithPrimitive<degx, degy, degz> spline(
                            pos, plo, infra.inv_cell_size_array());

                        // Read out particle velocity
                        amrex::GpuArray<amrex::Real, vdim> vel{ptd.rdata(ii.m_ivelx)[pp],
                                                               ptd.rdata(ii.m_ively)[pp],
                                                               ptd.rdata(ii.m_ivelz)[pp]};

                        apply_h_e_particle(vel, eA, spline, chargeOverMass, 0.5 * dt);

                        amrex::Real chargeWeight = charge * ptd.rdata(ii.m_iweight)[pp];

                        if constexpr (splittingScheme == SplittingScheme::HamiltonianSplittingExact)
                        {
                            apply_h_p_exact(pos, vel, infra, spline, infra.cell_size_array(), jA,
                                            bA, chargeOverMass, chargeWeight, dt);
                        }
                        else if constexpr (splittingScheme ==
                                           SplittingScheme::HamiltonianSplittingNestedRelativistic)
                        {
                            apply_h_p_nested_rel(pos, vel, infra, spline, infra.cell_size_array(),
                                                 jA, bA, chargeOverMass, chargeWeight, dt, c);
                        }
                        else if constexpr (splittingScheme ==
                                           SplittingScheme::HamiltonianSplittingNested)
                        {
                            apply_h_p_nested(pos, vel, infra, spline, infra.cell_size_array(), jA,
                                             bA, chargeOverMass, chargeWeight, dt);
                        }
                        AMREX_D_EXPR(ptd.rdata(ii.m_iposx)[pp] = pos[0],
                                     ptd.rdata(ii.m_iposy)[pp] = pos[1],
                                     ptd.rdata(ii.m_iposz)[pp] = pos[2]);

                        ptd.rdata(ii.m_ivelx)[pp] = vel[xDir];
                        ptd.rdata(ii.m_ively)[pp] = vel[yDir];
                        ptd.rdata(ii.m_ivelz)[pp] = vel[zDir];
                    });
            }
            particleSpecies->Redistribute();
        }
        J.post_particle_loop_sync();
    }
};

template <unsigned int vdim,
          int degx,
          int degy,
          int degz,
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
        amrex::Real const dt,
        std::vector<std::unique_ptr<Particle::ParticleSpecies<vdim, ndata>>>& particles,
        int const nSteps,
        int strangOrder,
        OperatorHamilton<vdim, degx, degy, degz> operatorHamilton) = 0;
};

// Cold plasma step
void apply_h_j(DeRhamField<Grid::dual, Space::face>& J,
               DeRhamField<Grid::dual, Space::face>& D,
               std::shared_ptr<FDDeRhamComplex> deRham,
               amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3> funcBEquilibrium,
               amrex::Real dt);

} // namespace Gempic::TimeLoop

#endif
