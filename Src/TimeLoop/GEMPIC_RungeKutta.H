#ifndef GEMPIC_RUNGEKUTTA_
#define GEMPIC_RUNGEKUTTA_

#include <cmath>
#include <vector>

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

namespace Gempic::TimeLoop
{
namespace Impl
{
inline AMREX_GPU_HOST_DEVICE void lsrk_stage_dk_particle (
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& R,
    amrex::GpuArray<amrex::Real, 3>& V,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& s2R,
    amrex::GpuArray<amrex::Real, 3>& s2V,
    amrex::GpuArray<amrex::Real, 3> const& velGC,
    amrex::Real accGC,
    amrex::Real dt,
    amrex::Real coeffsAStageIndex,
    amrex::Real coeffsBStageIndex)
{
    BL_PROFILE_HOST_DEVICE("Gempic::TimeLoop::Impl::lsrk_stage_dk_particle()");

    // grad Bpar = grad Bz is neglected for the moment and B_par^* = Bz (assuming initial B contains
    // background field)
    s2R[0] = coeffsAStageIndex * s2R[0] + dt * velGC[0];
    s2R[1] = coeffsAStageIndex * s2R[1] + dt * velGC[1];
    s2R[2] = coeffsAStageIndex * s2R[2] + dt * velGC[2];
    s2V[0] = 0;                                       // not used in DK model
    s2V[1] = 0;                                       // not used in DK model
    s2V[2] = coeffsAStageIndex * s2V[2] + dt * accGC; // Vpar
    for (int d = 0; d < AMREX_SPACEDIM; d++)
    {
        R[d] += coeffsBStageIndex * s2R[d];
    }
    V[0] += coeffsBStageIndex * s2V[0];
    V[1] += coeffsBStageIndex * s2V[1];
    V[2] += coeffsBStageIndex * s2V[2];
}

inline AMREX_GPU_HOST_DEVICE void lsrk_stage_kinetic_particle (
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& R,
    amrex::GpuArray<amrex::Real, 3>& V,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& s2R,
    amrex::GpuArray<amrex::Real, 3>& s2V,
    amrex::GpuArray<amrex::Real, 3> const& E,
    amrex::GpuArray<amrex::Real, 3> const& B,
    amrex::Real qom, // q/m
    amrex::Real dt,
    amrex::Real coeffsAStageIndex,
    amrex::Real coeffsBStageIndex)
{
    BL_PROFILE_HOST_DEVICE("Gempic::TimeLoop::Impl::lsrk_stage_kinetic_particle()");
    for (int d = 0; d < AMREX_SPACEDIM; d++)
    {
        s2R[d] = coeffsAStageIndex * s2R[d] + dt * V[d];
        R[d] += coeffsBStageIndex * s2R[d];
    }

    s2V[0] = coeffsAStageIndex * s2V[0] + dt * qom * (E[0] + V[1] * B[2] - V[2] * B[1]);
    s2V[1] = coeffsAStageIndex * s2V[1] + dt * qom * (E[1] + V[2] * B[0] - V[0] * B[2]);
    s2V[2] = coeffsAStageIndex * s2V[2] + dt * qom * (E[2] + V[0] * B[1] - V[1] * B[0]);
    V[0] += coeffsBStageIndex * s2V[0];
    V[1] += coeffsBStageIndex * s2V[1];
    V[2] += coeffsBStageIndex * s2V[2];
}

/**
 * @brief Computes the guiding center velocity and acceleration for a particle.
 *
 * This function calculates the guiding center velocity (`velGC`) and the guiding center
 * acceleration (`accGC`) for a charged particle based on its velocity, the electric
 * and magnetic fields at its position, and the particle's charge-to-mass ratio (`qom`).
 *
 * @param[out] velGC Guiding center velocity (3 components: x, y, z).
 * @param[out] accGC Guiding center acceleration (1 component: dv_parallel/dt).
 * @param[in] qom Charge-to-mass ratio of the particle.
 * @param[in] V Particle velocity (3 components, V[2] is parallel velocity).
 * @param[in] spline Spline object for field evaluation.
 * @param[in] eA Degrees of freedom of E as primal 1-form
 * @param[in] bA Degrees of freedom of B as primal 2-form.
 */
template <int degx, int degy, int degz>
inline AMREX_GPU_HOST_DEVICE void dxvpardt_drift_kinetic (
    amrex::GpuArray<amrex::Real, 3>& velGC,
    amrex::Real& accGC,
    amrex::Real const qom,
    amrex::GpuArray<amrex::Real, 3> const& V,
    Gempic::ParticleMeshCoupling::SplineBase<degx, degy, degz> const& spline,
    amrex::GpuArray<amrex::Array4<amrex::Real>, 3> const& eA,
    amrex::GpuArray<amrex::Array4<amrex::Real>, 3> const& bA)
{
    BL_PROFILE_HOST_DEVICE("Gempic::TimeLoop::Impl::dxvpardt_drift_kinetic");
    // grad Bpar = grad Bz is neglected for the moment and B_par^* = Bz (assuming initial B contains
    // background field)

    // Evaluate the electric field at the particle's position
    amrex::GpuArray<amrex::Real, 3> ep =
        spline.template eval_spline_field<Field::PrimalOneForm>(eA);

    // Evaluate the magnetic field at the particle's position
    amrex::GpuArray<amrex::Real, 3> bp =
        spline.template eval_spline_field<Field::PrimalTwoForm>(bA);

    // compute guiding center velocity and acceleration
    velGC[0] = (V[2] * bp[0] + ep[1]) / bp[2];
    velGC[1] = (V[2] * bp[1] - ep[0]) / bp[2];
    velGC[2] = V[2];
    accGC = qom * (bp[0] * ep[0] + bp[1] * ep[1] + bp[2] * ep[2]) / bp[2];
}
} //namespace Impl

/**
 * @brief Low storage Runge-Kutta method
 * The one stage method is explicit Euler
 * We use 2N storage Runge-Kutta methods (Williamson 1980, Carpenter-Kennedy 1984)
 * have a Butcher tableau which can be replaced by just two vectors of coefficients A and B
 * This allows to store only one array in addition to the solution array
 */
class RungeKutta
{
private:
    int m_stages{5};
    amrex::Vector<amrex::Real> m_coeffsA;
    amrex::Vector<amrex::Real> m_coeffsB;
    // amrex::Vector<amrex::Real> m_dtSequence;
    ComputationalDomain m_infra;
    std::shared_ptr<Gempic::Forms::FDDeRhamComplex> m_deRham;
    Gempic::Forms::DeRhamField<Grid::dual, Space::face> m_s2D;
    Gempic::Forms::DeRhamField<Grid::primal, Space::face> m_s2B;

public:
    /**
     * Default Constructor.
     */
    RungeKutta(ComputationalDomain infra, std::shared_ptr<Gempic::Forms::FDDeRhamComplex> deRham);

    // Getter function to access m_stages
    inline int get_stages () const { return m_stages; }

    inline double get_coeff_a (int stageIndex) const { return m_coeffsA[stageIndex]; }

    inline double get_coeff_b (int stageIndex) const { return m_coeffsB[stageIndex]; }

    void lsrk_stage_ampere(Gempic::Forms::DeRhamField<Grid::dual, Space::face>& D,
                           Gempic::Forms::DeRhamField<Grid::dual, Space::edge> const& H,
                           Gempic::Forms::DeRhamField<Grid::dual, Space::face> const& J,
                           amrex::Real dt,
                           int stageIndex);

    void lsrk_stage_faraday(Gempic::Forms::DeRhamField<Grid::primal, Space::face>& B,
                            Gempic::Forms::DeRhamField<Grid::primal, Space::edge> const& E,
                            amrex::Real dt,
                            int stageIndex);

    template <int degx, int degy, int degz, unsigned int vDim, unsigned int nData>
    void lsrk_vlasov_maxwell (
        std::vector<std::shared_ptr<Gempic::Particle::ParticleGroups<vDim, nData>>>& particleGroup,
        Gempic::Forms::DeRhamField<Grid::primal, Space::edge>& E,
        Gempic::Forms::DeRhamField<Grid::dual, Space::face>& D,
        Gempic::Forms::DeRhamField<Grid::primal, Space::face>& B,
        Gempic::Forms::DeRhamField<Grid::dual, Space::edge>& H,
        Gempic::Forms::DeRhamField<Grid::dual, Space::face>& J,
        amrex::Real dt)
    {
        BL_PROFILE("GEMPIC::RungeKutta::lsrk_vlasov_maxwell()");
        for (int stageIndex = 0; stageIndex < m_stages; stageIndex++)
        {
            amrex::Real const coeffsAStageIndex = m_coeffsA[stageIndex];
            amrex::Real const coeffsBStageIndex = m_coeffsB[stageIndex];
            // amrex::Real const dtStageIndex = m_dtSequence[stageIndex];
            // compute RHS for Ampere and do LSRK stage to update D
            lsrk_stage_ampere(D, H, J, dt, stageIndex);
            lsrk_stage_push_particle_deposit_j<degx, degy, degz>(
                particleGroup, E, B, J, dt, coeffsAStageIndex, coeffsBStageIndex);
            // compute RHS for Faraday and do LSRK stage to update B
            lsrk_stage_faraday(B, E, dt, stageIndex);

            //update H
            m_deRham->hodge(H, B, m_deRham->scaling_bto_h());
            //update E
            m_deRham->hodge(E, D, m_deRham->scaling_dto_e());

        } // end of loop over m_stages
    } // lsrk_vlasov_maxwell

    template <int degx, int degy, int degz, unsigned int vDim, unsigned int nData>
    void lsrk_dk_vlasov_maxwell (
        std::vector<std::shared_ptr<Gempic::Particle::ParticleGroups<vDim, nData>>>& particleGroup,
        Gempic::Forms::DeRhamField<Grid::primal, Space::edge>& E,
        Gempic::Forms::DeRhamField<Grid::dual, Space::face>& D,
        Gempic::Forms::DeRhamField<Grid::primal, Space::face>& B,
        Gempic::Forms::DeRhamField<Grid::dual, Space::edge>& H,
        Gempic::Forms::DeRhamField<Grid::dual, Space::face>& J,
        Gempic::Forms::DeRhamField<Grid::primal, Space::edge>& tensor,
        amrex::Real dt)
    {
        BL_PROFILE("GEMPIC::RungeKutta::lsrk_dk_vlasov_maxwell()");
        for (int stageIndex = 0; stageIndex < m_stages; stageIndex++)
        {
            amrex::Real const coeffsAStageIndex = m_coeffsA[stageIndex];
            amrex::Real const coeffsBStageIndex = m_coeffsB[stageIndex];
            // compute RHS for Ampere and do LSRK stage to update D
            lsrk_stage_ampere(D, H, J, dt, stageIndex);
            lsrk_stage_dk_push_particle_deposit_j<degx, degy, degz>(
                particleGroup, E, B, J, dt, coeffsAStageIndex, coeffsBStageIndex);
            // compute RHS for Faraday and do LSRK stage to update B
            lsrk_stage_faraday(B, E, dt, stageIndex);

            //update H
            m_deRham->hodge(H, B, m_deRham->scaling_bto_h());
            //update E
            m_deRham->hodge_dk(E, D, tensor, m_deRham->scaling_dto_e());

        } // end of loop over m_stages
    } //lsrk_dk_vlasov_maxwell

    template <int degx, int degy, int degz, unsigned int vDim, unsigned int nData>
    void lsrk_de_fi_vlasov_maxwell (
        std::vector<std::shared_ptr<Gempic::Particle::ParticleGroups<vDim, nData>>>& particleGroup,
        Gempic::Forms::DeRhamField<Grid::primal, Space::edge>& E,
        Gempic::Forms::DeRhamField<Grid::dual, Space::face>& D,
        Gempic::Forms::DeRhamField<Grid::primal, Space::face>& B,
        Gempic::Forms::DeRhamField<Grid::dual, Space::edge>& H,
        Gempic::Forms::DeRhamField<Grid::dual, Space::face>& J,
        Gempic::Forms::DeRhamField<Grid::primal, Space::edge>& tensor,
        amrex::Real dt)
    {
        BL_PROFILE("GEMPIC::RungeKutta::lsrk_DeFi_vlasov_maxwell()");
        for (int stageIndex = 0; stageIndex < m_stages; stageIndex++)
        {
            amrex::Real const coeffsAStageIndex = m_coeffsA[stageIndex];
            amrex::Real const coeffsBStageIndex = m_coeffsB[stageIndex];
            // compute RHS for Ampere and do LSRK stage to update D
            lsrk_stage_ampere(D, H, J, dt, stageIndex);
            lsrk_stage_de_fi_push_particle_deposit_j<degx, degy, degz>(
                particleGroup, E, B, J, dt, coeffsAStageIndex, coeffsBStageIndex);
            // compute RHS for Faraday and do LSRK stage to update B
            lsrk_stage_faraday(B, E, dt, stageIndex);

            //update H
            m_deRham->hodge(H, B, m_deRham->scaling_bto_h());
            //update E
            m_deRham->hodge_dk(E, D, tensor, m_deRham->scaling_dto_e());

        } // end of loop over m_stages
    } //lsrk_DeFi_vlasov_maxwell

    template <int degx, int degy, int degz, unsigned int vDim, unsigned int nData>
    void lsrk_stage_push_particle_deposit_j (
        std::vector<std::shared_ptr<Gempic::Particle::ParticleGroups<vDim, nData>>>& particleGroup,
        Gempic::Forms::DeRhamField<Grid::primal, Space::edge>& E,
        Gempic::Forms::DeRhamField<Grid::primal, Space::face>& B,
        Gempic::Forms::DeRhamField<Grid::dual, Space::face>& J,
        amrex::Real dt,
        amrex::Real coeffsAStageIndex,
        amrex::Real coeffsBStageIndex)
    {
        BL_PROFILE("GEMPIC::RungeKutta::lsrk_stage_push_particle_deposit_j()");
        // particle loop
        for (int i = 0; i < 3; i++)
        {
            J.m_data[i].setVal(0.0);
        }
        for (auto& particleSpecies : particleGroup)
        {
            lsrk_stage_push_particle_deposit_j_fully_kinetic<degx, degy, degz>(
                particleSpecies, E, B, J, dt, coeffsAStageIndex, coeffsBStageIndex);
        }
        J.post_particle_loop_sync();
    }

    template <int degx, int degy, int degz, unsigned int vDim, unsigned int nData>
    void lsrk_stage_dk_push_particle_deposit_j (
        std::vector<std::shared_ptr<Gempic::Particle::ParticleGroups<vDim, nData>>>& particleGroup,
        Gempic::Forms::DeRhamField<Grid::primal, Space::edge>& E,
        Gempic::Forms::DeRhamField<Grid::primal, Space::face>& B,
        Gempic::Forms::DeRhamField<Grid::dual, Space::face>& J,
        amrex::Real dt,
        amrex::Real const coeffsAStageIndex,
        amrex::Real const coeffsBStageIndex)
    {
        BL_PROFILE("GEMPIC::RungeKutta::lsrk_stage_dk_push_particle_deposit_j()");
        // particle loop
        for (int i = 0; i < 3; i++)
        {
            J.m_data[i].setVal(0.0);
        }
        for (auto& particleSpecies : particleGroup)
        {
            lsrk_stage_push_particle_deposit_j_drift_kinetic<degx, degy, degz>(
                particleSpecies, E, B, J, dt, coeffsAStageIndex, coeffsBStageIndex);
        }
        J.post_particle_loop_sync();
    }

    template <int degx, int degy, int degz, unsigned int vDim, unsigned int nData>
    void lsrk_stage_de_fi_push_particle_deposit_j (
        std::vector<std::shared_ptr<Gempic::Particle::ParticleGroups<vDim, nData>>>& particleGroup,
        Gempic::Forms::DeRhamField<Grid::primal, Space::edge>& E,
        Gempic::Forms::DeRhamField<Grid::primal, Space::face>& B,
        Gempic::Forms::DeRhamField<Grid::dual, Space::face>& J,
        amrex::Real dt,
        amrex::Real const coeffsAStageIndex,
        amrex::Real const coeffsBStageIndex)
    {
        BL_PROFILE("GEMPIC::RungeKutta::lsrk_stage_DeFi_push_particle_deposit_j()");
        // particle loop
        for (int i = 0; i < 3; i++)
        {
            J.m_data[i].setVal(0.0);
        }
        for (auto& particleSpecies : particleGroup)
        {
            if (particleSpecies->get_name() == "electrons")
            {
                lsrk_stage_push_particle_deposit_j_drift_kinetic<degx, degy, degz>(
                    particleSpecies, E, B, J, dt, coeffsAStageIndex, coeffsBStageIndex);
            }
            else
            {
                lsrk_stage_push_particle_deposit_j_fully_kinetic<degx, degy, degz>(
                    particleSpecies, E, B, J, dt, coeffsAStageIndex, coeffsBStageIndex);
            }
        } //end of loop over species
        J.post_particle_loop_sync();
    }

    template <int degx, int degy, int degz, unsigned int vDim, unsigned int nData>
    void lsrk_stage_push_particle_deposit_j_fully_kinetic (
        std::shared_ptr<Gempic::Particle::ParticleGroups<vDim, nData>>& particleSpecies,
        Gempic::Forms::DeRhamField<Grid::primal, Space::edge>& E,
        Gempic::Forms::DeRhamField<Grid::primal, Space::face>& B,
        Gempic::Forms::DeRhamField<Grid::dual, Space::face>& J,
        amrex::Real dt,
        amrex::Real coeffsAStageIndex,
        amrex::Real coeffsBStageIndex)
    {
        BL_PROFILE("GEMPIC::RungeKutta::lsrk_stage_push_particle_deposit_j_fully_kinetic()");
        amrex::Real charge = particleSpecies->get_charge();
        amrex::Real chargeMass = charge / particleSpecies->get_mass();

        for (auto& particleGrid : *particleSpecies)
        {
            long const np = particleGrid.numParticles();
            auto const ptd = particleGrid.GetParticleTile().getParticleTileData();
            auto const ii = particleSpecies->get_data_indices();

            amrex::GpuArray<amrex::Array4<amrex::Real>, 3> eA;
            amrex::GpuArray<amrex::Array4<amrex::Real>, 3> bA;
            amrex::GpuArray<amrex::Array4<amrex::Real>, 3> jarr;
            for (int cc = 0; cc < 3; cc++)
            {
                eA[cc] = (E.m_data[cc])[particleGrid].array();
                bA[cc] = (B.m_data[cc])[particleGrid].array();
                jarr[cc] = (J.m_data[cc])[particleGrid].array();
            }

            // loop over particles: add contribution of old particle position to J and push
            auto plo = m_infra.geometry().ProbLoArray();
            auto dxInverse = m_infra.inv_cell_size_array();
            auto dx = m_infra.cell_size_array();
            amrex::ParallelFor(
                np,
                [=] AMREX_GPU_DEVICE(long pp)
                {
                    // Read out particle data
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> R;
                    AMREX_D_EXPR(R[xDir] = ptd.rdata(ii.m_iposx)[pp],
                                 R[yDir] = ptd.rdata(ii.m_iposy)[pp],
                                 R[zDir] = ptd.rdata(ii.m_iposz)[pp]);
                    amrex::GpuArray<amrex::Real, 3> V{ptd.rdata(ii.m_ivelx)[pp],
                                                      ptd.rdata(ii.m_ively)[pp],
                                                      ptd.rdata(ii.m_ivelz)[pp]};
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> s2R{
                        AMREX_D_DECL(ptd.rdata(ii.m_is2X)[pp], ptd.rdata(ii.m_is2Y)[pp],
                                     ptd.rdata(ii.m_is2Z)[pp])};
                    amrex::GpuArray<amrex::Real, 3> s2V{ptd.rdata(ii.m_is2Vx)[pp],
                                                        ptd.rdata(ii.m_is2Vy)[pp],
                                                        ptd.rdata(ii.m_is2Vz)[pp]};

                    // evaluate the splines at particle position R
                    Gempic::ParticleMeshCoupling::SplineBase<degx, degy, degz> spline(R, plo,
                                                                                      dxInverse);
                    amrex::GpuArray<amrex::Real, vDim> efield =
                        spline.template eval_spline_field<Field::PrimalOneForm>(eA);
                    // evaluate the magnetic field at the particle position
                    amrex::GpuArray<amrex::Real, vDim> bfield =
                        spline.template eval_spline_field<Field::PrimalTwoForm>(bA);

                    Impl::lsrk_stage_kinetic_particle(R, V, s2R, s2V, efield, bfield, chargeMass,
                                                      dt, coeffsAStageIndex, coeffsBStageIndex);

                    // Update particle data
                    AMREX_D_EXPR(ptd.rdata(ii.m_iposx)[pp] = R[xDir],
                                 ptd.rdata(ii.m_iposy)[pp] = R[yDir],
                                 ptd.rdata(ii.m_iposz)[pp] = R[zDir]);
                    ptd.rdata(ii.m_ivelx)[pp] = V[0];
                    ptd.rdata(ii.m_ively)[pp] = V[1];
                    ptd.rdata(ii.m_ivelz)[pp] = V[2];
                    AMREX_D_EXPR(ptd.rdata(ii.m_is2X)[pp] = s2R[0],
                                 ptd.rdata(ii.m_is2Y)[pp] = s2R[1],
                                 ptd.rdata(ii.m_is2Z)[pp] = s2R[2]);
                    ptd.rdata(ii.m_is2Vx)[pp] = s2V[0];
                    ptd.rdata(ii.m_is2Vy)[pp] = s2V[1];
                    ptd.rdata(ii.m_is2Vz)[pp] = s2V[2];

                    // call deposit J
                    // evaluate the splines at particle position R
                    // get the spline at the particle's new position
                    Gempic::ParticleMeshCoupling::SplineBase<degx, degy, degz> newSpline(R, plo,
                                                                                         dxInverse);
                    deposit_twoform(jarr, newSpline, V, charge * ptd.rdata(ii.m_iweight)[pp]);
                }); // end of parallel_for
        } //end of loop over particles
        particleSpecies->Redistribute();
    }

    template <int degx, int degy, int degz, unsigned int vDim, unsigned int nData>
    void lsrk_stage_push_particle_deposit_j_drift_kinetic (
        std::shared_ptr<Gempic::Particle::ParticleGroups<vDim, nData>>& particleSpecies,
        Gempic::Forms::DeRhamField<Grid::primal, Space::edge>& E,
        Gempic::Forms::DeRhamField<Grid::primal, Space::face>& B,
        Gempic::Forms::DeRhamField<Grid::dual, Space::face>& J,
        amrex::Real dt,
        amrex::Real const coeffsAStageIndex,
        amrex::Real const coeffsBStageIndex)
    {
        BL_PROFILE("GEMPIC::RungeKutta::lsrk_stage_dk_push_particle_deposit_j_drift_kinetic()");
        amrex::Real charge = particleSpecies->get_charge();
        amrex::Real chargeMass = charge / particleSpecies->get_mass();

        for (auto& particleGrid : *particleSpecies)
        {
            long const np = particleGrid.numParticles();
            auto const ptd = particleGrid.GetParticleTile().getParticleTileData();
            auto const ii = particleSpecies->get_data_indices();

            amrex::GpuArray<amrex::Array4<amrex::Real>, 3> eA;
            amrex::GpuArray<amrex::Array4<amrex::Real>, 3> bA;
            amrex::GpuArray<amrex::Array4<amrex::Real>, 3> jarr;

            for (int cc = 0; cc < 3; cc++)
            {
                eA[cc] = (E.m_data[cc])[particleGrid].array();
                bA[cc] = (B.m_data[cc])[particleGrid].array();
                jarr[cc] = (J.m_data[cc])[particleGrid].array();
            }

            // loop over particles: add contribution of old particle position to J and push
            auto plo = m_infra.geometry().ProbLoArray();
            auto dxInverse = m_infra.inv_cell_size_array();
            amrex::ParallelFor(
                np,
                [=] AMREX_GPU_DEVICE(long pp)
                {
                    // Read out particle position
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> R{
                        AMREX_D_DECL(ptd.rdata(ii.m_iposx)[pp], ptd.rdata(ii.m_iposy)[pp],
                                     ptd.rdata(ii.m_iposz)[pp])};
                    amrex::GpuArray<amrex::Real, 3> V{ptd.rdata(ii.m_ivelx)[pp],
                                                      ptd.rdata(ii.m_ively)[pp],
                                                      ptd.rdata(ii.m_ivelz)[pp]};
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> s2R{
                        AMREX_D_DECL(ptd.rdata(ii.m_is2X)[pp], ptd.rdata(ii.m_is2Y)[pp],
                                     ptd.rdata(ii.m_is2Z)[pp])};
                    amrex::GpuArray<amrex::Real, 3> s2V{ptd.rdata(ii.m_is2Vx)[pp],
                                                        ptd.rdata(ii.m_is2Vy)[pp],
                                                        ptd.rdata(ii.m_is2Vz)[pp]};
                    // evaluate the splines at particle position R
                    Gempic::ParticleMeshCoupling::SplineBase<degx, degy, degz> spline(R, plo,
                                                                                      dxInverse);

                    // compute guiding center velocity and acceleration
                    amrex::GpuArray<amrex::Real, 3> velGC;
                    amrex::Real accGC;
                    Impl::dxvpardt_drift_kinetic(velGC, accGC, chargeMass, V, spline, eA, bA);

                    Impl::lsrk_stage_dk_particle(R, V, s2R, s2V, velGC, accGC, dt,
                                                 coeffsAStageIndex, coeffsBStageIndex);

                    // Update particle data
                    AMREX_D_EXPR(ptd.rdata(ii.m_iposx)[pp] = R[xDir],
                                 ptd.rdata(ii.m_iposy)[pp] = R[yDir],
                                 ptd.rdata(ii.m_iposz)[pp] = R[zDir]);
                    ptd.rdata(ii.m_ivelx)[pp] = V[0];
                    ptd.rdata(ii.m_ively)[pp] = V[1];
                    ptd.rdata(ii.m_ivelz)[pp] = V[2];
                    AMREX_D_EXPR(ptd.rdata(ii.m_is2X)[pp] = s2R[0],
                                 ptd.rdata(ii.m_is2Y)[pp] = s2R[1],
                                 ptd.rdata(ii.m_is2Z)[pp] = s2R[2]);
                    ptd.rdata(ii.m_is2Vx)[pp] = s2V[0];
                    ptd.rdata(ii.m_is2Vy)[pp] = s2V[1];
                    ptd.rdata(ii.m_is2Vz)[pp] = s2V[2];

                    // call deposit J
                    // evaluate the splines at particle position R
                    // get the spline at the particle's new position
                    Gempic::ParticleMeshCoupling::SplineBase<degx, degy, degz> newSpline(R, plo,
                                                                                         dxInverse);
                    // Only j_// !!!
                    deposit_twoform(jarr, newSpline, velGC, charge * ptd.rdata(ii.m_iweight)[pp]);
                }); // end of parallel_for
        } //end of loop over particles
        //redistribute particles
        particleSpecies->Redistribute();
    }
}; //class RungeKutta

/**
 * Runge Kutta Butcher Tableau handler
 *
 * @param nRKstages : number of stages (dof).
 */
template <int nRKstages>
struct RkButcherTableau
{
    amrex::GpuArray<amrex::GpuArray<amrex::Real, nRKstages>, nRKstages> m_a;
    amrex::GpuArray<amrex::Real, nRKstages> m_b;
    amrex::GpuArray<amrex::Real, nRKstages> m_c;

    constexpr int get_stages () const { return nRKstages; }
};

/**
 * Implicit Runge Kutta Butcher Tableau:
 * first order (backward) implicit Euler method
 */
struct ImplicitEulerRkButcherTableau : RkButcherTableau<1>
{
    amrex::GpuArray<amrex::GpuArray<amrex::Real, 1>, 1> m_a = {{{1}}};
    amrex::GpuArray<amrex::Real, 1> m_b = {1.0};
    amrex::GpuArray<amrex::Real, 1> m_c = {1.0};
};

/**
 * Explicit Runge Kutta Butcher Tableau:
 * first order (forward) explicit Euler method
 */
struct ExplicitEulerRkButcherTableau : RkButcherTableau<1>
{
    amrex::GpuArray<amrex::GpuArray<amrex::Real, 1>, 1> m_a = {{{0.0}}};
    amrex::GpuArray<amrex::Real, 1> m_b = {1.0};
    amrex::GpuArray<amrex::Real, 1> m_c = {0.0};
};

/**
 * Explicit Runge Kutta Butcher Tableau:
 * second order method
 */
struct RK2ButcherTableau : RkButcherTableau<2>
{
    //amrex::GpuArray<amrex::GpuArray<amrex::Real, 2>, 2> a = {{{{0.0, 0.0}}, {{0.5, 0.0}}}};
    amrex::GpuArray<amrex::GpuArray<amrex::Real, 2>, 2> m_a = {{{{0.0, 0.0}}, {{0.5, 0.0}}}};
    amrex::GpuArray<amrex::Real, 2> m_b = {0.0, 1.0};
    amrex::GpuArray<amrex::Real, 2> m_c = {0.0, 0.5};
};

/**
 * Explicit Runge Kutta Butcher Tableau:
 * third order RK3 method
 */
struct RK3ButcherTableau : RkButcherTableau<3>
{
    amrex::GpuArray<amrex::GpuArray<amrex::Real, 3>, 3> m_a = {
        {{{0.0, 0.0, 0.0}}, {{0.5, 0.0, 0.0}}, {{-1.0, 2.0, 0.0}}}};
    amrex::GpuArray<amrex::Real, 3> m_b = {{1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0}};
    amrex::GpuArray<amrex::Real, 3> m_c = {{0.0, 0.5, 1.0}};
};

/**
 * Explicit Runge Kutta Butcher Tableau:
 * third order SSP RK3 (Strong Stability Preserving)
 */
struct SspRK3ButcherTableau : RkButcherTableau<3>
{
    amrex::GpuArray<amrex::GpuArray<amrex::Real, 3>, 3> m_a = {
        {{{0.0, 0.0, 0.0}}, {{1.0, 0.0, 0.0}}, {{0.25, 0.25, 0.0}}}};
    amrex::GpuArray<amrex::Real, 3> m_b = {{1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0}};
    amrex::GpuArray<amrex::Real, 3> m_c = {{0.0, 1.0, 0.5}};
};

/**
 * Explicit Runge Kutta Butcher Tableau:
 * fourth order RK4 SSP
 */
struct RK4ButcherTableau : RkButcherTableau<4>
{
    amrex::GpuArray<amrex::GpuArray<amrex::Real, 4>, 4> m_a = {{{{0.0, 0.0, 0.0, 0.0}},
                                                                {{0.5, 0.0, 0.0, 0.0}},
                                                                {{0.0, 0.5, 0.0, 0.0}},
                                                                {{0.0, 0.0, 1.0, 0.0}}}};
    amrex::GpuArray<amrex::Real, 4> m_b = {{1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0}};
    amrex::GpuArray<amrex::Real, 4> m_c = {{0.0, 0.5, 0.5, 1.0}};
};

/**
 * Implicit-Explicit (IMEX) Runge Kutta Butcher Tableau handler
 */
template <int nDof>
struct ImexRkButcherTableau
{
    RkButcherTableau<nDof> m_implicitRkButcherTableau;
    RkButcherTableau<nDof> m_explicitRkButcherTableau;
};

/**
 * Implicit-Explicit (IMEX) Runge Kutta Butcher Tableau:
 * first order Euler method
 */
struct ImexEulerRkButcherTableau
{
    ImplicitEulerRkButcherTableau m_implicitRkButcherTableau;
    ExplicitEulerRkButcherTableau m_explicitRkButcherTableau;
};

/**
 * Explicit Runge Kutta Butcher Tableau for the IMEX
 * 2nd order Stiffly Accurate and Stability Preserving (Boscheri,Tavelli,Pareschi)
 */
struct SaspExplicitRK3ButcherTableau : RkButcherTableau<3>
{
    amrex::GpuArray<amrex::GpuArray<amrex::Real, 3>, 3> m_a = {
        {{{0.0, 0.0, 0.0}}, {{0.5, 0.0, 0.0}}, {{0.5, 0.5, 0.0}}}};
    amrex::GpuArray<amrex::Real, 3> m_b = {{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}};
    amrex::GpuArray<amrex::Real, 3> m_c = {{0.0, 0.5, 1.0}};
};
/**
 * Implicit Runge Kutta Butcher Tableau for the IMEX
 * 2nd order Stiffly Accurate and Stability Preserving (Boscheri,Tavelli,Pareschi)
 */
struct SaspImplicitRK3ButcherTableau : RkButcherTableau<3>
{
    amrex::GpuArray<amrex::GpuArray<amrex::Real, 3>, 3> m_a = {
        {{{0.25, 0.0, 0.0}}, {{0.0, 0.25, 0.0}}, {{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}}}};
    amrex::GpuArray<amrex::Real, 3> m_b = {{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}};
    amrex::GpuArray<amrex::Real, 3> m_c = {{0.25, 0.25, 1.0}};
};

/**
 * Implicit-Explicit (IMEX) Runge Kutta Butcher Tableau:
 * 2nd order Stiffly Accurate and Stability Preserving (Boscheri,Tavelli,Pareschi)
 */
struct SaspImexRK3ButcherTableau
{
    SaspImplicitRK3ButcherTableau m_implicitRkButcherTableau;
    SaspExplicitRK3ButcherTableau m_explicitRkButcherTableau;
};

enum class RungeKuttaTimeMode
{
    Explicit, // explicit RK
    Implicit, // (diagonally) impliciti RK
    ImEx      // (diagonally) implicit-explicit RK
};

enum class RungeKuttaTag
{
    // general methods (im,ex,imex)
    ExplicitEuler,
    RK2,
    RK3,
    RK4,
    SSPRK3,
    // only imex
    ImexEuler,
    SASPImExRK3
};

/**
 * @brief Type Selector for the Runge Kutta Butcher Tableau for RK_ButcherTableau
 * @tparam RungeKuttaTag : Runge Kutta enum class
 */
template <RungeKuttaTag>
struct RKTypeSelector;

//structure specialization
template <>
struct RKTypeSelector<RungeKuttaTag::ExplicitEuler>
{
    using selected_RK_ButcherTableau = ExplicitEulerRkButcherTableau;
    static constexpr int s_nStages{1};
    static constexpr int s_order{1};
    static constexpr RungeKuttaTimeMode s_timeMode{RungeKuttaTimeMode::Explicit};
};

template <>
struct RKTypeSelector<RungeKuttaTag::RK2>
{
    using selected_RK_ButcherTableau = RK2ButcherTableau;
    static constexpr int s_nStages{2};
    static constexpr int s_order{2};
    static constexpr RungeKuttaTimeMode s_timeMode{RungeKuttaTimeMode::Explicit};
};

template <>
struct RKTypeSelector<RungeKuttaTag::RK3>
{
    using selected_RK_ButcherTableau = RK3ButcherTableau;
    static constexpr int s_nStages{3};
    static constexpr int s_order{3};
    static constexpr RungeKuttaTimeMode s_timeMode{RungeKuttaTimeMode::Explicit};
};

template <>
struct RKTypeSelector<RungeKuttaTag::SSPRK3>
{
    using selected_RK_ButcherTableau = SspRK3ButcherTableau;
    static constexpr int s_nStages{3};
    static constexpr int s_order{3};
    static constexpr RungeKuttaTimeMode s_timeMode{RungeKuttaTimeMode::Explicit};
};

template <>
struct RKTypeSelector<RungeKuttaTag::RK4>
{
    using selected_RK_ButcherTableau = RK4ButcherTableau;
    static constexpr int s_nStages{4};
    static constexpr int s_order{4};
    static constexpr RungeKuttaTimeMode s_timeMode{RungeKuttaTimeMode::Explicit};
};

template <>
struct RKTypeSelector<RungeKuttaTag::ImexEuler>
{
    using selected_RK_ButcherTableau = ImexEulerRkButcherTableau;
    static constexpr int s_nStages{1};
    static constexpr int s_order{1};
    static constexpr RungeKuttaTimeMode s_timeMode{RungeKuttaTimeMode::ImEx};
};

template <>
struct RKTypeSelector<RungeKuttaTag::SASPImExRK3>
{
    using selected_RK_ButcherTableau = SaspImexRK3ButcherTableau;
    static constexpr int s_nStages{3};
    static constexpr int s_order{2};
    static constexpr RungeKuttaTimeMode s_timeMode{RungeKuttaTimeMode::ImEx};
};

/**
 * Base Handler structure for the physical state
 *
 * @param Derived : using CRTP
 */
template <typename Derived>
struct BaseFieldsHandlerStruct
{
    virtual ~BaseFieldsHandlerStruct() = default; // Virtual destructor
    // Assing function using CRTP
    void operator=(BaseFieldsHandlerStruct const& other)
    {
        static_cast<Derived*>(this)->assign(static_cast<Derived const&>(
            other)); // define template base class, defining the = operator, calling the assign
                     // function to be implemented in the derived classes.
    }

    /**
     * Addition and scaling function using CRTP
     *
     * @param Derived : using CRTP
     */
    void mult_and_add (amrex::Real alpha, std::shared_ptr<BaseFieldsHandlerStruct> const& other)
    {
        static_cast<Derived*>(this)->mult_and_add(alpha, static_cast<Derived const&>(other));
    }
};

/**
 * Explicit Runkge Kutta Class.
 *
 * @param FieldsHandler: physical state handler
 * @tparam userRkTag : Runge Kutta enum class
 */
template <class FieldsHandler, RungeKuttaTag userRkTag>
class ExplicitRK
{
public:
    static constexpr int s_nRKstages{RKTypeSelector<userRkTag>::s_nStages};
    static_assert(RKTypeSelector<userRkTag>::s_timeMode == RungeKuttaTimeMode::Explicit,
                  "Only Explicit mode is supported by ExplicitRK!");
    static_assert(s_nRKstages >= 0, "Only positive number of RK stages are supported!");
    using selectedRK_Tableau = typename RKTypeSelector<userRkTag>::selected_RK_ButcherTableau;
    using RKStagesArray = typename std::array<std::shared_ptr<FieldsHandler>, s_nRKstages>;
    // Create an instance of the selected type
    RkButcherTableau<s_nRKstages> m_rkTableau;

    void integrate_step(amrex::Real const time, amrex::Real const dt);

    void one_step_solve_dqdt_explicit(amrex::Real rkTime);

    void finalize_rk(amrex::Real const dt,
                     std::shared_ptr<FieldsHandler>& dst,
                     RKStagesArray const& srcArray);
    /**
     * @brief constructor of the ExplicitRK class
     * @param solutionInOutSrc : field struct, handling the initial and final state for the RK
     * time-step.
     * @param numSchemeSrc : numerical scheme struct, to be correctly designed.
     * @param drc : deRham complex used only to define the DeRhamFields.
     * @param rKreference : selected type for the RK Butcher Tableau.
     */
    ExplicitRK(std::shared_ptr<FieldsHandler>& solutionInOutSrc,
               NumericalScheme<FieldsHandler>& numSchemeSrc,
               std::shared_ptr<Gempic::Forms::FDDeRhamComplex>& drc,
               selectedRK_Tableau const& rKreference) :
        m_solutionInOut(solutionInOutSrc), m_scheme(numSchemeSrc)

    {
        m_solutionTmpExplicit = m_scheme.get_pointer_initial_field_explicit();
        m_timeDerivative = m_scheme.get_pointer_time_derivative_explicit();
        for (int i = 0; i < s_nRKstages; ++i)
        {
            m_timeDerivativeStages[i] = std::make_shared<FieldsHandler>(drc, m_scheme.get_ncomp());
            *(m_timeDerivativeStages[i]) = *m_timeDerivative;
        }

        for (int i = 0; i < s_nRKstages; ++i)
        {
            m_rkTableau.m_b[i] = rKreference.m_b[i];
            m_rkTableau.m_c[i] = rKreference.m_c[i];
            for (int j = 0; j < s_nRKstages; ++j)
            {
                m_rkTableau.m_a[i][j] = rKreference.m_a[i][j];
            }
        }
    }

private:
    std::shared_ptr<FieldsHandler> m_solutionInOut; // initial and final solution handler
    std::shared_ptr<FieldsHandler>
        m_solutionTmpExplicit; // this is the solution handler used in m_numScheme (to compute the
                               // timeDerivative)
    std::shared_ptr<FieldsHandler> m_timeDerivative; // this is the timeDerivative
    NumericalScheme<FieldsHandler>& m_scheme;
    RKStagesArray m_timeDerivativeStages{};
};

/**
 * @brief compute the time derivative of the explicit terms for a single step (e.g. a RK stage)
 * @param rkTime : time, typically corresponds to the RK stage.
 */
template <class FieldsHandler, RungeKuttaTag userRkTag>
void ExplicitRK<FieldsHandler, userRkTag>::one_step_solve_dqdt_explicit (amrex::Real rkTime)
{
    BL_PROFILE("GEMPIC::RungeKutta::ExplicitRK::one_step_solve_dqdt_explicit()");
    m_scheme.one_step_solve_dqdt_explicit(rkTime);
}

/**
 * @brief compute final state from the RK stages
 * @param dt : time step size
 * @param dst : destination FieldsHandler
 * @param srcArray : array of FieldsHandler with the RK stages
 */
template <class FieldsHandler, RungeKuttaTag userRkTag>
void ExplicitRK<FieldsHandler, userRkTag>::finalize_rk (amrex::Real const dt,
                                                       std::shared_ptr<FieldsHandler>& dst,
                                                       RKStagesArray const& srcArray)
{
    BL_PROFILE("GEMPIC::RungeKutta::ExplicitRK::finalize_rk()");
    for (int r = 0; r < s_nRKstages; r++)
    {
        amrex::Real const coeff = m_rkTableau.m_b[r] * dt;
        dst->mult_and_add(coeff, srcArray[r]);
    }
}

/**
 * @brief run RK method: evolve the field struct of a time-step 'dt' at time 'time' using the
 * specified RK algorithm
 * @param time : actual time
 * @param dt : time-step size
 */
template <class FieldsHandler, RungeKuttaTag userRkTag>
void ExplicitRK<FieldsHandler, userRkTag>::integrate_step (amrex::Real const time,
                                                          amrex::Real const dt)
{
    BL_PROFILE("GEMPIC::RungeKutta::ExplicitRK::integrate_step()");
    for (int s = 0; s < s_nRKstages; s++)
    {
        *m_solutionTmpExplicit = *m_solutionInOut;
        for (int r = 0; r < s; r++)
        {
            amrex::Real const coeff = m_rkTableau.m_a[s][r] * dt;
            m_solutionTmpExplicit->mult_and_add(coeff, m_timeDerivativeStages[r]);
        }
        //amrex::Real rkTime{time};
        amrex::Real timeE{time + m_rkTableau.m_c[s] * dt};
        one_step_solve_dqdt_explicit(timeE); // this will use m_solutionTmpExplicit as initial state
                                             // to
                                             // compute m_timeDerivative
        *(m_timeDerivativeStages[s]) = *m_timeDerivative;
    }
    finalize_rk(dt, m_solutionInOut, m_timeDerivativeStages);
    // we don't need fill boundary. it's done in one_step_solve_dqdt_explicit.
}

/**
 * Implicit-Explicit Runkge Kutta Class.
 *
 * @param FieldsHandler : physical state handler
 * @tparam userRkTag : Runge Kutta enum class
 */
template <class FieldsHandler, RungeKuttaTag userRkTag>
class ImexRk
{
public:
    static constexpr int s_nRKstages{RKTypeSelector<userRkTag>::s_nStages};
    static_assert(RKTypeSelector<userRkTag>::s_timeMode == RungeKuttaTimeMode::ImEx,
                  "Only ImEx mode is supported by ImexRk!");
    static_assert(s_nRKstages >= 0, "Only positive number of RK stages are supported!");
    using selectedRK_Tableau = typename RKTypeSelector<userRkTag>::selected_RK_ButcherTableau;
    using RKStagesArray = typename std::array<std::shared_ptr<FieldsHandler>, s_nRKstages>;
    // Create an instance of the selected type
    ImexRkButcherTableau<s_nRKstages> m_rkTableau;

    void integrate_step(amrex::Real const time, amrex::Real const dt);

    void one_step_solve_dqdt_semiimplicit(amrex::Real const timeI,
                                          amrex::Real const dtI,
                                          amrex::Real const timeE,
                                          amrex::Real const dtE);

    void finalize_rk(amrex::Real const dt,
                     std::shared_ptr<FieldsHandler>& dst,
                     RKStagesArray const& srcArray);

    ImexRk(std::shared_ptr<FieldsHandler>& solutionInOutSrc,
           NumericalScheme<FieldsHandler>& numSchemeSrc,
           std::shared_ptr<Gempic::Forms::FDDeRhamComplex>& drc,
           selectedRK_Tableau const& rKreference) :
        m_solutionInOut(solutionInOutSrc), // initial and final solution handler
        m_scheme(numSchemeSrc)

    {
        m_solutionTmpExplicit = m_scheme.get_pointer_initial_field_explicit();
        m_solutionTmpImplicit = m_scheme.get_pointer_initial_field_implicit();
        m_timeDerivative = m_scheme.get_pointer_time_derivative_implicit();

        for (int i = 0; i < s_nRKstages; ++i)
        {
            m_timeDerivativeStages[i] = std::make_shared<FieldsHandler>(drc, m_scheme.get_ncomp());
            *m_timeDerivativeStages[i] = *m_timeDerivative;
        }
        for (int i = 0; i < s_nRKstages; ++i)
        {
            m_rkTableau.m_explicitRkButcherTableau.m_b[i] =
                rKreference.m_explicitRkButcherTableau.m_b[i];
            m_rkTableau.m_explicitRkButcherTableau.m_c[i] =
                rKreference.m_explicitRkButcherTableau.m_c[i];
            for (int j = 0; j < s_nRKstages; ++j)
            {
                m_rkTableau.m_explicitRkButcherTableau.m_a[i][j] =
                    rKreference.m_explicitRkButcherTableau.m_a[i][j];
            }
            m_rkTableau.m_implicitRkButcherTableau.m_b[i] =
                rKreference.m_implicitRkButcherTableau.m_b[i];
            m_rkTableau.m_implicitRkButcherTableau.m_c[i] =
                rKreference.m_implicitRkButcherTableau.m_c[i];
            for (int j = 0; j < s_nRKstages; ++j)
            {
                m_rkTableau.m_implicitRkButcherTableau.m_a[i][j] =
                    rKreference.m_implicitRkButcherTableau.m_a[i][j];
                if (Gempic::Utils::Verbosity::level() > 0)
                {
                    amrex::Print()
                        << "(i,j) = (" << i << "," << j
                        << ");  b_i, a_ij = " << m_rkTableau.m_implicitRkButcherTableau.m_b[i]
                        << ", " << m_rkTableau.m_implicitRkButcherTableau.m_a[i][j] << std::endl;
                }
            }
        }
    }

private:
    std::shared_ptr<FieldsHandler>
        m_solutionInOut; // this is the initial and final state (by reference)
    std::shared_ptr<FieldsHandler> m_solutionTmpExplicit;
    std::shared_ptr<FieldsHandler> m_solutionTmpImplicit;
    std::shared_ptr<FieldsHandler> m_timeDerivative;
    NumericalScheme<FieldsHandler>& m_scheme;
    RKStagesArray m_timeDerivativeStages{};
};

/**
 * @brief compute the time derivative of the semi-implicit terms for a single step.
 * @param timeI : time of the implicit step
 * @param dtI : time-step size of the implicit step
 * @param timeE : time of the explicit step
 * @param dtE : time-step size of the explicit step
 */
template <class FieldsHandler, RungeKuttaTag userRkTag>
void ImexRk<FieldsHandler, userRkTag>::one_step_solve_dqdt_semiimplicit (amrex::Real const timeI,
                                                                        amrex::Real const dtI,
                                                                        amrex::Real const timeE,
                                                                        amrex::Real const dtE)
{
    BL_PROFILE("GEMPIC::RungeKutta::ImexRk::one_step_solve_dqdt_semiimplicit()");
    m_scheme.one_step_solve_dqdt_semiimplicit(timeI, dtI, timeE, dtE);
}

/**
 * @brief compute final state from the IMEX RK stages
 * @param dt : time step size
 * @param dst : destination FieldsHandler
 * @param srcArray : array of FieldsHandler with the RK stages
 */
template <class FieldsHandler, RungeKuttaTag userRkTag>
void ImexRk<FieldsHandler, userRkTag>::finalize_rk (amrex::Real const dt,
                                                   std::shared_ptr<FieldsHandler>& dst,
                                                   RKStagesArray const& srcArray)
{
    BL_PROFILE("GEMPIC::RungeKutta::ImexRk::finalize_rk()");
    for (int r = 0; r < s_nRKstages; r++)
    {
        amrex::Real const coeff = m_rkTableau.m_implicitRkButcherTableau.m_b[r] * dt;
        dst->mult_and_add(coeff, srcArray[r]);
    }
}

/**
 * @brief run IMEX RK method: evolve the field struct of a time-step 'dt' at time 'time' using the
 * specified RK algorithm
 * @param time : actual time
 * @param dt : time-step size
 */
template <class FieldsHandler, RungeKuttaTag userRkTag>
void ImexRk<FieldsHandler, userRkTag>::integrate_step (amrex::Real const time, amrex::Real const dt)
{
    BL_PROFILE("GEMPIC::RungeKutta::ImexRk::integrate_step()");
    // use  the multi components to cast the multi-stage vectors;
    if (Gempic::Utils::Verbosity::level() > 0)
    {
        amrex::Print() << "time = " << time << ",  dt = " << dt << std::endl;
    }
    for (int s = 0; s < s_nRKstages; s++)
    {
        *m_solutionTmpExplicit = *m_solutionInOut;
        for (int r = 0; r < s; r++)
        {
            amrex::Real const coeff = m_rkTableau.m_explicitRkButcherTableau.m_a[s][r] * dt;
            m_solutionTmpExplicit->mult_and_add(coeff, m_timeDerivativeStages[r]);
        }
        *m_solutionTmpImplicit = *m_solutionInOut;
        for (int r = 0; r < s; r++)
        {
            amrex::Real const coeff = m_rkTableau.m_implicitRkButcherTableau.m_a[s][r] * dt;
            m_solutionTmpImplicit->mult_and_add(coeff, m_timeDerivativeStages[r]);
        }
        amrex::Real timeE{time + m_rkTableau.m_explicitRkButcherTableau.m_c[s] * dt};
        amrex::Real timeI{time + m_rkTableau.m_implicitRkButcherTableau.m_c[s] * dt};
        amrex::Real dtE{m_rkTableau.m_explicitRkButcherTableau.m_a[s][s] * dt};
        amrex::Real dtI{m_rkTableau.m_implicitRkButcherTableau.m_a[s][s] * dt};
        one_step_solve_dqdt_semiimplicit(timeI, dtI, timeE, dtE);
        // this is a semi-implicit solver for a time-step
        // a_ii*dt, ending with a value for the time derivative.
        // the explicit step is solved at a different time!  time+tilde{c}*dt, tilde{a}_ii*dt
        *(m_timeDerivativeStages[s]) = *m_timeDerivative;
    }
    finalize_rk(dt, m_solutionInOut, m_timeDerivativeStages);
    // we don't need fill boundary. it's done in one_step_solve_dqdt_explicit.
}

} // namespace Gempic::TimeLoop
#endif
