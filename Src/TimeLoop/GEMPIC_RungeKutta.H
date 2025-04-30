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
    BL_PROFILE("Gempic::TimeLoop::Impl::lsrk_stage_dk_particle()");

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
    BL_PROFILE("Gempic::TimeLoop::Impl::lsrk_stage_kinetic_particle()");
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
    BL_PROFILE("Gempic::TimeLoop::Impl::dxvpardt_drift_kinetic");
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

            auto& particles = particleGrid.GetArrayOfStructs();
            auto partData = particles().data();
            auto& particleAttributes = particleGrid.GetStructOfArrays();
            auto vx = particleAttributes.GetRealData(0).data();
            auto vy = particleAttributes.GetRealData(1).data();
            auto vz = particleAttributes.GetRealData(2).data();
            auto weight = particleAttributes.GetRealData(3).data();

            auto s2X = particleAttributes.GetRealData(4).data();
            auto s2Y = particleAttributes.GetRealData(5).data();
            auto s2Z = particleAttributes.GetRealData(6).data();
            auto s2Vx = particleAttributes.GetRealData(7).data();
            auto s2Vy = particleAttributes.GetRealData(8).data();
            auto s2Vz = particleAttributes.GetRealData(9).data();

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
                    // Read out particle position
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> R;

                    amrex::GpuArray<amrex::Real, 3> V{vx[pp], vy[pp], vz[pp]};

                    for (unsigned int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        R[d] = partData[pp].pos(d);
                    }
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> s2R{
                        AMREX_D_DECL(s2X[pp], s2Y[pp], s2Z[pp])};
                    amrex::GpuArray<amrex::Real, 3> s2V{s2Vx[pp], s2Vy[pp], s2Vz[pp]};
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

                    for (unsigned int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        partData[pp].pos(d) = R[d];
                    }
                    vx[pp] = V[0];
                    vy[pp] = V[1];
                    vz[pp] = V[2];

                    AMREX_D_EXPR(s2X[pp] = s2R[0], s2Y[pp] = s2R[1], s2Z[pp] = s2R[2]);

                    s2Vx[pp] = s2V[0];
                    s2Vy[pp] = s2V[1];
                    s2Vz[pp] = s2V[2];

                    // call deposit J
                    // evaluate the splines at particle position R
                    // get the spline at the particle's new position
                    Gempic::ParticleMeshCoupling::SplineBase<degx, degy, degz> newSpline(R, plo,
                                                                                         dxInverse);
                    deposit_j(jarr, newSpline, V, charge * weight[pp]);
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

            auto& particles = particleGrid.GetArrayOfStructs();
            auto partData = particles().data();
            auto& particleAttributes = particleGrid.GetStructOfArrays();
            auto vx = particleAttributes.GetRealData(0).data();
            auto vy = particleAttributes.GetRealData(1).data();
            auto vz = particleAttributes.GetRealData(2).data(); //v_//
            auto weight = particleAttributes.GetRealData(3).data();

            auto s2X = particleAttributes.GetRealData(4).data();
            auto s2Y = particleAttributes.GetRealData(5).data();
            auto s2Z = particleAttributes.GetRealData(6).data();
            auto s2Vx = particleAttributes.GetRealData(7).data();
            auto s2Vy = particleAttributes.GetRealData(8).data();
            auto s2Vz = particleAttributes.GetRealData(9).data();

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
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> R;

                    amrex::GpuArray<amrex::Real, 3> V{vx[pp], vy[pp], vz[pp]};

                    for (unsigned int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        R[d] = partData[pp].pos(d);
                    }
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> s2R{
                        AMREX_D_DECL(s2X[pp], s2Y[pp], s2Z[pp])};
                    amrex::GpuArray<amrex::Real, 3> s2V{s2Vx[pp], s2Vy[pp], s2Vz[pp]};
                    // evaluate the splines at particle position R
                    Gempic::ParticleMeshCoupling::SplineBase<degx, degy, degz> spline(R, plo,
                                                                                      dxInverse);

                    // compute guiding center velocity and acceleration
                    amrex::GpuArray<amrex::Real, 3> velGC;
                    amrex::Real accGC;
                    Impl::dxvpardt_drift_kinetic(velGC, accGC, chargeMass, V, spline, eA, bA);

                    Impl::lsrk_stage_dk_particle(R, V, s2R, s2V, velGC, accGC, dt,
                                                 coeffsAStageIndex, coeffsBStageIndex);

                    for (unsigned int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        partData[pp].pos(d) = R[d];
                    }
                    vx[pp] = V[0];
                    vy[pp] = V[1];
                    vz[pp] = V[2];

                    AMREX_D_EXPR(s2X[pp] = s2R[0], s2Y[pp] = s2R[1], s2Z[pp] = s2R[2]);

                    s2Vx[pp] = s2V[0];
                    s2Vy[pp] = s2V[1];
                    s2Vz[pp] = s2V[2];

                    // call deposit J
                    // evaluate the splines at particle position R
                    // get the spline at the particle's new position
                    Gempic::ParticleMeshCoupling::SplineBase<degx, degy, degz> newSpline(R, plo,
                                                                                         dxInverse);
                    // Only j_// !!!
                    deposit_j(jarr, newSpline, velGC, charge * weight[pp]);
                }); // end of parallel_for
        } //end of loop over particles
        //redistribute particles
        particleSpecies->Redistribute();
    }
}; //class RungeKutta
} // namespace Gempic::TimeLoop
#endif
