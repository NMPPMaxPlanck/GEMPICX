/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef GEMPIC_DKQUASINEUTRALSOLVER_H_
#define GEMPIC_DKQUASINEUTRALSOLVER_H_

#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <AMReX.H>
#include <AMReX_MultiFab.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_DKQuasineutralCalculationE.H"
#include "GEMPIC_DKQuasineutralDebug.H"
#include "GEMPIC_DKQuasineutralTimestepping.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Particle.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_ParticleUtils.H"
#include "GEMPIC_RungeKutta.H"
#include "GEMPIC_Verbosity.H"

namespace Gempic::FieldSolvers
{

template <int hodgeDegree, unsigned int vdim, unsigned int ndata, int degx, int degy, int degz>
class DKQuasineutralSolver
{
private:
    ComputationalDomain m_infra;
    std::shared_ptr<Forms::FDDeRhamComplex> m_deRham;

    DKQuasineutralCalculationE<hodgeDegree, vdim, ndata, degx, degy, degz> m_calculationE;

    DKQuasineutralTimestepping<hodgeDegree, vdim, ndata, degx, degy, degz> m_timestepping;
    std::string m_timesteppingType{"onlyrk"};
    Gempic::TimeLoop::RungeKutta m_lsrk;

    Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face> m_s2B;

public:
    DKQuasineutralSolver(ComputationalDomain infr,
                         std::shared_ptr<Forms::FDDeRhamComplex> drc,
                         amrex::Real relTol = 1.e-12) :
        m_infra{infr},
        m_deRham{drc},
        m_calculationE(infr, drc, relTol),
        m_timestepping(infr, drc),
        m_lsrk(infr, drc),
        m_s2B(drc)
    {
        static_assert(vdim == 3, "vdim should be 3");

        Io::Parameters params("TimeLoop");
        params.get("timestepping", m_timesteppingType);
    }

    // Functions for time stepping begin
    void time_stepping_quasineutral_drift_electrons_kinetic_ions (
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& E,
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& estar,
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& D,
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& B,
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::edge>& H,
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& curlH,
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& J,
        std::vector<std::shared_ptr<Particle::ParticleSpecies<vdim, ndata>>>& partGr,
        int bdir,
        std::array<int, 2> const& pdir,
        amrex::Real dt,
        std::vector<Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell>>& rhoVec,
        std::vector<Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell>>& rhoMassVec,
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell>& rho)
    {
        Kokkos::Profiling::ScopedRegion region(
            "GEMPIC::DKQuasineutralSolver::time_stepping_quasineutral_drift_electrons_kinetic_ions("
            ")");

        // For the Faraday equation's Crank-Nicolson steps
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face> rhsFaradayCnEqn(m_deRham);

        int index = 0;
        int indexE = 0;
        for (auto& particleSpecies : partGr)
        {
            if (particleSpecies->get_name() == "electrons")
            {
                indexE = index;
            }
            index++;
        }

        if (m_timesteppingType == "onlyrk")
        {
            lsrk_quasineutral_drift_electrons_kinetic_ions_onlyrk_or_imex(
                E, estar, D, B, H, curlH, J, partGr, bdir, pdir, dt, rhoVec, rhoMassVec, rho);
        }
        else if (m_timesteppingType == "cnrkcn")
        {
            m_timestepping.de_ki_faraday_crank_nicolson_step(B, H, curlH, rhsFaradayCnEqn, bdir,
                                                             pdir, 0.5 * dt, rhoVec[indexE]);

            lsrk_quasineutral_drift_electrons_kinetic_ions_onlyrk_or_imex(
                E, estar, D, B, H, curlH, J, partGr, bdir, pdir, dt, rhoVec, rhoMassVec, rho);

            m_timestepping.de_ki_faraday_crank_nicolson_step(B, H, curlH, rhsFaradayCnEqn, bdir,
                                                             pdir, 0.5 * dt, rhoVec[indexE]);
        }
        else if (m_timesteppingType == "rkcnrk")
        {
            lsrk_quasineutral_drift_electrons_kinetic_ions_onlyrk_or_imex(
                E, estar, D, B, H, curlH, J, partGr, bdir, pdir, 0.5 * dt, rhoVec, rhoMassVec, rho);

            m_timestepping.de_ki_faraday_crank_nicolson_step(B, H, curlH, rhsFaradayCnEqn, bdir,
                                                             pdir, dt, rhoVec[indexE]);

            lsrk_quasineutral_drift_electrons_kinetic_ions_onlyrk_or_imex(
                E, estar, D, B, H, curlH, J, partGr, bdir, pdir, 0.5 * dt, rhoVec, rhoMassVec, rho);
        }
    }

    void lsrk_quasineutral_drift_electrons_kinetic_ions_onlyrk_or_imex (
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& E,
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& estar,
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& D,
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& B,
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::edge>& H,
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& curlH,
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& J,
        std::vector<std::shared_ptr<Particle::ParticleSpecies<vdim, ndata>>>& partGr,
        int bdir,
        std::array<int, 2> const& pdir,
        amrex::Real dt,
        std::vector<Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell>>& rhoVec,
        std::vector<Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell>>& rhoMassVec,
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell>& rho)
    {
        Kokkos::Profiling::ScopedRegion region(
            "GEMPIC::DKQuasineutralSolver::lsrk_quasineutral_drift_electrons_kinetic_ions_onlyrk_"
            "or_imex()");

        // For the E_perp equation
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face> rhsEperpEqn(m_deRham);

        // For the E_parallel equation
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face> rhsEparallelEqn(m_deRham);

        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face> minusJ(m_deRham);

        for (int stageIndex = 0; stageIndex < m_lsrk.get_stages(); stageIndex++)
        {
            Gempic::ParticleMeshCoupling::deposit_individual_particle_density<degx, degy, degz>(
                rhoVec, rho, rhoMassVec, partGr, m_infra);

            // Initialize E to zero before calculating it
            for (int i = 0; i < vdim; i++)
            {
                E.m_data[i].setVal(0.0);
                estar.m_data[i].setVal(0.0);
            }

            // Calculate RHS of the E_perp equation on dual face; solve for E_perp(t_n); check if
            // E_perp was calculated correctly
            m_calculationE.de_ki_calculate_rhs_eperp_eqn(rhsEperpEqn, E, B, H, curlH, J, partGr,
                                                         bdir);
            copy(minusJ, J);
            minusJ *= -1.0;

            m_calculationE.solve_de_fi_model_eperp(E, rhsEperpEqn, B, partGr, bdir, pdir, rhoVec);
            if (m_timesteppingType != "onlyrk")
            {
                m_calculationE.solve_de_fi_model_eperp(estar, minusJ, B, partGr, bdir, pdir,
                                                       rhoVec);
            }

            // Calculate RHS of the E_par equation on dual face; solve for E_par(t_n); check if
            // E_par was calculated correctly
            m_calculationE.de_ki_calculate_rhs_epar_eqn(rhsEparallelEqn, E, B, partGr, bdir, pdir);
            m_calculationE.solve_de_fi_model_epar(E, rhsEparallelEqn, partGr, bdir, rhoVec);

            // Calculate D(t_n)
            hodge(D, E, m_deRham->scaling_eto_d());

            // LSRK: Push particles using E(t_n) and B(t_n)
            m_timestepping.lsrk_de_fi_push_particles(partGr, E, B, dt, bdir, stageIndex);

            // LSRK: Update B(t_(n+1)) using curl of E(t_n)
            if (m_timesteppingType == "onlyrk")
            {
                m_timestepping.lsrk_update_b_from_curl_e(B, E, dt, stageIndex);
            }
            else
            {
                m_timestepping.lsrk_update_b_from_curl_e(B, estar, dt, stageIndex);
            }

            // update H(t_(n+1))
            hodge(H, B, m_deRham->scaling_bto_h());
        }
    }
    // Functions for time stepping end
};

} // namespace Gempic::FieldSolvers

#endif
