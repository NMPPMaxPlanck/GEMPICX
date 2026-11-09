/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef GEMPIC_QUASINEUTRALSOLVER_DEBUG_H
#define GEMPIC_QUASINEUTRALSOLVER_DEBUG_H

#include <AMReX_HypreSolver.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_Particle.H"
#include "GEMPIC_ParticleMeshCoupling.H"

#ifndef AMREX_USE_HYPRE
#error "HYPRE not enabled"
#endif

// Functions made for debugging start
namespace Gempic::FieldSolvers
{
template <unsigned int vdim, unsigned int ndata>
void check_energies_qn (
    Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& B,
    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::edge>& H,
    std::vector<std::shared_ptr<Gempic::Particle::ParticleSpecies<vdim, ndata>>>& particles,
    ComputationalDomain& mInfra)
{
    amrex::Real me{0.0};
    for (int dir = 0; dir < 3; dir++)
    {
        auto mask = amrex::OwnerMask(B.m_data[dir], mInfra.geometry().periodicity());
        me += 0.5 * amrex::MultiFab::Dot(*mask, B.m_data[dir], 0, H.m_data[dir], 0,
                                         B.m_data[dir].nComp(), 0, 0);
    }

    amrex::Real ke{0.0};

    for (auto& particleSpecies : particles)
    {
        amrex::Real mass = particleSpecies->get_mass();

        auto const ii = particleSpecies->get_data_indices();
        for (auto& pti : *particleSpecies)
        {
            long const np = pti.numParticles();
            auto* const weight = pti.GetStructOfArrays().GetRealData(ii.m_iweight).data();
            auto* const velx = pti.GetStructOfArrays().GetRealData(ii.m_ivelx).data();
            auto* const vely = pti.GetStructOfArrays().GetRealData(ii.m_ively).data();
            auto* const velz = pti.GetStructOfArrays().GetRealData(ii.m_ivelz).data();

            amrex::Real keSpecies = 0.0;

            Kokkos::parallel_reduce(
                "QuasineutralDebug::check_energies", np,
                KOKKOS_LAMBDA(long const pp, amrex::Real& localKe) {
                    amrex::GpuArray<amrex::Real, vdim> vel{velx[pp], vely[pp], velz[pp]};

                    localKe +=
                        weight[pp] * 0.5 * mass *
                        (vel[xDir] * vel[xDir] + vel[yDir] * vel[yDir] + vel[zDir] * vel[zDir]);
                },
                keSpecies);

            ke += keSpecies;
        }
    }

    amrex::ParallelDescriptor::ReduceRealSum({ke}, amrex::ParallelDescriptor::IOProcessorNumber());

    amrex::Print() << "(KE, ME):(" << ke << ", " << me << ")  TE: " << (ke + me) << std::endl;
}

template <unsigned int vdim, unsigned int ndata, int degx, int degy, int degz>
void check_e_related_norms_qn (
    Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& E,
    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell>& rho,
    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell>& rhoIdeal,
    std::shared_ptr<Gempic::Forms::FDDeRhamComplex> drc,
    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face> const& rhs,
    std::vector<std::shared_ptr<Gempic::Particle::ParticleSpecies<vdim, ndata>>>& particles,
    ComputationalDomain& mInfra)
{
    amrex::Print() << "\nError checks for E-related computations:";

    // lhs = CurlCurlE + rhoE = rhs check
    Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face> curlE(drc);
    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::edge> curlEProj(drc);
    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face> curlCurlE(drc);
    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face> rhoE(drc);

    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face> lhs(drc);

    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell> divRhoE(drc);
    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell> divLhs(drc);
    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell> divRhs(drc);

    curl(curlE, E);
    hodge(curlEProj, curlE, drc->scaling_bto_h());
    curl(curlCurlE, curlEProj);

    Gempic::ParticleMeshCoupling::deposit_rho_weighted_electric_field<degx, degy, degz, vdim,
                                                                      ndata>(rhoE, E, particles,
                                                                             mInfra);

    linear_combination(lhs, 1.0, curlCurlE, 1.0, rhoE);

    div(divRhoE, rhoE);
    div(divLhs, lhs);
    div(divRhs, rhs);

    amrex::Print() << "\nlhs_norm: (" << Utils::gempic_norm(lhs.m_data[xDir], mInfra, 1) << ","
                   << Utils::gempic_norm (lhs.m_data[yDir], mInfra, 1) << ","
                   << Utils::gempic_norm (lhs.m_data[zDir], mInfra, 1)
                   << "), lhs_div: " << Utils::gempic_norm(divLhs.m_data, mInfra, 1)
                   << ", rhoE_div: " << Utils::gempic_norm(divRhoE.m_data, mInfra, 1)
                   << "\nrhs_norm: (" << Utils::gempic_norm(rhs.m_data[xDir], mInfra, 1) << ","
                   << Utils::gempic_norm (rhs.m_data[yDir], mInfra, 1) << ","
                   << Utils::gempic_norm (rhs.m_data[zDir], mInfra, 1)
                   << "), rhs_div: " << Utils::gempic_norm(divRhs.m_data, mInfra, 1);

    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell> rhoError(drc);
    linear_combination(rhoError, 1.0, rho, -1.0, rhoIdeal);
    amrex::Real den = Utils::gempic_norm(rhoIdeal.m_data, mInfra, 1);
    if (den == 0.0) den = 1.0;
    amrex::Print() << "\nrho error %: "
                   << (Utils::gempic_norm(rhoError.m_data, mInfra, 1) / den) * 100.0 << "\n";
}

void check_b_calculation_qn(
    Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& E,
    Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& bOld,
    Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& B,
    std::shared_ptr<Gempic::Forms::FDDeRhamComplex> drc,
    amrex::Real dt,
    ComputationalDomain& mInfra);

void check_b_related_norms_qn(
    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& J,
    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell>& divJ,
    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& A,
    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell>& divA,
    Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& B,
    Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::cell>& divB,
    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::edge>& H,
    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& curlH,
    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& jMinusCurlH,
    ComputationalDomain& mInfra);

void check_j_change_norms_qn(
    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& jOld,
    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& jNew,
    std::shared_ptr<Gempic::Forms::FDDeRhamComplex> drc,
    ComputationalDomain& mInfra);
} //namespace Gempic::FieldSolvers
  // Functions made for debugging end
#endif
