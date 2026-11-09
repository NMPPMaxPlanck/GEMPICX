/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef GEMPIC_QUASINEUTRALSOLVER_DEBUG_H
#define GEMPIC_QUASINEUTRALSOLVER_DEBUG_H

#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <AMReX_HypreSolver.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_Particle.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_ParticleUtils.H" // DeltaF method

#ifndef AMREX_USE_HYPRE
#error "HYPRE not enabled"
#endif

// Functions made for debugging start
namespace Gempic::FieldSolvers
{
template <unsigned int vdim, unsigned int ndata>
void check_energies_dkqn (
    Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& B,
    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::edge>& H,
    std::vector<std::shared_ptr<Gempic::Particle::ParticleSpecies<vdim, ndata>>>& partGr,
    ComputationalDomain& mInfra,
    int bdir)
{
    Kokkos::Profiling::ScopedRegion region("check_energies_DKQN()");

    amrex::Real me{0.0};
    for (int dir = 0; dir < 3; dir++)
    {
        auto mask = amrex::OwnerMask(B.m_data[dir], mInfra.geometry().periodicity());
        me += 0.5 * amrex::MultiFab::Dot(*mask, B.m_data[dir], 0, H.m_data[dir], 0,
                                         B.m_data[dir].nComp(), 0, 0);
    }

    amrex::Real ke{0.0};

    for (auto& particleSpecies : partGr)
    {
        amrex::Real mass = particleSpecies->get_mass();

        if (particleSpecies->get_name() == "electrons")
        {
            amrex::ReduceOps<amrex::ReduceOpSum> reduceOps;
            using PTDType = typename Gempic::Particle::ParticleSpecies<
                vdim, ndata>::ParticleTileType::ConstParticleTileDataType;
            auto const ii = particleSpecies->get_data_indices();
            auto particleTmp = amrex::ParticleReduce<amrex::ReduceData<amrex::Real>>(
                *particleSpecies, 0, particleSpecies->finestLevel(),
                [=] AMREX_GPU_HOST_DEVICE(PTDType const& pt,
                                          int const pp) -> amrex::GpuTuple<amrex::Real>
                {
                    auto const weight = pt.rdata(ii.m_iweight)[pp];
                    amrex::Real weightEff = weight;
                    auto const velx = pt.rdata(ii.m_ivelx)[pp];
                    auto const vely = pt.rdata(ii.m_ively)[pp];
                    auto const velz = pt.rdata(ii.m_ivelz)[pp];
                    std::array<amrex::Real, vdim> vel{velx, vely, velz};

                    return {weightEff * 0.5 * mass * vel[bdir] * vel[bdir]};
                },
                reduceOps);

            ke += amrex::get<0>(particleTmp);
        }
        else
        { // "ions"
            amrex::ReduceOps<amrex::ReduceOpSum> reduceOps;
            using PTDType = typename Gempic::Particle::ParticleSpecies<
                vdim, ndata>::ParticleTileType::ConstParticleTileDataType;
            auto const ii = particleSpecies->get_data_indices();
            auto particleTmp = amrex::ParticleReduce<amrex::ReduceData<amrex::Real>>(
                *particleSpecies, 0, particleSpecies->finestLevel(),
                [=] AMREX_GPU_HOST_DEVICE(PTDType const& pt,
                                          int const pp) -> amrex::GpuTuple<amrex::Real>
                {
                    auto const weight = pt.rdata(ii.m_iweight)[pp];
                    amrex::Real weightEff = weight;

                    auto const velx = pt.rdata(ii.m_ivelx)[pp];
                    auto const vely = pt.rdata(ii.m_ively)[pp];
                    auto const velz = pt.rdata(ii.m_ivelz)[pp];
                    std::array<amrex::Real, vdim> vel{velx, vely, velz};

                    return {weightEff * 0.5 * mass * (velx * velx + vely * vely + velz * velz)};
                },
                reduceOps);
            ke += amrex::get<0>(particleTmp);
        }
    }
    // reduced sum over mpi ranks
    amrex::ParallelDescriptor::ReduceRealSum({ke}, amrex::ParallelDescriptor::IOProcessorNumber());
    amrex::Print() << "(KE, ME):(" << ke << ", " << me << ")  TE: " << (ke + me) << std::endl;
}

void check_rho_related_norms_dkqn(
    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell>& rho,
    std::vector<amrex::Real> rhoAnVec,
    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& J,
    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell>& divJ,
    ComputationalDomain& mInfra);
} //namespace Gempic::FieldSolvers
  // Functions made for debugging end
#endif