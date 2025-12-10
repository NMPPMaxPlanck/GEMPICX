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
void check_energies (
    Gempic::Forms::DeRhamField<Grid::primal, Space::face>& B,
    Gempic::Forms::DeRhamField<Grid::dual, Space::edge>& H,
    std::vector<std::shared_ptr<Gempic::Particle::ParticleSpecies<vdim, ndata>>>& particles,
    std::shared_ptr<Gempic::Forms::FDDeRhamComplex> drc,
    ComputationalDomain& mInfra)
{
    amrex::Real me{0.0};
    for (int dir = 0; dir < 3; dir++)
    {
        auto mask = amrex::OwnerMask(B.m_data[dir], mInfra.geometry().periodicity());
        me += 0.5 * amrex::MultiFab::Dot(*mask, B.m_data[dir], 0, H.m_data[dir], 0,
                                         B.m_data[dir].nComp(), 0, 0);
    }

    amrex::GpuArray<amrex::Real, 1> ke{0.0};
    auto* kePtr = ke.data();

    for (auto& particleSpecies : particles)
    {
        amrex::Real mass = particleSpecies->get_mass();

        for (auto& pti : *particleSpecies)
        {
            long const np = pti.numParticles();
            auto* const velx = pti.GetStructOfArrays().GetRealData(0).data();
            auto* const vely = pti.GetStructOfArrays().GetRealData(1).data();
            auto* const velz = pti.GetStructOfArrays().GetRealData(2).data();
            auto* const weight = pti.GetStructOfArrays().GetRealData(vdim).data();

            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{mInfra.geometry().ProbLoArray()};

            amrex::ParallelFor(
                np,
                [=] AMREX_GPU_DEVICE(long pp)
                {
                    amrex::GpuArray<amrex::Real, vdim> vel{velx[pp], vely[pp], velz[pp]};

                    *kePtr +=
                        weight[pp] * 0.5 * mass *
                        (vel[xDir] * vel[xDir] + vel[yDir] * vel[yDir] + vel[zDir] * vel[zDir]);
                });
        }
    }

    amrex::Array<amrex::Real, 1> keCpu;
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, ke.begin(), ke.end(), keCpu.begin());
    amrex::ParallelDescriptor::ReduceRealSum({keCpu[0]},
                                             amrex::ParallelDescriptor::IOProcessorNumber());

    amrex::Print() << "(KE, ME):(" << keCpu[0] << ", " << me << ")  TE: " << (keCpu[0] + me)
                   << std::endl;
}

template <unsigned int vdim, unsigned int ndata, int degx, int degy, int degz>
void check_e_related_norms (
    Gempic::Forms::DeRhamField<Grid::primal, Space::edge>& E,
    Gempic::Forms::DeRhamField<Grid::primal, Space::face>& B,
    Gempic::Forms::DeRhamField<Grid::dual, Space::edge>& H,
    Gempic::Forms::DeRhamField<Grid::dual, Space::face>& J,
    Gempic::Forms::DeRhamField<Grid::dual, Space::cell>& rho,
    Gempic::Forms::DeRhamField<Grid::dual, Space::cell>& rhoIdeal,
    std::shared_ptr<Gempic::Forms::FDDeRhamComplex> drc,
    Gempic::Forms::DeRhamField<Grid::dual, Space::face> const& rhs,
    std::vector<std::shared_ptr<Gempic::Particle::ParticleSpecies<vdim, ndata>>>& particles,
    ComputationalDomain& mInfra)
{
    amrex::Print() << "\nError checks for E-related computations:";

    // lhs = CurlCurlE + rhoE = rhs check
    Gempic::Forms::DeRhamField<Grid::primal, Space::face> curlE(drc);
    Gempic::Forms::DeRhamField<Grid::dual, Space::edge> curlEProj(drc);
    Gempic::Forms::DeRhamField<Grid::dual, Space::face> curlCurlE(drc);
    Gempic::Forms::DeRhamField<Grid::dual, Space::face> rhoE(drc);

    Gempic::Forms::DeRhamField<Grid::dual, Space::face> lhs(drc);

    Gempic::Forms::DeRhamField<Grid::dual, Space::cell> divRhoE(drc);
    Gempic::Forms::DeRhamField<Grid::dual, Space::cell> divLhs(drc);
    Gempic::Forms::DeRhamField<Grid::dual, Space::cell> divRhs(drc);

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

    Gempic::Forms::DeRhamField<Grid::dual, Space::cell> rhoError(drc);
    linear_combination(rhoError, 1.0, rho, -1.0, rhoIdeal);
    amrex::Real den = Utils::gempic_norm(rhoIdeal.m_data, mInfra, 1);
    if (den == 0.0) den = 1.0;
    amrex::Print() << "\nrho error %: "
                   << (Utils::gempic_norm(rhoError.m_data, mInfra, 1) / den) * 100.0 << "\n";
}

void check_b_calculation(Gempic::Forms::DeRhamField<Grid::primal, Space::edge>& E,
                         Gempic::Forms::DeRhamField<Grid::primal, Space::face>& bOld,
                         Gempic::Forms::DeRhamField<Grid::primal, Space::face>& B,
                         std::shared_ptr<Gempic::Forms::FDDeRhamComplex> drc,
                         amrex::Real dt,
                         ComputationalDomain& mInfra);

void check_b_related_norms(Gempic::Forms::DeRhamField<Grid::dual, Space::face>& J,
                           Gempic::Forms::DeRhamField<Grid::dual, Space::cell>& divJ,
                           Gempic::Forms::DeRhamField<Grid::dual, Space::face>& A,
                           Gempic::Forms::DeRhamField<Grid::dual, Space::cell>& divA,
                           Gempic::Forms::DeRhamField<Grid::primal, Space::face>& B,
                           Gempic::Forms::DeRhamField<Grid::primal, Space::cell>& divB,
                           Gempic::Forms::DeRhamField<Grid::dual, Space::edge>& H,
                           Gempic::Forms::DeRhamField<Grid::dual, Space::face>& curlH,
                           Gempic::Forms::DeRhamField<Grid::dual, Space::face>& jMinusCurlH,
                           std::shared_ptr<Gempic::Forms::FDDeRhamComplex> drc,
                           ComputationalDomain& mInfra);

void check_j_change_norms(Gempic::Forms::DeRhamField<Grid::dual, Space::face>& jOld,
                          Gempic::Forms::DeRhamField<Grid::dual, Space::face>& jNew,
                          std::shared_ptr<Gempic::Forms::FDDeRhamComplex> drc,
                          ComputationalDomain& mInfra);
} //namespace Gempic::FieldSolvers
  // Functions made for debugging end
#endif