
#include "GEMPIC_QuasineutralDebug.H"

#ifndef AMREX_USE_HYPRE
#error "HYPRE not enabled"
#endif

// Functions made for debugging start
namespace Gempic::FieldSolvers
{
using namespace Forms;
using namespace Particle;

void check_b_calculation (DeRhamField<Grid::primal, Space::edge>& E,
                          DeRhamField<Grid::primal, Space::face>& bOld,
                          DeRhamField<Grid::primal, Space::face>& B,
                          std::shared_ptr<FDDeRhamComplex> drc,
                          amrex::Real dt,
                          ComputationalDomain& mInfra)
{
    drc->add_dt_curl(bOld, E, -dt);

    DeRhamField<Grid::primal, Space::face> dB(drc);
    linear_combination(dB, 1.0, bOld, -1.0, B);
    amrex::Print() << "\nB difference %:" << " "
                   << (Utils::gempic_norm(dB.m_data[xDir], mInfra, 1) /
                       Utils::gempic_norm(bOld.m_data[xDir], mInfra, 1)) *
                          100.0
                   << " "
                   << (Utils::gempic_norm(dB.m_data[yDir], mInfra, 1) /
                       Utils::gempic_norm(bOld.m_data[yDir], mInfra, 1)) *
                          100.0
                   << " "
                   << (Utils::gempic_norm(dB.m_data[zDir], mInfra, 1) /
                       Utils::gempic_norm(bOld.m_data[zDir], mInfra, 1)) *
                          100.0;
}

void check_b_related_norms (DeRhamField<Grid::dual, Space::face>& J,
                            DeRhamField<Grid::dual, Space::cell>& divJ,
                            DeRhamField<Grid::dual, Space::face>& A,
                            DeRhamField<Grid::dual, Space::cell>& divA,
                            DeRhamField<Grid::primal, Space::face>& B,
                            DeRhamField<Grid::primal, Space::cell>& divB,
                            DeRhamField<Grid::dual, Space::edge>& H,
                            DeRhamField<Grid::dual, Space::face>& curlH,
                            DeRhamField<Grid::dual, Space::face>& jMinusCurlH,
                            std::shared_ptr<FDDeRhamComplex> drc,
                            ComputationalDomain& mInfra)
{
    amrex::Print() << "\nError checks for B-related computations:";

    drc->div(divJ, J);
    drc->div(divA, A);
    drc->div(divB, B);
    drc->curl(curlH, H);

    linear_combination(jMinusCurlH, 1.0, J, -1.0, curlH);

    amrex::Print() << "\nDivergences of (J,A,B): (" << Utils::gempic_norm(divJ.m_data, mInfra, 2)
                   << "," << Utils::gempic_norm(divA.m_data, mInfra, 2) << ","
                   << Utils::gempic_norm(divB.m_data, mInfra, 2) << ")";

    amrex::Print() << "\nJ_minus_curlH_norm: "
                   << Utils::gempic_norm (jMinusCurlH.m_data[xDir], mInfra, 1) +
                          Utils::gempic_norm(jMinusCurlH.m_data[yDir], mInfra, 1) +
                          Utils::gempic_norm(jMinusCurlH.m_data[zDir], mInfra, 1)
                   << "\n";
}

void check_j_change_norms (DeRhamField<Grid::dual, Space::face>& jOld,
                           DeRhamField<Grid::dual, Space::face>& jNew,
                           std::shared_ptr<FDDeRhamComplex> drc,
                           ComputationalDomain& mInfra)
{
    DeRhamField<Grid::dual, Space::face> dJ(drc);
    linear_combination(dJ, 1.0, jOld, -1.0, jNew);
    amrex::Print() << "\nJ change %:" << " "
                   << (Utils::gempic_norm(dJ.m_data[xDir], mInfra, 1) /
                       Utils::gempic_norm(jOld.m_data[xDir], mInfra, 1)) *
                          100.0
                   << " "
                   << (Utils::gempic_norm(dJ.m_data[yDir], mInfra, 1) /
                       Utils::gempic_norm(jOld.m_data[yDir], mInfra, 1)) *
                          100.0
                   << " "
                   << (Utils::gempic_norm(dJ.m_data[zDir], mInfra, 1) /
                       Utils::gempic_norm(jOld.m_data[zDir], mInfra, 1)) *
                          100.0;
}
} //namespace Gempic::FieldSolvers