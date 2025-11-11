#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Interpolation.H"
#include "GEMPIC_Parameters.H"

namespace Gempic::Forms
{

FDDeRhamComplex::FDDeRhamComplex(ComputationalDomain const& infra,
                                 int const hodgeDegree,
                                 int const maxSplineDegree,
                                 HodgeScheme hodgeScheme,
                                 int nComp) :
    DeRhamComplex::DeRhamComplex{infra, hodgeDegree, maxSplineDegree}
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::FDDeRhamComplex()");
    // Parameters used in the projection and hodge
    for (size_t dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        m_dr[dir] = infra.geometry().CellSize(dir);
    }
    m_nGhost = DeRhamComplex::m_nGhost[xDir];
    m_hodgeScheme = hodgeScheme;

    // Read the scaling factors from input file and compute value for the Hodge operator
    Gempic::Io::Parameters parameters{"FDDeRhamComplex"};
    m_sV = 1.0;
    parameters.get_or_set("sV", m_sV);
    m_sOmega = 1.0;
    parameters.get_or_set("sOmega", m_sOmega);

    // There is only one components in each MultiFab as the different components of the forms are
    // centered differently
    m_tempPrimalZeroForm.define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 1, 1)})),
        m_distriMap, nComp, m_nGhost);
    m_tempDualZeroForm.define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 0, 0)})),
        m_distriMap, nComp, m_nGhost);

    m_tempPrimalOneForm[xDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 1, 1)})),
        m_distriMap, nComp, m_nGhost);
    m_tempPrimalOneForm[yDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 0, 1)})),
        m_distriMap, nComp, m_nGhost);
    m_tempPrimalOneForm[zDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 1, 0)})),
        m_distriMap, nComp, m_nGhost);

    m_tempDualOneForm[xDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 0, 0)})),
        m_distriMap, nComp, m_nGhost);
    m_tempDualOneForm[yDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 1, 0)})),
        m_distriMap, nComp, m_nGhost);
    m_tempDualOneForm[zDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 0, 1)})),
        m_distriMap, nComp, m_nGhost);

    m_tempPrimalTwoForm[xDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 0, 0)})),
        m_distriMap, nComp, m_nGhost);
    m_tempPrimalTwoForm[yDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 1, 0)})),
        m_distriMap, nComp, m_nGhost);
    m_tempPrimalTwoForm[zDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 0, 1)})),
        m_distriMap, nComp, m_nGhost);

    m_tempDualTwoForm[xDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 1, 1)})),
        m_distriMap, nComp, m_nGhost);
    m_tempDualTwoForm[yDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 0, 1)})),
        m_distriMap, nComp, m_nGhost);
    m_tempDualTwoForm[zDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 1, 0)})),
        m_distriMap, nComp, m_nGhost);

    m_tempPrimalThreeForm.define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 0, 0)})),
        m_distriMap, nComp, m_nGhost);
    m_tempDualThreeForm.define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 1, 1)})),
        m_distriMap, nComp, m_nGhost);
}

FDDeRhamComplex::~FDDeRhamComplex() {}

amrex::Real FDDeRhamComplex::scaling_eto_d() { return 1 / (m_sOmega * m_sOmega); }

amrex::Real FDDeRhamComplex::scaling_dto_e() { return m_sOmega * m_sOmega; }

amrex::Real FDDeRhamComplex::scaling_bto_h() { return m_sV * m_sV / (m_sOmega * m_sOmega); }

} //namespace Gempic::Forms