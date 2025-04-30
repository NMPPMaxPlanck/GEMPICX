#include <AMReX.H>
#include <AMReX_BaseFwd.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

#include "GEMPIC_Config.H"
#include "GEMPIC_DiagnosticStrategies.H"

namespace Gempic::Io
{
ComputeDiagOutputProcessor::ComputeDiagOutputProcessor(int ncomp, amrex::IntVect crseRatio) :
    m_crseRatio(crseRatio), m_ncomp(ncomp)
{
    BL_PROFILE("ComputeDiagOutputProcessor::ComputeDiagOutputProcessor()");
}

int ComputeDiagOutputProcessor::n_comp() const { return m_ncomp; }

namespace Impl
{
RawOutputProcessor::RawOutputProcessor(amrex::MultiFab const& mfSrc,
                                       amrex::IntVect const crseRatio) :
    ComputeDiagOutputProcessor(mfSrc.nComp(), crseRatio),
    m_mfSrc(amrex::MultiFab(mfSrc, amrex::make_alias, 0, mfSrc.nComp()))
{
    BL_PROFILE("RawOutputProcessor::RawOutputProcessor()");
}

void RawOutputProcessor::operator ()(amrex::MultiFab& mfDst, int dcomp) const
{
    BL_PROFILE("RawOutputProcessor::operator()");
    // Copying raw MultiFab. You might think this a waste, but amrex's plot function does the same
    // if the MultiFab passed to it has ghost cells.
    amrex::Copy(mfDst, m_mfSrc, 0, dcomp, mfDst.nComp(), 0);
    // mf_dst.ParallelCopy( *m_mf_src, 0, dcomp, m_mf_src->nComp() );
}

CellCenterOutputProcessor::CellCenterOutputProcessor(amrex::MultiFab const& mfSrc,
                                                     amrex::Real scaling,
                                                     amrex::IntVect const crseRatio) :
    ComputeDiagOutputProcessor(mfSrc.nComp(), crseRatio),
    m_mfSrc(amrex::MultiFab(mfSrc, amrex::make_alias, 0, mfSrc.nComp()))
{
    BL_PROFILE("CellCenterOutputProcessor::CellCenterOutputProcessor()");
    m_scaling = scaling;
    amrex::IntVect const indexTypeSrc = mfSrc.boxArray().ixType().toIntVect();
    AMREX_D_TERM(m_ishift = indexTypeSrc[xDir];, m_jshift = indexTypeSrc[yDir];
                 , m_kshift = indexTypeSrc[zDir];)
}

void CellCenterOutputProcessor::operator ()(amrex::MultiFab& mfDst, int dcomp) const
{
    BL_PROFILE("CellCenterOutputProcessor::operator()");
    // In cartesian geometry, interpolate from simulation MultiFab, m_mfSrc,
    // to output diagnostic MultiFab, mfDst.
    int nComps = m_mfSrc.nComp();

    for (amrex::MFIter mfi(mfDst); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.tilebox();
        amrex::Array4 dst = mfDst[mfi].array();
        amrex::Array4 const src = m_mfSrc[mfi].array();

        // You shouldn't try to use private variables in a GPU loop -- CUDA disallows it
        amrex::Real scaling{m_scaling};
        amrex::Real ishift{m_ishift};
        amrex::Real jshift{m_jshift};
        amrex::Real kshift{m_kshift};

        ParallelFor(
            bx, nComps,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int srcComp)
            {
                dst(i, j, k, dcomp + srcComp) =
                    0.125 * scaling *
                    (src(i + ishift, j + jshift, k + kshift, srcComp) + src(i, j, k, srcComp) +
                     src(i + ishift, j + jshift, k, srcComp) + src(i, j, k + kshift, srcComp) +
                     src(i + ishift, j, k + kshift, srcComp) + src(i, j + jshift, k, srcComp) +
                     src(i, j + jshift, k + kshift, srcComp) + src(i + ishift, j, k, srcComp));
            });
    }
}
} //namespace Impl
} //namespace Gempic::Io
