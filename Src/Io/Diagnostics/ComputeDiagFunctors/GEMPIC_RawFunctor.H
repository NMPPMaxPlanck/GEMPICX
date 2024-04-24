#ifndef GEMPIC_RAWFUNCTOR_H_
#define GEMPIC_RAWFUNCTOR_H_

#include <AMReX.H>
#include <AMReX_BaseFwd.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

#include "GEMPIC_ComputeDiagFunctor.H"

/**
 * \brief Functor to cell-center MF and store result in mf_out.
 */
class RawFunctor final : public ComputeDiagFunctor
{
public:
    /** Constructor.
     *
     * \param[in] mfSrc source multifab.
     * \param[in] crseRatio for interpolating field values from the simulation MultiFab, srcMf,
                  to the output diagnostic MultiFab, mfDst.
     * \param[in] ncomp Number of component of mfSrc to cell-center in dst multifab.
     */
    RawFunctor(const amrex::MultiFab* const mfSrc, const amrex::IntVect crseRatio, int ncomp = 1) :
        ComputeDiagFunctor(ncomp, crseRatio), m_mfSrc(mfSrc)
    {
        BL_PROFILE("RawFunctor::RawFunctor()");
    }

    /** \brief Cell-center m_mfSrc and write the result in mfDst.
     *
     * In cylindrical geometry, by default this functor average all components
     * of a MultiFab and writes into one single component.
     *
     * \param[out] mfDst output MultiFab where the result is written
     * \param[in] dcomp first component of mfDst in which cell-centered
     *            data is stored
     */
    void operator()(amrex::MultiFab& mfDst, int dcomp) const override
    {
        BL_PROFILE("RawFunctor::operator()");
        // Copying raw file
        // std::cout<<"Reached copying files\n";
        amrex::Copy(mfDst, *m_mfSrc, 0, 0, 1, 0);
        // mf_dst.ParallelCopy( *m_mf_src, 0, dcomp, m_mf_src->nComp() );
    }

private:
    /** pointer to source multifab (can be multi-component) */
    amrex::MultiFab const* const m_mfSrc = nullptr;
};

#endif  // GEMPIC_RAWFUNCTOR_H_
