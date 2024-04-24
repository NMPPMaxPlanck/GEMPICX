#ifndef CELLCENTERFUNCTOR_H_
#define CELLCENTERFUNCTOR_H_

#include <AMReX.H>
#include <AMReX_BaseFwd.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

#include "GEMPIC_ComputeDiagFunctor.H"
#include "GEMPIC_Config.H"

/**
 * \brief Functor to cell-center MF and store result in mfOut.
 */
class CellCenterFunctor final : public ComputeDiagFunctor
{
public:
    /** Constructor.
     *
     * \param[in] mfSrc source multifab.
     * \param[in] scaling scaling factor.
     * \param[in] crseRatio for interpolating field values from the simulation MultiFab, mfSrc,
                  to the output diagnostic MultiFab, mfDst.
     * \param[in] ncomp Number of component of mfSrc to cell-center in dst multifab.
     */
    CellCenterFunctor(const amrex::MultiFab* const mfSrc,
                      amrex::Real scaling,
                      const amrex::IntVect crseRatio,
                      int ncomp = 1) :
        ComputeDiagFunctor(ncomp, crseRatio), m_mfSrc(mfSrc)
    {
        BL_PROFILE("CellCenterFunctor::CellCenterFunctor()");
        m_scaling = scaling;
        const amrex::IntVect indexTypeSrc = mfSrc->boxArray().ixType().toIntVect();
        AMREX_D_TERM(m_ishift = indexTypeSrc[xDir];, m_jshift = indexTypeSrc[yDir];
                     , m_kshift = indexTypeSrc[zDir];)
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
        BL_PROFILE("CellCenterFunctor::operator()");
        // In cartesian geometry, interpolate from simulation MultiFab, m_mfSrc,
        // to output diagnostic MultiFab, mfDst.
        for (amrex::MFIter mfi(mfDst); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.tilebox();
            amrex::Array4 dst = mfDst[mfi].array();
            amrex::Array4 const src = (*m_mfSrc)[mfi].array();

            // You shouldn't try to use private variables in a GPU loop -- CUDA disallows it
            amrex::Real scaling{m_scaling};
            amrex::Real ishift{m_ishift};
            amrex::Real jshift{m_jshift};
            amrex::Real kshift{m_kshift};

            ParallelFor(bx,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                            dst(i, j, k, dcomp) =
                                0.125 * scaling *
                                (src(i + ishift, j + jshift, k + kshift) + src(i, j, k) +
                                 src(i + ishift, j + jshift, k) + src(i, j, k + kshift) +
                                 src(i + ishift, j, k + kshift) + src(i, j + jshift, k) +
                                 src(i, j + jshift, k + kshift) + src(i + ishift, j, k));
                        });
        }
    }
    /** pointer to source multifab (can be multi-component) */
    amrex::MultiFab const* const m_mfSrc = nullptr;

private:
    // Scaling factor
    amrex::Real m_scaling{1.0};
    amrex::Real m_ishift{0}, m_jshift{0}, m_kshift{0};
};

#endif  // CELLCENTERFUNCTOR_H_
