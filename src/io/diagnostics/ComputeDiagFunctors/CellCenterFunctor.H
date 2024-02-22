#ifndef CELLCENTERFUNCTOR_H_
#define CELLCENTERFUNCTOR_H_

#include "ComputeDiagFunctor.H"

#include <AMReX.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BaseFwd.H>

/**
 * \brief Functor to cell-center MF and store result in mf_out.
 */
class
CellCenterFunctor final : public ComputeDiagFunctor
{
public:
    /** Constructor.
     *
     * \param[in] mf_src source multifab.
     * \param[in] scaling scaling factor.
     * \param[in] crse_ratio for interpolating field values from the simulation MultiFab, src_mf,
                  to the output diagnostic MultiFab, mf_dst.
     * \param[in] ncomp Number of component of mf_src to cell-center in dst multifab.
     */
    CellCenterFunctor(const amrex::MultiFab * const mf_src,
                    amrex::Real scaling,
                    const amrex::IntVect crse_ratio, 
                    int ncomp=1)
      : ComputeDiagFunctor(ncomp, crse_ratio), m_mf_src(mf_src) 
      {
          m_scaling = scaling;
          const amrex::IntVect indexType_src = mf_src->boxArray().ixType().toIntVect();
          AMREX_D_TERM(
            m_ishift = indexType_src[xDir];,
            m_jshift = indexType_src[yDir];,
            m_kshift = indexType_src[zDir];)
      }

    /** \brief Cell-center m_mf_src and write the result in mf_dst.
     *
     * In cylindrical geometry, by default this functor average all components
     * of a MultiFab and writes into one single component.
     *
     * \param[out] mf_dst output MultiFab where the result is written
     * \param[in] dcomp first component of mf_dst in which cell-centered
     *            data is stored
     */
    virtual void operator()(amrex::MultiFab& mf_dst, int dcomp) const override{
    // In cartesian geometry, interpolate from simulation MultiFab, m_mf_src,
    // to output diagnostic MultiFab, mf_dst.
    for (amrex::MFIter mfi(mf_dst); mfi.isValid(); ++mfi)
        {   
            const amrex::Box &bx = mfi.tilebox();
            amrex::Array4 dst = mf_dst[mfi].array();
            amrex::Array4 const src = (*m_mf_src)[mfi].array();

            // You shouldn't try to use private variables in a GPU loop -- CUDA disallows it
            amrex::Real scaling{m_scaling};
            amrex::Real ishift{m_ishift};
            amrex::Real jshift{m_jshift};
            amrex::Real kshift{m_kshift};

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                dst(i,j,k,dcomp) = 0.125 * scaling * (
                    src(i + ishift, j + jshift, k + kshift) 
                    + src(i + ishift, j + jshift, k)
                    + src(i + ishift, j, k)
                    + src(i, j + jshift, k)
                    + src(i, j + jshift, k + kshift)
                    + src(i, j, k + kshift)
                    + src(i + ishift, j, k + kshift)
                    + src(i, j, k)); 
            });
        }
    }
    /** pointer to source multifab (can be multi-component) */
    amrex::MultiFab const * const m_mf_src = nullptr;
private:
    // Scaling factor
    amrex::Real m_scaling{1.0};
    amrex::Real m_ishift{0}, m_jshift{0}, m_kshift{0};
};

#endif // CELLCENTERFUNCTOR_H_
