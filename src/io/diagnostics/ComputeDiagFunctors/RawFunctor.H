#ifndef GEMPIC_RAWFUNCTOR_H_
#define GEMPIC_RAWFUNCTOR_H_

#include "ComputeDiagFunctor.H"

#include <AMReX.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BaseFwd.H>

/**
 * \brief Functor to cell-center MF and store result in mf_out.
 */
class
RawFunctor final : public ComputeDiagFunctor
{
public:
    /** Constructor.
     *
     * \param[in] mf_src source multifab.
     * \param[in] crse_ratio for interpolating field values from the simulation MultiFab, src_mf,
                  to the output diagnostic MultiFab, mf_dst.
     * \param[in] ncomp Number of component of mf_src to cell-center in dst multifab.
     */
    RawFunctor(const amrex::MultiFab * const mf_src,
                      const amrex::IntVect crse_ratio, int ncomp=1)
      : ComputeDiagFunctor(ncomp, crse_ratio), m_mf_src(mf_src)
    {}

    /** \brief Cell-center m_mf_src and write the result in mf_dst.
     *
     * In cylindrical geometry, by default this functor average all components
     * of a MultiFab and writes into one single component.
     *
     * \param[out] mf_dst output MultiFab where the result is written
     * \param[in] dcomp first component of mf_dst in which cell-centered
     *            data is stored
     */
    virtual void operator()(amrex::MultiFab& mf_dst, int dcomp) const override
    {
        //Copying raw file
        //std::cout<<"Reached copying files\n";
        amrex::Copy(mf_dst, *m_mf_src, 0, 0, 1, 0);
        //mf_dst.ParallelCopy( *m_mf_src, 0, dcomp, m_mf_src->nComp() );
    }
private:
    /** pointer to source multifab (can be multi-component) */
    amrex::MultiFab const * const m_mf_src = nullptr;
};

#endif // GEMPIC_RAWFUNCTOR_H_
