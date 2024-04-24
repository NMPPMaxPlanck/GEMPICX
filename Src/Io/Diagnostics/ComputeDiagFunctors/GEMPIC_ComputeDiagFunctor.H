#ifndef COMPUTEDIAGFUNCTOR_H_
#define COMPUTEDIAGFUNCTOR_H_

#include <AMReX.H>
#include <AMReX_MultiFab.H>

/**
 * \brief Functor to compute a diagnostic and store the result in existing
 * MultiFab
 */
class ComputeDiagFunctor
{
public:
    ComputeDiagFunctor(int ncomp, amrex::IntVect crseRatio) : m_ncomp(ncomp), m_crseRatio(crseRatio)
    {
        BL_PROFILE("ComputeDiagFunctor::ComputeDiagFunctor()");
    }
    /** Virtual Destructor to handle clean destruction of derived classes */
    virtual ~ComputeDiagFunctor() = default;
    /** Compute a field and store the result in mfDst
     *
     * \param[out] mfDst output MultiFab where the result is written
     * \param[in] dcomp first component of mfDst in which the result is written
     *            to the output diagnostic MultiFab, mfDst.
     */
    virtual void operator()(amrex::MultiFab& mfDst, int dcomp) const = 0;
    /** Number of component from the input multifab to write to the output
     * multifab */
    int n_comp () const { return m_ncomp; }

    virtual void init_data () {}

private:
    /** Number of components of mfDst that this functor updates. */
    int m_ncomp;

protected:
    /** Coarsening ratio used to interpolate fields from simulation MultiFabs to output MultiFab. */
    amrex::IntVect m_crseRatio;
};

#endif  // COMPUTEDIAGFUNCTOR_H_
