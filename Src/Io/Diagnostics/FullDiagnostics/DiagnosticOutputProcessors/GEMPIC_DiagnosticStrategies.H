#ifndef DIAGNOSTICSTRATEGIES_H_
#define DIAGNOSTICSTRATEGIES_H_

#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_MultiFab.H>

#include "GEMPIC_ComputationalDomain.H"

namespace Gempic::Io
{
/**
 * \brief OutputProcessor to compute a diagnostic from an (alias to an) existing MultiFab and write
 * to another MultiFab, which gets printed.
 */
class ComputeDiagOutputProcessor
{
public:
    ComputeDiagOutputProcessor(int ncomp, amrex::IntVect crseRatio);

    /** Virtual Destructor to handle clean destruction of derived classes */
    virtual ~ComputeDiagOutputProcessor() = default;

    /** Number of component from the input MultiFab to write to the output
     * MultiFab */
    int n_comp() const;

    /** Reference computational domain to write to the output
     * MultiFab. This will give the Geometry to WriteSingleLevelPlotfile.
     * (temporary solution). */
    ComputationalDomain m_compDom;

    /** Compute a field and store the result in mfDst, which gets printed to an output file
     *
     * \param[out] mfDst output MultiFab where the result is written
     * \param[in] dcomp first component of mfDst in which the result is written
     *            to the output diagnostic MultiFab, mfDst.
     */
    virtual void operator()(amrex::MultiFab& mfDst, int dcomp) const = 0;

    virtual amrex::BoxArray get_box_array(amrex::IndexType& t) = 0;

protected:
    /** Coarsening ratio used to interpolate fields from simulation MultiFabs to output MultiFab. */
    amrex::IntVect m_crseRatio;

private:
    /** Number of components of mfDst that this outputProcessor updates. */
    int m_ncomp;
};

namespace Impl
{
/**
 * \brief OutputProcessor to store alias of mfSrc and copy to MultiFab for printing.
 */
class RawOutputProcessor final : public ComputeDiagOutputProcessor
{
public:
    /** Constructor.
     *
     * \param[in] mfSrc source MultiFab.
     * \param[in] crseRatio for interpolating field values from the simulation MultiFab, mfSrc,
                  to the output diagnostic MultiFab, mfDst.
     */
    RawOutputProcessor(amrex::MultiFab const& mfSrc, amrex::IntVect const crseRatio);

    /** \brief copy m_mfSrc to mfDst.
     *
     * \param[out] mfDst output MultiFab where the result is written
     * \param[in] dcomp first component of mfDst in which data is stored
     */
    void operator()(amrex::MultiFab& mfDst, int dcomp) const override;

    amrex::BoxArray get_box_array(amrex::IndexType& t) override;

private:
    /** Alias of source MultiFab (can be multi-component) */
    amrex::MultiFab const m_mfSrc;
};

/**
 * \brief OutputProcessor to store alias of mfSrc and interpolate to cell-center MultiFab.
 */
class CellCenterOutputProcessor final : public ComputeDiagOutputProcessor
{
public:
    /** Constructor.
     *
     * \param[in] mfSrc source MultiFab.
     * \param[in] scaling scaling factor.
     * \param[in] crseRatio for interpolating field values from the simulation MultiFab, mfSrc,
                  to the output diagnostic MultiFab, mfDst.
     */
    CellCenterOutputProcessor(amrex::MultiFab const& mfSrc,
                              amrex::Real scaling,
                              amrex::IntVect const crseRatio);

    /** \brief Cell-center m_mfSrc and write the result in mfDst.
     *
     * In cylindrical geometry, by default this outputProcessor average all components
     * of a MultiFab and writes into one single component.
     *
     * \param[out] mfDst output MultiFab where the result is written
     * \param[in] dcomp first component of mfDst in which cell-centered
     *            data is stored
     */
    void operator()(amrex::MultiFab& mfDst, int dcomp) const override;

    amrex::BoxArray get_box_array(amrex::IndexType& t) override;

private:
    /** Alias of source MultiFab (can be multi-component) */
    amrex::MultiFab const m_mfSrc;

    // Scaling factor
    amrex::Real m_scaling{1.0};
    amrex::Real m_ishift{0}, m_jshift{0}, m_kshift{0};
};
} //namespace Impl
} //namespace Gempic::Io

#endif // DIAGNOSTICSTRATEGIES_H_
