#include "GEMPIC_CustomDiagnosticStrategies.H"

//STRATEGY: free registry function + public static API on the class
//Define host-only registry accessors as free functions
namespace Gempic::Io::Detail
{

// Host-only registries (function-local statics)
inline std::map<std::string, Impl::ConstructorType>& class_registry ()
{
    static std::map<std::string, Impl::ConstructorType> reg;
    return reg;
}

inline std::map<std::string, Impl::CustomOperatorType>& operator_registry ()
{
    static std::map<std::string, Impl::CustomOperatorType> reg;
    return reg;
}

} //namespace Gempic::Io::Detail

namespace Gempic::Io
{

// Base class with class-factory registry (friend-free)
/** Constructor.
 *
 * \param[in] dataSrc source AnyFieldPtr register (type AnyFieldPtr).
 * \param[in] crseRatio for interpolating field values from the simulation MultiFab, srcMf,
              to the output diagnostic MultiFab, mfDst.
 */
CustomOutputProcessor::CustomOutputProcessor(Io::AnyFieldPtr const& dataSrc,
                                             amrex::IntVect const crseRatio) :
    ComputeDiagOutputProcessor{dataSrc.n_comp(), crseRatio}, m_dataSrc(&dataSrc)
{
    BL_PROFILE("ComputeDiagOutputProcessor::ComputeDiagOutputProcessor()");
}

/** \brief copy m_dataSrc to mfDst.
 *
 * \param[out] mfDst output MultiFab where the result is written
 */
void CustomOutputProcessor::operator ()(amrex::MultiFab& mfDst, int /*dcomp*/) const
{
    GEMPIC_ERROR("This operator should be overridden!");
}

amrex::BoxArray CustomOutputProcessor::get_box_array (amrex::IndexType& t)
{
    amrex::BoxArray bx = convert(m_compDom.m_grid, t);
    return bx;
}

//public static register_class/find_class
void CustomOutputProcessor::register_class (std::string key, Impl::ConstructorType ctor)
{
    auto& map = Detail::class_registry();
    if (map.contains(key)) GEMPIC_ERROR("Already defined custom diagnostic Id: '" + key + "'.");
    map.emplace(std::move(key), std::move(ctor));
}

Impl::ConstructorType const* CustomOutputProcessor::find_class(std::string_view const& key)
{
    auto& map = Detail::class_registry();
    auto it = map.find(std::string(key));
    return it == map.end() ? nullptr : &it->second;
}

/** Constructor.
 *
 * \param[in] dataSrc source AnyFieldPtr register (type AnyFieldPtr).
 * \param[in] crseRatio for interpolating field values from the simulation MultiFab, srcMf,
              to the output diagnostic MultiFab, mfDst.
 * \param[in] f the constructor of the custom outputProcessor chosen from the input file.
 */
CustomOperatorOutputProcessor::CustomOperatorOutputProcessor(Io::AnyFieldPtr const& dataSrc,
                                                             amrex::IntVect const crseRatio,
                                                             Impl::CustomOperatorType const& f) :
    CustomOutputProcessor{dataSrc, crseRatio}, m_f{f}
{
    BL_PROFILE("CustomOperatorOutputProcessor::CustomOperatorOutputProcessor()");

    amrex::IntVect const indexTypeSrc = dataSrc.box_array().ixType().toIntVect();
    AMREX_D_TERM(m_ishift = indexTypeSrc[xDir];, m_jshift = indexTypeSrc[yDir];
                 , m_kshift = indexTypeSrc[zDir];)
}

void CustomOperatorOutputProcessor::operator ()(amrex::MultiFab& mfDst, int /*dcomp*/) const
{
    BL_PROFILE("CustomOperatorOutputProcessor::operator()");
    int const nComps = this->m_dataSrc->n_comp();
    amrex::MultiFab const* srcMF = this->m_dataSrc->mf_ptr();
    AMREX_ALWAYS_ASSERT(srcMF != nullptr);

    for (amrex::MFIter mfi(mfDst, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.tilebox();
        amrex::Array4 dst = mfDst.array(mfi);
        amrex::Array4 const src = srcMF->const_array(mfi);

        m_f(bx, dst, src, nComps, this->m_dataSrc->scaling(), m_ishift, m_jshift, m_kshift);
    }
}

//public static register_operator/find_operator
void CustomOperatorOutputProcessor::register_operator (std::string key, Impl::CustomOperatorType op)
{
    auto& map = Detail::operator_registry();
    if (map.contains(key)) GEMPIC_ERROR("Already defined custom diagnostic Id: '" + key + "'.");
    map.emplace(std::move(key), std::move(op));
}

Impl::CustomOperatorType const* CustomOperatorOutputProcessor::find_operator(
    std::string_view const& key)
{
    auto& map = Detail::operator_registry();
    auto it = map.find(std::string(key));
    return it == map.end() ? nullptr : &it->second;
}

} // namespace Gempic::Io

namespace Gempic::Io::Impl
{
// Factory lookup (consult both registries): try class factories first, then operator kernels
ConstructorType match_custom_id (std::string const& key)
{
    if (auto const* ctor = Gempic::Io::CustomOutputProcessor::find_class(key)) return *ctor;

    if (auto const* op = Gempic::Io::CustomOperatorOutputProcessor::find_operator(key))
    {
        return [f = *op] (Io::AnyFieldPtr const& dataSrc,
                          amrex::IntVect crse) -> std::unique_ptr<ComputeDiagOutputProcessor>
        { return std::make_unique<Gempic::Io::CustomOperatorOutputProcessor>(dataSrc, crse, f); };
    }

    GEMPIC_ERROR("Unknown custom diagnostic Id: '" + key + "'.");
    return {}; // not reached
}

} // namespace Gempic::Io::Impl

namespace Gempic::Io
{

HighResSubcellOutputProcessor::HighResSubcellOutputProcessor(Io::AnyFieldPtr const& dataSrc,
                                                             amrex::IntVect const crseRatio) :
    CustomOutputProcessor{dataSrc, crseRatio}
{
    BL_PROFILE("HighResSubcellOutputProcessor::HighResSubcellOutputProcessor()");
    Io::Parameters paramsIO("FullDiagnostics.HighResSubcellOutputProcessor");
    amrex::IntVect sNSubCellTempvec;
    paramsIO.get("nSubCells", sNSubCellTempvec);
    amrex::IntVect const idxTypeSrc{dataSrc.box_array().ixType().toIntVect()};
    std::array<int, 3> nSubCell3d{1, 1, 1};
    for (int idir = 0; idir < AMREX_SPACEDIM; ++idir)
    {
        nSubCell3d[idir] = sNSubCellTempvec[idir];
    }

    Io::Parameters params("ComputationalDomain");
    amrex::Vector<int> maxGridSizeTmp;
    amrex::IntVect maxGridSize;
    params.get("maxGridSize", maxGridSizeTmp);
    // first create a coarse multifab

    int nGhost = 0;
    int nComp = dataSrc.n_comp();
    amrex::IndexType t = dataSrc.box_array().ixType();
    amrex::BoxArray ba = get_box_array(t);
    amrex::DistributionMapping const dm = dataSrc.get_distribution_mapping();
    // Then refine computational domain to create a refined multifab.
    for (int iDim = 0; iDim < AMREX_SPACEDIM; iDim++)
    {
        this->m_compDom.m_nCell[iDim] *= nSubCell3d[iDim];
        maxGridSize[iDim] = maxGridSizeTmp[iDim] * nSubCell3d[iDim];
    }

    // higher-boundary of domain
    amrex::Box domain;
    amrex::IntVect domLo(AMREX_D_DECL(0, 0, 0));
    domain.setSmall(domLo);
    amrex::IntVect domHi(AMREX_D_DECL(this->m_compDom.m_nCell[xDir] - 1,
                                      this->m_compDom.m_nCell[yDir] - 1,
                                      this->m_compDom.m_nCell[zDir] - 1));
    domain.setBig(domHi);
    this->m_compDom.m_grid.define(domain);
    this->m_compDom.m_grid.maxSize(maxGridSize);
    this->m_compDom.m_geom.Domain(domain);
    // do not change the distribution mapping needed to define MultiFabs!
    //compDom.m_distriMap.define(compDom.m_grid);
}

amrex::BoxArray HighResSubcellOutputProcessor::get_box_array (amrex::IndexType& t)
{
    amrex::IndexType tNew{t};
    tNew.clear(); // higher resolution cell-centered plotter;
    amrex::BoxArray bx = convert(this->m_compDom.m_grid, tNew);
    return bx;
};

void HighResSubcellOutputProcessor::interpolate_any_field_to_multifab (amrex::MultiFab& mfDst,
                                                                      int dcomp) const
{
    this->m_dataSrc->interpolate_any_field_to_multifab(mfDst, dcomp);
}

void HighResSubcellOutputProcessor::operator ()(amrex::MultiFab& mfDst, int dcomp) const
{
    BL_PROFILE("HighResSubcellOutputProcessor::operator()");
    mfDst.setVal(0.0);
    this->interpolate_any_field_to_multifab(mfDst, dcomp);
}

} //namespace Gempic::Io