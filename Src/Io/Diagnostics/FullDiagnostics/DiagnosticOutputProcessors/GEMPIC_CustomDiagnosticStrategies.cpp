#include <AMReX.H>
#include <AMReX_BaseFwd.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

#include "GEMPIC_Config.H"
#include "GEMPIC_CustomDiagnosticStrategies.H"
#include "GEMPIC_DiagnosticStrategies.H"
#include "GEMPIC_Interpolation.H"

namespace Gempic::Io
{
CustomOutputProcessor::CustomOutputProcessor(amrex::MultiFab const& mfSrc,
                                             amrex::Real const scaling,
                                             amrex::IntVect const crseRatio) :
    ComputeDiagOutputProcessor{mfSrc.nComp(), crseRatio},
    m_mfSrc{amrex::MultiFab(mfSrc, amrex::make_alias, 0, mfSrc.nComp())},
    m_scaling{scaling}
{
}

amrex::BoxArray CustomOutputProcessor::get_box_array (amrex::IndexType& t)
{
    amrex::BoxArray bx = convert(m_compDom.m_grid, t);
    return bx;
};

void CustomOutputProcessor::operator ()(amrex::MultiFab& mfDst, int /*dcomp*/) const
{
    GEMPIC_ERROR("This operator should be overridden!");
}

Impl::ConstructorType Impl::match_custom_id (std::string const& key)
{
    if (CustomOutputProcessor::s_outputProcessorClassMap.find(key) !=
        CustomOutputProcessor::s_outputProcessorClassMap.end())
    {
        return (CustomOutputProcessor::s_outputProcessorClassMap.at(key));
    }
    else if (CustomOperatorOutputProcessor::s_outputProcessorMap.find(key) !=
             CustomOperatorOutputProcessor::s_outputProcessorMap.end())
    {
        return [=] (amrex::MultiFab const& mfSrc, amrex::Real const scaling,
                    amrex::IntVect const crseRatio) -> std::unique_ptr<ComputeDiagOutputProcessor>
        {
            return std::make_unique<CustomOperatorOutputProcessor> (
                mfSrc, scaling, crseRatio,
                CustomOperatorOutputProcessor::s_outputProcessorMap.at(key));
        };
    }
    else
    {
        GEMPIC_ERROR("Unknown custom diagnostic Id: '" + key + "'.");
        return Impl::dynamic_convert<CustomOutputProcessor>;
    }
}

void add_output_processor (std::string const& key, Impl::CustomOperatorType const& f)
{
    if (CustomOperatorOutputProcessor::s_outputProcessorMap.find(key) !=
        CustomOperatorOutputProcessor::s_outputProcessorMap.end())
    {
        GEMPIC_ERROR("Already defined custom diagnostic Id: '" + key + "'.");
    }
    else
    {
        CustomOperatorOutputProcessor::s_outputProcessorMap.emplace(key, f);
    }
}

CustomOperatorOutputProcessor::CustomOperatorOutputProcessor(amrex::MultiFab const& mfSrc,
                                                             amrex::Real const scaling,
                                                             amrex::IntVect const crseRatio,
                                                             CustomOperatorType const& f) :
    CustomOutputProcessor{mfSrc, scaling, crseRatio}, m_f{f}
{
    BL_PROFILE("CustomOperatorOutputProcessor::CustomOperatorOutputProcessor()");

    amrex::IntVect const indexTypeSrc = mfSrc.boxArray().ixType().toIntVect();
    AMREX_D_TERM(m_ishift = indexTypeSrc[xDir];, m_jshift = indexTypeSrc[yDir];
                 , m_kshift = indexTypeSrc[zDir];)
}

void CustomOperatorOutputProcessor::operator ()(amrex::MultiFab& mfDst, int /*dcomp*/) const
{
    BL_PROFILE("CustomOperatorOutputProcessor::operator()");
    int nComps = m_mfSrc.nComp();

    for (amrex::MFIter mfi(mfDst); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.tilebox();
        amrex::Array4 dst = mfDst[mfi].array();
        amrex::Array4 const src = m_mfSrc[mfi].array();

        m_f(bx, dst, src, nComps, m_scaling, m_ishift, m_jshift, m_kshift);
    }
}

HighResSubcellOutputProcessor::HighResSubcellOutputProcessor(amrex::MultiFab const& mfSrc,
                                                             amrex::Real const scaling,
                                                             amrex::IntVect const crseRatio) :
    CustomOutputProcessor{mfSrc, scaling, crseRatio}
{
    BL_PROFILE("HighResSubcellOutputProcessor::HighResSubcellOutputProcessor()");
    Io::Parameters paramsIO("FullDiagnostics.HighResSubcellOutputProcessor");
    amrex::IntVect sNSubCellTempvec;
    paramsIO.get("nSubCells", sNSubCellTempvec);
    amrex::IntVect const idxTypeSrc{mfSrc.boxArray().ixType().toIntVect()};
    for (int idir = 0; idir < AMREX_SPACEDIM; ++idir)
    {
        m_sNSubCell3d[idir] = sNSubCellTempvec[idir];
    }

    Io::Parameters params("ComputationalDomain");
    amrex::Vector<int> maxGridSizeTmp;
    amrex::IntVect maxGridSize;
    params.get("maxGridSize", maxGridSizeTmp);
    // first create a coarse multifab

    int nGhost = 0;
    int nComp = mfSrc.nComp();
    amrex::IndexType t = mfSrc.boxArray().ixType();
    amrex::BoxArray ba = get_box_array(t);
    amrex::DistributionMapping const dm = mfSrc.DistributionMap();
    // Then refine computational domain to create a refined multifab.
    for (int iDim = 0; iDim < AMREX_SPACEDIM; iDim++)
    {
        m_compDom.m_nCell[iDim] *= m_sNSubCell3d[iDim];
        maxGridSize[iDim] = maxGridSizeTmp[iDim] * m_sNSubCell3d[iDim];
    }

    // higher-boundary of domain
    amrex::Box domain;
    amrex::IntVect domLo(AMREX_D_DECL(0, 0, 0));
    domain.setSmall(domLo);
    amrex::IntVect domHi(AMREX_D_DECL(m_compDom.m_nCell[xDir] - 1, m_compDom.m_nCell[yDir] - 1,
                                      m_compDom.m_nCell[zDir] - 1));
    domain.setBig(domHi);
    m_compDom.m_grid.define(domain);
    m_compDom.m_grid.maxSize(maxGridSize);
    m_compDom.m_geom.Domain(domain);
    // do not change the distribution mapping needed to define MultiFabs!
    //compDom.m_distriMap.define(compDom.m_grid);
    m_shiftV = {GEMPIC_D_PAD(idxTypeSrc[xDir], idxTypeSrc[yDir], idxTypeSrc[zDir])};
}

amrex::BoxArray HighResSubcellOutputProcessor::get_box_array (amrex::IndexType& t)
{
    amrex::IndexType tNew{t};
    tNew.clear(); // higher resolution cell-centered plotter;
    amrex::BoxArray bx = convert(m_compDom.m_grid, tNew);
    return bx;
};

void HighResSubcellOutputProcessor::operator ()(amrex::MultiFab& mfDst, int dcomp) const
{
    BL_PROFILE("HighResSubcellOutputProcessor::operator()");
    mfDst.setVal(0.0);

    for (amrex::MFIter mfiSrc(m_mfSrc); mfiSrc.isValid(); ++mfiSrc)
    {
        //// we should select here depending on m_shift_V = 0,0,0; 1,0,0; 0,1,0; ...
        if (m_shiftV == m_shiftCell) //m_myIdxTypeSrc.cellCentered())
        {
            this->io_interpolation<s_polyrecDegree, Grid::primal, Space::cell>(mfDst, mfiSrc);
        }
        else if (m_shiftV == m_shiftXFace)
        {
            this->io_interpolation<s_polyrecDegree, Grid::primal, Space::face, xDir>(mfDst, mfiSrc);
        }
        else if (m_shiftV == m_shiftYFace)
        {
            this->io_interpolation<s_polyrecDegree, Grid::primal, Space::face, yDir>(mfDst, mfiSrc);
        }
        else if (m_shiftV == m_shiftZFace)
        {
            this->io_interpolation<s_polyrecDegree, Grid::primal, Space::face, zDir>(mfDst, mfiSrc);
        }
        else if (m_shiftV == m_shiftXEdge)
        {
            this->io_interpolation<s_polyrecDegree, Grid::primal, Space::edge, xDir>(mfDst, mfiSrc);
        }
        else if (m_shiftV == m_shiftYEdge)
        {
            this->io_interpolation<s_polyrecDegree, Grid::primal, Space::edge, yDir>(mfDst, mfiSrc);
        }
        else if (m_shiftV == m_shiftZEdge)
        {
            this->io_interpolation<s_polyrecDegree, Grid::primal, Space::edge, zDir>(mfDst, mfiSrc);
        }
        else if (m_shiftV == m_shiftNode)
        {
            this->io_interpolation<s_polyrecDegree, Grid::primal, Space::node>(mfDst, mfiSrc);
        }
    }
}

// V0,V3 interpolation
template <int degree, Grid grid, Space space>
void HighResSubcellOutputProcessor::io_interpolation (amrex::MultiFab& mfDst,
                                                     amrex::MFIter& mfiSrc) const
{
    // change box upper limits to compute only the cell-centered output multifab:
    amrex::IntVect mShiftCellIv{AMREX_D_DECL(m_shiftCell[0], m_shiftCell[1], m_shiftCell[2])};
    int nComp = m_mfSrc.nComp();
    amrex::Box const& bxSrc = mfiSrc.tilebox(mShiftCellIv);

    amrex::Array4 dst = mfDst[mfiSrc].array();
    amrex::Array4 const src = m_mfSrc[mfiSrc].array();

    amrex::Real scaling{m_scaling};

    int abcTop{degree / 2};

    if constexpr (space == Space::node)
    {
        abcTop += 1;
    }

    std::array<int, 3> sNSubCellLoc = {m_sNSubCell3d};

    ParallelFor(
        bxSrc, nComp,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
        {
            int const iloc = sNSubCellLoc[0] * i;
            int const jloc = sNSubCellLoc[1] * j;
            int const kloc = sNSubCellLoc[2] * k;

            for (int iSub = 0; iSub < sNSubCellLoc[0]; ++iSub)
            {
                for (int jSub = 0; jSub < sNSubCellLoc[1]; ++jSub)
                {
                    for (int kSub = 0; kSub < sNSubCellLoc[2]; ++kSub)
                    {
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> rr = {AMREX_D_DECL(
                            (iSub + 0.5) / sNSubCellLoc[0], (jSub + 0.5) / sNSubCellLoc[1],
                            (kSub + 0.5) / sNSubCellLoc[2])};
                        amrex::Real fval = 0.0;
                        GEMPIC_D_EXCL(int b{0};, int c{0};, )
                        GEMPIC_D_LOOP_BEGIN(for (int a = 1 - degree / 2; a < abcTop; ++a),
                                            for (int b = 1 - degree / 2; b < abcTop; ++b),
                                            for (int c = 1 - degree / 2; c < abcTop; ++c))
                            fval +=
                                src(i + a, j + b, k + c, n) * scaling *
                                GEMPIC_D_MULT(
                                    (Gempic::Forms::choose_eval<degree, grid, space>(a, rr[xDir])),
                                    (Gempic::Forms::choose_eval<degree, grid, space>(b, rr[yDir])),
                                    (Gempic::Forms::choose_eval<degree, grid, space>(c, rr[zDir])));
                        GEMPIC_D_LOOP_END
                        dst(iloc + iSub, jloc + jSub, kloc + kSub, n) = fval;
                    }
                }
            }
        });
}

// V1,V2 interpolation
template <int degree, Grid grid, Space space, Direction dir>
void HighResSubcellOutputProcessor::io_interpolation (amrex::MultiFab& mfDst,
                                                     amrex::MFIter& mfiSrc) const
{
    // change box upper limits to compute only the cell-centered output multifab:
    amrex::IntVect mShiftCellIv{AMREX_D_DECL(m_shiftCell[0], m_shiftCell[1], m_shiftCell[2])};
    int nComp = m_mfSrc.nComp();
    amrex::Box const& bxSrc = mfiSrc.tilebox(mShiftCellIv);

    amrex::Array4 dst = mfDst[mfiSrc].array();
    amrex::Array4 const src = m_mfSrc[mfiSrc].array();

    amrex::Real scaling{m_scaling};
    std::array<int, 3> sNSubCellLoc = {m_sNSubCell3d};
    amrex::GpuArray<int, 3> dirShift = {0, 0, 0};
    if constexpr (space == Space::edge)
    {
        dirShift = {1, 1, 1};
        dirShift[dir] = 0;
    }
    else // if constexpr (space == Space::face)
    {
        dirShift[dir] = 1;
    }

    ParallelFor(
        bxSrc, nComp,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
        {
            int const iloc = sNSubCellLoc[0] * i;
            int const jloc = sNSubCellLoc[1] * j;
            int const kloc = sNSubCellLoc[2] * k;
            for (int iSub = 0; iSub < sNSubCellLoc[0]; ++iSub)
            {
                for (int jSub = 0; jSub < sNSubCellLoc[1]; ++jSub)
                {
                    for (int kSub = 0; kSub < sNSubCellLoc[2]; ++kSub)
                    {
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> rr = {AMREX_D_DECL(
                            (iSub + 0.5) / sNSubCellLoc[0], (jSub + 0.5) / sNSubCellLoc[1],
                            (kSub + 0.5) / sNSubCellLoc[2])};
                        amrex::Real fval = 0.0;
                        GEMPIC_D_EXCL(int b{0};, int c{0};, )
                        GEMPIC_D_LOOP_BEGIN(
                            for (int a = 1 - degree / 2; a < degree / 2 + dirShift[xDir]; ++a),
                            for (int b = 1 - degree / 2; b < degree / 2 + dirShift[yDir]; ++b),
                            for (int c = 1 - degree / 2; c < degree / 2 + dirShift[zDir]; ++c))
                            fval += src(i + a, j + b, k + c, n) * scaling *
                                    GEMPIC_D_MULT(
                                        (Gempic::Forms::choose_eval<degree, grid, space, dir, xDir>(
                                            a, rr[xDir])),
                                        (Gempic::Forms::choose_eval<degree, grid, space, dir, yDir>(
                                            b, rr[yDir])),
                                        (Gempic::Forms::choose_eval<degree, grid, space, dir, zDir>(
                                            c, rr[zDir])));
                        GEMPIC_D_LOOP_END
                        dst(iloc + iSub, jloc + jSub, kloc + kSub, n) = fval;
                    }
                }
            }
        });
}

} //namespace Gempic::Io
