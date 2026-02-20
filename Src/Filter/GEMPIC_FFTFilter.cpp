#include <AMReX_FFT.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Filter.H"
#include "GEMPIC_Parameters.H"

namespace Gempic::Filter
{
FourierFilter::FourierFilter(Gempic::ComputationalDomain const& compDom)
{
    Io::Parameters params("Filter", "class FourierFilter");
    params.get_or_set("enable", m_useFilter);
    if (m_useFilter)
    {
        m_periodicity = compDom.geometry().periodicity();
        GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE(
            m_periodicity.isAllPeriodic(),
            "Fourier filtering only possible in fully periodic domain");
        m_numCells = compDom.box().numPts();

        amrex::Vector<int> nMin, nMax;
        params.get("nMin", nMin);
        params.get("nMax", nMax);
        auto length = compDom.box().length();
        for (int i = 0; i < AMREX_SPACEDIM; i++)
        {
            m_n[i] = length[i];
            m_nMin[i] = nMin[i];
            m_nMax[i] = nMax[i];
        }

        m_r2c = std::make_unique<amrex::FFT::R2C<amrex::Real>>(compDom.box());
        // define cell-centered containers for storing data, declaring 1 ghost cell to be able to
        // copy back to MultiFab with original index type without data loss.
        m_tmpsrc.define(compDom.m_grid, compDom.m_distriMap, 1, 1);
        m_tmpdst.define(compDom.m_grid, compDom.m_distriMap, 1, 1);
    }
}

namespace Impl
{
void do_filter (amrex::MultiFab& dstmf, amrex::MultiFab const& srcmf, FourierFilter& f)
{
    AMREX_ALWAYS_ASSERT(dstmf.nComp() == srcmf.nComp());

    for (int comp{0}; comp < srcmf.nComp(); comp++)
    {
        for (amrex::MFIter mfi(f.m_tmpsrc); mfi.isValid(); ++mfi)
        {
            auto const& tmp = f.m_tmpsrc.array(mfi);
            auto const& src = srcmf.array(mfi);

            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                               { tmp(i, j, k) = src(i, j, k, comp); });
        }

        f.m_r2c->forwardThenBackward(
            f.m_tmpsrc, f.m_tmpdst,
            [n = f.m_n, nMin = f.m_nMin, nMax = f.m_nMax, numCells = f.m_numCells] AMREX_GPU_DEVICE(
                int nx, int j, int k, auto& sp)
            {
                // do actual filtering

                // in x-Direction only half of the Hermitian matrix is stored
                // -> indices are wave numbers
                if (nx < nMin[xDir] or nx > nMax[xDir])
                {
                    sp = 0;
                }
#if AMREX_SPACEDIM > 1
                // in y, and z-Direction the ifftshift of indices is computed to get the correct
                // wave numbers (without sign)
                else if (int ny = (j <= n[yDir] / 2) ? j : n[yDir] - j;
                         ny < nMin[yDir] or ny > nMax[yDir])
                {
                    sp = 0;
                }
#if AMREX_SPACEDIM > 2
                else if (int nz = (k <= n[zDir] / 2) ? k : n[zDir] - k;
                         nz < nMin[zDir] or nz > nMax[zDir])
                {
                    sp = 0;
                }
#endif
#endif
                else
                {
                    sp /= numCells;
                }
            });

        // copy back to
        f.m_tmpdst.FillBoundary(f.m_periodicity);
        for (amrex::MFIter mfi(dstmf); mfi.isValid(); ++mfi)
        {
            auto const& tmp = f.m_tmpdst.array(mfi);
            auto const& dst = dstmf.array(mfi);

            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                               { dst(i, j, k, comp) = tmp(i, j, k); });
        }
    }
}
} // namespace Impl
} //namespace Gempic::Filter
