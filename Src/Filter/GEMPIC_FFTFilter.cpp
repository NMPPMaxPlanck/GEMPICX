/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <AMReX_FFT.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Filter.H"
#include "GEMPIC_Parameters.H"

namespace Gempic::Filter
{
FourierFilter::FourierFilter(Io::Parameters& params, DiscreteGrid const& discreteGrid)
{
    amrex::Vector<int> nMin, nMax;
    params.get("Filter.Fourier.nMin", nMin);
    params.get("Filter.Fourier.nMax", nMax);
    for (auto dir : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
    {
        m_n[dir] = discreteGrid.n_cells(dir);
        m_nMin[dir] = nMin[dir];
        m_nMax[dir] = nMax[dir];
    }

    m_periodicity = Gempic::Impl::to_amrex_periodicity(discreteGrid);
    m_r2c = std::make_unique<amrex::FFT::R2C<amrex::Real>>(
        amrex::convert(Gempic::Impl::to_amrex_box(discreteGrid), amrex::IntVect::TheCellVector()));
}

FourierFilter::FourierFilter(Gempic::ComputationalDomain const& compDom)
{
    Io::Parameters params("Filter.Fourier", "class FourierFilter");
    m_periodicity = compDom.geometry().periodicity();
    GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE(m_periodicity.isAllPeriodic(),
                                      "Fourier filtering only possible in fully periodic domain");

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
}

void FourierFilter::operator ()(DiscreteField& dst, DiscreteField& src)
{
    Impl::do_filter(dst.multi_fab(), src.multi_fab(), *this);
}

void FourierFilter::operator ()(DiscreteVectorField& dst, DiscreteVectorField& src)
{
    for (auto dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        this->operator()(dst[dir], src[dir]);
    }
}

namespace Impl
{
void do_filter (amrex::MultiFab& dstmf, amrex::MultiFab const& srcmf, FourierFilter& f)
{
    AMREX_ALWAYS_ASSERT(dstmf.nComp() == srcmf.nComp());

    amrex::MultiFab tmpsrc{amrex::convert(srcmf.boxArray(), amrex::IntVect::TheCellVector()),
                           srcmf.DistributionMap(), 1, 0};
    amrex::MultiFab tmpdst{amrex::convert(dstmf.boxArray(), amrex::IntVect::TheCellVector()),
                           dstmf.DistributionMap(), 1, 0};
    // AMReX FFT requires cell-centered input. The source/destination may not be cell-centered,
    // which on a periodic domain means the high-boundary node is a duplicate of the low-boundary
    // node.
    // In the following we drop the duplicated high-boundary node by copying data into a
    // cell-centered MultiFab. The FFT is applied to the MultiFab with cell-centered data. Finally
    // the data is written to the result array and an OverrideSync is used to (in MPI terms)
    // broadcast the result back to the duplicated nodes.
    amrex::iMultiFab ownermask{dstmf.boxArray(), dstmf.DistributionMap(), 1, 0};
    ownermask.setVal(0);
    for (int comp{0}; comp < srcmf.nComp(); comp++)
    {
        for (amrex::MFIter mfi(tmpsrc); mfi.isValid(); ++mfi)
        {
            auto const& tmp = tmpsrc.array(mfi);
            auto const& src = srcmf.array(mfi);
            auto const& owner = ownermask.array(mfi);

            amrex::ParallelFor(mfi.validbox(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               {
                                   tmp(i, j, k) = src(i, j, k, comp);
                                   // The owner mask is the same for the src and dst MultiFab
                                   // since the src and dst MF of the FFT are the same.
                                   owner(i, j, k, comp) = 1;
                               });
        }

        f.m_r2c->forwardThenBackward(
            tmpsrc, tmpdst,
            [n = f.m_n, nMin = f.m_nMin, nMax = f.m_nMax] AMREX_GPU_DEVICE(int nx, int j, int k,
                                                                           auto& sp)
            {
                GEMPIC_D_EXCL(UNUSED(j);, UNUSED(k);, )
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
                    sp /= GEMPIC_D_MULT(n[Direction::xDir], n[Direction::yDir], n[Direction::zDir]);
                }
            });

        // copy back to
        for (amrex::MFIter mfi(tmpdst); mfi.isValid(); ++mfi)
        {
            auto const& tmp = tmpdst.array(mfi);
            auto const& dst = dstmf.array(mfi);

            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               { dst(i, j, k, comp) = tmp(i, j, k); });
        }
    }
    dstmf.OverrideSync(ownermask, f.m_periodicity);
}
} // namespace Impl
} //namespace Gempic::Filter
