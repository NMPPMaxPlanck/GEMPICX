#include <memory>

#include <AMReX.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Filter.H"

namespace Gempic::Filter
{
namespace Impl
{
void apply_stencil (amrex::MultiFab& dstmf,
                    amrex::MultiFab const& srcmf,
                    AMREX_D_DECL(amrex::Gpu::DeviceVector<amrex::Real> stencilX,
                                 amrex::Gpu::DeviceVector<amrex::Real> stencilY,
                                 amrex::Gpu::DeviceVector<amrex::Real> stencilZ))
{
    BL_PROFILE("Gempic::Filter::Impl::apply_stencil()");
    AMREX_D_TERM(amrex::Real const* AMREX_RESTRICT sx = stencilX.data();
                 , amrex::Real const* AMREX_RESTRICT sy = stencilY.data();
                 , amrex::Real const* AMREX_RESTRICT sz = stencilZ.data();)

    amrex::Dim3 slen{GEMPIC_D_PAD(static_cast<int>(stencilX.size()),
                                  static_cast<int>(stencilY.size()),
                                  static_cast<int>(stencilZ.size()))};

    for (amrex::MFIter mfi(dstmf); mfi.isValid(); ++mfi)
    {
        // src and dst are of type Array4 (Fortran ordering)
        auto const& src = srcmf.array(mfi);
        auto const& dst = dstmf.array(mfi);

        GEMPIC_D_EXCL(int iy{0};, int iz{0};, )
        amrex::ParallelFor(
            mfi.validbox(), srcmf.nComp(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
            {
                amrex::Real d{0.0};

                // 3 nested loop on 3D stencil
                GEMPIC_D_LOOP_BEGIN(for (int ix{0}; ix < slen.x; ++ix),
                                    for (int iy{0}; iy < slen.y; ++iy),
                                    for (int iz{0}; iz < slen.z; ++iz))
                    amrex::Real sss = GEMPIC_D_MULT(sx[ix], sy[iy], sz[iz]);

                    d += sss *
                         (GEMPIC_D_ADD(
                             src(i - ix, j - iy, k - iz, n) + src(i + ix, j - iy, k - iz, n),
                             src(i - ix, j + iy, k - iz, n) + src(i + ix, j + iy, k - iz, n),
                             src(i - ix, j - iy, k + iz, n) + src(i + ix, j - iy, k + iz, n) +
                                 src(i - ix, j + iy, k + iz, n) + src(i + ix, j + iy, k + iz, n)));
                GEMPIC_D_LOOP_END

                dst(i, j, k, n) = d;
            });
    }
}

// Only needs newS for optimization reasons that are a bit excessive
void convolve_filter (amrex::Vector<amrex::Real>& oldS,
                      amrex::Vector<amrex::Real>& newS,
                      amrex::Real const alpha,
                      int const ipass)
{
    BL_PROFILE("Gempic::Filter::Impl::convolve_filter()");
    amrex::Real const beta{0.5 - 0.5 * alpha};

    // element 0 has to be treated in its own way
    newS[0] = alpha * oldS[0] + (1 - alpha) * oldS[1];
    // For each element j, apply the filter to oldS to get newS[j]
    for (int j{1}; j < ipass; j++)
    {
        newS[j] = alpha * oldS[j] + beta * (oldS[j - 1] + oldS[j + 1]);
    }
    // final element has to be treated in its own way
    newS[ipass] = alpha * oldS[ipass] + beta * oldS[ipass - 1];

    oldS = newS;
}

void compute_stencil (amrex::Gpu::DeviceVector<amrex::Real>& stencil,
                      unsigned int const npass,
                      amrex::Real const alpha,
                      bool const doCompensation)
{
    BL_PROFILE("Gempic::Filter::Impl::compute_stencil()");
    amrex::Vector<amrex::Real> oldS(1u + npass + static_cast<size_t>(doCompensation), 0.);
    amrex::Vector<amrex::Real> newS(oldS.size(), 0.);

    oldS[0] = 1.;
    // Convolve the filter with itself npass times
    int const lastpass = static_cast<int>(npass + 1u);
    for (int ipass{1}; ipass < lastpass; ipass++)
    {
        convolve_filter(oldS, newS, alpha, ipass);
    }

    if (doCompensation)
    {
        amrex::Real compensationAlpha{0.5 * npass};

        // Convolve using different alpha
        convolve_filter(oldS, newS, compensationAlpha, lastpass);
    }

    // we use oldS here to make sure the stencil is correct even when npass = 0
    oldS[0] *= 0.5; // because we will use it twice
    stencil.resize(oldS.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, oldS.begin(), oldS.end(), stencil.begin());
    /// @todo: Device wide Gpu::synchronize() usually too excessive and might interfere with other
    /// libraries (e.g. MPI)
    /// https://amrex-codes.github.io/amrex/docs_html/GPU.html#stream-and-synchronization
    amrex::Gpu::synchronize();
}
} //namespace Impl

BilinearFilter::BilinearFilter()
{
    BL_PROFILE("Gempic::Filter::BilinearFilter::BilinearFilter()");
    Io::Parameters params("Filter", "class BilinearFilter");
    // Filter parameters
    // If true, a bilinear filter is used to smooth charge and currents
    params.get_or_set("enable", m_useFilter);
    if (m_useFilter)
    {
        params.get("nPass", m_nPass);

        // Do a compensation step?
        params.get_or_set("compensate", m_compensate);

        std::vector<int> nGhostVector;
        params.get("nGhost", nGhostVector);
        if (AMREX_D_TERM((nGhostVector[xDir] < (m_nPass[xDir] + m_compensate)),
                         || (nGhostVector[yDir] < (m_nPass[yDir] + m_compensate)),
                         || (nGhostVector[zDir] < (m_nPass[zDir] + m_compensate))))
        {
            AMREX_ALWAYS_ASSERT(
                "Grid is not large enough to contain the filter stencil. Try increasing the "
                "amount of extra ghost cells or decreasing the number of filter passes.\n");
        }
        compute_stencils();
    }
}

void BilinearFilter::do_filter (amrex::MultiFab& dstmf, amrex::MultiFab const& srcmf)
{
    AMREX_ALWAYS_ASSERT(dstmf.nComp() == srcmf.nComp());
    Impl::apply_stencil(dstmf, srcmf, AMREX_D_DECL(m_stencilX, m_stencilY, m_stencilZ));
}

void BilinearFilter::compute_stencils ()
{
    BL_PROFILE("Gempic::Filter::BilinearFilter::compute_stencils()");
    AMREX_D_DECL(Impl::compute_stencil(m_stencilX, m_nPass[xDir], m_alpha, m_compensate),
                 Impl::compute_stencil(m_stencilY, m_nPass[yDir], m_alpha, m_compensate),
                 Impl::compute_stencil(m_stencilZ, m_nPass[zDir], m_alpha, m_compensate));
}
} //namespace Gempic::Filter