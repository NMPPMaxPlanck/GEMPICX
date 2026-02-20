/* Copyright 2019 Maxence Thevenet, Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef GEMPIC_FILTER_H_
#define GEMPIC_FILTER_H_

#include <limits>

#include <AMReX_Array4.H>
#include <AMReX_BaseFwd.H>
#include <AMReX_Box.H>
#include <AMReX_Config.H>
#ifdef AMREX_USE_FFT
#include <AMReX_FFT.H>
#endif
#include <AMReX_GpuContainers.H>
#include <AMReX_MFIter.H>
#include <AMReX_REAL.H>

#include "GEMPIC_Config.H"
#include "GEMPIC_Fields.H"

namespace Gempic::Filter
{
namespace Impl
{
void apply_stencil(amrex::MultiFab& dstmf,
                   amrex::MultiFab const& srcmf,
                   AMREX_D_DECL(amrex::Gpu::DeviceVector<amrex::Real> stencilX,
                                amrex::Gpu::DeviceVector<amrex::Real> stencilY,
                                amrex::Gpu::DeviceVector<amrex::Real> stencilZ));

// Only needs newS for optimization reasons that are a bit excessive
void convolve_filter(amrex::Vector<amrex::Real>& oldS,
                     amrex::Vector<amrex::Real>& newS,
                     amrex::Real const alpha,
                     int const ipass);

void compute_stencil(amrex::Gpu::DeviceVector<amrex::Real>& stencil,
                     unsigned int const npass,
                     amrex::Real const alpha,
                     bool const doCompensation);
} // namespace Impl

class Filter
{
public:
    virtual ~Filter() = default;

    virtual void do_filter(
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::node>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::node> const& src) = 0;
    virtual void do_filter(
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge> const& src) = 0;
    virtual void do_filter(
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face> const& src) = 0;
    virtual void do_filter(
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::cell>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::cell> const& src) = 0;
    virtual void do_filter(
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::node>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::node> const& src) = 0;
    virtual void do_filter(
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::edge>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::edge> const& src) = 0;
    virtual void do_filter(
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face> const& src) = 0;
    virtual void do_filter(
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell> const& src) = 0;

    template <Forms::Grid grid, Forms::Space space>
    void apply (Gempic::Forms::DeRhamField<grid, space>& dst,
                Gempic::Forms::DeRhamField<grid, space> const& src)
    {
        BL_PROFILE("Gempic::Filter::Filter::apply()");

        if (!m_useFilter)
        {
            // just copy
            copy(dst, src);
            return;
        }
        do_filter(dst, src);
    }

protected:
    bool m_useFilter{false};
};

class BilinearFilter : public Filter
{
public:
    BilinearFilter();

    void do_filter (
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::node>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::node> const& src) final
    {
        do_filter(dst.m_data, src.m_data);
    }
    void do_filter (
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge> const& src) final
    {
        do_filter_vector(dst, src);
    }
    void do_filter (
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face> const& src) final
    {
        do_filter_vector(dst, src);
    }
    void do_filter (
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::cell>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::cell> const& src) final
    {
        do_filter(dst.m_data, src.m_data);
    }
    void do_filter (
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::node>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::node> const& src) final
    {
        do_filter(dst.m_data, src.m_data);
    }
    void do_filter (
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::edge>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::edge> const& src) final
    {
        do_filter_vector(dst, src);
    }
    void do_filter (
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face> const& src) final
    {
        do_filter_vector(dst, src);
    }
    void do_filter (
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell> const& src) final
    {
        do_filter(dst.m_data, src.m_data);
    }

private:
    template <Forms::Grid grid, Forms::Space space>
        requires Forms::OneOrTwoFormSpaceC<space>
    void do_filter_vector (Gempic::Forms::DeRhamField<grid, space>& dst,
                           Gempic::Forms::DeRhamField<grid, space> const& src)
    {
        for (int d{0}; d < 3; ++d)
        {
            do_filter(dst.m_data[d], src.m_data[d]);
        }
    }
    void do_filter(amrex::MultiFab& dstmf, amrex::MultiFab const& srcmf);
    void compute_stencils();

    // Stencil along each direction.
    // in 2D, stencilZ is not initialized.
    amrex::Gpu::DeviceVector<amrex::Real> AMREX_D_DECL(m_stencilX, m_stencilY, m_stencilZ);
    // Filter parameter (0.5 for bilinear filters)
    amrex::Real const m_alpha{0.5};
    // Do a compensation step?
    bool m_compensate{false};
    amrex::Array<int, AMREX_SPACEDIM> m_nPass{AMREX_D_DECL(1, 1, 1)};
};

#ifdef AMREX_USE_FFT
class FourierFilter;
namespace Impl
{
void do_filter(amrex::MultiFab& dst, amrex::MultiFab const& src, FourierFilter& filter);
} // namespace Impl

class FourierFilter : public Filter
{
public:
    FourierFilter(Gempic::ComputationalDomain const& compDom);

    void do_filter (
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::node>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::node> const& src) final
    {
        Impl::do_filter(dst.m_data, src.m_data, *this);
    }
    void do_filter (
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge> const& src) final
    {
        do_filter_vector(dst, src);
    }
    void do_filter (
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face> const& src) final
    {
        do_filter_vector(dst, src);
    }
    void do_filter (
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::cell>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::cell> const& src) final
    {
        Impl::do_filter(dst.m_data, src.m_data, *this);
    }
    void do_filter (
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::node>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::node> const& src) final
    {
        Impl::do_filter(dst.m_data, src.m_data, *this);
    }
    void do_filter (
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::edge>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::edge> const& src) final
    {
        do_filter_vector(dst, src);
    }
    void do_filter (
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face> const& src) final
    {
        do_filter_vector(dst, src);
    }
    void do_filter (
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell> const& src) final
    {
        Impl::do_filter(dst.m_data, src.m_data, *this);
    }

    friend void Impl::do_filter(amrex::MultiFab& dst,
                                amrex::MultiFab const& src,
                                FourierFilter& filter);

private:
    template <Forms::Grid grid, Forms::Space space>
        requires Forms::OneOrTwoFormSpaceC<space>
    void do_filter_vector (Gempic::Forms::DeRhamField<grid, space>& dst,
                           Gempic::Forms::DeRhamField<grid, space> const& src)
    {
        for (int d{0}; d < 3; ++d)
        {
            Impl::do_filter(dst.m_data[d], src.m_data[d], *this);
        }
    }

    std::unique_ptr<amrex::FFT::R2C<amrex::Real>> m_r2c;
    std::array<int, AMREX_SPACEDIM> m_n, m_nMin, m_nMax;
    amrex::Periodicity m_periodicity;
    int m_numCells{0};
    amrex::MultiFab m_tmpsrc, m_tmpdst;
};
#endif

inline std::unique_ptr<Filter> make_filter (Gempic::ComputationalDomain const& compDom)
{
    Gempic::Io::Parameters params("Filter");
    std::string filterType{"Bilinear"};
    params.get_or_set("type", filterType);

    if (filterType == "Bilinear")
    {
        return std::make_unique<BilinearFilter>();
    }
    else if (filterType == "Fourier")
    {
#ifdef AMREX_USE_FFT
        return std::make_unique<FourierFilter>(compDom);
#else
        GEMPIC_ERROR("FFT was not compiled with GEMPICX so the Fourier filter is not available");
#endif
    }
    else
    {
        GEMPIC_ERROR("Filter type '" + filterType + "' is not implemented!");
    }
    exit(1); // Calms the compiler even though we don't technically return anything
}

} // namespace Gempic::Filter

#endif // #ifndef GEMPIC_FILTER_H_
