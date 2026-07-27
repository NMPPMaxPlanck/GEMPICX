/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
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
        do_filter(dst, src);
    }
};

std::unique_ptr<Filter> make_filter(Gempic::ComputationalDomain const& compDom);

class NoFilter : public Filter
{
public:
    NoFilter() = default;

    void do_filter(
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::node>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::node> const& src) final;
    void do_filter(
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge> const& src) final;
    void do_filter(
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face> const& src) final;
    void do_filter(
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::cell>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::cell> const& src) final;
    void do_filter(
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::node>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::node> const& src) final;
    void do_filter(
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::edge>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::edge> const& src) final;
    void do_filter(
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face> const& src) final;
    void do_filter(
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell>& dst,
        Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell> const& src) final;

    void operator()(Gempic::DiscreteField& dst, Gempic::DiscreteField& src);
    void operator()(Gempic::DiscreteVectorField& dst, Gempic::DiscreteVectorField& src);

private:
    void do_filter(amrex::MultiFab& dst, amrex::MultiFab const& src);
};

class BilinearFilter : public Filter
{
public:
    BilinearFilter(Io::Parameters&);
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

    void operator()(Gempic::DiscreteField& dst, Gempic::DiscreteField& src);
    void operator()(Gempic::DiscreteVectorField& dst, Gempic::DiscreteVectorField& src);

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

    std::array<amrex::Gpu::DeviceVector<amrex::Real>, AMREX_SPACEDIM> m_stencil;
    // Filter parameter (0.5 for bilinear filters)
    amrex::Real const m_alpha{0.5};
    // Do a compensation step
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
    FourierFilter(Io::Parameters& params, DiscreteGrid const& grid);
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

    void operator()(DiscreteField& dst, DiscreteField& src);
    void operator()(DiscreteVectorField& dst, DiscreteVectorField& src);

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
};
#endif

} // namespace Gempic::Filter

#endif // #ifndef GEMPIC_FILTER_H_
