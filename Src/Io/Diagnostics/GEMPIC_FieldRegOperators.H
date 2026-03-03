/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef GEMPIC_FIELDREGOPERATORS_H_
#define GEMPIC_FIELDREGOPERATORS_H_

#include <map>
#include <memory>
#include <string>
#include <utility>

#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include "GEMPIC_Fields.H"        // for Field types
#include "GEMPIC_Interpolation.H" // for interpolate_field_to_multifab_degree_selector

namespace Gempic::Io::Detail
{

// ----------------------------
// C++20 detection helpers
template <class T>
inline constexpr bool hasMfDirV = requires(T const& t, Direction d) { t.multi_fab(d); };

template <class T>
AMREX_FORCE_INLINE amrex::MultiFab const& subfield_mf (T const& t, Direction dir)
{
    if constexpr (hasMfDirV<T>)
    {
        return t.multi_fab(dir); // 1-/2-form
    }
    else
    {
        return t.multi_fab(); // 0-/3-form
    }
}

// ----------------------------
// Collapsed builders for type-erased ops

using InterpFn = void (*)(amrex::MultiFab&, void const*, amrex::Real, int, Direction);
using RawFn = void (*)(amrex::MultiFab&, void const*, int, Direction);
using CellCenteredFn = void (*)(amrex::MultiFab&, void const*, amrex::Real, int, Direction);

struct FieldOps
{
    InterpFn m_interp{};
    RawFn m_raw{};
    CellCenteredFn m_cell{};
};

template <typename T>
InterpFn make_interp_fn ()
{
    return +[] (amrex::MultiFab& dst, void const* p, amrex::Real scaling, int dcomp, Direction dir)
    {
        auto const& field = *static_cast<T const*>(p);
        Gempic::Forms::interpolate_field_to_multifab_degree_selector(dst, field, scaling, dcomp,
                                                                     dir);
    };
}

template <typename T>
RawFn make_raw_fn ()
{
    return +[] (amrex::MultiFab& dst, void const* p, int dcomp, Direction dir)
    {
        auto const& field = *static_cast<T const*>(p);
        Gempic::Forms::rawdata_field_to_multifabs(dst, field, dcomp, dir);
    };
}

template <typename T>
CellCenteredFn make_cellcentered_fn ()
{
    return +[] (amrex::MultiFab& dst, void const* p, amrex::Real scaling, int dcomp, Direction dir)
    {
        auto const& field = *static_cast<T const*>(p);
        Gempic::Forms::field_to_cellcentered_multifabs(dst, field, scaling, dcomp, dir);
    };
}

template <typename T>
FieldOps make_field_ops ()
{
    return FieldOps{make_interp_fn<T>(), make_raw_fn<T>(), make_cellcentered_fn<T>()};
}

} //namespace Gempic::Io::Detail

#endif