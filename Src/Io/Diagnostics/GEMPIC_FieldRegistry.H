/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef GEMPIC_FIELDREGISTRY_H_
#define GEMPIC_FIELDREGISTRY_H_

#include <map>
#include <memory>
#include <string>
#include <utility>

#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include "GEMPIC_FieldRegOperators.H" // for Field types

// ====================================================================
// ----------------------------
// Public Type-erased Registry API
// ----------------------------
namespace Gempic::Io
{

struct Concept
{
    virtual ~Concept() = default;

    // Geometry/layout
    virtual int n_comp(Direction) const = 0;
    virtual amrex::BoxArray box_array(Direction) const = 0;
    virtual amrex::IndexType ix_type(Direction) const = 0;
    virtual amrex::DistributionMapping dm(Direction) const = 0;
    virtual amrex::Geometry geom() const = 0;
    virtual amrex::MultiFab const* mf_ptr(Direction) const = 0;

    // Operations used by diagnostics
    virtual void interpolate_to_multifab(amrex::MultiFab& dst,
                                         amrex::Real scaling,
                                         int dcomp,
                                         Direction) const = 0;
    virtual void rawdata_to_multifab(amrex::MultiFab& dst, int dcomp, Direction) const = 0;
    virtual void to_cellcentered_multifab(amrex::MultiFab& dst,
                                          amrex::Real scaling,
                                          int dcomp,
                                          Direction) const = 0;
};

template <typename T>
struct Model : Concept
{
    explicit Model(T* ptr) : m_ptr(ptr), m_ops(Detail::make_field_ops<T>()) // use collapsed builder
    {
    }

    int n_comp (Direction dir) const override { return Detail::subfield_mf(*m_ptr, dir).nComp(); }

    amrex::BoxArray box_array (Direction dir) const override
    {
        return Detail::subfield_mf(*m_ptr, dir).boxArray();
    }

    amrex::IndexType ix_type (Direction dir) const override
    {
        return Detail::subfield_mf(*m_ptr, dir).ixType();
    }

    amrex::DistributionMapping dm (Direction dir) const override
    {
        return Detail::subfield_mf(*m_ptr, dir).DistributionMap();
    }

    amrex::Geometry geom () const override { return m_ptr->get_geom(); }

    amrex::MultiFab const* mf_ptr (Direction dir) const override
    {
        return &Detail::subfield_mf(*m_ptr, dir);
    }

    void interpolate_to_multifab (amrex::MultiFab& dst,
                                  amrex::Real scaling,
                                  int dcomp,
                                  Direction dir) const override
    {
        m_ops.m_interp(dst, m_ptr, scaling, dcomp, dir);
    }

    void rawdata_to_multifab (amrex::MultiFab& dst, int dcomp, Direction dir) const override
    {
        m_ops.m_raw(dst, m_ptr, dcomp, dir);
    }

    void to_cellcentered_multifab (amrex::MultiFab& dst,
                                   amrex::Real scaling,
                                   int dcomp,
                                   Direction dir) const override
    {
        m_ops.m_cell(dst, m_ptr, scaling, dcomp, dir);
    }

    T* get_ptr () const { return m_ptr; } // getter

private:
    T* m_ptr;
    Detail::FieldOps m_ops;
};
// ----------------------------
// Pointer-based type-erased wrapper (now owns dir + scaling)
struct AnyFieldPtr
{
    AnyFieldPtr() = default;

    template <typename T>
    explicit AnyFieldPtr(T* f, amrex::Real scaling = 1.0, Direction dir = noDir) :
        m_self(std::make_shared<Model<T>>(f)), m_scaling(scaling), m_dir(dir)
    {
    }

    // ---- metadata
    amrex::Real scaling () const noexcept { return m_scaling; }
    Direction dir () const noexcept { return m_dir; }

    // ---- structural queries
    int n_comp () const { return m_self->n_comp(m_dir); }
    amrex::BoxArray box_array () const { return m_self->box_array(m_dir); }
    amrex::IndexType ix_type () const { return m_self->ix_type(m_dir); }
    amrex::DistributionMapping get_distribution_mapping () const { return m_self->dm(m_dir); }
    amrex::Geometry geom () const { return m_self->geom(); }

    amrex::MultiFab const* mf_ptr () const { return m_self->mf_ptr(m_dir); }

    amrex::MultiFab make_mf_alias (int scomp, int ncomp) const
    {
        auto const* src = mf_ptr();
        AMREX_ALWAYS_ASSERT(src != nullptr);
        return amrex::MultiFab(*src, amrex::make_alias, scomp, ncomp);
    }

    // ---- diagnostics operations
    void interpolate_any_field_to_multifab (amrex::MultiFab& dst, int dcomp) const
    {
        m_self->interpolate_to_multifab(dst, m_scaling, dcomp, m_dir);
    }

    void any_field_to_rawdata_multifabs (amrex::MultiFab& dst, int dcomp) const
    {
        m_self->rawdata_to_multifab(dst, dcomp, m_dir);
    }

    void any_field_to_cellcentered_multifabs (amrex::MultiFab& dst, int dcomp) const
    {
        m_self->to_cellcentered_multifab(dst, m_scaling, dcomp, m_dir);
    }

    // ---- optional downcast
    template <typename T>
    T const* as_ptr_or_null () const
    {
        auto m = std::dynamic_pointer_cast<Model<T>>(m_self);
        return m ? m->get_ptr() : nullptr;
    }

private:
    std::shared_ptr<Concept> m_self;
    amrex::Real m_scaling{1.0};
    Direction m_dir{xDir};
};

// -----------------------------------------------------------------------------
// FieldRegistry types (map-based)
using Key = std::string;
using FieldDataLoc = AnyFieldPtr;
using FieldRegistry = std::map<Key, FieldDataLoc>;

// -----------------------------------------------------------------------------
// Free helper functions for field registry

// Register a field (emplace, returns true if inserted)
inline bool register_field (FieldRegistry& registry, Key const& name, AnyFieldPtr field)
{
    auto [it, inserted] = registry.try_emplace(name, std::move(field));
    return inserted;
}

// Overload for raw pointer + scaling/direction
template <typename T>
inline bool register_field (
    FieldRegistry& registry, Key const& name, T* ptr, amrex::Real scaling, Direction dir = noDir)
{
    return register_field(registry, name, AnyFieldPtr{ptr, scaling, dir});
}

// Find a field by name (returns pointer to AnyFieldPtr or nullptr)
inline FieldDataLoc const* find_field (FieldRegistry const& registry, Key const& name)
{
    auto it = registry.find(name);
    return it == registry.end() ? nullptr : &it->second;
}

// Get pointer to the concrete field type (or nullptr if type mismatch or not found)
template <typename FieldType>
inline FieldType const* get_field_ptr_or_null (FieldRegistry const& registry, Key const& name)
{
    if (auto const* entry = find_field(registry, name))
    {
        return entry->template as_ptr_or_null<FieldType>();
    }
    return nullptr;
}

// -----------------------------------------------------------------------------
// Overload/template helper for registering forms
template <typename SrcFormType>
void register_form (FieldRegistry& reg, std::string const& name, SrcFormType& f)
{
    if constexpr (Forms::is_one_or_two_form<SrcFormType::s_form>())
    {
        register_field(reg, name + "x", &f, f.get_scaling(xDir), xDir);
        register_field(reg, name + "y", &f, f.get_scaling(yDir), yDir);
        register_field(reg, name + "z", &f, f.get_scaling(zDir), zDir);
    }
    else
    {
        register_field(reg, name, &f, f.get_scaling(), xDir);
    }
}

// Helper function so that fewer overloads are needed
template <typename SrcFormType, typename... Args>
SrcFormType registered_form (FieldRegistry& reg, std::string const& name, Args&&... args)
{
    SrcFormType f(args...);
    register_form(reg, name, f);
    return f;
}

} // namespace Gempic::Io

#endif // GEMPIC_FIELDREGISTRY_H_
