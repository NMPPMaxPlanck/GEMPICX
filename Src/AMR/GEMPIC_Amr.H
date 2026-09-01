/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef GempicAmr_H_
#define GempicAmr_H_

#include <filesystem>

#include <AMReX_AmrCore.H>
#include <AMReX_PlotFileUtil.H>

#include "GEMPIC_AmrLevelInterface.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Parameters.H"

namespace Gempic::Amr
{

amrex::AmrInfo create_amr_info(Io::Parameters& params);
amrex::Geometry create_geometry(Io::Parameters& params);

class GempicAmrCore : public amrex::AmrCore
{
public:
    GempicAmrCore(Io::Parameters& params);
    ~GempicAmrCore() override {};

    void MakeNewLevelFromCoarse(int /*lev*/,
                                amrex::Real /*time*/,
                                amrex::BoxArray const& /*ba*/,
                                amrex::DistributionMapping const& /*dm*/) override {};

    void RemakeLevel(int /*lev*/,
                     amrex::Real /*time*/,
                     amrex::BoxArray const& /*ba*/,
                     amrex::DistributionMapping const& /*dm*/) override {};

    void ClearLevel(int /*lev*/) override {};

    // Make a new level from scratch using provided BoxArray and DistributionMapping.
    // Only used during initialization.
    void MakeNewLevelFromScratch(int /*lev*/,
                                 amrex::Real /*time*/,
                                 amrex::BoxArray const& /*ba*/,
                                 amrex::DistributionMapping const& /*dm*/) override {};

    // Tag all cells for refinement
    void ErrorEst(int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow) override;

private:
    amrex::Parser m_taggingParser;
    amrex::Vector<amrex::Real> m_taggingLevels;
};

/**************************************************************************************************
 * Multi Level Field Datastructure                                                                *
 **************************************************************************************************/

template <typename Form>
struct MLForm : public std::vector<Form>
{
    // Scalar interface for 0-forms and 3-forms, vector for 1-forms and 2-forms
    using LevelInterfaceType =
        std::conditional_t<std::is_same_v<Form, Gempic::PrimalZeroForm> ||
                               std::is_same_v<Form, Gempic::PrimalThreeForm> ||
                               std::is_same_v<Form, Gempic::DualZeroForm> ||
                               std::is_same_v<Form, Gempic::DualThreeForm>,
                           Impl::LevelInterface,               // scalar
                           std::array<Impl::LevelInterface, 3> // vector
                           >;

    std::shared_ptr<GempicAmrCore> m_amr;
    std::string m_label;
    std::optional<std::vector<LevelInterfaceType>> m_levelInterfaces;

    MLForm(std::shared_ptr<GempicAmrCore> const& amr, std::string label) :
        m_amr{amr}, m_label{label}, m_levelInterfaces{std::nullopt}
    {
    }
};

// Primal forms
using MLPrimalZeroForm = MLForm<Gempic::PrimalZeroForm>;
using MLPrimalOneForm = MLForm<Gempic::PrimalOneForm>;
using MLPrimalTwoForm = MLForm<Gempic::PrimalTwoForm>;
using MLPrimalThreeForm = MLForm<Gempic::PrimalThreeForm>;

// Dual forms
using MLDualZeroForm = MLForm<Gempic::DualZeroForm>;
using MLDualOneForm = MLForm<Gempic::DualOneForm>;
using MLDualTwoForm = MLForm<Gempic::DualTwoForm>;
using MLDualThreeForm = MLForm<Gempic::DualThreeForm>;

template <typename T>
concept MLFormC = std::same_as<T, MLPrimalZeroForm> || std::same_as<T, MLPrimalOneForm> ||
                  std::same_as<T, MLPrimalTwoForm> || std::same_as<T, MLPrimalThreeForm> ||
                  std::same_as<T, MLDualZeroForm> || std::same_as<T, MLDualOneForm> ||
                  std::same_as<T, MLDualTwoForm> || std::same_as<T, MLDualThreeForm>;

template <typename T>
concept MLPrimalFormC = std::same_as<T, MLPrimalZeroForm> || std::same_as<T, MLPrimalOneForm> ||
                        std::same_as<T, MLPrimalTwoForm> || std::same_as<T, MLPrimalThreeForm>;

template <typename T>
concept MLDualFormC = MLFormC<T> && !MLPrimalFormC<T>;

template <typename T>
concept MLScalarFormC = std::same_as<T, MLPrimalZeroForm> || std::same_as<T, MLPrimalThreeForm> ||
                        std::same_as<T, MLDualZeroForm> || std::same_as<T, MLDualThreeForm>;

template <typename T>
concept MLVectorFormC = MLFormC<T> && !MLScalarFormC<T>;

template <MLScalarFormC T>
std::vector<Impl::LevelInterface> make_level_interfaces (
    T const& field, std::shared_ptr<Gempic::Amr::GempicAmrCore> const& amr, bool buildCoveredMask)
{
    std::vector<Impl::LevelInterface> result;
    result.reserve(amr->finestLevel());
    for (int lev = 0; lev < amr->finestLevel(); lev++)
    {
        result.push_back(Impl::LevelInterface(field[lev + 1].multi_fab(), field[lev].multi_fab(),
                                              amr->refRatio(lev), amr->Geom(lev + 1),
                                              buildCoveredMask));
    }
    return result;
}

template <MLVectorFormC T>
std::vector<std::array<Impl::LevelInterface, 3>> make_level_interfaces (
    T const& field, std::shared_ptr<Gempic::Amr::GempicAmrCore> const& amr, bool buildCoveredMask)
{
    std::vector<std::array<Impl::LevelInterface, 3>> result;
    result.reserve(amr->finestLevel());
    for (int lev = 0; lev < amr->finestLevel(); lev++)
    {
        result.push_back(
            {Impl::LevelInterface(field[lev + 1].multi_fab(xDir), field[lev].multi_fab(xDir),
                                  amr->refRatio(lev), amr->Geom(lev + 1), buildCoveredMask),
             Impl::LevelInterface(field[lev + 1].multi_fab(yDir), field[lev].multi_fab(yDir),
                                  amr->refRatio(lev), amr->Geom(lev + 1), buildCoveredMask),
             Impl::LevelInterface(field[lev + 1].multi_fab(zDir), field[lev].multi_fab(zDir),
                                  amr->refRatio(lev), amr->Geom(lev + 1), buildCoveredMask)});
    }
    return result;
}

class MLFiniteDifferenceDeRhamSpaces
{
public:
    MLFiniteDifferenceDeRhamSpaces(
        Io::Parameters& param,
        std::shared_ptr<GempicAmrCore> amr,
        GaussLegendreQuadrature const& integrator = GaussLegendreQuadrature{5});

    MLPrimalZeroForm create_primal_zero_form(std::string const& label,
                                             Gempic::Impl::BoundaryConditionConfiguration bcConf,
                                             bool buildLevelInterface = true,
                                             bool buildCoveredMask = true) const;
    MLPrimalOneForm create_primal_one_form(
        std::string const& label,
        std::array<Gempic::Impl::BoundaryConditionConfiguration, 3> bcConf,
        bool buildLevelInterface = true,
        bool buildCoveredMask = true) const;
    MLPrimalTwoForm create_primal_two_form(
        std::string const& label,
        std::array<Gempic::Impl::BoundaryConditionConfiguration, 3> bcConf,
        bool buildLevelInterface = true,
        bool buildCoveredMask = true) const;
    MLPrimalThreeForm create_primal_three_form(std::string const& label,
                                               Gempic::Impl::BoundaryConditionConfiguration bcConf,
                                               bool buildLevelInterface = true,
                                               bool buildCoveredMask = true) const;

    MLDualZeroForm create_dual_zero_form(std::string const& label,
                                         Gempic::Impl::BoundaryConditionConfiguration bcConf,
                                         bool buildLevelInterface = true,
                                         bool buildCoveredMask = true) const;
    MLDualOneForm create_dual_one_form(
        std::string const& label,
        std::array<Gempic::Impl::BoundaryConditionConfiguration, 3> bcConf,
        bool buildLevelInterface = true,
        bool buildCoveredMask = true) const;
    MLDualTwoForm create_dual_two_form(
        std::string const& label,
        std::array<Gempic::Impl::BoundaryConditionConfiguration, 3> bcConf,
        bool buildLevelInterface = true,
        bool buildCoveredMask = true) const;
    MLDualThreeForm create_dual_three_form(std::string const& label,
                                           Gempic::Impl::BoundaryConditionConfiguration bcConf,
                                           bool buildLevelInterface = true,
                                           bool buildCoveredMask = true) const;

private:
    std::shared_ptr<GempicAmrCore> m_amr;
    std::vector<Gempic::FiniteDifferenceDeRhamSpaces> m_deRham;
};

/**************************************************************************************************
 * Level-wise Operations (Derivatives, Arithmetics, Projections, ...)                             *
 **************************************************************************************************/

#define CHECK_NUMBER_OF_LEVELS(t, u)                                                              \
    GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE(                                                            \
        (t).size() == (u).size(), "Multi-level fields do not have the same number of levels: " #t \
                                  ".size() (" +                                                   \
                                      std::to_string((t).size()) + ") != " #u ".size() (" +       \
                                      std::to_string((u).size()) + ")")

template <MLFormC T>
    requires requires(typename T::reference t) {
        t.apply_boundary_conditions({AMREX_D_DECL(0, 0, 0)});
    }
void apply_boundary_conditions (T& t, std::array<size_t, AMREX_SPACEDIM> width)
{
    for (auto& lev : t)
    {
        lev.apply_boundary_conditions(width);
    }
}

template <MLFormC T, typename F>
    requires requires(typename T::reference t, F& f) { project(t, f); }
void project (T& t, F& f)
{
    for (auto& lev : t) project(lev, f);
}

template <MLFormC T, typename F>
    requires requires(typename T::reference t, F& f) { project(t, f, std::declval<amrex::Real>()); }
void project (T& t, F& f, amrex::Real time)
{
    for (auto& lev : t) project(lev, f, time);
}

template <MLFormC T, MLFormC U>
    requires requires(typename T::reference t, typename U::reference u) { grad(t, u); }
void grad (T& t, U& u)
{
    CHECK_NUMBER_OF_LEVELS(t, u);
    for (int lev = 0; lev < static_cast<int>(t.size()); lev++) grad(t[lev], u[lev]);
}

template <MLFormC T, MLFormC U>
    requires requires(typename T::reference t, typename U::reference u) { curl(t, u); }
void curl (T& t, U& u)
{
    CHECK_NUMBER_OF_LEVELS(t, u);
    for (int lev = 0; lev < static_cast<int>(t.size()); lev++) curl(t[lev], u[lev]);
}

template <MLFormC T, MLFormC U>
    requires requires(typename T::reference t, typename U::reference u) { div(t, u); }
void div (T& t, U& u)
{
    CHECK_NUMBER_OF_LEVELS(t, u);
    for (int lev = 0; lev < static_cast<int>(t.size()); lev++) div(t[lev], u[lev]);
}

template <MLFormC T, MLFormC U>
    requires requires(typename T::reference t, typename U::reference u) { operator+=(t, u); }
void operator+=(T& t, U& u)
{
    CHECK_NUMBER_OF_LEVELS(t, u);
    for (int lev = 0; lev < static_cast<int>(t.size()); lev++) t[lev] += u[lev];
}

template <MLFormC T, MLFormC U>
    requires requires(typename T::reference t, typename U::reference u) { operator-=(t, u); }
void operator-=(T& t, U& u)
{
    CHECK_NUMBER_OF_LEVELS(t, u);
    for (int lev = 0; lev < static_cast<int>(t.size()); lev++) t[lev] -= u[lev];
}

template <MLFormC T>
    requires requires(typename T::reference t) { operator*=(t, std::declval<amrex::Real>()); }
void operator*=(T& t, amrex::Real factor)
{
    for (int lev = 0; lev < static_cast<int>(t.size()); lev++) t[lev] *= factor;
}

template <MLFormC T, MLFormC U>
    requires requires(typename T::reference t, typename U::reference u) { copy(t, u); }
void copy (T& t, U& u)
{
    CHECK_NUMBER_OF_LEVELS(t, u);
    for (int lev = 0; lev < static_cast<int>(t.size()); lev++) copy(t[lev], u[lev]);
}

template <MLFormC T, MLFormC U>
    requires requires(typename T::reference t, typename U::reference u) { dot_product(t, u); }
amrex::Real dot_product (T& t, U& u)
{
    CHECK_NUMBER_OF_LEVELS(t, u);
    amrex::Real res = 0;
    for (int lev = 0; lev < static_cast<int>(t.size()); lev++) res += dot_product(t[lev], u[lev]);
    return res;
}

template <MLFormC T, MLFormC U>
    requires requires(typename T::reference t, typename U::reference u) { dot_product(t, u); }
amrex::Real dot_product (T& t, U& u, Direction dir)
{
    CHECK_NUMBER_OF_LEVELS(t, u);
    amrex::Real res = 0;
    for (int lev = 0; lev < static_cast<int>(t.size()); lev++)
    {
        res += dot_product(t[lev][dir], u[lev][dir]);
    }
    return res;
}

template <MLFormC T, MLFormC U>
    requires requires(typename T::reference t, typename U::reference u) {
        Gempic::l_inf_error(t, u);
    }
amrex::Real l_inf_error (T& t, U& u)
{
    CHECK_NUMBER_OF_LEVELS(t, u);
    amrex::Real res = 0;
    for (int lev = 0; lev < static_cast<int>(t.size()); lev++)
    {
        if constexpr (MLVectorFormC<T>)
        {
            res = std::max(res, std::ranges::max(Gempic::l_inf_error(t[lev], u[lev])));
        }
        else
        {
            res = std::max(res, Gempic::l_inf_error(t[lev], u[lev]));
        }
    }
    return res;
}

template <MLFormC T, MLFormC U, MLFormC V>
    requires requires(typename T::reference t, typename U::reference u, typename V::reference v) {
        linear_combination(t, std::declval<amrex::Real>(), u, std::declval<amrex::Real>(), v);
    }
void linear_combination (T& t, amrex::Real a, U& u, amrex::Real b, V& v)
{
    CHECK_NUMBER_OF_LEVELS(t, u);
    CHECK_NUMBER_OF_LEVELS(t, v);
    for (int lev = 0; lev < static_cast<int>(t.size()); lev++)
    {
        linear_combination(t[lev], a, u[lev], b, v[lev]);
    }
}

template <MLScalarFormC T>
void fill_all (T& t, amrex::Real value)
{
    for (int lev = 0; lev < static_cast<int>(t.size()); lev++)
    {
        t[lev].multi_fab().setVal(value);
    }
}

template <MLVectorFormC T>
void fill_all (T& t, amrex::Real value)
{
    for (int lev = 0; lev < static_cast<int>(t.size()); lev++)
    {
        for (auto dir : {xDir, yDir, zDir})
        {
            t[lev].multi_fab(dir).setVal(value);
        }
    }
}

template <MLFormC T>
    requires requires(typename T::reference t) { fill_random(t); }
void fill_random (T& t)
{
    for (int lev = 0; lev < static_cast<int>(t.size()); lev++) fill_random(t[lev]);
}

template <MLScalarFormC T>
void override_sync (T& t)
{
    for (int lev = 0; lev < static_cast<int>(t.size()); lev++)
    {
        t[lev].multi_fab().OverrideSync();
    }
}

template <MLVectorFormC T>
void override_sync (T& t)
{
    for (int lev = 0; lev < static_cast<int>(t.size()); lev++)
    {
        for (auto dir : {xDir, yDir, zDir})
        {
            t[lev].multi_fab(dir).OverrideSync();
        }
    }
}

template <MLScalarFormC T>
amrex::Real sum_unique (T& t)
{
    amrex::Real sum = 0;
    for (int lev = 0; lev < static_cast<int>(t.size()); lev++)
    {
        sum += t[lev].multi_fab().sum_unique(
            0, false, Gempic::Impl::to_amrex_periodicity(t[lev].discrete_grid()));
    }
    amrex::ParallelDescriptor::ReduceRealSum(sum);
    return sum;
}

/**************************************************************************************************
 * Operations involving Level Interfaces (Conforming Projections, Hodges (with corrections), ...) *
 **************************************************************************************************/

#define CHECK_LEVEL_INTERFACES(t)                                        \
    GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE((t).m_levelInterfaces.has_value(), \
                                      "Level interfaces are not built for field: " #t)
namespace Impl
{
template <typename T>
void do_hodge_correction (T& form,
                          Impl::LevelInterface const& li,
                          amrex::BoxArray const& ba,
                          amrex::DistributionMapping const& dm,
                          amrex::Vector<int> const& boxIdx,
                          bool isFine,
                          bool isPrimal)
{
    for (amrex::MFIter mfi(ba, dm); mfi.isValid(); ++mfi)
    {
        amrex::Real factor;
        if (isPrimal)
        {
            factor = (isFine ? 1.0 * (1 << AMREX_SPACEDIM) /
                                   li.m_orientationFine[mfi.index()].m_coveredNeighbors
                             : 1.0 * (1 << AMREX_SPACEDIM) /
                                   ((1 << AMREX_SPACEDIM) -
                                    li.m_orientationCoarse[mfi.index()].m_coveredNeighbors));
        }
        else
        {
            factor = (isFine ? 1.0 * li.m_orientationFine[mfi.index()].m_coveredNeighbors /
                                   (1 << AMREX_SPACEDIM)
                             : 1.0 *
                                   ((1 << AMREX_SPACEDIM) -
                                    li.m_orientationCoarse[mfi.index()].m_coveredNeighbors) /
                                   (1 << AMREX_SPACEDIM));
        }
        form.select_box(boxIdx[mfi.LocalIndex()]);

        ParallelFor(mfi.validbox(),
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) { form(i, j, k) *= factor; });
    }
}

template <MLFormC T>
void hodge_correction (T& mlform)
{
    BL_PROFILE("Amr::hodge_correction");
    CHECK_LEVEL_INTERFACES(mlform);
    for (int lev = 0; lev < static_cast<int>(mlform.size()) - 1; ++lev)
    {
        if constexpr (MLVectorFormC<T>)
        {
            for (auto dir : {xDir, yDir, zDir})
            {
                auto const& li = mlform.m_levelInterfaces.value()[lev][dir];
                do_hodge_correction(mlform[lev + 1][dir], li, li.m_baFine, li.m_dmFine,
                                    li.m_fineIdx, true, MLPrimalFormC<T>);
                do_hodge_correction(mlform[lev][dir], li, li.m_baCoarse, li.m_dmCoarse,
                                    li.m_coarseIdx, false, MLPrimalFormC<T>);
            }
        }
        else
        {
            auto const& li = mlform.m_levelInterfaces.value()[lev];
            do_hodge_correction(mlform[lev + 1], li, li.m_baFine, li.m_dmFine, li.m_fineIdx, true,
                                MLPrimalFormC<T>);
            do_hodge_correction(mlform[lev], li, li.m_baCoarse, li.m_dmCoarse, li.m_coarseIdx,
                                false, MLPrimalFormC<T>);
        }
    }
}

template <typename T>
void do_zero_covered_coarse (T& form, Impl::LevelInterface const& li)
{
    for (amrex::MFIter mfi(form.multi_fab()); mfi.isValid(); ++mfi)
    {
        form.select_box(mfi);
        amrex::Array4<int const> const& mask = li.m_coarseCoveredMask->const_array(mfi);

        ParallelFor(mfi.validbox(),
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) { form(i, j, k) *= mask(i, j, k); });
    }
}

// make fine box arrays with odd ends coarsenable by adapting the low and high of every box
// version for summing/averaging shared dof on the boundary
amrex::BoxArray coarsen_fine_summation(amrex::BoxArray const& ba, amrex::IntVect const refRatio);

// make fine box arrays with odd ends coarsenable by adapting the low and high of every box
// version for interpolating inbetween shared dof
// interpolation is applied per cell to fill all hanging fine dof
amrex::BoxArray coarsen_fine_interpolation(amrex::BoxArray const& ba,
                                           amrex::IntVect const refRatio);

// shifts for summing fine dof that belong to a single coarse dof. In 3D:
// point values -> one zero shift
// edge integrals -> two shifts
// face integrals -> four shifts
struct SummationShifts
{
    std::array<amrex::IntVect, 1 << (AMREX_SPACEDIM - 1)> m_shifts;
    int m_num;
};

SummationShifts build_summation_shifts(amrex::IntVect const boxLength,
                                       amrex::IntVect const idxType,
                                       amrex::IntVect const refRatio);

// Shifts for interpolating to fine dof that do not align with a coarse dof in at least one nodal
// direction.
// Shifts give the relative position of the fine dof that have not been updated by an averaging.
// For a fully nodal field the shifts correspond to
//  - a linear interpolation on edges (2D)
//  - a bilinear interpolation on faces (3D)
struct InterpolationShifts
{
    std::array<amrex::IntVect, AMREX_D_PICK(0, 1, 5)> m_shifts;
    std::array<amrex::IntVect, 1 << (AMREX_SPACEDIM - 1)> m_cornerShifts;
    int m_numShifts;
    int m_numCornerShifts;
    std::array<int, AMREX_SPACEDIM> m_tangentialDirs;
    int m_numTangentialDirs;
};

InterpolationShifts build_interpolation_shifts(amrex::IntVect const boxLength,
                                               amrex::IntVect const idxType,
                                               amrex::IntVect const refRatio);

template <typename T>
void do_conforming_projection (T& coarseForm, T& fineForm, Impl::LevelInterface const& li)
{
    amrex::IntVect idxType = coarseForm.multi_fab().ixType().toIntVect();
    amrex::IntVect refRatio = li.m_refRatio;

    // create temporary multifabs to parallel copy data of other refinement level
    amrex::MultiFab fineData;   // for coarse side
    amrex::MultiFab coarseData; // for fine side

    // these box computations could be moved to 'LevelInterface' since they are the same every time
    auto baFineSummation = coarsen_fine_summation(li.m_baFine, refRatio);
    auto baFineInterpolation = coarsen_fine_interpolation(li.m_baFine, refRatio);
    // cannot refine boxArray with single dof in nodal direction. use workaround via boxList
    auto tmp = li.m_baCoarse.boxList();
    tmp.refine(refRatio);
    auto baCoarse = amrex::BoxArray(std::move(tmp));

    int nComp{1};
    int nGhost{0};
    fineData.define(baCoarse, li.m_dmCoarse, nComp, nGhost, amrex::MFInfo());
    coarseData.define(baFineSummation, li.m_dmFine, nComp, nGhost, amrex::MFInfo());

    // both copy operations need to completed before proceeding to prevent race conditions
    fineData.ParallelCopy(fineForm.multi_fab(),
                          Gempic::Impl::to_amrex_periodicity(fineForm.discrete_grid()));
    coarseData.ParallelCopy(coarseForm.multi_fab(),
                            Gempic::Impl::to_amrex_periodicity(coarseForm.discrete_grid()));

    // looping over coarse interface
    for (amrex::MFIter mfi(li.m_baCoarse, li.m_dmCoarse); mfi.isValid(); ++mfi)
    {
        amrex::Array4<amrex::Real const> const& arrFine = fineData.const_array(mfi);
        coarseForm.select_box(li.m_coarseIdx[mfi.LocalIndex()]);

        auto shifts = build_summation_shifts(mfi.validbox().length(), idxType, refRatio);

        ParallelFor(mfi.validbox(),
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        amrex::IntVect iv{AMREX_D_DECL(i, j, k)};
                        amrex::IntVect fineIv = iv * refRatio;
                        amrex::Real sum = 0;
                        for (int ii = 0; ii < shifts.m_num; ii++)
                        {
                            sum += arrFine(fineIv + shifts.m_shifts[ii]);
                        }
                        coarseForm(iv) = 0.5 * (coarseForm(iv) + sum);
                    });
    }

    // looping over fine interface
    for (amrex::MFIter mfi(baFineSummation, li.m_dmFine); mfi.isValid(); ++mfi)
    {
        amrex::Array4<amrex::Real const> const& arrCoarse = coarseData.const_array(mfi);
        fineForm.select_box(li.m_fineIdx[mfi.LocalIndex()]);

        auto shifts = build_summation_shifts(mfi.validbox().length(), idxType, refRatio);

        ParallelFor(mfi.validbox(),
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        amrex::IntVect iv{AMREX_D_DECL(i, j, k)};
                        amrex::IntVect fineIv = iv * refRatio;
                        amrex::Real sum = 0;
                        for (int ii = 0; ii < shifts.m_num; ii++)
                        {
                            sum += fineForm(fineIv + shifts.m_shifts[ii]);
                        }
                        amrex::Real value = 0.5 / shifts.m_num * (sum + arrCoarse(iv));
                        for (int ii = 0; ii < shifts.m_num; ii++)
                        {
                            fineForm(fineIv + shifts.m_shifts[ii]) = value;
                        }
                    });
    }

    // need a second pass over fine in nodal direction to update inbetween fine nodes with
    // (bi-)linear interpolation
    for (amrex::MFIter mfi(baFineInterpolation, li.m_dmFine); mfi.isValid(); ++mfi)
    {
        auto box = mfi.validbox();
// in cell-centered directions with refinement, both dof have to be updated -> refine box before
// parallel for
#if AMREX_SPACEDIM == 3
        auto localRefRatio = amrex::IntVect::TheUnitVector(); // store refinement that has to be
                                                              // applied in parallel for
        for (auto dir : {xDir, yDir, zDir})
        {
            if (idxType[dir] == 0)
            {
                auto tmp = amrex::IntVect::TheUnitVector();
                tmp[dir] = refRatio[dir];
                box.refine(tmp);
            }
            else
            {
                localRefRatio[dir] = refRatio[dir];
            }
        }
#endif
        fineForm.select_box(li.m_fineIdx[mfi.LocalIndex()]);

        auto shifts = build_interpolation_shifts(box.length(), idxType, refRatio);

        if (shifts.m_numTangentialDirs > 0)
        {
            ParallelFor(box,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                            amrex::IntVect iv{AMREX_D_DECL(i, j, k)};
#if AMREX_SPACEDIM == 3
                            amrex::IntVect fineIv = iv * localRefRatio;
#else
                    amrex::IntVect fineIv = iv * refRatio;
#endif

                            for (int ii = 0; ii < shifts.m_numShifts; ii++)
                            {
                                amrex::Real value = 0;
                                auto shift = shifts.m_shifts[ii];

                                for (int c = 0; c < shifts.m_numCornerShifts; c++)
                                {
                                    auto corner = shifts.m_cornerShifts[c];

                                    auto diff = shift - corner;
                                    amrex::Real interpolationWeight =
                                        GEMPIC_D_MULT(1.0 - 0.5 * amrex::Math::abs(diff[xDir]),
                                                      1.0 - 0.5 * amrex::Math::abs(diff[yDir]),
                                                      1.0 - 0.5 * amrex::Math::abs(diff[zDir]));
                                    value += interpolationWeight * fineForm(fineIv + corner);
                                }
                                fineForm(fineIv + shifts.m_shifts[ii]) = value;
                            }
                        });
        }
    }
}

template <typename T>
void do_conforming_projection_adjoint (T& coarseForm, T& fineForm, Impl::LevelInterface const& li)
{
    amrex::IntVect idxType = coarseForm.multi_fab().ixType().toIntVect();
    amrex::IntVect refRatio = li.m_refRatio;

    // create temporary multifabs to parallel copy data of other refinement level
    amrex::MultiFab fineData;   // for coarse side
    amrex::MultiFab coarseData; // for fine side
    amrex::MultiFab fineTmp;

    auto baFineSummation = coarsen_fine_summation(li.m_baFine, refRatio);
    auto baFineInterpolation = coarsen_fine_interpolation(li.m_baFine, refRatio);
    // cannot refine boxArray with single dof in nodal direction. use workaround via boxList
    auto tmp = li.m_baCoarse.boxList();
    tmp.refine(refRatio);
    auto baCoarse = amrex::BoxArray(std::move(tmp));

    int nComp{1};
    int nGhost{0};
    fineData.define(baCoarse, li.m_dmCoarse, nComp, nGhost, amrex::MFInfo());
    coarseData.define(baFineSummation, li.m_dmFine, nComp, nGhost, amrex::MFInfo());
    /// @todo: this tmp multifab is the complete fine level which is excessive
    fineTmp.define(fineForm.multi_fab().boxArray(), fineForm.multi_fab().DistributionMap(), nComp,
                   nGhost, amrex::MFInfo());
    fineTmp.setVal(0.0);

    // overlap copy of coarse data with local computation on fine
    coarseData.ParallelCopy_nowait(coarseForm.multi_fab(),
                                   Gempic::Impl::to_amrex_periodicity(coarseForm.discrete_grid()));

    // looping over fine interface - reverse interpolation
    for (amrex::MFIter mfi(baFineInterpolation, li.m_dmFine); mfi.isValid(); ++mfi)
    {
        auto box = mfi.validbox();
// in cell-centered directions with refinement, both dof have to be updated -> refine box before
// parallel for
#if AMREX_SPACEDIM == 3
        auto localRefRatio = amrex::IntVect::TheUnitVector(); // store refinement that has to be
                                                              // applied in parallel for
        for (auto dir : {xDir, yDir, zDir})
        {
            if (idxType[dir] == 0)
            {
                auto tmp = amrex::IntVect::TheUnitVector();
                tmp[dir] = refRatio[dir];
                box.refine(tmp);
            }
            else
            {
                localRefRatio[dir] = refRatio[dir];
            }
        }
#endif
        amrex::Array4<amrex::Real> const& arrTmp = fineTmp.array(li.m_fineIdx[mfi.LocalIndex()]);
        fineForm.select_box(li.m_fineIdx[mfi.LocalIndex()]);

        auto shifts = build_interpolation_shifts(box.length(), idxType, refRatio);

        int threshold = 0;
        // edges will already be updated by surrounding faces, exclude them by requiring two
        // tangential directions
        // same could be done in primal projection, but there it is just computing values
        // again without harm
        if (AMREX_SPACEDIM == 3 and idxType == amrex::IntVect::TheUnitVector()) threshold = 1;
        if (shifts.m_numTangentialDirs > threshold)
        {
            ParallelFor(
                box,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    amrex::IntVect iv{AMREX_D_DECL(i, j, k)};
#if AMREX_SPACEDIM == 3
                    amrex::IntVect fineIv = iv * localRefRatio;
#else
                    amrex::IntVect fineIv = iv * refRatio;
#endif

                    for (int ii = 0; ii < shifts.m_numShifts; ii++)
                    {
                        auto shift = shifts.m_shifts[ii];

                        for (int c = 0; c < shifts.m_numCornerShifts; c++)
                        {
                            auto corner = shifts.m_cornerShifts[c];

                            auto diff = shift - corner;
                            amrex::Real interpolationWeight =
                                GEMPIC_D_MULT(1.0 - 0.5 * amrex::Math::abs(diff[xDir]),
                                              1.0 - 0.5 * amrex::Math::abs(diff[yDir]),
                                              1.0 - 0.5 * amrex::Math::abs(diff[zDir]));

                            // operations on edges between cells are applied twice, once from each
                            // side on GPU that could happen simultaneously so we halve the weight
                            // and postpone setting the fine nodes to zero
                            if (AMREX_SPACEDIM == 3 and
                                idxType == amrex::IntVect::TheUnitVector() and
                                interpolationWeight > 0)
                            {
                                interpolationWeight = 0.25;
                            }

                            amrex::Gpu::Atomic::Add(&arrTmp(fineIv + corner),
                                                    interpolationWeight * fineForm(fineIv + shift));
                        }

                        if (not(AMREX_SPACEDIM == 3 and idxType == amrex::IntVect::TheUnitVector()))
                        {
                            fineForm(fineIv + shift) = 0;
                        }
                    }
                });
        }
    }
    if (AMREX_SPACEDIM == 3 and idxType == amrex::IntVect::TheUnitVector())
    {
        for (amrex::MFIter mfi(baFineInterpolation, li.m_dmFine); mfi.isValid(); ++mfi)
        {
            auto box = mfi.validbox();
            fineForm.select_box(li.m_fineIdx[mfi.LocalIndex()]);
            auto shifts = build_interpolation_shifts(box.length(), idxType, refRatio);

            if (shifts.m_numTangentialDirs > 0) // only skip vertices, has to be applied to edges
            {
                ParallelFor(box,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k)
                            {
                                amrex::IntVect iv{AMREX_D_DECL(i, j, k)};
                                amrex::IntVect fineIv = iv * refRatio;

                                for (int ii = 0; ii < shifts.m_numShifts; ii++)
                                {
                                    fineForm(fineIv + shifts.m_shifts[ii]) = 0;
                                }
                            });
            }
        }
    }
    fineForm.multi_fab().ParallelCopy(fineTmp,
                                      Gempic::Impl::to_amrex_periodicity(fineForm.discrete_grid()),
                                      amrex::FabArrayBase::ADD);

    fineData.ParallelCopy(fineForm.multi_fab(),
                          Gempic::Impl::to_amrex_periodicity(fineForm.discrete_grid()));

    coarseData.ParallelCopy_finish();

    // looping over coarse interface
    for (amrex::MFIter mfi(li.m_baCoarse, li.m_dmCoarse); mfi.isValid(); ++mfi)
    {
        amrex::Array4<amrex::Real const> const& arrFine = fineData.const_array(mfi);
        coarseForm.select_box(li.m_coarseIdx[mfi.LocalIndex()]);

        auto shifts = build_summation_shifts(mfi.validbox().length(), idxType, refRatio);

        ParallelFor(mfi.validbox(),
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        amrex::IntVect iv{AMREX_D_DECL(i, j, k)};
                        amrex::IntVect fineIv = iv * refRatio;
                        amrex::Real sum = 0;
                        for (int ii = 0; ii < shifts.m_num; ii++)
                        {
                            sum += arrFine(fineIv + shifts.m_shifts[ii]);
                        }
                        coarseForm(iv) = 0.5 * (coarseForm(iv) + sum / shifts.m_num);
                    });
    }

    // looping over fine interface
    for (amrex::MFIter mfi(baFineSummation, li.m_dmFine); mfi.isValid(); ++mfi)
    {
        amrex::Array4<amrex::Real const> const& arrCoarse = coarseData.const_array(mfi);
        fineForm.select_box(li.m_fineIdx[mfi.LocalIndex()]);

        auto shifts = build_summation_shifts(mfi.validbox().length(), idxType, refRatio);

        ParallelFor(mfi.validbox(),
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        amrex::IntVect iv{AMREX_D_DECL(i, j, k)};
                        amrex::IntVect fineIv = iv * refRatio;
                        amrex::Real sum = 0;
                        for (int ii = 0; ii < shifts.m_num; ii++)
                        {
                            sum += fineForm(fineIv + shifts.m_shifts[ii]);
                        }
                        amrex::Real value = 0.5 * (arrCoarse(iv) + sum / shifts.m_num);
                        for (int ii = 0; ii < shifts.m_num; ii++)
                        {
                            fineForm(fineIv + shifts.m_shifts[ii]) = value;
                        }
                    });
    }
}
} // namespace Impl

template <MLFormC T, MLFormC U>
    requires requires(typename T::reference t, typename U::reference u) { hodge(t, u); } &&
             requires(T& t) { Impl::hodge_correction(t); }
void hodge (T& t, U& u)
{
    CHECK_NUMBER_OF_LEVELS(t, u);
    for (int lev = 0; lev < static_cast<int>(t.size()); lev++) hodge(t[lev], u[lev]);
    Impl::hodge_correction(t);
}

template <MLFormC T>
void zero_covered_coarse (T& mlform)
{
    BL_PROFILE("Amr::zero_covered_coarse");
    CHECK_LEVEL_INTERFACES(mlform);
    for (int lev = 0; lev < static_cast<int>(mlform.size()) - 1; ++lev)
    {
        if constexpr (MLVectorFormC<T>)
        {
            for (auto dir : {xDir, yDir, zDir})
            {
                GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE(
                    mlform.m_levelInterfaces.value()[lev][dir].m_coarseCoveredMask != nullptr,
                    "LevelInterface needs coveredCoarseMask for zero_covered_coarse.");
                Impl::do_zero_covered_coarse(mlform[lev][dir], mlform.m_levelInterfaces[lev][dir]);
            }
        }
        else
        {
            GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE(
                mlform.m_levelInterfaces.value()[lev].m_coarseCoveredMask != nullptr,
                "LevelInterface needs coveredCoarseMask for zero_covered_coarse.");
            Impl::do_zero_covered_coarse(mlform[lev], mlform.m_levelInterfaces.value()[lev]);
        }
    }
}

template <MLPrimalFormC T>
void conforming_projection (T& mlform)
{
    BL_PROFILE("Amr::conforming_projection(primal)");
    for (int lev = 0; lev < static_cast<int>(mlform.size()) - 1; ++lev)
    {
        if constexpr (MLVectorFormC<T>)
        {
            for (auto dir : {xDir, yDir, zDir})
            {
                auto const& li = mlform.m_levelInterfaces.value()[lev][dir];
                Impl::do_conforming_projection(mlform[lev][dir], mlform[lev + 1][dir], li);
            }
        }
        else
        {
            auto const& li = mlform.m_levelInterfaces.value()[lev];
            Impl::do_conforming_projection(mlform[lev], mlform[lev + 1], li);
        }
    }
}

template <MLDualFormC T>
void conforming_projection (T& mlform)
{
    BL_PROFILE("Amr::conforming_projection(dual)");
    for (int lev = 0; lev < static_cast<int>(mlform.size()) - 1; ++lev)
    {
        if constexpr (MLVectorFormC<T>)
        {
            for (auto dir : {xDir, yDir, zDir})
            {
                auto const& li = mlform.m_levelInterfaces.value()[lev][dir];
                Impl::do_conforming_projection_adjoint(mlform[lev][dir], mlform[lev + 1][dir], li);
            }
        }
        else
        {
            auto const& li = mlform.m_levelInterfaces.value()[lev];
            Impl::do_conforming_projection_adjoint(mlform[lev], mlform[lev + 1], li);
        }
    }
}

/**************************************************************************************************
 * I/O of Multi Level Fields                                                                      *
 **************************************************************************************************/

template <MLScalarFormC T>
void serialize (T& form, DiscreteTime const& time, std::filesystem::path folderName = "Checkpoints")
{
    amrex::Vector<int> levelStep(form.m_amr->finestLevel() + 1, time.current_step());
    int padLen{6};
    std::string filename =
        // Empty string "" to get appropriate trailing separator
        amrex::Concatenate((folderName / form.m_label / "").string(), time.current_step(), padLen);

    amrex::Vector<amrex::MultiFab const*> tmp;
    for (int lev = 0; lev <= form.m_amr->finestLevel(); lev++)
    {
        tmp.push_back(&form[lev].multi_fab());
    }

    amrex::WriteMultiLevelPlotfile(filename, form.m_amr->finestLevel() + 1, tmp,
                                   amrex::Vector<std::string>{form.m_label}, form.m_amr->Geom(),
                                   time.t(), levelStep, form.m_amr->refRatio());
}

template <MLVectorFormC T>
void serialize (T& form, DiscreteTime const& time, std::filesystem::path folderName = "Checkpoints")
{
    amrex::Vector<int> levelStep(form.m_amr->finestLevel() + 1, time.current_step());
    std::array<std::string, 3> dirNames{"X", "Y", "Z"};

    for (auto dir : {xDir, yDir, zDir})
    {
        std::string filename = amrex::Concatenate(
            (folderName / (form.m_label + dirNames[dir]) / "").string(), time.current_step(), 6);
        amrex::Vector<amrex::MultiFab const*> tmp;
        for (int lev = 0; lev <= form.m_amr->finestLevel(); lev++)
        {
            tmp.push_back(&form[lev].multi_fab(dir));
        }

        amrex::WriteMultiLevelPlotfile(filename, form.m_amr->finestLevel() + 1, tmp,
                                       amrex::Vector<std::string>{form.m_label + dirNames[dir]},
                                       form.m_amr->Geom(), time.t(), levelStep,
                                       form.m_amr->refRatio());
    }
}

template <MLScalarFormC T>
void deserialize (T& form,
                  DiscreteTime const& time,
                  std::filesystem::path folderName = "Checkpoints")
{
    std::string filename =
        amrex::Concatenate((folderName / form.m_label / "").string(), time.current_step(), 6);

    for (int lev = 0; lev <= form.m_amr->finestLevel(); lev++)
    {
        amrex::VisMF::Read(form[lev].multi_fab(),
                           amrex::MultiFabFileFullPrefix(lev, filename, "Level_"));
    }
}

template <MLVectorFormC T>
void deserialize (T& form,
                  DiscreteTime const& time,
                  std::filesystem::path folderName = "Checkpoints")
{
    std::array<std::string, 3> dirNames{"X", "Y", "Z"};

    for (auto dir : {xDir, yDir, zDir})
    {
        std::string filename = amrex::Concatenate(
            (folderName / (form.m_label + dirNames[dir]) / "").string(), time.current_step(), 6);
        for (int lev = 0; lev <= form.m_amr->finestLevel(); lev++)
        {
            amrex::VisMF::Read(form[lev].multi_fab(dir),
                               amrex::MultiFabFileFullPrefix(lev, filename));
        }
    }
}

amrex::Vector<amrex::MultiFab> make_diagnostic_multifab(std::shared_ptr<GempicAmrCore> const& amr,
                                                        int numComp);

void write_plot_file(amrex::Vector<amrex::MultiFab>& diagnostics,
                     std::shared_ptr<GempicAmrCore> const& amr,
                     DiscreteTime const& time,
                     amrex::Vector<std::string> varNames,
                     std::string name = "Plots/plt");

amrex::Real diagnostic_scale(DiscreteField const& field);

inline void interpolate_cell_center (DiscreteField& field, amrex::MultiFab& diagnostic, int idx)
{
    amrex::Real scale = 1.0;
    for (auto dir : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
    {
        if (field.dof_category(dir) == DiscreteField::DOFCategory::CenteredLineIntegral)
        {
            scale *= 1.0 / field.discrete_grid().dx(dir);
        }
    }
    auto grid = field.discrete_grid();
    std::array<amrex::Real, 3> shift{GEMPIC_D_PAD(static_cast<double>(grid.position(xDir)),
                                                  static_cast<double>(grid.position(yDir)),
                                                  static_cast<double>(grid.position(zDir)))};
    amrex::Real ishift{shift[xDir]};
    amrex::Real jshift{shift[yDir]};
    amrex::Real kshift{shift[zDir]};

    for (amrex::MFIter mfi(diagnostic); mfi.isValid(); ++mfi)
    {
        field.select_box(mfi);
        amrex::Array4<amrex::Real> const& res = diagnostic.array(mfi);

        ParallelFor(mfi.validbox(),
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        res(i, j, k, idx) =
                            0.125 * scale *
                            (field(i, j, k) + field(i + ishift, j, k) + field(i, j + jshift, k) +
                             field(i + ishift, j + jshift, k) + field(i, j, k + kshift) +
                             field(i + ishift, j, k + kshift) + field(i, j + jshift, k + kshift) +
                             field(i + ishift, j + jshift, k + kshift));
                    });
    }
}

template <MLVectorFormC T>
void interpolate_cell_center (T& mlfield,
                              amrex::Vector<amrex::MultiFab>& diagnostics,
                              amrex::Vector<std::string>& varNames,
                              int startIdx)
{
    GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE(diagnostics[0].nComp() >= startIdx + 4,
                                      "Diagnostic multifab does not have enough components.");
    varNames.resize(startIdx + 4);
    varNames[startIdx + 0] = mlfield.m_label + "x";
    varNames[startIdx + 1] = mlfield.m_label + "y";
    varNames[startIdx + 2] = mlfield.m_label + "z";
    varNames[startIdx + 3] = mlfield.m_label + "abs";

    for (int level = 0; level < static_cast<int>(mlfield.size()); level++)
    {
        for (auto dir : {xDir, yDir, zDir})
        {
            interpolate_cell_center(mlfield[level][dir], diagnostics[level], startIdx + dir);
        }

        for (amrex::MFIter mfi(diagnostics[level]); mfi.isValid(); ++mfi)
        {
            amrex::Array4<amrex::Real> const& res = diagnostics[level].array(mfi);
            ParallelFor(mfi.validbox(),
                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                            res(i, j, k, startIdx + 3) =
                                std::sqrt(res(i, j, k, startIdx) * res(i, j, k, startIdx) +
                                          res(i, j, k, startIdx + 1) * res(i, j, k, startIdx + 1) +
                                          res(i, j, k, startIdx + 2) * res(i, j, k, startIdx + 2));
                        });
        }
    }
}

template <MLScalarFormC T>
void interpolate_cell_center (T& mlfield,
                              amrex::Vector<amrex::MultiFab>& diagnostics,
                              amrex::Vector<std::string>& varNames,
                              int startIdx)
{
    GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE(diagnostics[0].nComp() >= startIdx + 1,
                                      "Diagnostic multifab does not have enough components.");
    varNames.resize(startIdx + 1);
    varNames[startIdx] = mlfield.m_label;

    for (int level = 0; level < static_cast<int>(mlfield.size()); level++)
    {
        interpolate_cell_center(mlfield[level], diagnostics[level], startIdx);
    }
}

} // namespace Gempic::Amr

#endif
