/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef GEMPIC_FIELDS_H_
#define GEMPIC_FIELDS_H_

#include <memory>
#include <type_traits>
#include <variant>

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>

#include "GEMPIC_BoundaryConditions.H"
#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_DeRhamComplex.H"
#include "GEMPIC_NumTools.H"
#include "GEMPIC_Parameters.H"

namespace Gempic
{

namespace Impl
{

struct DiscreteFieldData
{
    std::string m_label;
    amrex::MultiFab m_data;
    DiscreteGrid m_discreteGrid;
    std::array<int, AMREX_SPACEDIM> m_dofCategory{};
    std::array<int, AMREX_SPACEDIM> m_haloWidth{AMREX_D_DECL(0, 0, 0)};
    amrex::PhysBCFunct<amrex::GpuBndryFuncFab<Forms::BoundaryCondition>> m_boundaryCondition;
};
// Boundary conditions are currently only implemented for number of components nComp==1
// Multiple components are not allowed in an amrex::MultiFab
struct BoundaryConditionConfiguration
{
    Forms::Grid m_grid{};
    int m_extrapolationDegree{};
    // By default we use periodic boundary conditions
    amrex::BCRec m_bcRec{
        AMREX_D_DECL(amrex::BCType::int_dir, amrex::BCType::int_dir, amrex::BCType::int_dir),
        AMREX_D_DECL(amrex::BCType::int_dir, amrex::BCType::int_dir, amrex::BCType::int_dir)};
};
} //namespace Impl

// clang-format off
/**
 * @class DiscreteField
 * @brief The discretized scalar field connects degrees of freedom indexed 
 *        by integers with the domain defined by @ref discrete_grid
 * 
 * @details The underlying datastructure which stores degrees of freedom is an amrex::MultiFab.
 *          Therefore, to actually access degrees of freedom, a box needs to be selected with @ref select_box.
 *          Then data stored in the @ref DiscreteField can be accessed using @ref DiscreteField::operator()(int, int, int, int) const
 *          The following example provides an idea how to use @ref DiscreteField.
 *          Note that the developer of an operation on the field is responsible to ensure the availability of required data.
 * 
 *          @snippet Forms/Fields_test.cpp DiscreteFieldExample.DiscreteDerivative
 * 
 *          List of concepts to be handled by the developer of an operation:
 *          - Ghost cells using @ref apply_boundary_conditions
 */
// clang-format on
class DiscreteField
{
public:
    // clang-format off
    /**
     *  @brief The degree of freedom (DoF) stored in the DiscreteField can be a pointwise or a 1-D
     *         integral centered at the position associated through @ref DiscreteGrid. Currently this is
     *         only required for the DeRhamFields of the finite-difference de Rham complex.
     * @todo In the future, the DOFCategory could be expressed as a type trait that
     *       e.g. contains the integration rule used for each DoF.
     */
    // clang-format on
    enum class DOFCategory
    {
        // clang-format off
        /**
         * @brief Assume \f$x_i\f$ is a location on @ref DiscreteGrid::location_1d
         *        Then a pointwise DoF stored in @ref DiscreteField::operator()(int, int, int, int) const
         *        is given by \f$f(x_i)\f$
         */
        // clang-format on
        PointValue,
        // clang-format off
        /**
         * @brief Assume \f$x_i\f$ is a location on @ref DiscreteGrid::location_1d
         *        and \f$\Delta x\f$ is the width of a cell given by @ref DiscreteGrid::dx.
         *        Then a centered integral DoF stored in @ref DiscreteField::operator()(int, int, int, int) const
         *        is given by \f$\int_{x_i-\Delta x/2}^{x_i+\Delta x/2}f(x) \mathrm{d}x\f$
         */
        // clang-format on
        CenteredLineIntegral
    };

    //!@cond EXCLUDE_API_DOC
    DiscreteField() = delete;
    DiscreteField(std::string const& label,
                  Io::Parameters& params,
                  DiscreteGrid const& discreteGrid,
                  std::array<DOFCategory, AMREX_SPACEDIM> const& dofCategory,
                  Impl::BoundaryConditionConfiguration const& boundaryConditionConfiguration =
                      Impl::BoundaryConditionConfiguration{});
    DiscreteField(std::string const& label,
                  DiscreteGrid const& discreteGrid,
                  std::array<DOFCategory, AMREX_SPACEDIM> const& dofCategory,
                  amrex::BoxArray ba,
                  amrex::DistributionMapping dm,
                  Impl::BoundaryConditionConfiguration const& boundaryConditionConfiguration =
                      Impl::BoundaryConditionConfiguration{});
    //!@endcond

    std::string label() const;
    //! Discretized domain which maps indexable degrees of freedom to a continuous domain
    AMREX_GPU_HOST_DEVICE
    DiscreteGrid const& discrete_grid () const { return m_discreteGrid; }
    //! @cond EXCLUDE_API_DOC
    amrex::MultiFab const& multi_fab() const;
    amrex::MultiFab& multi_fab();
    amrex::Array4<amrex::Real> const& view () const { return m_view; }
    //! @endcond

    //! Access indexable degrees of freedom stored in the @ref DiscreteField
    AMREX_GPU_HOST_DEVICE amrex::Real& operator()(int ix1, int ix2, int ix3) const
    {
        return m_view(ix1, ix2, ix3);
    }
    //! Describes the degree of freedom stored by the @ref DiscreteField
    DOFCategory dof_category (Direction dir) const
    {
        return static_cast<DOFCategory>(m_data->m_dofCategory[dir]);
    };
    // clang-format off
    //! Select a box from an amrex::MultiFab to use in the @ref DiscreteField::operator()(int, int, int, int) const
    // clang-format on
    amrex::Box const& select_box(amrex::MFIter const& mfi);
    amrex::Box const& select_box(int index);
    //! Currently selected box of the amrex::MultiFab.
    amrex::Box const& selected_box() const;
    // clang-format off
    /**
     * @brief Apply boundary conditions to the @p width ghost cells on all patches of the domain covered by the @ref DiscreteField.
     *
     * If the patches have fewer ghost cells than requested by @p width, they are automatically resized to have exactly @p width ghost cells.
     *
     * The boundary defined as follows:
     * - Patches with neighbouring valid domain are filled using values from the corresponding neighbouring patches.
     * - Boundaries outside the valid domain are filled according to the configured @ref gempic::BoundaryCondition.
     *   Depending on the chosen boundary condition, this may also modify degrees of freedom inside the valid domain.
     *
     * @param width The number of ghost-cells of freedom to which the boundary condition should be applied.
     */
    // clang-format on
    void apply_boundary_conditions(std::array<size_t, AMREX_SPACEDIM> width);

private:
    //!@cond EXCLUDE_API_DOC
    DiscreteField(std::shared_ptr<Impl::DiscreteFieldData> const& data);
    //! @endcond
    std::shared_ptr<Impl::DiscreteFieldData> m_data;
    // GPU accessible members
    DiscreteGrid m_discreteGrid;
    amrex::Array4<amrex::Real> m_view;
    int m_selectedBoxIdx;
};

namespace Impl
{
enum class GhostRegion
{
    low,
    up
};
amrex::Box selected_ghost_box(DiscreteField const& df,
                              std::array<size_t, AMREX_SPACEDIM> const& ghostSize,
                              Direction const& dir,
                              Impl::GhostRegion const region);
} //namespace Impl

// clang-format off
/**
 * @class DiscreteVectorField
 * @brief The discretized vector field connects degrees of freedom of three directions indexed by integers with the domain defined by @ref discrete_grid.
 *        The indexing in different directions may differ.
 * 
 * @details The underlying datastructure which stores degrees of freedom is an amrex::MultiFab.
 *          Therefore, to actually access degrees of freedoms, a box needs to be selected with @ref select_box.
 *          Then data stored in the @ref DiscreteVectorField can be accessed using @ref DiscreteVectorField::operator()(Direction, int, int, int, int) const
 *          The following example provides an idea how to use @ref DiscreteVectorField.
 *          Note that the developer of an operation on the vector field is responsible to ensure the availability of required data.
 * 
 *          @snippet Forms/Fields_test.cpp DiscreteVectorFieldExample.DiscreteDivergence
 * 
 *          List of concepts to be handled by the developer of an operation:
 *          - Ghost cells using @ref apply_boundary_conditions
 */
// clang-format on
class DiscreteVectorField
{
public:
    //!@cond EXCLUDE_API_DOC
    DiscreteVectorField() = delete;
    DiscreteVectorField(
        std::string const& label,
        Io::Parameters& params,
        std::array<DiscreteGrid, 3> const& discreteGrid,
        std::array<std::array<DiscreteField::DOFCategory, AMREX_SPACEDIM>, 3> const& dofCategory,
        std::array<Impl::BoundaryConditionConfiguration, 3> const& boundaryConditionConfiguration =
            std::array<Impl::BoundaryConditionConfiguration, 3>{});
    DiscreteVectorField(
        std::string const& label,
        std::array<DiscreteGrid, 3> const discreteGrid,
        std::array<std::array<DiscreteField::DOFCategory, AMREX_SPACEDIM>, 3> const& dofCategory,
        amrex::BoxArray ba,
        amrex::DistributionMapping dm,
        std::array<Impl::BoundaryConditionConfiguration, 3> const& boundaryConditionConfiguration =
            std::array<Impl::BoundaryConditionConfiguration, 3>{});
    //!@endcond

    std::string label () const { return *m_label; }
    //clang-format off
    /**
     * @brief Discretized domain which maps indexable degrees of freedom to a continuous domain
     * @param dir @ref DiscreteGrid of the amrex::MultiFab in the corresponding direction
     */
    //clang-format on
    AMREX_GPU_HOST_DEVICE
    DiscreteGrid const& discrete_grid (Direction dir) const
    {
        return m_fields[dir].discrete_grid();
    };
    //! @cond EXCLUDE_API_DOC
    amrex::MultiFab const& multi_fab(Direction dir) const;
    amrex::MultiFab& multi_fab(Direction dir);
    amrex::Array4<amrex::Real> const& view (Direction dir) const { return m_fields[dir].view(); }
    //! @endcond
    DiscreteField& operator[](Direction dir) { return m_fields[dir]; };

    //! Access indexable degrees of freedom stored in the @ref DiscreteVectorField
    AMREX_GPU_HOST_DEVICE amrex::Real& operator()(Direction dir, int ix1, int ix2, int ix3) const
    {
        return m_fields[dir](ix1, ix2, ix3);
    }
    DiscreteField::DOFCategory dof_category (Direction field, Direction grid) const
    {
        return m_fields[field].dof_category(grid);
    };

    // clang-format off
    //! Select a box for all amrex::MultiFab in all directions to use in the @ref DiscreteVectorField::operator()(Direction, int, int, int, int) const
    // clang-format on
    std::array<amrex::Box, 3> const select_box(amrex::MFIter const& mfi);
    std::array<amrex::Box, 3> const select_box(int index);
    //! Currently selected boxes of amrex::MultiFabs in given direction.
    amrex::Box const& selected_box(Direction dir) const;
    // clang-format off
    /**
     * @brief Apply boundary conditions to the @p width ghost cells on all patches of the domain covered by and all directions of the @ref DiscreteVectorField.
     *
     * If the patches have fewer ghost cells than requested by @p width, they are automatically resized to have exactly @p width ghost cells.
     *
     * The boundary defined as follows:
     * - Patches with neighbouring valid domain are filled using values from the corresponding neighbouring patches.
     * - Boundaries outside the valid domain are filled according to the configured @ref gempic::BoundaryCondition.
     *   Depending on the chosen boundary condition, this may also modify degrees of freedom inside the valid domain.
     *
     * @param width The number of ghost-cells of freedom to which the boundary condition should be applied.
     */
    // clang-format on
    void apply_boundary_conditions(std::array<size_t, AMREX_SPACEDIM> width);

private:
    // std::string is not GPU capable.
    // For now wrap into a shared pointer to allow copying the data to GPU
    std::shared_ptr<std::string> m_label{};
    std::array<DiscreteField, 3> m_fields;
};

void operator*=(DiscreteVectorField&, amrex::Real const&);
void operator+=(DiscreteVectorField&, DiscreteVectorField const&);

template <typename F>
concept ScalarAnalyticalExpressionSpace =
    std::is_invocable_r_v<amrex::Real, F, AMREX_D_DECL(amrex::Real, amrex::Real, amrex::Real)>;
template <typename F>
concept ScalarAnalyticalExpressionSpaceTime =
    std::is_invocable_r_v<amrex::Real,
                          F,
                          AMREX_D_DECL(amrex::Real, amrex::Real, amrex::Real),
                          amrex::Real>;
template <typename F>
concept VectorAnalyticalExpressionSpace =
    std::is_invocable_r_v<amrex::Real,
                          F,
                          Direction,
                          AMREX_D_DECL(amrex::Real, amrex::Real, amrex::Real)>;
template <typename F>
concept VectorAnalyticalExpressionSpaceTime =
    std::is_invocable_r_v<amrex::Real,
                          F,
                          Direction,
                          AMREX_D_DECL(amrex::Real, amrex::Real, amrex::Real),
                          amrex::Real>;
/**
 * @brief Fill a @ref DiscreteField with values from @p f at the location of
 *        each cell (or grid point) on the computational domain.
 *
 *        Assuming `AMREX_SPACEDIM=3D` the following code is equivalent to
 *        \f[
 *        f(x,y,z) = sin(k_{0,x} x)sin(k_{0,y} y)sin(k_{0,z} z) \quad \text{with} \quad k_{0,i} =
 * \frac{2\pi}{L_i} \f] where \f$f(x,y,z)\f$ corresponds to a @ref DiscreteField
 *        @snippet Forms/Fields_test.cpp DiscreteFieldExample.Fill
 *
 * @tparam F A callable type or function with a signature compatible with the dimensionality
 *         of @p field , such as:
 *         @code{.cpp}
 *         amrex::Real(AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z));
 *         @endcode
 */
template <ScalarAnalyticalExpressionSpace F>
void fill (DiscreteField& field, F const& f)
{
    for (amrex::MFIter mfi{field.multi_fab()}; mfi.isValid(); ++mfi)
    {
        field.select_box(mfi);
        DiscreteGrid const& dg = field.discrete_grid();
        ParallelFor(mfi.validbox(),
                    [=] AMREX_GPU_DEVICE(int ix, int iy, int iz)
                    {
                        amrex::Real x = dg.location_1d(xDir, ix);
#if AMREX_SPACEDIM > 1
                        amrex::Real y = dg.location_1d(yDir, iy);
#endif
#if AMREX_SPACEDIM == 3
                        amrex::Real z = dg.location_1d(zDir, iz);
#endif
                        field(ix, iy, iz) = f(AMREX_D_DECL(x, y, z));
                    });
    }
}
template <ScalarAnalyticalExpressionSpaceTime F>
void fill (DiscreteField& field, F const& f, double time)
{
    auto timeIndependent = [=] AMREX_GPU_HOST_DEVICE(AMREX_D_DECL(amrex::Real x, amrex::Real y,
                                                                  amrex::Real z)) -> amrex::Real
    { return f(AMREX_D_DECL(x, y, z), time); };
    fill(field, timeIndependent);
}
void fill_zero(DiscreteField&);

// clang-format off
/**
 * @brief Fill a @ref DiscreteVectorField with values from @p f at the
 *        location of each cell (or grid point) in the computational domain for 
 *        every direction @ref Direction of the vector.
 *
 *        Assuming `AMREX_SPACEDIM=3D` the following code is equivalent to
 *        \f[
 *        \vec f(x,y,z) = 
 *        \begin{pmatrix}
 *          sin(k_{0,x} x)sin(k_{0,y} y)sin(k_{0,z} z)\\
 *          cos(k_{0,x} x)cos(k_{0,y} y)sin(k_{0,z} z)\\
 *          sin(k_{0,x} x)cos(k_{0,y} y)sin(k_{0,z} z)\\
 *        \end{pmatrix}\quad
 *        \text{with}\quad k_{0,i} = \frac{2\pi}{L_i}
 *        \f]
 *        where \f$\vec f(x,y,z)\f$ corresponds to a @ref DiscreteVectorField
 *        @snippet Forms/Fields_test.cpp DiscreteVectorFieldExample.Fill
 *
 * @tparam F A callable type or function with a signature compatible with the dimensionality of
 *         @p field , such as:
 *         @code{.cpp}
 *         amrex::Real(Direction dir, AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z));
 *         @endcode
 */
// clang-format on
template <VectorAnalyticalExpressionSpace F>
void fill (DiscreteVectorField& field, F const& f)
{
    for (Direction dir : {xDir, yDir, zDir})
    {
        auto fdir = [=] AMREX_GPU_HOST_DEVICE(
                        AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z)) -> amrex::Real
        { return f(dir, AMREX_D_DECL(x, y, z)); };
        fill(field[dir], fdir);
    }
}
template <VectorAnalyticalExpressionSpaceTime F>
void fill (DiscreteVectorField& field, F const& f, double time)
{
    auto timeIndependent =
        [=] AMREX_GPU_HOST_DEVICE(
            Direction dir, AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z)) -> amrex::Real
    { return f(dir, AMREX_D_DECL(x, y, z), time); };
    fill(field, timeIndependent);
}
void fill_zero(DiscreteVectorField&);

/**
 * @brief Provide an interface for functions parsed from the input file to
 *        @ref fill(DiscreteField&, F const&)
 */
class DiscreteFieldFunctionParser
{
public:
    //!@cond EXCLUDE_API_DOC
    DiscreteFieldFunctionParser() = delete;
    //!@endcond
    /**
     * @brief Create a function parser to fill a @ref DiscreteField
     *
     * @param label The identifier of a function in the parameter file parsed through
     *              `Io::Parameters`
     */
    DiscreteFieldFunctionParser(std::string const& label, Io::Parameters& params);
    /**
     * @brief Provide an interface for functions parsed from the input file to
     *        @ref fill(DiscreteField&, F const&)
     */
    AMREX_GPU_HOST_DEVICE
    amrex::Real operator()(AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z),
                           amrex::Real t) const
    {
        return m_function(AMREX_D_DECL(x, y, z), t);
        ;
    }

private:
    amrex::ParserExecutor<AMREX_SPACEDIM + 1> m_function{};
    amrex::Parser m_parser{};
};

/**
 * @brief Provide an interface for functions parsed from the input file to
 *        @ref fill(DiscreteVectorField&, F const&)
 */
class DiscreteVectorFieldFunctionParser
{
public:
    //!@cond EXCLUDE_API_DOC
    DiscreteVectorFieldFunctionParser() = delete;
    //!@endcond
    /**
     * @brief Create a function parser to fill a @ref DiscreteVectorField
     *
     * @param label The identifier of three functions in the parameter file parsed through
     *              `Io::Parameters` where the position of the string in the array
     *              corresponds to @ref Direction of the @ref DiscreteVectorField
     */
    DiscreteVectorFieldFunctionParser(std::array<std::string, 3> const& label,
                                      Io::Parameters& params);
    /**
     * @brief Provide an interface for functions parsed from the input file to
     *        @ref fill(DiscreteVectorField&, F const&)
     */
    AMREX_GPU_HOST_DEVICE
    amrex::Real operator()(Direction dir,
                           AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z),
                           amrex::Real t) const
    {
        return m_function[dir](AMREX_D_DECL(x, y, z), t);
    }

private:
    std::array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3> m_function{};
    std::array<amrex::Parser, 3> m_parser{};
};

bool is_nan(DiscreteField&);
amrex::Real l_inf_error(DiscreteField& a, DiscreteField& b);
std::array<bool, 3> is_nan(DiscreteVectorField&);
std::array<amrex::Real, 3> l_inf_error(DiscreteVectorField&, DiscreteVectorField&);

namespace Impl
{
amrex::Real dot(DiscreteField& a, DiscreteField& b);
amrex::Real dot(DiscreteVectorField& a, DiscreteVectorField& b);
} // namespace Impl
} //namespace Gempic

namespace Gempic::Forms
{
// forward definition
class DeRhamComplex;

template <Grid grid>
constexpr Grid get_opposite_grid ()
{
    if constexpr (grid == Grid::primal)
    {
        return Grid::dual;
    }
    else if constexpr (grid == Grid::dual)
    {
        return Grid::primal;
    }
}

template <Space space>
constexpr Space get_opposite_space ()
{
    if constexpr (space == Space::node)
    {
        return Space::cell;
    }
    else if constexpr (space == Space::cell)
    {
        return Space::node;
    }
    else if constexpr (space == Space::face)
    {
        return Space::edge;
    }
    else if constexpr (space == Space::edge)
    {
        return Space::face;
    }
}

template <Grid grid>
constexpr Grid oppositeGrid{get_opposite_grid<grid>()};

template <Space space>
constexpr Space oppositeSpace{get_opposite_space<space>()};

// Primary template (disabled)
template <Grid grid, Space space>
struct DeRhamField
{
    static_assert(gempic_always_false<grid>(), "DeRhamField not implemented for this grid");
    static_assert(gempic_always_false<space>(), "DeRhamField not implemented for this space");
};

// One- or Two-form, i.e., space is either edge or face
template <Grid grid, Space space>
    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
struct DeRhamField<grid, space> : DeRhamFormBase<to_sequence<grid>(), to_form<space>()>
{
    static constexpr Grid s_grid = grid;
    static constexpr Space s_space = space;
    std::array<amrex::Real, 3> m_scaling;

    std::shared_ptr<DeRhamComplex> m_deRham;
    DeRhamComplex const& get_de_rham() const;
    amrex::MultiFab const& multi_fab(Direction dir) const;
    amrex::MultiFab& multi_fab(Direction dir);
    amrex::Array<amrex::MultiFab, 3> m_data;
    std::unique_ptr<amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BoundaryCondition>>> m_bcFuncX;
    std::unique_ptr<amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BoundaryCondition>>> m_bcFuncY;
    std::unique_ptr<amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BoundaryCondition>>> m_bcFuncZ;

    // avoid implicit conversions like DeRhamFieldBase<Grid::primal, Space::node> field = drc;
    explicit DeRhamField(std::shared_ptr<DeRhamComplex> drc, int nVar = 1);
    DeRhamField(std::shared_ptr<DeRhamComplex> drc, std::array<int, 3> ncomponents);
    DeRhamField(std::shared_ptr<DeRhamComplex> drc,
                amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>& func);
    DeRhamField(
        std::shared_ptr<DeRhamComplex> drc,
        amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>> const& funcs);
    DeRhamField(std::shared_ptr<DeRhamComplex> drc, DeRhamField<grid, space> const& src);
    DeRhamField(std::shared_ptr<DeRhamComplex> drc,
                amrex::MultiFab const& mfIn1,
                amrex::MultiFab const& mfIn2,
                amrex::MultiFab const& mfIn3);

    void fill_boundary();
    void fill_boundary_nowait();
    void fill_boundary_finish();
    void apply_bc();
    int get_ncomp(int dir) const;
    int get_hodge_degree() const;
    int get_ncells(int dir) const;
    amrex::IntVect get_ncells() const;
    amrex::Geometry const get_geom() const;
    amrex::Real get_scaling(Direction idir) const;

    void post_particle_loop_sync();
    void operator*=(amrex::Real dt);
    void operator*=(amrex::ParserExecutor<AMREX_SPACEDIM + 1> func);
    void operator+=(DeRhamField<grid, space>&);
    void operator-=(DeRhamField<grid, space>&);
    DeRhamField<grid, space> operator-(DeRhamField<grid, space>&);
    void operator+=(amrex::Real c);
    void operator+=(amrex::Vector<amrex::Real> c);
    void operator-=(amrex::Real c);
};

// Zero- or Three-form, i.e., space is either node or cell
template <Grid grid, Space space>
    requires(is_zero_or_three_form<space>()) // Ensures that space is a zero- or three-form
struct DeRhamField<grid, space> : DeRhamFormBase<to_sequence<grid>(), to_form<space>()>
{
    static constexpr Grid s_grid = grid;
    static constexpr Space s_space = space;
    amrex::Real m_scaling;
    int m_hodgeDegree; // if we want it runtime

    std::shared_ptr<DeRhamComplex> m_deRham;
    DeRhamComplex const& get_de_rham() const;

    amrex::MultiFab const& multi_fab() const;
    amrex::MultiFab& multi_fab();
    amrex::MultiFab m_data;
    std::unique_ptr<amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BoundaryCondition>>> m_bcFunc;

    DeRhamField(std::shared_ptr<DeRhamComplex> drc, int nVar = 1);
    DeRhamField(std::shared_ptr<DeRhamComplex> drc,
                amrex::ParserExecutor<AMREX_SPACEDIM + 1>& func);
    DeRhamField(std::shared_ptr<DeRhamComplex> drc,
                amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>> const& funcs);
    DeRhamField(std::shared_ptr<DeRhamComplex> drc, DeRhamField<grid, space> const& src);
    DeRhamField(std::shared_ptr<DeRhamComplex> drc, amrex::MultiFab const& mfIn);

    void fill_boundary();
    void fill_boundary_nowait();
    void fill_boundary_finish();
    void apply_bc();
    int get_ncomp() const;
    int get_hodge_degree() const;
    int get_ncells(int dir) const;
    amrex::IntVect get_ncells() const;
    amrex::Geometry const get_geom() const;
    amrex::Real get_scaling() const;

    void post_particle_loop_sync();
    void operator*=(amrex::Real dt);
    void operator*=(amrex::ParserExecutor<AMREX_SPACEDIM + 1> func);
    void operator+=(DeRhamField<grid, space>&);
    void operator-=(DeRhamField<grid, space>&);
    DeRhamField<grid, space> operator-(DeRhamField<grid, space>&);
    void operator+=(amrex::Real c);
    void operator-=(amrex::Real c);
};

/**
 * @brief helper, getting the multifab from the field type FT
 * @tparam FT field type
 * @param pf pointer to the field
 * @param dir direction (for vector fields)
 * @return multifab
 */
template <class FT>
GEMPIC_FORCE_INLINE static amrex::MultiFab const& multi_fab (FT const* pf, Direction dir)
{
    if constexpr (requires(FT const& f, Direction d) { f.multi_fab(d); })
    {
        return pf->multi_fab(dir); // vector form (1-/2-form)
    }
    else
    {
        return pf->multi_fab(); // scalar form (0-/3-form)
    }
}

} //namespace Gempic::Forms

namespace Gempic::Forms
{
/**
 * @brief Base class for discrete de Rham Complexes. The member functions with the `inline`
 * specifier implement the methods of the elements of the Complex (the functions of the fields). The
 * projections are virtual functions, whose details are implemented in the inherited subclasses.
 *
 */
class DeRhamComplex
{
private:
    // Indices
    amrex::IndexType m_primalNodeIndex;
    amrex::Array<amrex::IndexType, 3> m_primalEdgeIndex;
    amrex::Array<amrex::IndexType, 3> m_primalFaceIndex;
    amrex::IndexType m_primalCellIndex;

protected:
    static constexpr std::array<int, 0> s_validDegreesNoScheme{};
    // Computational grid
    amrex::BoxArray m_grid;
    amrex::DistributionMapping m_distriMap;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_dx;
    int m_hodgeDegree;

    // Number of ghost cells along each direction
    amrex::IntVect m_nGhost;
    amrex::IntVect m_nGhostRequired;

public:
    // ToDo: Temporary moved to be public until ComputationalDomain is removed
    amrex::Geometry m_geom;
    /// Constructor for the DeRhamComplex class
    /// @param infra The computational domain needed to construct the class correctly
    /// @param hodgeDegree The degree of the Hodge scheme, used for a lower limit of ghost cells
    /// @param maxSplineDegree The degree of the splines, used for a lower limit of ghost cells
    DeRhamComplex(ComputationalDomain const& infra,
                  int const hodgeDegree,
                  int const maxSplineDegree);

    virtual HodgeScheme get_hodge_scheme() const = 0;
    virtual amrex::MultiFab& get_temp_primal_zero_form() = 0;
    virtual amrex::MultiFab& get_temp_dual_zero_form() = 0;
    virtual amrex::Array<amrex::MultiFab, 3>& get_temp_primal_one_form() = 0;
    virtual amrex::Array<amrex::MultiFab, 3>& get_temp_dual_one_form() = 0;
    virtual amrex::Array<amrex::MultiFab, 3>& get_temp_primal_two_form() = 0;
    virtual amrex::Array<amrex::MultiFab, 3>& get_temp_dual_two_form() = 0;
    virtual amrex::MultiFab& get_temp_primal_three_form() = 0;
    virtual amrex::MultiFab& get_temp_dual_three_form() = 0;

    virtual ~DeRhamComplex() {}

    using Primal0Form = DeRhamField<Grid::primal, Space::node>;
    using Primal1Form = DeRhamField<Grid::primal, Space::edge>;
    using Primal2Form = DeRhamField<Grid::primal, Space::face>;
    using Primal3Form = DeRhamField<Grid::primal, Space::cell>;
    using Dual0Form = DeRhamField<Grid::dual, Space::node>;
    using Dual1Form = DeRhamField<Grid::dual, Space::edge>;
    using Dual2Form = DeRhamField<Grid::dual, Space::face>;
    using Dual3Form = DeRhamField<Grid::dual, Space::cell>;

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> get_dx () const { return m_dx; };
    /// Getter for hodge degree
    int get_hodge_degree () const { return m_hodgeDegree; }

    /// Getter for nCells
    amrex::IntVect get_ncells () const { return m_geom.Domain().length(); }
    int get_ncells (int dir) const { return get_ncells()[dir]; }

    /// Getter for the number of ghost cells
    amrex::IntVect const& get_n_ghost () const { return m_nGhost; }

    /// Getter for the periodicity
    amrex::Periodicity const get_periodicity () const { return m_geom.periodicity(); }

    /// Getter for prob lo
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const get_prob_lo () const
    {
        return m_geom.ProbLoArray();
    }

    /// Getter for geom
    amrex::Geometry const get_geom () const { return m_geom; }

    amrex::DistributionMapping const& get_distribution_mapping () const { return m_distriMap; }

    /// @brief Initializes fields and ties them to the grid.
    ///        Called from the DeRhamField constructors.
    /// @param field The field to be initialized
    /// @param nVar  The number of variables of the field
    ///              (times one component for 0-/3-forms, times three components for 1-/2-forms)
    void initialize_field (DeRhamField<Grid::primal, Space::node>& field, int nVar = 1)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex::initialize_field(primal,node,field)");
        field.m_data.define(amrex::convert(m_grid, m_primalNodeIndex), m_distriMap, nVar, m_nGhost);
        field.m_data.setVal(0.0);
    }

    // Function overloads for the initializations
    void initialize_field (DeRhamField<Grid::primal, Space::edge>& field, int nVar = 1)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex::initialize_field(primal,edge,field)");
        field.m_data[xDir].define(amrex::convert(m_grid, m_primalEdgeIndex[xDir]), m_distriMap,
                                  nVar, m_nGhost);
        field.m_data[xDir].setVal(0.0);
        field.m_data[yDir].define(amrex::convert(m_grid, m_primalEdgeIndex[yDir]), m_distriMap,
                                  nVar, m_nGhost);
        field.m_data[yDir].setVal(0.0);
        field.m_data[zDir].define(amrex::convert(m_grid, m_primalEdgeIndex[zDir]), m_distriMap,
                                  nVar, m_nGhost);
        field.m_data[zDir].setVal(0.0);
    }

    void initialize_field (DeRhamField<Grid::primal, Space::face>& field, int nVar = 1)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex::initialize_field(primal,face,field)");
        field.m_data[xDir].define(amrex::convert(m_grid, m_primalFaceIndex[xDir]), m_distriMap,
                                  nVar, m_nGhost);
        field.m_data[xDir].setVal(0.0);
        field.m_data[yDir].define(amrex::convert(m_grid, m_primalFaceIndex[yDir]), m_distriMap,
                                  nVar, m_nGhost);
        field.m_data[yDir].setVal(0.0);
        field.m_data[zDir].define(amrex::convert(m_grid, m_primalFaceIndex[zDir]), m_distriMap,
                                  nVar, m_nGhost);
        field.m_data[zDir].setVal(0.0);
    }

    void initialize_field (DeRhamField<Grid::primal, Space::edge>& field,
                           std::array<int, 3> ncomponents)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex::initialize_field(primal,edge,field)");
        field.m_data[xDir].define(amrex::convert(m_grid, m_primalEdgeIndex[xDir]), m_distriMap,
                                  ncomponents[xDir], m_nGhost);
        field.m_data[xDir].setVal(0.0);
        field.m_data[yDir].define(amrex::convert(m_grid, m_primalEdgeIndex[yDir]), m_distriMap,
                                  ncomponents[yDir], m_nGhost);
        field.m_data[yDir].setVal(0.0);
        field.m_data[zDir].define(amrex::convert(m_grid, m_primalEdgeIndex[zDir]), m_distriMap,
                                  ncomponents[zDir], m_nGhost);
        field.m_data[zDir].setVal(0.0);
    }

    void initialize_field (DeRhamField<Grid::primal, Space::face>& field,
                           std::array<int, 3> ncomponents)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex::initialize_field(primal,face,field)");
        field.m_data[xDir].define(amrex::convert(m_grid, m_primalFaceIndex[xDir]), m_distriMap,
                                  ncomponents[xDir], m_nGhost);
        field.m_data[xDir].setVal(0.0);
        field.m_data[yDir].define(amrex::convert(m_grid, m_primalFaceIndex[yDir]), m_distriMap,
                                  ncomponents[yDir], m_nGhost);
        field.m_data[yDir].setVal(0.0);
        field.m_data[zDir].define(amrex::convert(m_grid, m_primalFaceIndex[zDir]), m_distriMap,
                                  ncomponents[zDir], m_nGhost);
        field.m_data[zDir].setVal(0.0);
    }

    void initialize_field (DeRhamField<Grid::primal, Space::cell>& field, int nVar = 1)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex::initialize_field(primal,cell,field)");
        field.m_data.define(amrex::convert(m_grid, m_primalCellIndex), m_distriMap, nVar, m_nGhost);
        field.m_data.setVal(0.0);
    }

    // Centering is always set by the primal grid
    void initialize_field (DeRhamField<Grid::dual, Space::node>& field, int nVar = 1)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex::initialize_field(dual,node,field)");
        field.m_data.define(amrex::convert(m_grid, m_primalCellIndex), m_distriMap, nVar, m_nGhost);
        field.m_data.setVal(0.0);
    }

    void initialize_field (DeRhamField<Grid::dual, Space::edge>& field, int nVar = 1)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex::initialize_field(dual,edge,field)");
        field.m_data[xDir].define(amrex::convert(m_grid, m_primalFaceIndex[xDir]), m_distriMap,
                                  nVar, m_nGhost);
        field.m_data[xDir].setVal(0.0);
        field.m_data[yDir].define(amrex::convert(m_grid, m_primalFaceIndex[yDir]), m_distriMap,
                                  nVar, m_nGhost);
        field.m_data[yDir].setVal(0.0);
        field.m_data[zDir].define(amrex::convert(m_grid, m_primalFaceIndex[zDir]), m_distriMap,
                                  nVar, m_nGhost);
        field.m_data[zDir].setVal(0.0);
    }

    void initialize_field (DeRhamField<Grid::dual, Space::face>& field, int nVar = 1)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex::initialize_field(dual,face,field)");
        field.m_data[xDir].define(amrex::convert(m_grid, m_primalEdgeIndex[xDir]), m_distriMap,
                                  nVar, m_nGhost);
        field.m_data[xDir].setVal(0.0);
        field.m_data[yDir].define(amrex::convert(m_grid, m_primalEdgeIndex[yDir]), m_distriMap,
                                  nVar, m_nGhost);
        field.m_data[yDir].setVal(0.0);
        field.m_data[zDir].define(amrex::convert(m_grid, m_primalEdgeIndex[zDir]), m_distriMap,
                                  nVar, m_nGhost);
        field.m_data[zDir].setVal(0.0);
    }

    void initialize_field (DeRhamField<Grid::dual, Space::edge>& field,
                           std::array<int, 3> ncomponents)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex::initialize_field(dual,edge,field)");
        field.m_data[xDir].define(amrex::convert(m_grid, m_primalFaceIndex[xDir]), m_distriMap,
                                  ncomponents[xDir], m_nGhost);
        field.m_data[xDir].setVal(0.0);
        field.m_data[yDir].define(amrex::convert(m_grid, m_primalFaceIndex[yDir]), m_distriMap,
                                  ncomponents[yDir], m_nGhost);
        field.m_data[yDir].setVal(0.0);
        field.m_data[zDir].define(amrex::convert(m_grid, m_primalFaceIndex[zDir]), m_distriMap,
                                  ncomponents[zDir], m_nGhost);
        field.m_data[zDir].setVal(0.0);
    }

    void initialize_field (DeRhamField<Grid::dual, Space::face>& field,
                           std::array<int, 3> ncomponents)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex::initialize_field(dual,face,field)");
        field.m_data[xDir].define(amrex::convert(m_grid, m_primalEdgeIndex[xDir]), m_distriMap,
                                  ncomponents[xDir], m_nGhost);
        field.m_data[xDir].setVal(0.0);
        field.m_data[yDir].define(amrex::convert(m_grid, m_primalEdgeIndex[yDir]), m_distriMap,
                                  ncomponents[yDir], m_nGhost);
        field.m_data[yDir].setVal(0.0);
        field.m_data[zDir].define(amrex::convert(m_grid, m_primalEdgeIndex[zDir]), m_distriMap,
                                  ncomponents[zDir], m_nGhost);
        field.m_data[zDir].setVal(0.0);
    }

    void initialize_field (DeRhamField<Grid::dual, Space::cell>& field, int nVar = 1)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex::initialize_field(dual,cell,field)");
        field.m_data.define(amrex::convert(m_grid, m_primalNodeIndex), m_distriMap, nVar, m_nGhost);
        field.m_data.setVal(0.0);
    }

    /// Function overloads for FillBoundary. Exists for 1-/2-forms and 0-/3-forms
    // fillBoundary for One- or Two-forms
    template <Grid grid, Space space>
        requires OneOrTwoFormSpaceC<space>
    void fill_boundary (DeRhamField<grid, space>& field)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex<isOneOrTwoForm>::fill_boundary(field)");
        field.m_data[xDir].FillBoundary(m_geom.periodicity());
        field.m_data[yDir].FillBoundary(m_geom.periodicity());
        field.m_data[zDir].FillBoundary(m_geom.periodicity());
    }

    /// @cond EXCLUDE_API_DOC
    // fillBoundary for Zero- or Three-forms
    template <Grid grid, Space space>
        requires ZeroOrThreeFormSpaceC<space>
    void fill_boundary (DeRhamField<grid, space>& field)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex<isZeroOrThreeForm>::fill_boundary(field)");
        field.m_data.FillBoundary(m_geom.periodicity());
    }
    /// @endcond

    /// Function overloads for FillBoundary_nowait. Exists for 1-/2-forms and 0-/3-forms
    // fillBoundary_nowait for One- or Two-forms
    template <Grid grid, Space space>
        requires OneOrTwoFormSpaceC<space>
    void fill_boundary_nowait (DeRhamField<grid, space>& field)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex<isOneOrTwoForm>::fill_boundary_nowait(field)");
        field.m_data[xDir].FillBoundary_nowait(m_geom.periodicity());
        field.m_data[yDir].FillBoundary_nowait(m_geom.periodicity());
        field.m_data[zDir].FillBoundary_nowait(m_geom.periodicity());
    }

    /// @cond EXCLUDE_API_DOC
    // fillBoundary_nowait for Zero- or Three-forms
    template <Grid grid, Space space>
        requires ZeroOrThreeFormSpaceC<space>
    void fill_boundary_nowait (DeRhamField<grid, space>& field)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex<isZeroOrThreeForm>::fill_boundary_nowait(field)");
        field.m_data.FillBoundary_nowait(m_geom.periodicity());
    }
    /// @endcond
    /// Function overloads for FillBoundary_nowait. Exists for 1-/2-forms and 0-/3-forms
    // fillBoundary_nowait for One- or Two-forms
    template <Grid grid, Space space>
        requires OneOrTwoFormSpaceC<space>
    void fill_boundary_finish (DeRhamField<grid, space>& field)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex<isOneOrTwoForm>::fill_boundary_finish()");
        field.m_data[xDir].FillBoundary_finish();
        field.m_data[yDir].FillBoundary_finish();
        field.m_data[zDir].FillBoundary_finish();
    }

    /// @cond EXCLUDE_API_DOC
    // fillBoundary_nowait for Zero- or Three-forms
    template <Grid grid, Space space>
        requires ZeroOrThreeFormSpaceC<space>
    void fill_boundary_finish (DeRhamField<grid, space>& field)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex<isZeroOrThreeForm>::fill_boundary_finish()");
        field.m_data.FillBoundary_finish();
    }
    /// @endcond

    /**
     * @brief Apply boundary conditions and extrapolate to ghost cells. Exists for 1-/2-forms and
     * 0-/3-forms.
     *
     * @details
     * If at least one dimension is non-periodic, primal degrees of freedom on the corresponding
     * boundaries are set to the condition specified in the input file. A polynomial extrapolation
     * of order (Hodge degree -1) is then performed to fill ghost cells. For dual forms, only the
     * extrapolation is applied.
     *
     * @param field Discrete form
     */
    template <Grid grid, Space space>
        requires OneOrTwoFormSpaceC<space>
    void apply_bc (DeRhamField<grid, space>& field)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex<isOneOrTwoForm>::apply_bc(field)");
        if (not m_geom.isAllPeriodic())
        {
            // call bcFunc with MultiFab& mf, int icomp, int ncomp, IntVect const& nghost, Real
            // time, int bccomp
            field.m_bcFuncX->operator()(field.m_data[xDir], 0, 1, m_nGhostRequired, 0., 0);
            field.m_bcFuncY->operator()(field.m_data[yDir], 0, 1, m_nGhostRequired, 0., 0);
            field.m_bcFuncZ->operator()(field.m_data[zDir], 0, 1, m_nGhostRequired, 0., 0);

            field.fill_boundary();
        }
    }

    /// @cond EXCLUDE_API_DOC
    // applyBC for Zero- or Three-forms
    template <Grid grid, Space space>
        requires ZeroOrThreeFormSpaceC<space>
    void apply_bc (DeRhamField<grid, space>& field)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex<isZeroOrThreeForm>::apply_bc(field)");
        if (not m_geom.isAllPeriodic())
        {
            field.m_bcFunc->operator()(field.m_data, 0, 1, m_nGhostRequired, 0., 0);

            field.fill_boundary();
        }
    }
    /// @endcond
};

// Partial specialization for any DeRhamField<G,S>
template <Grid g, Space s>
struct field_to_drc<DeRhamField<g, s>>
{
    using type = DeRhamComplex; // always maps to the same DeRhamComplex
};

// Constructors of DeRhamFields after DeRhamComplex to avoid using a member of an incomplete type
template <Grid grid, Space space>
    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
DeRhamField<grid, space>::DeRhamField(std::shared_ptr<DeRhamComplex> drc, int nVar) :
    DeRhamField{drc, std::array<int, 3>{nVar, nVar, nVar}}
{
}

template <Grid grid, Space space>
    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
DeRhamField<grid, space>::DeRhamField(std::shared_ptr<DeRhamComplex> drc,
                                      std::array<int, 3> ncomponents) :
    m_deRham{drc}
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::DeRhamField(drc)");
    m_deRham->initialize_field(*this, ncomponents);

    if (not m_deRham->get_geom().isAllPeriodic())
    {
        amrex::Vector<amrex::BCRec> bcRec =
            load_boundary_condition(m_deRham->get_geom(), "Default");
        amrex::GpuBndryFuncFab<BoundaryCondition> bcX(
            BoundaryCondition{m_data[xDir].ixType(), m_deRham->get_hodge_degree() - 1, grid});
        amrex::GpuBndryFuncFab<BoundaryCondition> bcY(
            BoundaryCondition{m_data[yDir].ixType(), m_deRham->get_hodge_degree() - 1, grid});
        amrex::GpuBndryFuncFab<BoundaryCondition> bcZ(
            BoundaryCondition{m_data[zDir].ixType(), m_deRham->get_hodge_degree() - 1, grid});
        m_bcFuncX = std::make_unique<amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BoundaryCondition>>>(
            m_deRham->get_geom(), bcRec, bcX);
        m_bcFuncY = std::make_unique<amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BoundaryCondition>>>(
            m_deRham->get_geom(), bcRec, bcY);
        m_bcFuncZ = std::make_unique<amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BoundaryCondition>>>(
            m_deRham->get_geom(), bcRec, bcZ);
    }

    auto dxInv = m_deRham->m_geom.InvCellSizeArray();

    if constexpr (space == Space::edge)
    {
        m_scaling = {GEMPIC_D_PAD_ONE(dxInv[xDir], dxInv[yDir], dxInv[zDir])};
    }
    else if constexpr (space == Space::face)
    {
        m_scaling = {GEMPIC_D_MULT(1.0, dxInv[yDir], dxInv[zDir]),
                     GEMPIC_D_MULT(dxInv[xDir], 1.0, dxInv[zDir]),
                     GEMPIC_D_MULT(dxInv[xDir], dxInv[yDir], 1.0)};
    }
}

template <Grid grid, Space space>
    requires(is_zero_or_three_form<space>()) // Ensures that space is a zero- or three-form
DeRhamField<grid, space>::DeRhamField(std::shared_ptr<DeRhamComplex> drc, int nVar) : m_deRham{drc}
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::DeRhamField(drc)");
    m_deRham->initialize_field(*this, nVar);

    if (not m_deRham->get_geom().isAllPeriodic())
    {
        amrex::Vector<amrex::BCRec> bcRec =
            load_boundary_condition(m_deRham->get_geom(), "Default");
        amrex::GpuBndryFuncFab<BoundaryCondition> bc{
            BoundaryCondition(m_data.ixType(), m_deRham->get_hodge_degree() - 1, grid)};
        m_bcFunc = std::make_unique<amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BoundaryCondition>>>(
            m_deRham->get_geom(), bcRec, bc);
    }

    auto dxInv = m_deRham->m_geom.InvCellSizeArray();

    if constexpr (space == Space::node) // node field
    {
        m_scaling = 1.0;
    }
    else if constexpr (space == Space::cell) // cell field
    {
        m_scaling = GEMPIC_D_MULT(dxInv[xDir], dxInv[yDir], dxInv[zDir]);
    }
}

template <Grid grid, Space space>
    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
DeRhamField<grid, space>::DeRhamField(
    std::shared_ptr<DeRhamComplex> drc,
    amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>& func) :
    DeRhamField{drc, 1}
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::DeRhamField(drc,func)");
    projection(func, 0.0, *this);
}

template <Grid grid, Space space>
    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
DeRhamField<grid, space>::DeRhamField(
    std::shared_ptr<DeRhamComplex> drc,
    amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>> const& funcs) :
    DeRhamField{drc, static_cast<int>(funcs.size())}
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::DeRhamField(drc,funcs)");
    projection(funcs, 0.0, *this);
}

template <Grid grid, Space space>
    requires(is_zero_or_three_form<space>()) // Ensures that space is a zero- or three-form
inline DeRhamField<grid, space>::DeRhamField (std::shared_ptr<DeRhamComplex> drc,
                                             amrex::ParserExecutor<AMREX_SPACEDIM + 1>& func) :
    DeRhamField{drc, 1}
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::DeRhamField(drc,func)");
    projection(func, 0.0, *this);
}

template <Grid grid, Space space>
    requires(is_zero_or_three_form<space>()) // Ensures that space is a zero- or three-form
inline DeRhamField<grid, space>::DeRhamField (
    std::shared_ptr<DeRhamComplex> drc,
    amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>> const& funcs) :
    DeRhamField{drc, static_cast<int>(funcs.size())}
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::DeRhamField(drc,funcs)");
    projection(funcs, 0.0, *this);
}

/// @cond EXCLUDE_API_DOC
//template <Grid grid, Space space>
//    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
//DeRhamField<grid, space>::DeRhamField(
//    std::shared_ptr<DeRhamComplex> drc,
//    amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>& func,
//    std::string const& diagName) :
//    DeRhamField{drc, diagName, 1}
//{
//    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::DeRhamField(drc,func,diagName)");
//    projection(func, 0.0, *this);
//}

//template <Grid grid, Space space>
//    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
//DeRhamField<grid, space>::DeRhamField(
//    std::shared_ptr<DeRhamComplex> drc,
//    amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>> const& funcs,
//    std::string const& diagName) :
//    DeRhamField{drc, diagName, static_cast<int>(funcs.size())}
//{
//    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::DeRhamField(drc,funcs,diagName)");
//    projection(funcs, 0.0, *this);
//}

template <Grid grid, Space space>
    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
DeRhamField<grid, space>::DeRhamField(std::shared_ptr<DeRhamComplex> drc,
                                      amrex::MultiFab const& mfIn1,
                                      amrex::MultiFab const& mfIn2,
                                      amrex::MultiFab const& mfIn3) :
    DeRhamField{drc, mfIn1.nComp()}
{
    m_data[xDir] = amrex::MultiFab(mfIn1, amrex::make_alias, 0, mfIn1.nComp());
    m_data[yDir] = amrex::MultiFab(mfIn2, amrex::make_alias, 0, mfIn2.nComp());
    m_data[zDir] = amrex::MultiFab(mfIn3, amrex::make_alias, 0, mfIn3.nComp());
}

/**
 * @brief make alias Field constructor
 * @param drc
 * @param src
 */
template <Grid grid, Space space>
    requires(is_zero_or_three_form<space>()) // Ensures that space is a zero- or three-form
DeRhamField<grid, space>::DeRhamField(std::shared_ptr<DeRhamComplex> drc,
                                      DeRhamField<grid, space> const& src) :
    DeRhamField{drc, src.get_ncomp()}
{
    m_data = amrex::MultiFab(src.multi_fab(), amrex::make_alias, 0, src.get_ncomp());
}

template <Grid grid, Space space>
    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
DeRhamField<grid, space>::DeRhamField(std::shared_ptr<DeRhamComplex> drc,
                                      DeRhamField<grid, space> const& src) :
    DeRhamField{drc,
                std::array<int, 3>{src.get_ncomp(xDir), src.get_ncomp(yDir), src.get_ncomp(zDir)}}
{
    m_data[xDir] = amrex::MultiFab(src.multi_fab(xDir), amrex::make_alias, 0, src.get_ncomp(xDir));
    m_data[yDir] = amrex::MultiFab(src.multi_fab(yDir), amrex::make_alias, 0, src.get_ncomp(yDir));
    m_data[zDir] = amrex::MultiFab(src.multi_fab(zDir), amrex::make_alias, 0, src.get_ncomp(zDir));
}

//template <Grid grid, Space space>
//    requires(is_zero_or_three_form<space>()) // Ensures that space is a zero- or three-form
//inline DeRhamField<grid, space>::DeRhamField(std::shared_ptr<DeRhamComplex> drc,
//                                             amrex::ParserExecutor<AMREX_SPACEDIM + 1>& func,
//                                             std::string const& diagName) :
//    DeRhamField{drc, diagName, 1}
//{
//    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::DeRhamField(drc,func,diagName)");
//    projection(func, 0.0, *this);
//}

//template <Grid grid, Space space>
//    requires(is_zero_or_three_form<space>()) // Ensures that space is a zero- or three-form
//inline DeRhamField<grid, space>::DeRhamField(
//    std::shared_ptr<DeRhamComplex> drc,
//    amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>> const& funcs,
//    std::string const& diagName) :
//    DeRhamField{drc, diagName, static_cast<int>(funcs.size())}
//{
//    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::DeRhamField(drc,funcs,diagName)");
//    projection(funcs, 0.0, *this);
//}

template <Grid grid, Space space>
    requires(is_zero_or_three_form<space>()) // Ensures that space is a zero- or three-form
DeRhamField<grid, space>::DeRhamField(std::shared_ptr<DeRhamComplex> drc,
                                      amrex::MultiFab const& mfIn) :
    DeRhamField{drc, mfIn.nComp()}
{
    m_data = amrex::MultiFab(mfIn, amrex::make_alias, 0, mfIn.nComp());
}

template <Grid grid, Space space>
    requires(is_zero_or_three_form<space>()) // Ensures that space is a zero- or three-form
int DeRhamField<grid, space>::get_ncomp () const
{
    return m_data.nComp();
}

template <Grid grid, Space space>
    requires(is_zero_or_three_form<space>()) // Ensures that space is a zero- or three-form
int DeRhamField<grid, space>::get_hodge_degree () const
{
    return m_deRham->get_hodge_degree();
}

template <Grid grid, Space space>
    requires(is_zero_or_three_form<space>()) // Ensures that space is a zero- or three-form
int DeRhamField<grid, space>::get_ncells (int dir) const
{
    return m_deRham->get_ncells(dir);
}
template <Grid grid, Space space>
    requires(is_zero_or_three_form<space>()) // Ensures that space is a zero- or three-form
amrex::IntVect DeRhamField<grid, space>::get_ncells () const
{
    return m_deRham->get_ncells();
}
template <Grid grid, Space space>
    requires(is_zero_or_three_form<space>()) // Ensures that space is a zero- or three-form
amrex::Geometry const DeRhamField<grid, space>::get_geom() const
{
    return m_deRham->get_geom();
}
template <Grid grid, Space space>
    requires(is_zero_or_three_form<space>()) // Ensures that space is a zero- or three-form
amrex::Real DeRhamField<grid, space>::get_scaling () const
{
    return m_scaling;
}

template <Grid grid, Space space>
    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
int DeRhamField<grid, space>::get_ncomp (int dir) const
{
    return m_data[dir].nComp();
}

template <Grid grid, Space space>
    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
int DeRhamField<grid, space>::get_hodge_degree () const
{
    return m_deRham->get_hodge_degree();
}

template <Grid grid, Space space>
    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
int DeRhamField<grid, space>::get_ncells (int dir) const
{
    return m_deRham->get_ncells(dir);
}
template <Grid grid, Space space>
    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
amrex::IntVect DeRhamField<grid, space>::get_ncells () const
{
    return m_deRham->get_ncells();
}
template <Grid grid, Space space>
    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
amrex::Geometry const DeRhamField<grid, space>::get_geom() const
{
    return m_deRham->get_geom();
}
template <Grid grid, Space space>
    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
amrex::Real DeRhamField<grid, space>::get_scaling (Direction idir) const
{
    return m_scaling[idir];
}
// Synchronization necessary after cpu and gpu particle loops
template <Grid grid, Space space>
    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
inline void DeRhamField<grid, space>::post_particle_loop_sync()
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::post_particle_loop_sync()");
    for (int comp = 0; comp < 3; ++comp)
    {
        m_data[comp].SumBoundary(0, m_data[comp].nComp(), m_deRham->get_n_ghost(),
                                 amrex::IntVect::Zero, m_deRham->get_periodicity());
        // IMPORTANT: SumBoundary does not guarantee the same order of summation for overlapping
        // degrees of freedom on all contributing boxes. The resulting differences in J are on the
        // order of machine precision but can accumulate in the electric field D. OverrideSync
        // ensures that the fields are consistent on all boxes. For the same reason, a fill_boundary
        // is used instead of directly filling ghost cells in SumBoundary.
        m_data[comp].OverrideSync(m_deRham->get_periodicity());
    }
    fill_boundary();
}

template <Grid grid, Space space>
    requires(is_zero_or_three_form<space>()) // Ensures that space is a zero- or three-form
inline void DeRhamField<grid, space>::post_particle_loop_sync()
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::post_particle_loop_sync()");
    m_data.SumBoundary(0, m_data.nComp(), m_deRham->get_n_ghost(), amrex::IntVect::Zero,
                       m_deRham->get_periodicity());
    // See comment on OverrideSync in function for OneOrTwoForm above
    m_data.OverrideSync(m_deRham->get_periodicity());
    fill_boundary();
}

// FillBoundary of DeRhamFields following the same logic as above
template <Grid grid, Space space>
    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
inline void DeRhamField<grid, space>::fill_boundary()
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::fill_boundary()");
    m_deRham->fill_boundary(*this);
}

template <Grid grid, Space space>
    requires(is_zero_or_three_form<space>()) // Ensures that space is a zero- or three-form
inline void DeRhamField<grid, space>::fill_boundary()
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::fill_boundary()");
    m_deRham->fill_boundary(*this);
}

// FillBoundary_nowait of DeRhamFields following the same logic as above
template <Grid grid, Space space>
    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
inline void DeRhamField<grid, space>::fill_boundary_nowait()
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::fill_boundary_nowait()");
    m_deRham->fill_boundary_nowait(*this);
}

template <Grid grid, Space space>
    requires(is_zero_or_three_form<space>()) // Ensures that space is a zero- or three-form
inline void DeRhamField<grid, space>::fill_boundary_nowait()
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::fill_boundary()");
    m_deRham->fill_boundary_nowait(*this);
}

// FillBoundary_finish of DeRhamFields following the same logic as above
template <Grid grid, Space space>
    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
inline void DeRhamField<grid, space>::fill_boundary_finish()
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::fill_boundary_finish()");
    m_deRham->fill_boundary_finish(*this);
}

template <Grid grid, Space space>
    requires(is_zero_or_three_form<space>()) // Ensures that space is a zero- or three-form
inline void DeRhamField<grid, space>::fill_boundary_finish()
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::fill_boundary_finish()");
    m_deRham->fill_boundary_finish(*this);
}

// ApplyBC of DeRhamFields
template <Grid grid, Space space>
    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
inline void DeRhamField<grid, space>::apply_bc()
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::apply_bc()");
    m_deRham->apply_bc(*this);
}

template <Grid grid, Space space>
    requires(is_zero_or_three_form<space>()) // Ensures that space is a zero- or three-form
inline void DeRhamField<grid, space>::apply_bc()
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::apply_bc()");
    m_deRham->apply_bc(*this);
}

// Multiply by scalar
template <Grid grid, Space space>
    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
inline void DeRhamField<grid, space>::operator*=(amrex::Real dt)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::operator*=()");
    auto nGhost = m_deRham->get_n_ghost();
    for (int comp = 0; comp < 3; ++comp)
    {
        m_data[comp].mult(dt, nGhost[xDir]);
    }
}

template <Grid grid, Space space>
    requires(is_zero_or_three_form<space>()) // Ensures that space is a zero- or three-form
inline void DeRhamField<grid, space>::operator*=(amrex::Real dt)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::operator*=()");
    auto nGhost = m_deRham->get_n_ghost();
    m_data.mult(dt, nGhost[xDir]);
}

// Multiply by parsed function
template <Grid grid, Space space>
    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
inline void DeRhamField<grid, space>::operator*=(amrex::ParserExecutor<AMREX_SPACEDIM + 1> func)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::operator*=(function)");

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = m_deRham->m_geom.CellSizeArray();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const r0 = m_deRham->get_prob_lo();

    int nComps{m_data[xDir].nComp()};

    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(m_data[comp], true); mfi.isValid(); ++mfi)
        {
            amrex::Box const& bx = mfi.tilebox();
            amrex::Array4<amrex::Real> const& form = (m_data[comp])[mfi].array();

            ParallelFor(
                bx, nComps,
                [r0, comp, dx, form, func] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                {
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r;
                    if constexpr ((grid == Grid::primal && space == Space::edge) ||
                                  (grid == Grid::dual && space == Space::face))
                    {
                        // midpoints of primal edge / dual face
                        r = {AMREX_D_DECL(r0[xDir] + (i + (comp == xDir ? 0.5 : 0)) * dx[xDir],
                                          r0[yDir] + (j + (comp == yDir ? 0.5 : 0)) * dx[yDir],
                                          r0[zDir] + (k + (comp == zDir ? 0.5 : 0)) * dx[zDir])};
                    }
                    else
                    {
                        // midpoints of primal face / dual edge
                        r = {AMREX_D_DECL(
                            r0[xDir] + (i + ((comp == yDir || comp == zDir) ? 0.5 : 0)) * dx[xDir],
                            r0[yDir] + (j + ((comp == xDir || comp == zDir) ? 0.5 : 0)) * dx[yDir],
                            r0[zDir] +
                                (k + ((comp == xDir || comp == yDir) ? 0.5 : 0)) * dx[zDir])};
                    }
                    form(i, j, k, n) =
                        form(i, j, k, n) * func(AMREX_D_DECL(r[xDir], r[yDir], r[zDir]), 0.);
                });
        }
    }
    this->fill_boundary();
}

template <Grid grid, Space space>
    requires(is_zero_or_three_form<space>()) // Ensures that space is a zero- or three-form
inline void DeRhamField<grid, space>::operator*=(amrex::ParserExecutor<AMREX_SPACEDIM + 1> func)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::operator*=(function)");

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = m_deRham->m_geom.CellSizeArray();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const r0 = m_deRham->get_prob_lo();

    int nComps{m_data.nComp()};

    for (amrex::MFIter mfi(m_data, true); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const& form = (m_data)[mfi].array();

        ParallelFor(bx, nComps,
                    [r0, dx, form, func] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r;
                        if constexpr ((grid == Grid::primal && space == Space::node) ||
                                      (grid == Grid::dual && space == Space::cell))
                        {
                            // primal nodes / dual cell midpoints
                            r = {AMREX_D_DECL(r0[xDir] + i * dx[xDir], r0[yDir] + j * dx[yDir],
                                              r0[zDir] + k * dx[zDir])};
                        }
                        else
                        {
                            // primal cell midpoints / dual nodes
                            r = {AMREX_D_DECL(r0[xDir] + (i + 0.5) * dx[xDir],
                                              r0[yDir] + (j + 0.5) * dx[yDir],
                                              r0[zDir] + (k + 0.5) * dx[zDir])};
                        }
                        form(i, j, k, n) =
                            form(i, j, k, n) * func(AMREX_D_DECL(r[xDir], r[yDir], r[zDir]), 0.);
                    });
    }
    this->fill_boundary();
}

// plusEqual
template <Grid grid, Space space>
    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
inline void DeRhamField<grid, space>::operator+=(DeRhamField<grid, space>& form)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::operator+=(MultiFab)");
    auto nGhost = m_deRham->get_n_ghost();
    for (int comp = 0; comp < 3; ++comp)
    {
        m_data[comp].plus(form.m_data[comp], 0, m_data[comp].nComp(), nGhost[xDir]);
    }
}

template <Grid grid, Space space>
    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
inline void DeRhamField<grid, space>::operator+=(amrex::Real c)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::operator+=(amrex::Real)");
    auto nGhost = m_deRham->get_n_ghost();
    for (int comp = 0; comp < 3; ++comp)
    {
        m_data[comp].plus(c, nGhost[xDir]);
    }
}

template <Grid grid, Space space>
    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
inline void DeRhamField<grid, space>::operator+=(amrex::Vector<amrex::Real> c)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::operator+=(amrex::Real)");
    auto nGhost = m_deRham->get_n_ghost();
    for (int comp = 0; comp < 3; ++comp)
    {
        m_data[comp].plus(c[comp], nGhost[xDir]);
    }
}

template <Grid grid, Space space>
    requires(is_zero_or_three_form<space>()) // Ensures that space is a one- or two-form
inline void DeRhamField<grid, space>::operator+=(DeRhamField<grid, space>& form)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::operator+=(MultiFab)");
    auto nGhost = m_deRham->get_n_ghost();
    m_data.plus(form.m_data, 0, m_data.nComp(), nGhost[xDir]);
}

template <Grid grid, Space space>
    requires(is_zero_or_three_form<space>()) // Ensures that space is a zero- or three-form
inline void DeRhamField<grid, space>::operator+=(amrex::Real c)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::operator+=(amrex::Real)");
    auto nGhost = m_deRham->get_n_ghost();
    m_data.plus(c, nGhost[xDir]);
}

// minusEqual
template <Grid grid, Space space>
    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
inline void DeRhamField<grid, space>::operator-=(DeRhamField<grid, space>& form)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::operator-=(MultiFab)");
    auto nGhost = m_deRham->get_n_ghost();
    for (int comp = 0; comp < 3; ++comp)
    {
        m_data[comp].minus(form.m_data[comp], 0, m_data[comp].nComp(), nGhost[xDir]);
    }
}

template <Grid grid, Space space>
    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
inline void DeRhamField<grid, space>::operator-=(amrex::Real c)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::operator-=(amrex::Real)");
    auto nGhost = m_deRham->get_n_ghost();
    for (int comp = 0; comp < 3; ++comp)
    {
        m_data[comp].plus(-c, nGhost[xDir]);
    }
}

template <Grid grid, Space space>
    requires(is_zero_or_three_form<space>()) // Ensures that space is a zero- or three-form
inline void DeRhamField<grid, space>::operator-=(DeRhamField<grid, space>& form)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::operator-=(MultiFab)");
    auto nGhost = m_deRham->get_n_ghost();
    m_data.minus(form.m_data, 0, m_data.nComp(), nGhost[xDir]);
}

template <Grid grid, Space space>
    requires(is_zero_or_three_form<space>()) // Ensures that space is a zero- or three-form
inline void DeRhamField<grid, space>::operator-=(amrex::Real c)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::operator-=(amrex::Real)");
    auto nGhost = m_deRham->get_n_ghost();
    m_data.plus(-c, nGhost[xDir]);
}

// minus
template <Grid grid, Space space>
    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
inline DeRhamField<grid, space> DeRhamField<grid, space>::operator-(DeRhamField<grid, space>& form)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::operator-(MultiFab)");
    DeRhamField<grid, space> result(m_deRham);
    auto nGhost = m_deRham->get_n_ghost();

    for (int comp = 0; comp < 3; ++comp)
    {
        amrex::Copy(result.m_data[comp], m_data[comp], 0, 0, m_data[comp].nComp(), nGhost);
        amrex::Subtract(result.m_data[comp], form.m_data[comp], 0, 0, form.m_data[comp].nComp(),
                        nGhost);
    }
    return result;
}

template <Grid grid, Space space>
    requires(is_zero_or_three_form<space>()) // Ensures that space is a zero- or three-form
inline DeRhamField<grid, space> DeRhamField<grid, space>::operator-(DeRhamField<grid, space>& form)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::operator-(MultiFab)");
    DeRhamField<grid, space> result(m_deRham);

    amrex::Copy(result.m_data, m_data, 0, 0, m_data.nComp(), 0);
    amrex::Subtract(result.m_data, form.m_data, 0, 0, form.m_data.nComp(), 0);

    return result;
}

///@endcond

template <Grid grid, Space space>
    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
DeRhamComplex const& DeRhamField<grid, space>::get_de_rham() const
{
    if (!m_deRham)
    {
        throw std::runtime_error("m_deRham is not initialized.");
    }
    return *m_deRham;
}
template <Grid grid, Space space>
    requires(is_zero_or_three_form<space>()) // Ensures that space is a zero- or three-form
DeRhamComplex const& DeRhamField<grid, space>::get_de_rham() const
{
    if (!m_deRham)
    {
        throw std::runtime_error("m_deRham is not initialized.");
    }
    return *m_deRham;
}

template <Grid grid, Space space>
    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
amrex::MultiFab const& DeRhamField<grid, space>::multi_fab(Direction dir) const
{
    return m_data[dir];
}
template <Grid grid, Space space>
    requires(is_one_or_two_form<space>()) // Ensures that space is a one- or two-form
amrex::MultiFab& DeRhamField<grid, space>::multi_fab (Direction dir)
{
    return m_data[dir];
}
template <Grid grid, Space space>
    requires(is_zero_or_three_form<space>()) // Ensures that space is a zero- or three-form
amrex::MultiFab const& DeRhamField<grid, space>::multi_fab() const
{
    return m_data;
}
template <Grid grid, Space space>
    requires(is_zero_or_three_form<space>()) // Ensures that space is a zero- or three-form
amrex::MultiFab& DeRhamField<grid, space>::multi_fab ()
{
    return m_data;
}

} // namespace Gempic::Forms

namespace Gempic::Forms::Impl
{
using namespace Forms;
template <class T>
struct UnwrapReference
{
    using type = T;
};
template <class U>
struct UnwrapReference<std::reference_wrapper<U>>
{
    using type = U&;
};

template <class T>
struct UnwrapRefDecay : UnwrapReference<std::decay_t<T>>
{
};

template <class T>
using unwrap_reference_t = typename UnwrapReference<T>::type;

template <class T>
using unwrap_ref_decay_t = typename UnwrapRefDecay<T>::type;
} // namespace Gempic::Forms::Impl

#endif
