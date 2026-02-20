#ifndef GEMPIC_FDDERHAMCOMPLEX_H_
#define GEMPIC_FDDERHAMCOMPLEX_H_

#include <AMReX_SPACE.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_DeRhamComplex.H"
#include "GEMPIC_FieldMethods.H"

namespace Gempic
{
//!@cond EXCLUDE_API_DOC
namespace Impl
{
template <VectorAnalyticalExpressionSpace F, typename Integrator>
void fill_with_integrated_edges (DiscreteVectorField& field,
                                 F const& funcs,
                                 Integrator const& integrate = GaussLegendreQuadrature{10})
{
    auto f = [=] AMREX_GPU_HOST_DEVICE(Direction dir,
                                       AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z))
    {
        std::array<amrex::Real, 3> r{GEMPIC_D_PAD(x, y, z)};
        std::array<amrex::Real, 3> dx{GEMPIC_D_PAD_ONE(field.discrete_grid(xDir).dx(xDir),
                                                       field.discrete_grid(yDir).dx(yDir),
                                                       field.discrete_grid(zDir).dx(zDir))};
        auto f1D = [&, r] (amrex::Real val)
        {
            std::array<amrex::Real, 3> tmp{r};
            tmp[dir] = val;
            return funcs(dir, AMREX_D_DECL(tmp[xDir], tmp[yDir], tmp[zDir]));
        };
        return integrate.line(r[dir], dx[dir] * 0.5, f1D);
    };
    fill(field, f);
}
template <VectorAnalyticalExpressionSpace F, typename Integrator>
void fill_with_integrated_faces (DiscreteVectorField& field,
                                 F const& funcs,
                                 Integrator const& integrate = GaussLegendreQuadrature{10})
{
    // Direction dir is the direction of the normal vector of the face
    auto f = [=] AMREX_GPU_HOST_DEVICE(Direction dir,
                                       AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z))
    {
        std::array<amrex::Real, 3> r{GEMPIC_D_PAD(x, y, z)};
        auto f2D = [&, r] (amrex::Real val1, amrex::Real val2)
        {
            std::array<amrex::Real, 3> tmp{r};
            switch (dir)
            {
                case Direction::xDir:
                {
                    tmp[Direction::yDir] = val1;
                    tmp[Direction::zDir] = val2;
                    break;
                }
                case Direction::yDir:
                {
                    tmp[Direction::xDir] = val1;
                    tmp[Direction::zDir] = val2;
                    break;
                }
                case Direction::zDir:
                {
                    tmp[Direction::xDir] = val1;
                    tmp[Direction::yDir] = val2;
                    break;
                }
            }
            return funcs(dir, AMREX_D_DECL(tmp[xDir], tmp[yDir], tmp[zDir]));
        };
        // Direction dir is the direction which is normal to the integrated face
        auto getFaceDirections = [=] (std::array<amrex::Real, 3> r,
                                      Direction dir) -> std::array<amrex::Real, 2>
        {
            switch (dir)
            {
                case Direction::xDir:
                    return std::array<amrex::Real, 2>{r[yDir], r[zDir]};
                case Direction::yDir:
                    return std::array<amrex::Real, 2>{r[xDir], r[zDir]};
                case Direction::zDir:
                    return std::array<amrex::Real, 2>{r[xDir], r[yDir]};
                default: // silence warning "-Werror=return-type"
                    return std::array<amrex::Real, 2>{
                        std::numeric_limits<amrex::Real>::quiet_NaN(),
                        std::numeric_limits<amrex::Real>::quiet_NaN()};
            }
        };
        std::array<amrex::Real, 2> r2D{getFaceDirections(r, dir)};
        std::array<amrex::Real, 2> dx{
            getFaceDirections({GEMPIC_D_PAD_ONE(field.discrete_grid(xDir).dx(xDir),
                                                field.discrete_grid(yDir).dx(yDir),
                                                field.discrete_grid(zDir).dx(zDir))},
                              dir)};
        for (int i = 0; i < 2; i++) dx[i] = dx[i] / 2.0;
        return integrate.surface(r2D, dx, f2D);
    };
    fill(field, f);
}
template <ScalarAnalyticalExpressionSpace F, typename Integrator>
void fill_with_integrated_cells (DiscreteField& field,
                                 F const& funcs,
                                 Integrator const& integrate = GaussLegendreQuadrature{10})
{
    std::array<amrex::Real, 3> dx{GEMPIC_D_PAD_ONE(field.discrete_grid().dx(Direction::xDir),
                                                   field.discrete_grid().dx(Direction::yDir),
                                                   field.discrete_grid().dx(Direction::zDir))};
    for (int i = 0; i < 3; i++) dx[i] = dx[i] / 2.0;
    auto f = [=] AMREX_GPU_HOST_DEVICE(AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z))
    {
        auto f3D = [&] (amrex::Real x, amrex::Real y, amrex::Real z)
        { return funcs(AMREX_D_DECL(x, y, z)); };
        return integrate.volume({GEMPIC_D_PAD(x, y, z)}, dx, f3D);
    };
    fill(field, f);
}
template <Sequence s, Form f>
class DeRhamScalarField : public DiscreteField
{
public:
    explicit DeRhamScalarField(DiscreteField&& field, GaussLegendreQuadrature const& integrator) :
        DiscreteField{std::move(field)}, m_integrator{integrator} {};
    /**
     * Explicitly delete move constructors to disable implicit conversions.
     * This only seems to be an issue for the move-constructor. Other operations are
     * not compiled even if the default constructor/assignment is not deleted.
     * (e.g. copy-constructor, copy-assignment, move-assignment)
     */
    template <Sequence so, Form fo>
        requires(so != s || fo != f)
    DeRhamScalarField(DeRhamScalarField<so, fo> const&) = delete;
    GaussLegendreQuadrature const& integrator () { return m_integrator; }

private:
    //! @todo If it is possible this should be typed into the @ref DiscreteFields::DOFCategory
    GaussLegendreQuadrature m_integrator;
};
template <Sequence s, Form f>
struct DeRhamVectorField : public DiscreteVectorField
{
public:
    explicit DeRhamVectorField(
        DiscreteVectorField&& field,
        // If it is possible this should be typed into the DOFCategory of DiscreteFields
        GaussLegendreQuadrature const& integrator) :
        DiscreteVectorField{std::move(field)}, m_integrator{integrator} {};
    /**
     * Explicitly delete move constructors to disable implicit conversions.
     * This only seems to be an issue for the move-constructor. Other operations are
     * not compiled even if the default constructor/assignment is not deleted.
     * (e.g. copy-constructor, copy-assignment, move-assignment)
     */
    template <Sequence so, Form fo>
        requires(so != s || fo != f)
    DeRhamVectorField(DeRhamVectorField<so, fo> const&) = delete;
    GaussLegendreQuadrature const& integrator () { return m_integrator; }

private:
    //! @todo If it is possible this should be typed into the @ref DiscreteFields::DOFCategory
    GaussLegendreQuadrature m_integrator;
};
} //namespace Impl

using PrimalZeroForm = Impl::DeRhamScalarField<Sequence::primal, Form::zero>;
using PrimalOneForm = Impl::DeRhamVectorField<Sequence::primal, Form::one>;
using PrimalTwoForm = Impl::DeRhamVectorField<Sequence::primal, Form::two>;
using PrimalThreeForm = Impl::DeRhamScalarField<Sequence::primal, Form::three>;
using DualZeroForm = Impl::DeRhamScalarField<Sequence::dual, Form::zero>;
using DualOneForm = Impl::DeRhamVectorField<Sequence::dual, Form::one>;
using DualTwoForm = Impl::DeRhamVectorField<Sequence::dual, Form::two>;
using DualThreeForm = Impl::DeRhamScalarField<Sequence::dual, Form::three>;
//!@endcond

class FiniteDifferenceDeRhamSpaces
{
public:
    FiniteDifferenceDeRhamSpaces(GaussLegendreQuadrature const& integrator);
    FiniteDifferenceDeRhamSpaces(GaussLegendreQuadrature const integrator,
                                 DiscreteGrid grid,
                                 amrex::BoxArray ba,
                                 amrex::DistributionMapping dm);
    PrimalZeroForm create_primal_zero_form(std::string const& label,
                                           Impl::BoundaryConditionConfiguration bcConf) const;
    PrimalOneForm create_primal_one_form(
        std::string const& label, std::array<Impl::BoundaryConditionConfiguration, 3> bcConf) const;
    PrimalTwoForm create_primal_two_form(
        std::string const& label, std::array<Impl::BoundaryConditionConfiguration, 3> bcConf) const;
    PrimalThreeForm create_primal_three_form(std::string const& label,
                                             Impl::BoundaryConditionConfiguration bcConf) const;
    DualZeroForm create_dual_zero_form(std::string const& label,
                                       Impl::BoundaryConditionConfiguration bcConf) const;
    DualOneForm create_dual_one_form(
        std::string const& label, std::array<Impl::BoundaryConditionConfiguration, 3> bcConf) const;
    DualTwoForm create_dual_two_form(
        std::string const& label, std::array<Impl::BoundaryConditionConfiguration, 3> bcConf) const;
    DualThreeForm create_dual_three_form(std::string const& label,
                                         Impl::BoundaryConditionConfiguration bcConf) const;
    using DOFCategories = std::array<DiscreteField::DOFCategory, AMREX_SPACEDIM>;

private:
    // These need to be equivalent for all operations done with the DeRhamComplex.
    // Therefore, it should not be necessary to pass them around in operations.
    // Maybe there are better solutions than static member variables
    GaussLegendreQuadrature m_integrator;
    DiscreteGrid m_grid;
    amrex::BoxArray m_boxArray;
    amrex::DistributionMapping m_distributionMapping;
    DOFCategories const point_value() const;
    std::array<DOFCategories, 3> const edge_centered_integral() const;
    std::array<DOFCategories, 3> const face_centered_integral() const;
    DOFCategories const cell_centered_integral() const;
};

template <typename FDForm, ScalarAnalyticalExpressionSpace F>
    requires std::is_same_v<FDForm, PrimalZeroForm> || std::is_same_v<FDForm, DualZeroForm>
void project (FDForm& field, F const& f)
{
    fill(field, f);
}
template <typename FDForm, VectorAnalyticalExpressionSpace F>
    requires std::is_same_v<FDForm, PrimalOneForm> || std::is_same_v<FDForm, DualOneForm>
void project (FDForm& field, F const& f)
{
    Impl::fill_with_integrated_edges(field, f, field.integrator());
}
template <typename FDForm, VectorAnalyticalExpressionSpace F>
    requires std::is_same_v<FDForm, PrimalTwoForm> || std::is_same_v<FDForm, DualTwoForm>
void project (FDForm& field, F const& f)
{
    Impl::fill_with_integrated_faces(field, f, field.integrator());
}
template <typename FDForm, ScalarAnalyticalExpressionSpace F>
    requires std::is_same_v<FDForm, PrimalThreeForm> || std::is_same_v<FDForm, DualThreeForm>
void project (FDForm& field, F const& f)
{
    Impl::fill_with_integrated_cells(field, f, field.integrator());
}

template <typename FDForm, ScalarAnalyticalExpressionSpaceTime F>
    requires std::is_same_v<FDForm, PrimalZeroForm> || std::is_same_v<FDForm, PrimalThreeForm> ||
             std::is_same_v<FDForm, DualZeroForm> || std::is_same_v<FDForm, DualThreeForm>
void project (FDForm& field, F const& f, amrex::Real const& time)
{
    auto timeIndependent = [=] AMREX_GPU_HOST_DEVICE(AMREX_D_DECL(amrex::Real x, amrex::Real y,
                                                                  amrex::Real z)) -> amrex::Real
    { return f(AMREX_D_DECL(x, y, z), time); };
    project(field, timeIndependent);
}
template <typename FDForm, VectorAnalyticalExpressionSpaceTime F>
    requires std::is_same_v<FDForm, PrimalOneForm> || std::is_same_v<FDForm, PrimalTwoForm> ||
             std::is_same_v<FDForm, DualOneForm> || std::is_same_v<FDForm, DualTwoForm>
void project (FDForm& field, F const& f, amrex::Real const& time)
{
    auto timeIndependent =
        [=] AMREX_GPU_HOST_DEVICE(
            Direction dir, AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z)) -> amrex::Real
    { return f(dir, AMREX_D_DECL(x, y, z), time); };
    project(field, timeIndependent);
}

void grad(PrimalOneForm& of, PrimalZeroForm& zf);
void curl(PrimalTwoForm& tf, PrimalOneForm& of);
void div(PrimalThreeForm& threeform, PrimalTwoForm& twoform);
void grad(DualOneForm& of, DualZeroForm& zf);
void curl(DualTwoForm& tf, DualOneForm& of);
void div(DualThreeForm& threeform, DualTwoForm& twoform);
namespace Impl
{
void finite_difference_hodge(PrimalZeroForm& p, DualThreeForm& d, size_t degree);
void finite_difference_hodge(PrimalOneForm& p, DualTwoForm& d, size_t degree);
void finite_difference_hodge(PrimalTwoForm& p, DualOneForm& d, size_t degree);
void finite_difference_hodge(PrimalThreeForm& p, DualZeroForm& d, size_t degree);
void finite_difference_hodge(DualZeroForm& d, PrimalThreeForm& p, size_t degree);
void finite_difference_hodge(DualOneForm& d, PrimalTwoForm& p, size_t degree);
void finite_difference_hodge(DualTwoForm& d, PrimalOneForm& p, size_t degree);
void finite_difference_hodge(DualThreeForm& d, PrimalZeroForm& p, size_t degree);
} //namespace Impl
void hodge(PrimalZeroForm& p, DualThreeForm& d);
void hodge(PrimalOneForm& p, DualTwoForm& d);
void hodge(PrimalTwoForm& p, DualOneForm& d);
void hodge(PrimalThreeForm& p, DualZeroForm& d);
void hodge(DualZeroForm& d, PrimalThreeForm& p);
void hodge(DualOneForm& d, PrimalTwoForm& p);
void hodge(DualTwoForm& d, PrimalOneForm& p);
void hodge(DualThreeForm& d, PrimalZeroForm& p);

} //namespace Gempic

namespace Gempic::Forms
{
/// FDDeRhamComplex
class FDDeRhamComplex : public DeRhamComplex
{
private:
    amrex::RealVect m_dr;
    HodgeScheme m_hodgeScheme;

    // Temporary MultiFabs used in the Hodges
    amrex::MultiFab m_tempPrimalZeroForm;
    amrex::MultiFab m_tempDualZeroForm;
    amrex::Array<amrex::MultiFab, 3> m_tempPrimalOneForm;
    amrex::Array<amrex::MultiFab, 3> m_tempDualOneForm;
    amrex::Array<amrex::MultiFab, 3> m_tempPrimalTwoForm;
    amrex::Array<amrex::MultiFab, 3> m_tempDualTwoForm;
    amrex::MultiFab m_tempPrimalThreeForm;
    amrex::MultiFab m_tempDualThreeForm;

    // define scaling parameters for the Hodges
    amrex::Real m_sV;
    amrex::Real m_sOmega;

public:
    /// FDDeRhamComplex constructor
    /// @param infra
    /// @param hodgeDegree
    /// @param maxSplineDegree
    /// @param hodgeScheme
    FDDeRhamComplex(ComputationalDomain const& infra,
                    int const hodgeDegree,
                    int const maxSplineDegree,
                    HodgeScheme const hodgeScheme = HodgeScheme::FDHodge,
                    int nComp = 1);
    HodgeScheme get_hodge_scheme () const override { return m_hodgeScheme; }
    amrex::MultiFab& get_temp_primal_zero_form () override { return m_tempPrimalZeroForm; }
    amrex::MultiFab& get_temp_dual_zero_form () override { return m_tempDualZeroForm; }
    amrex::Array<amrex::MultiFab, 3>& get_temp_primal_one_form () override
    {
        return m_tempPrimalOneForm;
    }
    amrex::Array<amrex::MultiFab, 3>& get_temp_dual_one_form () override
    {
        return m_tempDualOneForm;
    }
    amrex::Array<amrex::MultiFab, 3>& get_temp_primal_two_form () override
    {
        return m_tempPrimalTwoForm;
    }
    amrex::Array<amrex::MultiFab, 3>& get_temp_dual_two_form () override
    {
        return m_tempDualTwoForm;
    }
    amrex::MultiFab& get_temp_primal_three_form () override { return m_tempPrimalThreeForm; }
    amrex::MultiFab& get_temp_dual_three_form () override { return m_tempDualThreeForm; }

    ~FDDeRhamComplex() override;

    // Getter function to access m_sV
    inline amrex::Real get_s_v () const { return m_sV; }
    inline amrex::Real get_s_omega () const { return m_sOmega; }

    /**
     * @brief Returns the scaling factor for the Hodges \f$I_0\f$
     */
    amrex::Real scaling_dto_e();
    amrex::Real scaling_eto_d();
    amrex::Real scaling_bto_h();

private:
};

/**
 * @brief Actual specialization of FDDeRhamComplex class for specific hodge
 * @tparam hodgeSchemeLoc the actual hodge scheme we use to specialize the DeRhamComplex
 * @note FDDeRhamComplex is a deRham complex class assuming a single MF data structure.
 */
template <HodgeScheme hodgeSchemeLoc, Grid grid, Space space>
class DeRhamFieldMF
{
    static_assert(gempic_always_false<hodgeSchemeLoc>(),
                  "DeRhamFieldMF is not implemented for this TagType!");
};

template <Grid grid, Space space>
class DeRhamFieldMF<HodgeScheme::FDHodge, grid, space> : public DeRhamField<grid, space>
{
public:
    // Inherit all constructors from DeRhamField<Grid, Space>
    using DeRhamField<grid, space>::DeRhamField;
};

template <Grid grid, Space space>
class DeRhamFieldMF<HodgeScheme::GDECHodge, grid, space> : public DeRhamField<grid, space>
{
public:
    // Inherit all constructors from DeRhamField<Grid, Space>
    using DeRhamField<grid, space>::DeRhamField;
};

template <Grid grid, Space space>
class DeRhamFieldMF<HodgeScheme::GDECLumpHodge, grid, space> : public DeRhamField<grid, space>
{
public:
    // Inherit all constructors from DeRhamField<Grid, Space>
    using DeRhamField<grid, space>::DeRhamField;
};

template <Grid grid, Space space>
class DeRhamFieldMF<HodgeScheme::GalerkinHodge, grid, space> : public DeRhamField<grid, space>
{
public:
    // Inherit all constructors from DeRhamField<Grid, Space>
    using DeRhamField<grid, space>::DeRhamField;
};

/**
 * @brief Actual specialization of FDDeRhamComplex class for specific hodge
 * @tparam hodgeSchemeLoc the actual hodge scheme we use to specialize the DeRhamComplex
 * @note FDDeRhamComplex is a deRham complex class assuming a single MF data structure.
 */
template <HodgeScheme hodgeSchemeLoc>
class DeRhamComplexMF
{
    static_assert(gempic_always_false<hodgeSchemeLoc>(),
                  "DeRhamComplexMF is not implemented for this TagType!");
};

template <>
class DeRhamComplexMF<HodgeScheme::FDHodge>
    : public FDDeRhamComplex,
      public std::enable_shared_from_this<DeRhamComplexMF<HodgeScheme::FDHodge>>
{
public:
    static constexpr HodgeScheme s_hodgeScheme = HodgeScheme::FDHodge;
    static constexpr auto s_validDegrees{valid_degrees<s_hodgeScheme>()};
    DeRhamComplexMF(ComputationalDomain const& infra,
                    int const hodgeDegree,
                    int const maxSplineDegree,
                    int nComp = 1) :
        FDDeRhamComplex::FDDeRhamComplex(
            infra, hodgeDegree, maxSplineDegree, s_hodgeScheme, nComp) {};
};

template <>
class DeRhamComplexMF<HodgeScheme::GalerkinHodge>
    : public FDDeRhamComplex,
      public std::enable_shared_from_this<DeRhamComplexMF<HodgeScheme::GalerkinHodge>>
{
public:
    static constexpr HodgeScheme s_hodgeScheme = HodgeScheme::GalerkinHodge;
    static constexpr auto s_validDegrees{valid_degrees<s_hodgeScheme>()};

    DeRhamComplexMF(ComputationalDomain const& infra,
                    int const hodgeDegree,
                    int const maxSplineDegree,
                    int nComp = 1) :
        FDDeRhamComplex::FDDeRhamComplex(
            infra, hodgeDegree, maxSplineDegree, s_hodgeScheme, nComp) {};
};

template <>
class DeRhamComplexMF<HodgeScheme::GDECHodge>
    : public FDDeRhamComplex,
      public std::enable_shared_from_this<DeRhamComplexMF<HodgeScheme::GDECHodge>>
{
public:
    static constexpr HodgeScheme s_hodgeScheme = HodgeScheme::GDECHodge;
    using Primal0Form = DeRhamFieldMF<s_hodgeScheme, Grid::primal, Space::node>;
    using Primal1Form = DeRhamFieldMF<s_hodgeScheme, Grid::primal, Space::edge>;
    using Primal2Form = DeRhamFieldMF<s_hodgeScheme, Grid::primal, Space::face>;
    using Primal3Form = DeRhamFieldMF<s_hodgeScheme, Grid::primal, Space::cell>;
    using Dual0Form = DeRhamFieldMF<s_hodgeScheme, Grid::dual, Space::node>;
    using Dual1Form = DeRhamFieldMF<s_hodgeScheme, Grid::dual, Space::edge>;
    using Dual2Form = DeRhamFieldMF<s_hodgeScheme, Grid::dual, Space::face>;
    using Dual3Form = DeRhamFieldMF<s_hodgeScheme, Grid::dual, Space::cell>;

    static constexpr auto s_validDegrees{valid_degrees<s_hodgeScheme>()};
    DeRhamComplexMF(ComputationalDomain const& infra,
                    int const hodgeDegree,
                    int const maxSplineDegree,
                    int nComp = 1) :
        FDDeRhamComplex::FDDeRhamComplex(
            infra, hodgeDegree, maxSplineDegree, s_hodgeScheme, nComp) {};
};

template <>
class DeRhamComplexMF<HodgeScheme::GDECLumpHodge>
    : public FDDeRhamComplex,
      public std::enable_shared_from_this<DeRhamComplexMF<HodgeScheme::GDECLumpHodge>>
{
public:
    static constexpr HodgeScheme s_hodgeScheme = HodgeScheme::GDECLumpHodge;
    static constexpr auto s_validDegrees{valid_degrees<s_hodgeScheme>()};
    DeRhamComplexMF(ComputationalDomain const& infra,
                    int const hodgeDegree,
                    int const maxSplineDegree,
                    int nComp = 1) :
        FDDeRhamComplex::FDDeRhamComplex(
            infra, hodgeDegree, maxSplineDegree, s_hodgeScheme, nComp) {};
};

// alias for MF deRham complexes
using FDDeRhamComplexMF = DeRhamComplexMF<HodgeScheme::FDHodge>;
using FDGalerkinDeRhamComplexMF = DeRhamComplexMF<HodgeScheme::GalerkinHodge>;
using GDECDeRhamComplexMF = DeRhamComplexMF<HodgeScheme::GDECHodge>;
using GDECLDeRhamComplexMF = DeRhamComplexMF<HodgeScheme::GDECLumpHodge>;

// alias for MF deRham fields
template <Grid grid, Space space>
using FDDeRhamFieldMF = DeRhamFieldMF<HodgeScheme::FDHodge, grid, space>;
template <Grid grid, Space space>
using GDECDeRhamFieldMF = DeRhamFieldMF<HodgeScheme::GDECHodge, grid, space>;

// Partial specialization for any FDDeRhamField<G,S>
template <Grid g, Space s>
struct field_to_drc<FDDeRhamFieldMF<g, s>>
{
    using type = FDDeRhamComplexMF; // always maps to the same DeRhamComplex
};
// Partial specialization for any FDDeRhamField<G,S>
template <Grid g, Space s>
struct field_to_drc<GDECDeRhamFieldMF<g, s>>
{
    using type = GDECDeRhamComplexMF; // always maps to the same DeRhamComplex
};

template <typename DstFormType, typename SrcFormType>
    requires MutualDual<SrcFormType, DstFormType> && HasHodgeScheme<field_to_drc_t<SrcFormType>>
void hodge (DstFormType& dstForm, SrcFormType const& srcForm, amrex::Real weight = 1.)
{
    BL_PROFILE("hodge()");
    hodge_degree_selector<field_to_drc_t<SrcFormType>::s_hodgeScheme>(dstForm, srcForm, weight);
}

} // namespace Gempic::Forms

#endif
