/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef GEMPIC_FIELDMETHODS_H_
#define GEMPIC_FIELDMETHODS_H_

#include "GEMPIC_Fields.H"
#include "GEMPIC_Interpolation.H"
#include "GEMPIC_NumericalIntegrationDifferentiation.H"

namespace Gempic
{
template <class T>
concept HasScalarMF = requires(T const& f) {
    { f.multi_fab() } -> std::same_as<amrex::MultiFab const&>;
};

template <class T>
concept HasScaling = requires(T const& f, int s) {
    { f.scaling(s) } -> std::convertible_to<amrex::Real>;
};

template <class T>
concept HasDiscreteGrid = requires(T const& f) {
    { f.discrete_grid() } -> std::same_as<DiscreteGrid>;
} || requires(T const& f) {
    { f.discrete_grid() } -> std::same_as<DiscreteGrid const&>;
};
} //namespace Gempic
namespace Gempic::Forms
{
/**
 * @brief Computes the Restriction operator \f$R_0\f$
 *
 * @details
 * Using the geometric degrees of freedeom:
 * Evaluates func at time t on the nodes of the primal grid
 *
 * For dimensions 1 and 2, the 0-form is taken from the "stacked" de Rham complex
 *
 * @param funcs ParserExecutor, functions to be projected
 * @param t Real, time at which func is to be evaluated
 * @param field Gempic::Forms::DeRhamField<primal, Space::node>, 0-form \f$u^0\f$ holding the node
 * values
 */
void projection(amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>> const& funcs,
                amrex::Real t,
                Gempic::Forms::DeRhamComplex::Primal0Form& field);

inline void projection (amrex::ParserExecutor<AMREX_SPACEDIM + 1>& func,
                        amrex::Real t,
                        DeRhamComplex::Primal0Form& field)
{
    amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>> funcs{func};
    projection(funcs, t, field);
}

/**
 * @brief Computes the Restriction operator \f$R_3\f$
 *
 * @details
 * Using the geometric degrees of freedeom:
 * Integrates func at time t over the cells of the primal grid by using a Gauss-Legendre quadrature
 * rule
 *
 * For dimensions 1 and 2, the 3-form is taken from the "stacked" de Rham complex
 *
 * @param funcs ParserExecutor, functions to be projected
 * @param t Real, time at which func is to be evaluated
 * @param field Gempic::Forms::DeRhamField<primal, Space::cell>, 3-form \f$u^3\f$ holding the cell
 * integrals
 * @param gaussNodes int, number of Gauss nodes to be used for quadrature
 */
void projection(amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>> const& funcs,
                amrex::Real t,
                Gempic::Forms::DeRhamComplex::Primal3Form& field,
                int gaussNodes = 6);

inline void projection (amrex::ParserExecutor<AMREX_SPACEDIM + 1>& func,
                        amrex::Real t,
                        DeRhamComplex::Primal3Form& field,
                        int gaussNodes = 6)
{
    amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>> funcs{func};
    projection(funcs, t, field, gaussNodes);
}

/**
* @brief Computes the Restriction operator \f$\tilde{R}_0\f$
*
* @details
* Using the geometric degrees of freedom:
* Evaluates func at time t on the nodes of the dual grid
*
* For dimensions 1 and 2, the 0-form is taken from the "stacked" de Rham complex
*

* @param funcs ParserExecutor, functions to be projected
* @param t Real, time at which func is to be evaluated
* @param field Gempic::Forms::DeRhamField<dual, Space::node>, 0-form \f$\tilde{u}^0\f$ holding the
node values
*/
void projection(amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>> const& funcs,
                amrex::Real t,
                Gempic::Forms::DeRhamComplex::Dual0Form& field);

inline void projection (amrex::ParserExecutor<AMREX_SPACEDIM + 1>& func,
                        amrex::Real t,
                        DeRhamComplex::Dual0Form& field)
{
    amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>> funcs{func};
    projection(funcs, t, field);
}

/**
 * @brief Computes the Restriction operator \f$\tilde{R}_3\f$
 *
 * @details
 * Using the geometric degrees of freedom:
 * Integrates func at time t over the cells of the dual grid by using a Gauss-Legendre quadrature
 * rule
 *
 * For dimensions 1 and 2, the 3-form is taken from the "stacked" de Rham complex
 *
 * @param funcs ParserExecutor, functions to be projected
 * @param t Real, time at which func is to be evaluated
 * @param field Gempic::Forms::DeRhamField<dual, Space::cell>, 3-form \f$\tilde{u}^3\f$ holding the
 * cell integrals
 * @param gaussNodes int, number of Gauss nodes to be used for quadrature
 */
void projection(amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>> const& funcs,
                amrex::Real t,
                Gempic::Forms::DeRhamComplex::Dual3Form& field,
                int gaussNodes = 6);

inline void projection (amrex::ParserExecutor<AMREX_SPACEDIM + 1>& func,
                        amrex::Real t,
                        DeRhamComplex::Dual3Form& field,
                        int gaussNodes = 6)
{
    amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>> funcs{func};
    projection(funcs, t, field, gaussNodes);
}

/**
 * @brief Computes the Restriction operator \f$\tilde{R}_1\f$
 *
 * @details
 * Using the geometric degrees of freedeom:
 * Integrates func at time t over the edges of the dual grid by using a Gauss-Legendre quadrature
 * rule
 *
 * For dimensions 1 and 2, the 1-form is taken from the "stacked" de Rham complex
 *
 * @param funcs ParserExecutor, functions to be projected
 * @param t Real, time at which func is to be evaluated
 * @param field Gempic::Forms::DeRhamField<dual, Space::edge>, 1-form \f$\tilde{u}^1\f$ holding the
 * edge integrals
 * @param gaussNodes int, number of Gauss nodes to be used for quadrature
 */
void projection(
    amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>> const& funcs,
    amrex::Real t,
    Gempic::Forms::DeRhamComplex::Dual1Form& field,
    int gaussNodes = 6);

inline void projection (amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>& func,
                        amrex::Real t,
                        DeRhamComplex::Dual1Form& field,
                        int gaussNodes = 6)
{
    amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>> funcs{func};
    projection(funcs, t, field, gaussNodes);
}

/**
 * @brief Computes the Restriction operator \f$R_2\f$
 *
 * @details
 * Using the geometric degrees of freedeom:
 * Integrates func at time t over the faces of the primal grid by using a Gauss-Legendre quadrature
 * rule
 *
 * For dimensions 1 and 2, the 2-form is taken from the "stacked" de Rham complex
 *
 * @param funcs ParserExecutor, functions to be projected
 * @param t Real, time at which func is to be evaluated
 * @param field Gempic::Forms::DeRhamField<primal, Space::face>, 2-form \f$u^2\f$ holding the face
 * integrals
 * @param gaussNodes int, number of Gauss nodes to be used for quadrature
 */
void projection(
    amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>> const& funcs,
    amrex::Real t,
    Gempic::Forms::DeRhamComplex::Primal2Form& field,
    int gaussNodes = 6);

inline void projection (amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>& func,
                        amrex::Real t,
                        DeRhamComplex::Primal2Form& field,
                        int gaussNodes = 6)
{
    amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>> funcs{func};
    projection(funcs, t, field, gaussNodes);
}

/**
 * @brief Computes the Restriction operator \f$R_1\f$
 *
 * @details
 * Using the geometric degrees of freedeom:
 * Integrates func at time t over the edges of the primal grid by using a Gauss-Legendre quadrature
 * rule
 *
 * For dimensions 1 and 2, the 1-form is taken from the "stacked" de Rham complex
 *
 * @param funcs ParserExecutor, functions to be projected
 * @param t Real, time at which func is to be evaluated
 * @param field Gempic::Forms::DeRhamField<primal, Space::edge>, 1-form \f$u^1\f$ holding the edge
 * integrals
 * @param gaussNodes int, number of Gauss nodes to be used for quadrature
 */
void projection(
    amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>> const& funcs,
    amrex::Real t,
    Gempic::Forms::DeRhamComplex::Primal1Form& field,
    int gaussNodes = 6);

inline void projection (amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>& func,
                        amrex::Real t,
                        DeRhamComplex::Primal1Form& field,
                        int gaussNodes = 6)
{
    amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>> funcs{func};
    projection(funcs, t, field, gaussNodes);
}

/**
 * @brief Computes the Restriction operator \f$\tilde{R}_2\f$
 *
 * @details
 * Using the geometric degrees of freedeom:
 * Integrates func at time t over the faces of the dual grid by using a Gauss-Legendre quadrature
 * rule
 *
 * For dimensions 1 and 2, the 2-form is taken from the "stacked" de Rham complex
 *
 * @param funcs ParserExecutor, functions to be projected
 * @param t Real, time at which func is to be evaluated
 * @param field Gempic::Forms::DeRhamField<dual, Space::face>, 2-form \f$\tilde{u}^2\f$ holding the
 * face integrals
 * @param gaussNodes int, number of Gauss nodes to be used for quadrature
 */
void projection(
    amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>> const& funcs,
    amrex::Real t,
    Gempic::Forms::DeRhamComplex::Dual2Form& field,
    int gaussNodes = 6);

inline void projection (amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>& func,
                        amrex::Real t,
                        DeRhamComplex::Dual2Form& field,
                        int gaussNodes = 6)
{
    amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>> funcs{func};
    projection(funcs, t, field, gaussNodes);
}

/// Copy of fields
/// @tparam OneTwoForm specialized function for one- or two- forms
/// @param fieldOut The field to be copied into
/// @param fieldIn The field to be copied from
template <OneTwoFormC OneTwoFormOut, OneTwoFormC OneTwoFormIn>
inline void copy (OneTwoFormOut& fieldOut, OneTwoFormIn const& fieldIn)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::copy");
    auto nGhost = fieldIn.m_deRham->get_n_ghost();
    for (int comp = 0; comp < 3; ++comp)
    {
        amrex::Copy(fieldOut.m_data[comp], fieldIn.m_data[comp], 0, 0, fieldIn.m_data[comp].nComp(),
                    nGhost);
    }
}

template <ZeroThreeFormC ZeroThreeFormOut, ZeroThreeFormC ZeroThreeFormIn>
inline void copy (ZeroThreeFormOut& fieldOut, ZeroThreeFormIn const& fieldIn)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::copy");
    auto nGhost = fieldIn.m_deRham->get_n_ghost();
    amrex::Copy(fieldOut.m_data, fieldIn.m_data, 0, 0, fieldIn.m_data.nComp(), nGhost);
}

/**
 * Performs a linear combination operation on two DeRhamField objects for one or two forms.
 *
 * @param result The result of the linear combination operation.
 * @param a The scaling factor for the first field.
 * @param field1 The first DeRhamField object.
 * @param b The scaling factor for the second field.
 * @param field2 The second DeRhamField object.
 */
template <ZeroThreeFormC ZeroThreeForm1, ZeroThreeFormC ZeroThreeForm2>
inline void linear_combination (ZeroThreeForm1& result,
                                amrex::Real a,
                                ZeroThreeForm1 const& field1,
                                amrex::Real b,
                                ZeroThreeForm2 const& field2)
{
    BL_PROFILE("Gempic::Forms::linearCombination()");
    auto nGhost = result.m_deRham->get_n_ghost();
    amrex::LinComb(result.m_data, a, field1.m_data, 0, b, field2.m_data, 0, 0,
                   field1.m_data.nComp(), nGhost);
}

/**
 * Performs a linear combination operation on two DeRhamField objects for one or two forms.
 *
 * @param result The result of the linear combination operation.
 * @param a The scaling factor for the first field.
 * @param field1 The first DeRhamField object.
 * @param b The scaling factor for the second field.
 * @param field2 The second DeRhamField object.
 */
template <OneTwoFormC OneTwoForm1, OneTwoFormC OneTwoForm2>
inline void linear_combination (OneTwoForm1& result,
                                amrex::Real a,
                                OneTwoForm1 const& field1,
                                amrex::Real b,
                                OneTwoForm2 const& field2)
{
    BL_PROFILE("Gempic::Forms::linearCombination()");
    auto nGhost = result.m_deRham->get_n_ghost();
    for (int comp = 0; comp < 3; ++comp)
    {
        amrex::LinComb(result.m_data[comp], a, field1.m_data[comp], 0, b, field2.m_data[comp], 0, 0,
                       field1.m_data[comp].nComp(), nGhost);
    }
}

// multiply and add for fields
/**
 * Multiply and add for operation on two DeRhamField objects for one or two forms.
 *
 * @param field1 The destination DeRhamField object.
 * @param a The scaling factor for the second field.
 * @param field2 The second DeRhamField object.
 */
template <ZeroThreeFormC ZeroThreeForm1, ZeroThreeFormC ZeroThreeForm2>
inline void multiply_and_add (ZeroThreeForm1& field1, amrex::Real a, ZeroThreeForm2 const& field2)
{
    BL_PROFILE("Gempic::Forms::multiply_and_add()");
    auto nGhost = field1.m_deRham->get_n_ghost();
    mult_and_add(field1.m_data, a, field2.m_data, 0, 0, field2.m_data.nComp(), nGhost);
}

/**
 * Multiply and add for operation on two DeRhamField objects for one or two forms.
 *
 * @param field1 The destination DeRhamField object.
 * @param a The scaling factor for the second field.
 * @param field2 The second DeRhamField object.
 */
template <OneTwoFormC OneTwoForm1, OneTwoFormC OneTwoForm2>
inline void multiply_and_add (OneTwoForm1& field1, amrex::Real a, OneTwoForm2 const& field2)
{
    BL_PROFILE("Gempic::Forms::multiply_and_add()");
    auto nGhost = field1.m_deRham->get_n_ghost();
    for (int comp = 0; comp < 3; ++comp)
    {
        mult_and_add(field1.m_data[comp], a, field2.m_data[comp], 0, 0, field2.m_data[comp].nComp(),
                     nGhost);
    }
}

/**
 * @brief Computes the dot product of two fields on the same grid.
 *
 * @tparam grid Grid type.
 * @tparam space Space type.
 * @tparam std::enable_if_t<is_one_or_two_form<space>> Enables the function only if space is one or
 * two form.
 * @param field1 The first field.
 * @param field2 The second field.
 *
 * @return The dot product of the fields.
 */
template <OneTwoFormC OneTwoForm1, OneTwoFormC OneTwoForm2>
inline amrex::Real dot_product (OneTwoForm1 const& field1, OneTwoForm2 const& field2)
{
    bool const skipMpiCommTrue = true;
    // Compute the dot product by summing the dot product of each component.
    amrex::Real dotProd{0};              // Initialize the dot product to 0.
    for (int comp = 0; comp < 3; ++comp) // Loop over each component.
    {
        auto mask = amrex::OwnerMask(field1.m_data[comp], field1.m_deRham->get_periodicity());
        dotProd += amrex::MultiFab::Dot(*mask, field1.m_data[comp], 0, field2.m_data[comp], 0,
                                        field1.m_data[comp].nComp(), 0, skipMpiCommTrue);
    }
    amrex::ParallelAllReduce::Sum(dotProd, amrex::ParallelContext::CommunicatorSub());
    return dotProd; // Return the computed dot product.
}

/**
 * @brief Computes the dot product of two DeRham fields on the same grid and space
 * for zero and three forms
 * @tparam grid The grid type of the DeRham field.
 * @tparam space The space type of the DeRham field.
 * @param field1 The first DeRham field.
 * @param field2 The second DeRham field.
 *
 * @return The dot product of the two fields.
 */
template <ZeroThreeFormC ZeroThreeForm1, ZeroThreeFormC ZeroThreeForm2>
inline amrex::Real dot_product (ZeroThreeForm1 const& field1, ZeroThreeForm2 const& field2)
{
    auto mask = amrex::OwnerMask(field1.m_data, field1.m_deRham->get_periodicity());
    amrex::Real dotProd =
        amrex::MultiFab::Dot(*mask, field1.m_data, 0, field2.m_data, 0, field1.m_data.nComp(), 0);
    return dotProd;
}

//---------------------
// exterior derivatives
//---------------------

/**
 * @brief Computes the discrete curl \f$\mathbb{C}\f$ on the primal grid
 *
 * @details
 * Using the geometric degrees of freedom. No approximations involved.
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * @param[out] twoForm DeRhamField<Grid::primal, Space::face>, 2-form \f$\mathbb{C} u^1\f$ holding
 * the resulting face integrals
 * @param oneForm DeRhamField<Grid::primal, Space::edge>, 1-form \f$u^1\f$ holding the edge
 * integrals
 */
template <PrimalTwoFormC TwoFormType, PrimalOneFormC OneFormType>
void curl (TwoFormType& twoForm, OneFormType const& oneForm)
{
    BL_PROFILE("Gempic::Forms::curl(primal,face,primal,edge)");
    int nComps{oneForm.m_data[xDir].nComp()};
    // Component-0 of curl
    for (amrex::MFIter mfi(twoForm.m_data[xDir]); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.validbox();

        amrex::Array4<amrex::Real> const& twoForm0 = (twoForm.m_data[xDir])[mfi].array();
#if (AMREX_SPACEDIM > 2)
        amrex::Array4<amrex::Real const> const& oneForm1 =
            (oneForm.m_data[yDir])[mfi].const_array();
#endif
#if (AMREX_SPACEDIM > 1)
        amrex::Array4<amrex::Real const> const& oneForm2 =
            (oneForm.m_data[zDir])[mfi].const_array();
#endif
        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        twoForm0(i, j, k, n) =
                            GEMPIC_D_ADD(0., oneForm2(i, j + 1, k, n) - oneForm2(i, j, k, n),
                                         -oneForm1(i, j, k + 1, n) + oneForm1(i, j, k, n));
                    });
    }
    twoForm.m_data[xDir].FillBoundary_nowait(twoForm.m_deRham->m_geom.periodicity());

    // Component-1 of curl
    for (amrex::MFIter mfi(twoForm.m_data[yDir]); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.validbox();

        amrex::Array4<amrex::Real> const& twoForm1 = (twoForm.m_data[yDir])[mfi].array();
        amrex::Array4<amrex::Real const> const& oneForm2 =
            (oneForm.m_data[zDir])[mfi].const_array();
#if (AMREX_SPACEDIM > 2)
        amrex::Array4<amrex::Real const> const& oneForm0 =
            (oneForm.m_data[xDir])[mfi].const_array();
#endif
        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        twoForm1(i, j, k, n) =
                            GEMPIC_D_ADD(-oneForm2(i + 1, j, k, n) + oneForm2(i, j, k, n), 0.,
                                         oneForm0(i, j, k + 1, n) - oneForm0(i, j, k, n));
                    });
    }
    twoForm.m_data[yDir].FillBoundary_nowait(twoForm.m_deRham->m_geom.periodicity());

    // Component-2 of curl
    for (amrex::MFIter mfi(twoForm.m_data[zDir]); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.validbox();

        amrex::Array4<amrex::Real> const& twoForm2 = (twoForm.m_data[zDir])[mfi].array();
#if (AMREX_SPACEDIM > 1)
        amrex::Array4<amrex::Real const> const& oneForm0 =
            (oneForm.m_data[xDir])[mfi].const_array();
#endif
        amrex::Array4<amrex::Real const> const& oneForm1 =
            (oneForm.m_data[yDir])[mfi].const_array();
        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        twoForm2(i, j, k, n) =
                            GEMPIC_D_ADD(oneForm1(i + 1, j, k, n) - oneForm1(i, j, k, n),
                                         -oneForm0(i, j + 1, k, n) + oneForm0(i, j, k, n), 0.);
                        ;
                    });
    }
    twoForm.m_data[zDir].FillBoundary_nowait(twoForm.m_deRham->m_geom.periodicity());

    // Wait for completed communication of guard data
    twoForm.m_data[xDir].FillBoundary_finish();
    twoForm.m_data[yDir].FillBoundary_finish();
    twoForm.m_data[zDir].FillBoundary_finish();
}

/**
 * @brief Adds dt times the discrete curl \f$\mathbb{C}\f$ on the primal grid
 *
 * @details
 * Using the geometric degrees of freedom. No approximations involved.
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * @param[out] twoForm DeRhamField<Grid::primal, Space::face>, 2-form \f$\mathbb{C} u^1\f$ holding
 * the resulting face integrals
 * @param oneForm DeRhamField<Grid::primal, Space::edge>, 1-form \f$u^1\f$ holding the edge
 * integrals
 * @param dt time step by which curl is multiplied. Can be negative for subtraction.
 */
template <PrimalTwoFormC TwoFormType, PrimalOneFormC OneFormType>
void add_dt_curl (TwoFormType& twoForm, OneFormType const& oneForm, amrex::Real dt)
{
    BL_PROFILE("Gempic::Forms::add_dt_curl(primal,face,primal,edge)");
    int nComps{oneForm.m_data[xDir].nComp()};
    // Component-0 of curl
    for (amrex::MFIter mfi(twoForm.m_data[xDir]); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.validbox();

        amrex::Array4<amrex::Real> const& twoForm0 = (twoForm.m_data[xDir])[mfi].array();
#if (AMREX_SPACEDIM > 2)
        amrex::Array4<amrex::Real const> const& oneForm1 =
            (oneForm.m_data[yDir])[mfi].const_array();
#endif
#if (AMREX_SPACEDIM > 1)
        amrex::Array4<amrex::Real const> const& oneForm2 =
            (oneForm.m_data[zDir])[mfi].const_array();
#endif
        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        twoForm0(i, j, k, n) +=
                            dt * GEMPIC_D_ADD(0., oneForm2(i, j + 1, k, n) - oneForm2(i, j, k, n),
                                              -oneForm1(i, j, k + 1, n) + oneForm1(i, j, k, n));
                    });
    }
    twoForm.m_data[xDir].FillBoundary_nowait(twoForm.m_deRham->m_geom.periodicity());

    // Component-1 of curl
    for (amrex::MFIter mfi(twoForm.m_data[yDir]); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.validbox();

        amrex::Array4<amrex::Real> const& twoForm1 = (twoForm.m_data[yDir])[mfi].array();
        amrex::Array4<amrex::Real const> const& oneForm2 =
            (oneForm.m_data[zDir])[mfi].const_array();
#if (AMREX_SPACEDIM > 2)
        amrex::Array4<amrex::Real const> const& oneForm0 =
            (oneForm.m_data[xDir])[mfi].const_array();
#endif
        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        twoForm1(i, j, k, n) +=
                            dt * GEMPIC_D_ADD(-oneForm2(i + 1, j, k, n) + oneForm2(i, j, k, n), 0.,
                                              oneForm0(i, j, k + 1, n) - oneForm0(i, j, k, n));
                    });
    }
    twoForm.m_data[yDir].FillBoundary_nowait(twoForm.m_deRham->m_geom.periodicity());

    // Component-2 of curl
    for (amrex::MFIter mfi(twoForm.m_data[zDir]); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.validbox();

        amrex::Array4<amrex::Real> const& twoForm2 = (twoForm.m_data[zDir])[mfi].array();
#if (AMREX_SPACEDIM > 1)
        amrex::Array4<amrex::Real const> const& oneForm0 =
            (oneForm.m_data[xDir])[mfi].const_array();
#endif
        amrex::Array4<amrex::Real const> const& oneForm1 =
            (oneForm.m_data[yDir])[mfi].const_array();
        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        twoForm2(i, j, k, n) +=
                            dt * GEMPIC_D_ADD(oneForm1(i + 1, j, k, n) - oneForm1(i, j, k, n),
                                              -oneForm0(i, j + 1, k, n) + oneForm0(i, j, k, n), 0.);
                        ;
                    });
    }
    twoForm.m_data[zDir].FillBoundary_nowait(twoForm.m_deRham->m_geom.periodicity());

    // Wait for completed communication of guard data
    twoForm.m_data[xDir].FillBoundary_finish();
    twoForm.m_data[yDir].FillBoundary_finish();
    twoForm.m_data[zDir].FillBoundary_finish();
}

/**
 * @brief Computes the discrete curl \f$\mathbb{C}\f$ on the dual grid
 *
 * @details
 * Using the geometric degrees of freedom. No approximations involved.
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * @param[out] twoForm DeRhamField<Grid::dual, Space::face>, 2-form \f$\tilde{C} \tilde{u}^1\f$
 * holding the resulting face integrals
 * @param oneForm DeRhamField<Grid::dual, Space::edge>, 1-form \f$\tilde{u}^1\f$ holding the edge
 * integrals
 */
template <DualTwoFormC TwoFormType, DualOneFormC OneFormType>
void curl (TwoFormType& twoForm, OneFormType const& oneForm)
{
    BL_PROFILE("Gempic::Forms::curl(dual,face,dual,edge)");
    int nComps{oneForm.m_data[xDir].nComp()};
    // Component-0 of curl
    for (amrex::MFIter mfi(twoForm.m_data[xDir]); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.validbox();

        amrex::Array4<amrex::Real> const& twoForm0 = (twoForm.m_data[xDir])[mfi].array();
#if (AMREX_SPACEDIM > 2)
        amrex::Array4<amrex::Real const> const& oneForm1 =
            (oneForm.m_data[yDir])[mfi].const_array();
#endif
#if (AMREX_SPACEDIM > 1)
        amrex::Array4<amrex::Real const> const& oneForm2 =
            (oneForm.m_data[zDir])[mfi].const_array();
#endif
        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        twoForm0(i, j, k, n) =
                            GEMPIC_D_ADD(0., oneForm2(i, j, k, n) - oneForm2(i, j - 1, k, n),
                                         -oneForm1(i, j, k, n) + oneForm1(i, j, k - 1, n));
                    });
    }

    twoForm.m_data[xDir].FillBoundary_nowait(twoForm.m_deRham->m_geom.periodicity());

    // Component-1 of curl
    for (amrex::MFIter mfi(twoForm.m_data[yDir]); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.validbox();

        amrex::Array4<amrex::Real> const& twoForm1 = (twoForm.m_data[yDir])[mfi].array();
        amrex::Array4<amrex::Real const> const& oneForm2 =
            (oneForm.m_data[zDir])[mfi].const_array();
#if (AMREX_SPACEDIM > 2)
        amrex::Array4<amrex::Real const> const& oneForm0 =
            (oneForm.m_data[xDir])[mfi].const_array();
#endif
        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        twoForm1(i, j, k, n) =
                            GEMPIC_D_ADD(-oneForm2(i, j, k, n) + oneForm2(i - 1, j, k, n), 0.,
                                         oneForm0(i, j, k, n) - oneForm0(i, j, k - 1, n));
                    });
    }

    twoForm.m_data[yDir].FillBoundary_nowait(twoForm.m_deRham->m_geom.periodicity());

    // Component-2 of curl
    for (amrex::MFIter mfi(twoForm.m_data[zDir]); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.validbox();

        amrex::Array4<amrex::Real> const& twoForm2 = (twoForm.m_data[zDir])[mfi].array();
#if (AMREX_SPACEDIM > 1)
        amrex::Array4<amrex::Real const> const& oneForm0 =
            (oneForm.m_data[xDir])[mfi].const_array();
#endif
        amrex::Array4<amrex::Real const> const& oneForm1 =
            (oneForm.m_data[yDir])[mfi].const_array();

        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        twoForm2(i, j, k, n) =
                            GEMPIC_D_ADD(oneForm1(i, j, k, n) - oneForm1(i - 1, j, k, n),
                                         -oneForm0(i, j, k, n) + oneForm0(i, j - 1, k, n), 0.);
                    });
    }

    twoForm.m_data[zDir].FillBoundary_nowait(twoForm.m_deRham->m_geom.periodicity());

    // Wait for completed communication of guard data
    twoForm.m_data[xDir].FillBoundary_finish();
    twoForm.m_data[yDir].FillBoundary_finish();
    twoForm.m_data[zDir].FillBoundary_finish();
}

/**
 * @brief Adds dt times the discrete curl \f$\mathbb{C}\f$ on the dual grid
 *
 * @details
 * Using the geometric degrees of freedom. No approximations involved.
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * @param[out] twoForm DeRhamField<Grid::dual, Space::face>, 2-form \f$\tilde{C} \tilde{u}^1\f$
 * holding the resulting face integrals
 * @param oneForm DeRhamField<Grid::dual, Space::edge>, 1-form \f$\tilde{u}^1\f$ holding the edge
 * integrals
 * @param dt time step by which curl is multiplied. Can be negative for subtraction.
 */
template <DualTwoFormC TwoFormType, DualOneFormC OneFormType>
void add_dt_curl (TwoFormType& twoForm, OneFormType const& oneForm, amrex::Real dt)
{
    BL_PROFILE("Gempic::Forms::add_dt_curl(dual,face,dual,edge)");
    int nComps{oneForm.m_data[xDir].nComp()};
    // Component-0 of curl
    for (amrex::MFIter mfi(twoForm.m_data[xDir]); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.validbox();

        amrex::Array4<amrex::Real> const& twoForm0 = (twoForm.m_data[xDir])[mfi].array();
#if (AMREX_SPACEDIM > 2)
        amrex::Array4<amrex::Real const> const& oneForm1 =
            (oneForm.m_data[yDir])[mfi].const_array();
#endif
#if (AMREX_SPACEDIM > 1)
        amrex::Array4<amrex::Real const> const& oneForm2 =
            (oneForm.m_data[zDir])[mfi].const_array();
#endif
        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        twoForm0(i, j, k, n) +=
                            dt * GEMPIC_D_ADD(0., oneForm2(i, j, k, n) - oneForm2(i, j - 1, k, n),
                                              -oneForm1(i, j, k, n) + oneForm1(i, j, k - 1, n));
                    });
    }

    twoForm.m_data[xDir].FillBoundary_nowait(twoForm.m_deRham->m_geom.periodicity());

    // Component-1 of curl
    for (amrex::MFIter mfi(twoForm.m_data[yDir]); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.validbox();

        amrex::Array4<amrex::Real> const& twoForm1 = (twoForm.m_data[yDir])[mfi].array();
        amrex::Array4<amrex::Real const> const& oneForm2 =
            (oneForm.m_data[zDir])[mfi].const_array();
#if (AMREX_SPACEDIM > 2)
        amrex::Array4<amrex::Real const> const& oneForm0 =
            (oneForm.m_data[xDir])[mfi].const_array();
#endif
        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        twoForm1(i, j, k, n) +=
                            dt * GEMPIC_D_ADD(-oneForm2(i, j, k, n) + oneForm2(i - 1, j, k, n), 0.,
                                              oneForm0(i, j, k, n) - oneForm0(i, j, k - 1, n));
                    });
    }

    twoForm.m_data[yDir].FillBoundary_nowait(twoForm.m_deRham->m_geom.periodicity());

    // Component-2 of curl
    for (amrex::MFIter mfi(twoForm.m_data[zDir]); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.validbox();

        amrex::Array4<amrex::Real> const& twoForm2 = (twoForm.m_data[zDir])[mfi].array();
#if (AMREX_SPACEDIM > 1)
        amrex::Array4<amrex::Real const> const& oneForm0 =
            (oneForm.m_data[xDir])[mfi].const_array();
#endif
        amrex::Array4<amrex::Real const> const& oneForm1 =
            (oneForm.m_data[yDir])[mfi].const_array();

        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        twoForm2(i, j, k, n) +=
                            dt * GEMPIC_D_ADD(oneForm1(i, j, k, n) - oneForm1(i - 1, j, k, n),
                                              -oneForm0(i, j, k, n) + oneForm0(i, j - 1, k, n), 0.);
                    });
    }

    twoForm.m_data[zDir].FillBoundary_nowait(twoForm.m_deRham->m_geom.periodicity());

    // Wait for completed communication of guard data
    twoForm.m_data[xDir].FillBoundary_finish();
    twoForm.m_data[yDir].FillBoundary_finish();
    twoForm.m_data[zDir].FillBoundary_finish();
}

/**
 * @brief Computes the discrete gradient \f$\mathbb{G}\f$ on the primal grid
 *
 * @details
 * Using the geometric degrees of freedom. No approximations involved.
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * @param[out] oneForm DeRhamField<Grid::primal, Space::edge>, 1-form \f$\mathbb{G} u^0\f$ holding
 * the resulting edge integrals
 * @param zeroForm DeRhamField<Grid::primal, Space::node>, 0-form \f$u^0\f$ holding the node
 * values
 */
template <PrimalOneFormC OneFormType, PrimalZeroFormC ZeroFormType>
void grad (OneFormType& oneForm, ZeroFormType const& zeroForm)
{
    BL_PROFILE("Gempic::Forms::grad(primal,edge,primal,node)");
    int nComps{zeroForm.m_data.nComp()};
    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(oneForm.m_data[comp]); mfi.isValid(); ++mfi)
        {
            amrex::Box const& bx = mfi.validbox();

            amrex::Array4<amrex::Real const> const& zeroFormMF =
                (zeroForm.m_data)[mfi].const_array();
            amrex::Array4<amrex::Real> const& oneFormMF = (oneForm.m_data[comp])[mfi].array();

            // Calculate gradient component by component
            if (comp == xDir)
            {
                ParallelFor(bx, nComps,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                            {
                                oneFormMF(i, j, k, n) =
                                    zeroFormMF(i + 1, j, k, n) - zeroFormMF(i, j, k, n);
                            });
            }

            if (comp == yDir)
            {
                ParallelFor(bx, nComps,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                            {
                                oneFormMF(i, j, k, n) = GEMPIC_D_ADD(
                                    0., zeroFormMF(i, j + 1, k, n) - zeroFormMF(i, j, k, n), 0.);
                            });
            }

            if (comp == zDir)
            {
                ParallelFor(bx, nComps,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                            {
                                oneFormMF(i, j, k, n) = GEMPIC_D_ADD(
                                    0., 0., zeroFormMF(i, j, k + 1, n) - zeroFormMF(i, j, k, n));
                            });
            }
        }
        oneForm.m_data[comp].FillBoundary_nowait(oneForm.m_deRham->m_geom.periodicity());
    }
    oneForm.m_data[xDir].FillBoundary_finish();
    oneForm.m_data[yDir].FillBoundary_finish();
    oneForm.m_data[zDir].FillBoundary_finish();
}

/**
 * @brief Computes a * grad phi on the primal grid for some given constant a
 *
 * @details
 * Using the geometric degrees of freedom. No approximations involved.
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * @param[out] oneForm DeRhamField<Grid::primal, Space::edge>, 1-form \f$\mathbb{G} u^0\f$ holding
 * the resulting edge integrals
 * @param zeroForm DeRhamField<Grid::primal, Space::node>, 0-form \f$u^0\f$ holding the node
 * values
 * @param a amrex::Real, constant to be multiplied with the gradient
 */
template <PrimalOneFormC OneFormType, PrimalZeroFormC ZeroFormType>
void a_times_grad (OneFormType& oneForm, ZeroFormType const& zeroForm, amrex::Real a)
{
    BL_PROFILE("Gempic::Forms::a_times_grad(primal,edge,primal,node)");
    int nComps{zeroForm.m_data.nComp()};
    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(oneForm.m_data[comp]); mfi.isValid(); ++mfi)
        {
            amrex::Box const& bx = mfi.validbox();

            amrex::Array4<amrex::Real const> const& zeroFormMF =
                (zeroForm.m_data)[mfi].const_array();
            amrex::Array4<amrex::Real> const& oneFormMF = (oneForm.m_data[comp])[mfi].array();

            // Calculate gradient component by component
            if (comp == xDir)
            {
                ParallelFor(bx, nComps,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                            {
                                oneFormMF(i, j, k, n) =
                                    a * (zeroFormMF(i + 1, j, k, n) - zeroFormMF(i, j, k, n));
                            });
            }

            if (comp == yDir)
            {
                ParallelFor(
                    bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        oneFormMF(i, j, k, n) =
                            a * (GEMPIC_D_ADD(
                                    0., zeroFormMF(i, j + 1, k, n) - zeroFormMF(i, j, k, n), 0.));
                    });
            }

            if (comp == zDir)
            {
                ParallelFor(
                    bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        oneFormMF(i, j, k, n) =
                            a * (GEMPIC_D_ADD(0., 0.,
                                              zeroFormMF(i, j, k + 1, n) - zeroFormMF(i, j, k, n)));
                    });
            }
        }
        oneForm.m_data[comp].FillBoundary_nowait(oneForm.m_deRham->m_geom.periodicity());
    }
    oneForm.m_data[xDir].FillBoundary_finish();
    oneForm.m_data[yDir].FillBoundary_finish();
    oneForm.m_data[zDir].FillBoundary_finish();
}

/**
 * @brief Computes the discrete gradient \f$\mathbb{G}\f$ on the dual grid
 *
 * @details
 * Using the geometric degrees of freedom. No approximations involved.
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * @param[out] oneForm DeRhamField<Grid::dual, Space::edge>, 1-form \f$\tilde{G} \tilde{u}^0\f$
 * holding the resulting edge integrals
 * @param zeroForm DeRhamField<Grid::dual, Space::node>, 0-form \f$\tilde{u}^0\f$ holding the node
 * values
 */
template <DualOneFormC OneFormType, DualZeroFormC ZeroFormType>
void grad (OneFormType& oneForm, ZeroFormType const& zeroForm)
{
    BL_PROFILE("Gempic::Forms::grad(dual,edge,dual,node)");
    int nComps{zeroForm.m_data.nComp()};
    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(oneForm.m_data[comp]); mfi.isValid(); ++mfi)
        {
            amrex::Box const& bx = mfi.validbox();

            amrex::Array4<amrex::Real const> const& zeroFormMF =
                (zeroForm.m_data)[mfi].const_array();
            amrex::Array4<amrex::Real> const& oneFormMF = (oneForm.m_data[comp])[mfi].array();

            // Calculate gradient component by component
            if (comp == xDir)
            {
                ParallelFor(bx, nComps,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                            {
                                oneFormMF(i, j, k, n) =
                                    zeroFormMF(i, j, k, n) - zeroFormMF(i - 1, j, k, n);
                            });
            }

            if (comp == yDir)
            {
                ParallelFor(bx, nComps,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                            {
                                oneFormMF(i, j, k, n) = GEMPIC_D_ADD(
                                    0., zeroFormMF(i, j, k, n) - zeroFormMF(i, j - 1, k, n), 0.);
                            });
            }

            if (comp == zDir)
            {
                ParallelFor(bx, nComps,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                            {
                                oneFormMF(i, j, k, n) = GEMPIC_D_ADD(
                                    0., 0., zeroFormMF(i, j, k, n) - zeroFormMF(i, j, k - 1, n));
                            });
            }
        }
        oneForm.m_data[comp].FillBoundary_nowait(oneForm.m_deRham->m_geom.periodicity());
    }

    oneForm.m_data[xDir].FillBoundary_finish();
    oneForm.m_data[yDir].FillBoundary_finish();
    oneForm.m_data[zDir].FillBoundary_finish();
}

/**
 * @brief Computes a times the discrete gradient \f$a \mathbb{G}\f$ on the dual grid
 *
 * @details
 * Using the geometric degrees of freedom. No approximations involved.
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * @param[out] oneForm DeRhamField<Grid::dual, Space::edge>, 1-form \f$\tilde{G} \tilde{u}^0\f$
 * holding the resulting edge integrals
 * @param zeroForm DeRhamField<Grid::dual, Space::node>, 0-form \f$\tilde{u}^0\f$ holding the node
 * values
 * @param a Multiplication factor
 */
template <DualOneFormC OneFormType, DualZeroFormC ZeroFormType>
void a_times_grad (OneFormType& oneForm, ZeroFormType const& zeroForm, amrex::Real a)
{
    BL_PROFILE("Gempic::Forms::a_times_grad(dual,edge,dual,node)");
    int nComps{zeroForm.m_data.nComp()};
    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(oneForm.m_data[comp]); mfi.isValid(); ++mfi)
        {
            amrex::Box const& bx = mfi.validbox();

            amrex::Array4<amrex::Real const> const& zeroFormMF =
                (zeroForm.m_data)[mfi].const_array();
            amrex::Array4<amrex::Real> const& oneFormMF = (oneForm.m_data[comp])[mfi].array();

            // Calculate gradient component by component
            if (comp == xDir)
            {
                ParallelFor(bx, nComps,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                            {
                                oneFormMF(i, j, k, n) =
                                    a * (zeroFormMF(i, j, k, n) - zeroFormMF(i - 1, j, k, n));
                            });
            }

            if (comp == yDir)
            {
                ParallelFor(
                    bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        oneFormMF(i, j, k, n) =
                            a * (GEMPIC_D_ADD(
                                    0., zeroFormMF(i, j, k, n) - zeroFormMF(i, j - 1, k, n), 0.));
                    });
            }

            if (comp == zDir)
            {
                ParallelFor(
                    bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        oneFormMF(i, j, k, n) =
                            a * (GEMPIC_D_ADD(0., 0.,
                                              zeroFormMF(i, j, k, n) - zeroFormMF(i, j, k - 1, n)));
                    });
            }
        }
        oneForm.m_data[comp].FillBoundary_nowait(oneForm.m_deRham->m_geom.periodicity());
    }

    oneForm.m_data[xDir].FillBoundary_finish();
    oneForm.m_data[yDir].FillBoundary_finish();
    oneForm.m_data[zDir].FillBoundary_finish();
}

/**
 * @brief Computes the discrete divergence \f$\mathbb{D}\f$ on the primal grid
 *
 * @details
 * Using the geometric degrees of freedom. No approximations involved.
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * @param[out] threeForm DeRhamField<Grid::primal, Space::cell>, 3-form \f$\mathbb{D} u^2\f$
 * holding the resulting cell integrals
 * @param twoForm DeRhamField<Grid::primal, Space::face>, 2-form \f$u^2\f$ holding the face
 * integrals
 */
template <PrimalThreeFormC ThreeFormType, PrimalTwoFormC TwoFormType>
void div (ThreeFormType& threeForm, TwoFormType const& twoForm)
{
    BL_PROFILE("Gempic::Forms::div(primal,cell,primal,face)");
    int nComps{twoForm.m_data[xDir].nComp()};
    for (amrex::MFIter mfi(threeForm.m_data); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.validbox();

        amrex::Array4<amrex::Real const> const& twoFormMF0 =
            (twoForm.m_data[xDir])[mfi].const_array();
#if (AMREX_SPACEDIM > 1)
        amrex::Array4<amrex::Real const> const& twoFormMF1 =
            (twoForm.m_data[yDir])[mfi].const_array();
#endif
#if (AMREX_SPACEDIM > 2)
        amrex::Array4<amrex::Real const> const& twoFormMF2 =
            (twoForm.m_data[zDir])[mfi].const_array();
#endif
        amrex::Array4<amrex::Real> const& threeFormMF = (threeForm.m_data)[mfi].array();

        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        threeFormMF(i, j, k, n) =
                            GEMPIC_D_ADD(twoFormMF0(i + 1, j, k, n) - twoFormMF0(i, j, k, n),
                                         twoFormMF1(i, j + 1, k, n) - twoFormMF1(i, j, k, n),
                                         twoFormMF2(i, j, k + 1, n) - twoFormMF2(i, j, k, n));
                    });
    }
    threeForm.fill_boundary();
}

/**
 * @brief Computes the discrete divergence \f$\mathbb{D}\f$ on the dual grid
 *
 * @details
 * Using the geometric degrees of freedom. No approximations involved.
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * @param twoForm DeRhamField<Grid::primal, Space::dual>, 2-form \f$\tilde{u}^2\f$ holding the
 * face integrals
 * @param[out] threeForm DeRhamField<Grid::primal, Space::dual>, 3-form \f$\tilde{D}
 * \tilde{u}^2\f$ holding the resulting cell integrals
 */
template <DualThreeFormC ThreeFormType, DualTwoFormC TwoFormType>
void div (ThreeFormType& threeForm, TwoFormType const& twoForm)
{
    BL_PROFILE("Gempic::Forms::div(dual,cell,dual,face)");
    int nComps{twoForm.m_data[xDir].nComp()};
    for (amrex::MFIter mfi(threeForm.m_data); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.validbox();

        amrex::Array4<amrex::Real const> const& twoFormMF0 =
            (twoForm.m_data[xDir])[mfi].const_array();
#if (AMREX_SPACEDIM > 1)
        amrex::Array4<amrex::Real const> const& twoFormMF1 =
            (twoForm.m_data[yDir])[mfi].const_array();
#endif
#if (AMREX_SPACEDIM > 2)
        amrex::Array4<amrex::Real const> const& twoFormMF2 =
            (twoForm.m_data[zDir])[mfi].const_array();
#endif
        amrex::Array4<amrex::Real> const& threeFormMF = (threeForm.m_data)[mfi].array();

        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        threeFormMF(i, j, k, n) =
                            GEMPIC_D_ADD(twoFormMF0(i, j, k, n) - twoFormMF0(i - 1, j, k, n),
                                         twoFormMF1(i, j, k, n) - twoFormMF1(i, j - 1, k, n),
                                         twoFormMF2(i, j, k, n) - twoFormMF2(i, j, k - 1, n));
                    });
    }
    threeForm.fill_boundary();
}

/**
 * @brief compute stencil coefficients for the V0 1d mass matrix for GDEC finite elements
 * @tparam intDegree interpolating polynomial degree
 * @return array of stencil coefficients
 */
template <int intDegree>
auto inline compute_v0_mass_matrix_1d_stencil_gdec()
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::compute_V0_mass_matrix_1d_stencil()");
    // Select stencil according to degree

    static constexpr int massStencilLength{2 * intDegree + 1};
    static constexpr int nDofs{intDegree + 1};
    static constexpr int nQuadraturePoints{nDofs};
    static constexpr int leftStencilIdx{-intDegree};
    using GL = Gempic::GaussQuadratureUnit<nQuadraturePoints>;
    // polynomial indexes goes from {-n,...,0,...,n+1}, in total 2n+2 values.  n = (intDegree - 1) /
    // 2. cell indexes goes from {-n - 1,...,0,...,n}, in total 2n+2 values.
    static constexpr int nShift{(intDegree - 1) / 2};

    // loop along the stencil of the Mass matrix
    // this corresponds to the loop along the neighbor geometric dofs.
    //
    // for the nodal basis function, if we assume ell=0 is the index of the geometric dof; ell=0
    // will be also the adjacent cell at the right of the ell-th node. ell+sigma is the index of the
    // neighbor dof on the stencil. then we must compute the integral  int (  Lambda_{ell}
    // Lambda_{ell+sigma}  ) using quadrature rule. Lambda_i are pw polynomials with compact support
    // over an even number of cells (intDegree+1 ==hodgeDegree) centered at the i-th node. On every
    // cell, Lambda_i is defined by the Lagrange interpolating polynomial built with reference to a
    // central stencil of intDegree+1 nodes, centered at the local cell. which one? The one
    // interpolating 1 at the i-th node, and zero otherwise.

    std::array<amrex::Real, massStencilLength> stencil1d;
    // quadrature on unit domain [0,1]

    // a: stencil index
    for (int a = 0, sigma = leftStencilIdx; a < massStencilLength; ++a, ++sigma)
    {
        stencil1d[a] = 0.0;
        for (int iCell = -nShift - 1; iCell <= nShift; ++iCell)
        {
            int rightCellJ{sigma + nShift};
            int leftCellJ{sigma - nShift - 1};
            // if the iCell-th cell is not in the support of LambdaJ, then, continue.
            if (iCell > rightCellJ || iCell < leftCellJ)
            {
                continue;
            }
            int iCellJ = iCell - sigma;
            int polyIndex0 = -iCell;  // iCell is exactly the distance to the ell-th node ==
                                      // polyIndex for eval_form
            int polyIndexJ = -iCellJ; // iCellJ is exactly the distance to the j-th node ==
                                      // polyIndex for eval_form
            amrex::Real val{0.0};
            for (int g = 0; g < nQuadraturePoints; ++g)
            {
                amrex::Real lambda0 = eval_lagrange<intDegree + 1>(polyIndex0, GL::s_nodes[g]);
                amrex::Real lambdaJ = eval_lagrange<intDegree + 1>(polyIndexJ, GL::s_nodes[g]);
                val += GL::s_weights[g] * lambda0 * lambdaJ;
            }
            stencil1d[a] += val;
        }
    }
    return stencil1d;
}

/**
 * @brief pre-compute matrix of coefficients necessary to evaluate V0 1d weighted mass matrix for
 * GDEC finite elements. evaluating the basis elements at quadrature points in every cell in the
 * support of a basis element.
 * @tparam intDegree intepolating polynomial degree
 * @return matrix of coefficients
 */
template <int intDegree>
std::array<std::array<std::array<amrex::Real, intDegree + 1>, intDegree + 1>, 2 * intDegree + 1>
compute_v0_weighted_mass_matrix_1d_stencil_gdec ()
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::compute_v0_weighted_mass_matrix_1d_stencil_gdec()");
    // Select stencil according to degree
    static constexpr int massStencilLength{2 * intDegree + 1};
    static constexpr int nQuadraturePoints{intDegree + 1};
    using GL = Gempic::GaussQuadratureUnit<nQuadraturePoints>;
    static constexpr int leftStencilIdx{-intDegree};
    static constexpr int nShift{
        (intDegree - 1) /
        2}; // polyIndexes goes from {-n,...,0,...,n+1}, in total 2n+2 values.  intDegree +1

    // for the nodal basis function, we assume ell=0 is the index of the geometric dof; ell=0 will
    // be also the adjacent cell at the right of the ell-th node.

    // loop along the stencil of the Mass matrix
    // this corresponds to the loop along the adjacent geometric dofs, ell+sigma is the index of the
    // adjacent dof we must compute the integral  int (  Lambda_{ell} Lambda_{ell+sigma}  ) using
    // quadrature rule. Lambda_i are pw polynomials with compact support over an even number of
    // cells (intDegree+1 ==hodgeDegree) centered at the i-th node. On every cell, Lambda_i is
    // defined by the Lagrange interpolating polynomial built with reference to a central stencil of
    // intDegree+1 nodes, centered at the local cell. which one? The one interpolating 1 at the
    // i-th node, and zero otherwise.

    std::array<std::array<std::array<amrex::Real, nQuadraturePoints>, intDegree + 1>,
               massStencilLength>
        stencil1d;
    // a: stencil index
    for (int a = 0, sigma = leftStencilIdx; a < massStencilLength; ++a, ++sigma)
    {
        for (int iCell = -nShift - 1, cellIdx = 0; iCell <= nShift; ++iCell, ++cellIdx)
        {
            int rightCellJ{sigma + nShift};
            int leftCellJ{sigma - nShift - 1};
            // if the iCell-th cell is not in the support of LambdaJ, then, continue.
            if (iCell > rightCellJ || iCell < leftCellJ)
            {
                for (int g = 0; g < nQuadraturePoints; ++g)
                {
                    stencil1d[a][cellIdx][g] = 0.;
                }
                continue;
            }
            int iCellJ = iCell - sigma;
            int polyIndex0 = -iCell;  // iCell is exactly the distance to the ell-th node ==
                                      // polyIndex for eval_form
            int polyIndexJ = -iCellJ; // iCellJ is exactly the distance to the j-th node ==
                                      // polyIndex for eval_form
            for (int g = 0; g < nQuadraturePoints; ++g)
            {
                amrex::Real lambda0 = eval_lagrange<intDegree + 1>(polyIndex0, GL::s_nodes[g]);
                amrex::Real lambdaJ = eval_lagrange<intDegree + 1>(polyIndexJ, GL::s_nodes[g]);
                stencil1d[a][cellIdx][g] = GL::s_weights[g] * lambda0 * lambdaJ;
            }
        }
    }
    return stencil1d;
}

/**
 * @brief compute stencil coefficients for the V1 1d mass matrix for GDEC finite elements
 * @tparam intDegree intepolating polynomial degree
 * @return array of stencil coefficients
 */
template <int intDegree>
auto inline compute_v1_mass_matrix_1d_stencil_gdec()
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::compute_v1_mass_matrix_1d_stencil_gdec()");
    // Select stencil according to degree

    constexpr int degree{intDegree - 1};
    constexpr int massStencilLength{2 * intDegree - 1};
    constexpr int nDofs{intDegree};
    constexpr int nQuadraturePoints{nDofs};
    using GL = Gempic::GaussQuadratureUnit<nQuadraturePoints>;
    constexpr int leftStencilIdx{-degree};
    // polynomial indexes goes from {-n,...,0,...,n}, in total 2n+1 values.  n = (intDegree - 1) / 2
    // = degree / 2. cell indexes goes from {-n,...,0,...,n}, in total 2n+2 values.
    constexpr int nShift{degree / 2};

    // loop along the stencil of the Mass matrix
    // this corresponds to the loop along the neighbor geometric dofs
    //
    // for the baricentric basis function, we assume ell=0 is the index of the geometric dof, and
    // polynomial indexes match cell indexes.
    // ell+sigma is the index of the neighbor dof
    // then we must compute the integral  int (  Lambda_{ell} Lambda_{ell+sigma}  ) using
    // quadrature rule. Lambda_i are pw polynomials with compact support over an odd number of
    // cells (degree+1 ==hodgeDegree-1) centered at the i-th cell. On every cell, Lambda_i is
    // defined by the histopolating polynomial built with reference to a central stencil of
    // degree+1 nodes, centered at the local cell. which one? The one integrating to 1 at the
    // i-th cell, and zero otherwise.

    std::array<amrex::Real, massStencilLength> stencil1d;
    // a: stencil index
    for (int a = 0, sigma = leftStencilIdx; a < massStencilLength; ++a, ++sigma)
    {
        stencil1d[a] = 0.0;
        for (int iCell = -nShift; iCell <= nShift; ++iCell)
        {
            int rightCellJ{sigma + nShift};
            int leftCellJ{sigma - nShift};
            // if the iCell-th cell is not in the support of LambdaJ, then, continue.
            if (iCell > rightCellJ || iCell < leftCellJ)
            {
                continue;
            }
            int iCellJ = iCell - sigma;
            int polyIndex0 = -iCell;  // iCell is exactly the distance to the ell-th node ==
                                      // polyIndex for eval_form
            int polyIndexJ = -iCellJ; // iCellJ is exactly the distance to the j-th node ==
                                      // polyIndex for eval_form
            amrex::Real val{0.0};
            for (int g = 0; g < nQuadraturePoints; ++g)
            {
                amrex::Real lambda0 = eval_edge<degree + 2>(polyIndex0, GL::s_nodes[g]);
                amrex::Real lambdaJ = eval_edge<degree + 2>(polyIndexJ, GL::s_nodes[g]);
                val += GL::s_weights[g] * lambda0 * lambdaJ;
            }
            stencil1d[a] += val;
        }
    }
    return stencil1d;
}

/**
 * @brief pre-compute matrix of coefficients necessary to evaluate V1 1d weighted mass matrix for
 * GDEC finite elements. evaluating the basis elements at quadrature points in every cell in the
 * support of a basis element.
 * @tparam intDegree intepolating polynomial degree
 * @return matrix of coefficients
 */
template <int intDegree>
std::array<std::array<std::array<amrex::Real, intDegree + 1>, intDegree>, 2 * intDegree - 1>
compute_v1_weighted_mass_matrix_1d_stencil_gdec ()
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::compute_v1_weighted_mass_matrix_1d_stencil_gdec()");
    // Select stencil according to degree
    constexpr int degree{intDegree - 1};
    constexpr int massStencilLength{2 * intDegree - 1};
    constexpr int nQuadraturePoints{intDegree + 1};
    using GL = Gempic::GaussQuadratureUnit<nQuadraturePoints>;
    constexpr int leftStencilIdx{-degree};
    constexpr int nShift{degree /
                         2}; // polyIndexes goes from {-n,...,0,...,n}, in total 2n+1 values.

    // for the nodal basis function, we assume ell=0 is the index of the geometric dof; ell=0 will
    // be also the adjacent cell at the right of the ell-th node.

    // loop along the stencil of the Mass matrix
    // this corresponds to the loop along the adjacent geometric dofs, ell+sigma is the index of the
    // adjacent dof we must compute the integral  int (  Lambda_{ell} Lambda_{ell+sigma}  ) using
    // quadrature rule. Lambda_i are pw polynomials with compact support over an odd number of
    // cells (degree+1 ==hodgeDegree-1) centered at the i-th cell. On every cell, Lambda_i is
    // defined by the histopolating polynomial built with reference to a central stencil of
    // degree+1 nodes, centered at the local cell. which one? The one integrating to 1 at the
    // i-th cell, and zero otherwise.

    std::array<std::array<std::array<amrex::Real, nQuadraturePoints>, degree + 1>,
               massStencilLength>
        stencil1d;
    // a: stencil index
    for (int a = 0, sigma = leftStencilIdx; a < massStencilLength; ++a, ++sigma)
    {
        for (int iCell = -nShift, cellIdx = 0; iCell <= nShift; ++iCell, ++cellIdx)
        {
            int rightCellJ{sigma + nShift};
            int leftCellJ{sigma - nShift};
            // if the iCell-th cell is not in the support of LambdaJ, then, continue.
            if (iCell > rightCellJ || iCell < leftCellJ)
            {
                for (int g = 0; g < nQuadraturePoints; ++g)
                {
                    stencil1d[a][cellIdx][g] = 0.0;
                }
                continue;
            }
            int iCellJ = iCell - sigma;
            int polyIndex0 = -iCell;  // iCell is exactly the distance to the ell-th node ==
                                      // polyIndex for eval_form
            int polyIndexJ = -iCellJ; // iCellJ is exactly the distance to the j-th node ==
                                      // polyIndex for eval_form
            for (int g = 0; g < nQuadraturePoints; ++g)
            {
                amrex::Real lambda0 = eval_edge<degree + 2>(polyIndex0, GL::s_nodes[g]);
                amrex::Real lambdaJ = eval_edge<degree + 2>(polyIndexJ, GL::s_nodes[g]);
                stencil1d[a][cellIdx][g] = GL::s_weights[g] * lambda0 * lambdaJ;
            }
        }
    }
    return stencil1d;
}

/**
 * @brief Fills the arrays of Hodge stencils.
 *
 * @details Computes stencils for periodic boundary conditions for either
 * finite difference of Galerkin Hodge
 *
 * @tparam degree int, degree of basis functions used
 * @tparam scheme int, 0 for FDHodge, 1 for Galerkin Hodge
 * @return std::tuple containing the stencils in the following order:
 *                      stencilNodeToCell, stencilCellToNode
 */
// NOLINTBEGIN(readability-magic-numbers)
template <int degree, HodgeScheme scheme>
inline auto get_hodge_stencils ()
{
    BL_PROFILE("Gempic::Forms::get_hodge_stencils()");
    if constexpr (scheme == HodgeScheme::FDHodge)
    {
        constexpr unsigned int stencilLength{degree - 1};
        amrex::GpuArray<amrex::Real, stencilLength> stencilNodeToCell;
        amrex::GpuArray<amrex::Real, stencilLength> stencilCellToNode;

        switch (degree)
        {
            case 2:
                stencilNodeToCell[0] = 1.0;
                stencilCellToNode[0] = 1.0;
                break;
            case 4:
                stencilNodeToCell[0] = 1. / 24.;
                stencilNodeToCell[1] = 22. / 24.;
                stencilNodeToCell[2] = 1. / 24.;
                stencilCellToNode[0] = -1. / 24.;
                stencilCellToNode[1] = 13. / 12.;
                stencilCellToNode[2] = -1. / 24.;
                break;
            case 6:
                stencilNodeToCell[0] = -17. / 5760.;
                stencilNodeToCell[1] = 154. / 2880.;
                stencilNodeToCell[2] = 863. / 960.;
                stencilNodeToCell[3] = 154. / 2880.;
                stencilNodeToCell[4] = -17. / 5760.;
                stencilCellToNode[0] = 3. / 640.;
                stencilCellToNode[1] = -29. / 480.;
                stencilCellToNode[2] = 1067. / 960.;
                stencilCellToNode[3] = -29. / 480.;
                stencilCellToNode[4] = 3. / 640.;
                break;
            default:
                AMREX_ASSERT("Degree not implemented");
                break;
        }

        return std::make_tuple(stencilNodeToCell, stencilCellToNode);
    }
    if constexpr (scheme == HodgeScheme::GalerkinHodge)
    {
        constexpr unsigned int stencilLength{2 * degree - 1};
        amrex::GpuArray<amrex::Real, stencilLength> stencilNodeToCell;
        amrex::GpuArray<amrex::Real, stencilLength> stencilCellToNode;

        switch (degree)
        {
            case 2:
                stencilCellToNode[0] = 1. / 8.;
                stencilCellToNode[1] = 3. / 4.;
                stencilCellToNode[2] = 1. / 8.;
                stencilNodeToCell = stencilCellToNode;
                break;
            case 4:
                stencilCellToNode[0] = 121. / 46080.;
                stencilCellToNode[1] = -959. / 23040.;
                stencilCellToNode[2] = 4663. / 46080.;
                stencilCellToNode[3] = 10087. / 11520.;
                stencilCellToNode[4] = 4663. / 46080.;
                stencilCellToNode[5] = -959. / 23040.;
                stencilCellToNode[6] = 121. / 46080.;
                stencilNodeToCell = stencilCellToNode;
                break;
            case 6:
                stencilCellToNode[0] = 365053. / 3715891200.;
                stencilCellToNode[1] = -2981533. / 1857945600.;
                stencilCellToNode[2] = 55805753. / 3715891200.;
                stencilCellToNode[3] = -27050951. / 464486400.;
                stencilCellToNode[4] = 147416293. / 1857945600.;
                stencilCellToNode[5] = 864656441. / 928972800.;
                stencilCellToNode[6] = 147416293. / 1857945600.;
                stencilCellToNode[7] = -27050951. / 464486400.;
                stencilCellToNode[8] = 55805753. / 3715891200.;
                stencilCellToNode[9] = -2981533. / 1857945600.;
                stencilCellToNode[10] = 365053. / 3715891200.;
                stencilNodeToCell = stencilCellToNode;
                break;
            default:
                AMREX_ASSERT("Degree not implemented");
                break;
        }

        return std::make_tuple(stencilNodeToCell, stencilCellToNode);
    }
    if constexpr (scheme == HodgeScheme::GDECHodge)
    {
        // the so called degree is actually the order of accuracy (="degree"+1)
        constexpr unsigned int intDegree{degree - 1};
        constexpr unsigned int order{intDegree + 1};
        constexpr unsigned int stencilLength{2 * intDegree + 1};
        // V0 or V1 refer to the 1D complex V0 --dx--> V1
        amrex::GpuArray<amrex::Real, stencilLength> stencilHodgeV0;
        amrex::GpuArray<amrex::Real, stencilLength> stencilHodgeV1;

        switch (order)
        {
            case 2:
                stencilHodgeV0[0] = 1. / 6.;
                stencilHodgeV0[1] = 2. / 3.;
                stencilHodgeV0[2] = 1. / 6.;
                stencilHodgeV1[0] = 0.;
                stencilHodgeV1[1] = 1.;
                stencilHodgeV1[2] = 0.;
                break;
            case 4:
                stencilHodgeV0[0] = 31. / 15120.;
                stencilHodgeV0[1] = -3. / 70.;
                stencilHodgeV0[2] = 257. / 1680.;
                stencilHodgeV0[3] = 733. / 945.;
                stencilHodgeV0[4] = 257. / 1680.;
                stencilHodgeV0[5] = -3. / 70.;
                stencilHodgeV0[6] = 31. / 15120.;

                stencilHodgeV1[0] = 0.;
                stencilHodgeV1[1] = -7. / 360.;
                stencilHodgeV1[2] = -1. / 180.;
                stencilHodgeV1[3] = 21. / 20.;
                stencilHodgeV1[4] = -1. / 180.;
                stencilHodgeV1[5] = -7. / 360.;
                stencilHodgeV1[6] = 0.;
                break;
            case 6:
                stencilHodgeV0[0] = 313. / 4435200.;
                stencilHodgeV0[1] = -3937. / 3326400.;
                stencilHodgeV0[2] = 59053. / 4435200.;
                stencilHodgeV0[3] = -12421. / 207900.;
                stencilHodgeV0[4] = 906919. / 6652800.;
                stencilHodgeV0[5] = 455963. / 554400.;
                stencilHodgeV0[6] = 906919. / 6652800.;
                stencilHodgeV0[7] = -12421. / 207900.;
                stencilHodgeV0[8] = 59053. / 4435200.;
                stencilHodgeV0[9] = -3937. / 3326400.;
                stencilHodgeV0[10] = 313. / 4435200.;

                stencilHodgeV1[0] = 0.;
                stencilHodgeV1[1] = -89. / 129600.;
                stencilHodgeV1[2] = 4547. / 453600.;
                stencilHodgeV1[3] = -4003. / 113400.;
                stencilHodgeV1[4] = -9691. / 453600.;
                stencilHodgeV1[5] = 99307. / 90720.;
                stencilHodgeV1[6] = -9691. / 453600.;
                stencilHodgeV1[7] = -4003. / 113400.;
                stencilHodgeV1[8] = 4547. / 453600.;
                stencilHodgeV1[9] = -89. / 129600.;
                stencilHodgeV1[10] = 0.;
                break;
            default:
                if (order % 2 != 0)
                {
                    AMREX_ASSERT("Impossible error! ODD orders not implemented");
                    amrex::Abort("Impossible error! ODD orders not implemented");
                }
                else // even number
                {
                    auto stencil0tmp = compute_v0_mass_matrix_1d_stencil_gdec<intDegree>();
                    auto stencil1tmp = compute_v1_mass_matrix_1d_stencil_gdec<intDegree>();
                    for (int i = 0; i < stencilLength; ++i)
                    {
                        stencilHodgeV0[i] = stencil0tmp[i];
                    }
                    stencilHodgeV1[0] = 0.;
                    stencilHodgeV1[stencilLength - 1] = 0.;
                    for (int i = 1; i < stencilLength - 1; ++i)
                    {
                        stencilHodgeV1[i] = stencil1tmp[i - 1];
                    }
                }
        }

        return std::make_tuple(stencilHodgeV0, stencilHodgeV1);
    }
    if constexpr (scheme == HodgeScheme::GDECLumpHodge)
    {
        // the so called degree is actually the order of accuracy (="degree"+1)
        constexpr unsigned int intDegree{degree - 1};
        constexpr unsigned int order{intDegree + 1};
        constexpr unsigned int stencilLength{2 * intDegree + 1};
        // V0 or V1 refer to the 1D complex V0 --dx--> V1
        amrex::GpuArray<amrex::Real, stencilLength> stencilHodgeV0;
        amrex::GpuArray<amrex::Real, stencilLength> stencilHodgeV1;

        switch (order)
        {
            case 2:
                stencilHodgeV0[0] = 0.;
                stencilHodgeV0[1] = 1.;
                stencilHodgeV0[2] = 0.;
                stencilHodgeV1[0] = 0.;
                stencilHodgeV1[1] = 1.;
                stencilHodgeV1[2] = 0.;
                break;
            case 4:
                stencilHodgeV0[0] = 0.;
                stencilHodgeV0[1] = 0.;
                stencilHodgeV0[2] = 0.;
                stencilHodgeV0[3] = 1.;
                stencilHodgeV0[4] = 0.;
                stencilHodgeV0[5] = 0.;
                stencilHodgeV0[6] = 0.;

                stencilHodgeV1[0] = 0.;
                stencilHodgeV1[1] = -7. / 360.;
                stencilHodgeV1[2] = -1. / 180.;
                stencilHodgeV1[3] = 21. / 20.;
                stencilHodgeV1[4] = -1. / 180.;
                stencilHodgeV1[5] = -7. / 360.;
                stencilHodgeV1[6] = 0.;
                break;
            case 6:
                stencilHodgeV0[0] = 0.;
                stencilHodgeV0[1] = 0.;
                stencilHodgeV0[2] = 0.;
                stencilHodgeV0[3] = 0.;
                stencilHodgeV0[4] = 0.;
                stencilHodgeV0[5] = 1.;
                stencilHodgeV0[6] = 0.;
                stencilHodgeV0[7] = 0.;
                stencilHodgeV0[8] = 0.;
                stencilHodgeV0[9] = 0.;
                stencilHodgeV0[10] = 0.;

                stencilHodgeV1[0] = 0.;
                stencilHodgeV1[1] = -89. / 129600.;
                stencilHodgeV1[2] = 4547. / 453600.;
                stencilHodgeV1[3] = -4003. / 113400.;
                stencilHodgeV1[4] = -9691. / 453600.;
                stencilHodgeV1[5] = 99307. / 90720.;
                stencilHodgeV1[6] = -9691. / 453600.;
                stencilHodgeV1[7] = -4003. / 113400.;
                stencilHodgeV1[8] = 4547. / 453600.;
                stencilHodgeV1[9] = -89. / 129600.;
                stencilHodgeV1[10] = 0.;
                break;
            default:
                AMREX_ASSERT("order not implemented");
                break;
        }

        return std::make_tuple(stencilHodgeV0, stencilHodgeV1);
    }
}
// NOLINTEND(readability-magic-numbers)

/**
 * @brief Computes a matrix-vector product along 1 spatial direction.
 *
 * @details
 * Computes M*u for all edges in a given direction, where M is given
 * by a stencil and u are the values of the input along one edge.
 *
 * @tparam stencilLength int, length of the stencil array defining the matrix
 * @tparam direction Direction, direction along which matrix vector product is calculated
 * @param geom Geometry, needed for FillBoundary
 * @param stencil GpuArray, defining the matrix
 * @param multiInput MultiFab, containing the values of the vector being multiplied
 * @param multiOutput MultiFab, containing the result
 */

template <Direction direction, int hodgeDegree, unsigned int stencilLength>
inline void apply_1d_hodge (amrex::Geometry geom,
                            amrex::GpuArray<amrex::Real, stencilLength> stencil,
                            amrex::MultiFab const& multiInput,
                            amrex::MultiFab& multiOutput)
{
    BL_PROFILE("Gempic::Forms::apply_1d_hodge()");
    int nComps{multiInput.nComp()};
    for (amrex::MFIter mfi(multiOutput); mfi.isValid(); ++mfi)
    {
        amrex::Box bx = mfi.validbox();

        // Grow box to include sufficient number of ghost cells for stencil application in following
        // non-periodic dimensions. In the periodic case, this is handled by FillBoundary.
#if AMREX_SPACEDIM >= 2
        constexpr int nGrow = hodgeDegree / 2 - 1;
        if constexpr (direction == xDir)
        {
            if (!geom.isPeriodic(yDir))
            {
                bx.grow(yDir, nGrow);
            }
        }
#endif
#if AMREX_SPACEDIM == 3
        if constexpr (direction == xDir || direction == yDir)
        {
            if (!geom.isPeriodic(zDir))
            {
                bx.grow(zDir, nGrow);
            }
        }
#endif

        amrex::Array4<amrex::Real const> const& vecIn = multiInput[mfi].const_array();
        amrex::Array4<amrex::Real> const& vecOut = multiOutput[mfi].array();

        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        amrex::Real val = 0.0;

                        for (int d = 0; d < stencilLength; d++)
                        {
                            int shift = d - (stencilLength + 1) / 2 + 1;
                            val += stencil[d] * vecIn(i + (direction == xDir ? shift : 0),
                                                      j + (direction == yDir ? shift : 0),
                                                      k + (direction == zDir ? shift : 0), n);
                        }
                        vecOut(i, j, k, n) = val;
                    });
    }
    multiOutput.FillBoundary(geom.periodicity());
}

template <int hodgeDegree, HodgeScheme hodgeScheme, typename SrcForm, typename DstForm>
    requires MutualDual<SrcForm, DstForm>
void hodge_operator (DstForm f0, SrcForm f1, amrex::Real weight = 1.0)
{
    static_assert(gempic_always_false<hodgeScheme>(),
                  "DeRhamComplexChain is not implemented for this hodgeScheme and pairs!");
}

/**
 * @brief Computes the Finite Difference Hodge \f$\tilde{\mathbb{H}}_3\f$.
 *
 * @details
 * Computes \f$\tilde{\mathbb{H}}_3\f$ by multiplying the 1D stencils along
 * each direction separately
 * Supports a constant weight function of the form w(x,y,z) = w
 *
 * Boundary conditions are periodic in all directions
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * @tparam hodgeDegree int, degree of basis functions used
 * @tparam hodgeScheme HodgeScheme, determines the stencils used for the Hodge operator
 * @param threeForm DeRhamComplex::Dual3Form, 3-form \f$\tilde{u}^3\f$
 * @param zeroForm DeRhamComplex::Primal0Form, resulting 0-form \f$u^0\f$
 * @param weight Real, constant weight function
 */
template <int hodgeDegree, HodgeScheme hodgeScheme, DeRhamComplexC DRC = DeRhamComplex>
void hodge_operator (typename DRC::Primal0Form& zeroForm,
                     typename DRC::Dual3Form const& threeForm,
                     amrex::Real weight = 1.0)
{
    BL_PROFILE("hodge_operator(dual,cell,primal,node,weight)");
    // Select stencil according to degree
    auto [stencilNodeToCell, stencilCellToNode] = get_hodge_stencils<hodgeDegree, hodgeScheme>();

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dr{threeForm.m_deRham->get_dx()};
    amrex::Geometry const geom = threeForm.m_deRham->m_geom;

#if AMREX_SPACEDIM > 1
    amrex::MultiFab& tempPrimalZeroForm = zeroForm.m_deRham->get_temp_primal_zero_form();
#endif

    AMREX_D_TERM(
        (apply_1d_hodge<xDir, hodgeDegree>(
            geom, stencilCellToNode, threeForm.m_data,
            AMREX_D_PICK(zeroForm.m_data, tempPrimalZeroForm, zeroForm.m_data)));
        , (apply_1d_hodge<yDir, hodgeDegree>(
              geom, stencilCellToNode, AMREX_D_PICK((void), tempPrimalZeroForm, zeroForm.m_data),
              AMREX_D_PICK((void), zeroForm.m_data, tempPrimalZeroForm)));
        , (apply_1d_hodge<zDir, hodgeDegree>(geom, stencilCellToNode, tempPrimalZeroForm,
                                             zeroForm.m_data)););

    zeroForm.m_data.mult(weight / GEMPIC_D_MULT(dr[xDir], dr[yDir], dr[zDir]));

    zeroForm.fill_boundary();
}

/**
 * @brief Computes the Finite Difference Hodge \f$\mathbb{H}_3\f$.
 *
 * @details
 * Computes \f$\mathbb{H}_3\f$ by multiplying the 1D stencils along
 * each direction separately
 * Supports a constant weight function of the form w(x,y,z) = w
 *
 * Boundary conditions are periodic in all directions
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * @tparam hodgeDegree int, degree of basis functions used
 * @tparam hodgeScheme HodgeScheme, determines the stencils used for the Hodge operator
 * @param threeForm DeRhamComplex::Primal3Form, 3-form \f$u^3\f$
 * @param zeroForm DeRhamComplex::Dual0Form, resulting 0-form \f$\tilde{u}^0\f$
 * @param weight Real, constant weight function
 */
template <int hodgeDegree, HodgeScheme hodgeScheme, DeRhamComplexC DRC = DeRhamComplex>
void hodge_operator (typename DRC::Dual0Form& zeroForm,
                     typename DRC::Primal3Form const& threeForm,
                     amrex::Real weight = 1.0)
{
    BL_PROFILE("hodge_operator(primal,cell,dual,node,weight)");
    // Select stencil according to degree
    auto [stencilNodeToCell, stencilCellToNode] = get_hodge_stencils<hodgeDegree, hodgeScheme>();

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dr{threeForm.m_deRham->get_dx()};
    amrex::Geometry const geom = threeForm.m_deRham->m_geom;

#if AMREX_SPACEDIM > 1
    amrex::MultiFab& tempDualZeroForm = zeroForm.m_deRham->get_temp_dual_zero_form();
#endif

    AMREX_D_TERM(
        (apply_1d_hodge<xDir, hodgeDegree>(
            geom, stencilCellToNode, threeForm.m_data,
            AMREX_D_PICK(zeroForm.m_data, tempDualZeroForm, zeroForm.m_data)));
        , (apply_1d_hodge<yDir, hodgeDegree>(
              geom, stencilCellToNode, AMREX_D_PICK((void), tempDualZeroForm, zeroForm.m_data),
              AMREX_D_PICK((void), zeroForm.m_data, tempDualZeroForm)));
        , (apply_1d_hodge<zDir, hodgeDegree>(geom, stencilCellToNode, tempDualZeroForm,
                                             zeroForm.m_data)););

    zeroForm.m_data.mult(weight / GEMPIC_D_MULT(dr[xDir], dr[yDir], dr[zDir]));

    zeroForm.fill_boundary();
}

/**
 * @brief Computes the Finite Difference Hodge \f$\tilde{\mathbb{H}}_2\f$.
 *
 * @details
 * Computes \f$\tilde{\mathbb{H}}_2\f$ by multiplying the 1D stencils along
 * each direction separately
 * Supports a constant weight function of the form w(x,y,z) = w
 *
 * Boundary conditions are periodic in all directions
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * @tparam hodgeDegree int, degree of basis functions used
 * @tparam hodgeScheme HodgeScheme, determines the stencils used for the Hodge operator
 * @param twoForm DRC::Dual2Form, 2-form \f$\tilde{u}^2\f$
 * @param oneForm DRC::Primal1Form, resulting 1-form \f$u^1\f$
 * @param weight Real, constant weight function
 */
template <int hodgeDegree, HodgeScheme hodgeScheme, DeRhamComplexC DRC = DeRhamComplex>
void hodge_operator (typename DRC::Primal1Form& oneForm,
                     typename DRC::Dual2Form const& twoForm,
                     amrex::Real weight = 1.0)
{
    BL_PROFILE("hodge_operator(dual,face,primal,edge,weight)");
    // Select stencil according to degree
    auto [stencilNodeToCell, stencilCellToNode] = get_hodge_stencils<hodgeDegree, hodgeScheme>();

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dr{twoForm.m_deRham->get_dx()};
    amrex::Geometry const geom = twoForm.m_deRham->m_geom;

#if AMREX_SPACEDIM > 1
    amrex::Array<amrex::MultiFab, 3>& tempPrimalOneForm =
        oneForm.m_deRham->get_temp_primal_one_form();
#endif

    for (int comp = 0; comp < 3; ++comp)
    {
        AMREX_D_TERM((apply_1d_hodge<xDir, hodgeDegree>(
            geom, (comp == xDir ? stencilNodeToCell : stencilCellToNode), twoForm.m_data[comp],
            AMREX_D_PICK(oneForm.m_data[comp], tempPrimalOneForm[comp], oneForm.m_data[comp])));
                     , (apply_1d_hodge<yDir, hodgeDegree>(
                           geom, (comp == yDir ? stencilNodeToCell : stencilCellToNode),
                           AMREX_D_PICK((void), tempPrimalOneForm[comp], oneForm.m_data[comp]),
                           AMREX_D_PICK((void), oneForm.m_data[comp], tempPrimalOneForm[comp])));
                     , (apply_1d_hodge<zDir, hodgeDegree>(
                           geom, (comp == zDir ? stencilNodeToCell : stencilCellToNode),
                           tempPrimalOneForm[comp], oneForm.m_data[comp])););

        if (comp == xDir)
        {
            oneForm.m_data[comp].mult(dr[xDir] * weight / GEMPIC_D_MULT(1, dr[yDir], dr[zDir]));
        }
        if (comp == yDir)
        {
            oneForm.m_data[comp].mult(GEMPIC_D_MULT(1, dr[yDir], 1) * weight /
                                      GEMPIC_D_MULT(dr[xDir], 1, dr[zDir]));
        }
        if (comp == zDir)
        {
            oneForm.m_data[comp].mult(GEMPIC_D_MULT(1, 1, dr[zDir]) * weight /
                                      GEMPIC_D_MULT(dr[xDir], dr[yDir], 1));
        }
    }

    oneForm.fill_boundary();
}

/**
 * @brief Computes the Finite Difference Hodge \f$\mathbb{H}_2\f$.
 *
 * @details
 * Computes \f$\mathbb{H}_2\f$ by multiplying the 1D stencils along
 * each direction separately
 * Supports a constant weight function of the form w(x,y,z) = w
 *
 * Boundary conditions are periodic in all directions
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * Parameters:
 * @tparam hodgeDegree int, degree of basis functions used
 * @tparam hodgeScheme HodgeScheme, determines the stencils used for the Hodge operator
 * @param twoForm DRC::Primal2Form, 2-form \f$u^2\f$
 * @param oneForm DRC::Dual1Form, resulting 1-form \f$\tilde{u}^1\f$
 * @param weight Real, constant weight function
 */
template <int hodgeDegree, HodgeScheme hodgeScheme, DeRhamComplexC DRC = DeRhamComplex>
void hodge_operator (typename DRC::Dual1Form& oneForm,
                     typename DRC::Primal2Form const& twoForm,
                     amrex::Real weight = 1.0)
{
    BL_PROFILE("hodge_operator(primal,face,dual,edge,weight)");
    // select stencil according to degree
    auto [stencilNodeToCell, stencilCellToNode] = get_hodge_stencils<hodgeDegree, hodgeScheme>();

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dr{twoForm.m_deRham->get_dx()};
    amrex::Geometry const geom = twoForm.m_deRham->m_geom;

#if AMREX_SPACEDIM > 1
    amrex::Array<amrex::MultiFab, 3>& tempDualOneForm = oneForm.m_deRham->get_temp_dual_one_form();
#endif

    for (int comp = 0; comp < 3; ++comp)
    {
        AMREX_D_TERM((apply_1d_hodge<xDir, hodgeDegree>(
            geom, (comp == xDir ? stencilNodeToCell : stencilCellToNode), twoForm.m_data[comp],
            AMREX_D_PICK(oneForm.m_data[comp], tempDualOneForm[comp], oneForm.m_data[comp])));
                     , (apply_1d_hodge<yDir, hodgeDegree>(
                           geom, (comp == yDir ? stencilNodeToCell : stencilCellToNode),
                           AMREX_D_PICK((void), tempDualOneForm[comp], oneForm.m_data[comp]),
                           AMREX_D_PICK((void), oneForm.m_data[comp], tempDualOneForm[comp])));
                     , (apply_1d_hodge<zDir, hodgeDegree>(
                           geom, (comp == zDir ? stencilNodeToCell : stencilCellToNode),
                           tempDualOneForm[comp], oneForm.m_data[comp])););

        if (comp == xDir)
        {
            oneForm.m_data[comp].mult(dr[xDir] * weight / GEMPIC_D_MULT(1, dr[yDir], dr[zDir]));
        }
        if (comp == yDir)
        {
            oneForm.m_data[comp].mult(GEMPIC_D_MULT(1, dr[yDir], 1) * weight /
                                      GEMPIC_D_MULT(dr[xDir], 1, dr[zDir]));
        }
        if (comp == zDir)
        {
            oneForm.m_data[comp].mult(GEMPIC_D_MULT(1, 1, dr[zDir]) * weight /
                                      GEMPIC_D_MULT(dr[xDir], dr[yDir], 1));
        }
    }

    oneForm.fill_boundary();
}

/**
 * @brief Computes the Finite Difference Hodge \f$\tilde{\mathbb{H}}_1\f$.
 *
 * @details
 * Computes \f$\tilde{\mathbb{H}}_1\f$ by multiplying the 1D stencils along
 * each direction separately
 * Supports a constant weight function of the form w(x,y,z) = w
 *
 * Boundary conditions are periodic in all directions
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * @tparam hodgeDegree int, degree of basis functions used
 * @tparam hodgeScheme HodgeScheme, determines the stencils used for the Hodge operator
 * @param oneForm DRC::Dual1Form, 1-form \f$\tilde{u}^1\f$
 * @param twoForm DRC::Primal2Form, resulting 2-form \f$u^2\f$
 * @param weight Real, constant weight function
 */
template <int hodgeDegree, HodgeScheme hodgeScheme, DeRhamComplexC DRC = DeRhamComplex>
void hodge_operator (typename DRC::Primal2Form& twoForm,
                     typename DRC::Dual1Form const& oneForm,
                     amrex::Real weight = 1.0)
{
    BL_PROFILE("hodge_operator(dual,edge,primal,face,weight)");
    // Select stencil according to degree
    auto [stencilNodeToCell, stencilCellToNode] = get_hodge_stencils<hodgeDegree, hodgeScheme>();

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dr{oneForm.m_deRham->get_dx()};
    amrex::Geometry const geom = oneForm.m_deRham->m_geom;

#if AMREX_SPACEDIM > 1
    amrex::Array<amrex::MultiFab, 3>& tempPrimalTwoForm =
        twoForm.m_deRham->get_temp_primal_two_form();
#endif

    for (int comp = 0; comp < 3; ++comp)
    {
        AMREX_D_TERM((apply_1d_hodge<xDir, hodgeDegree>(
            geom, (comp == xDir ? stencilCellToNode : stencilNodeToCell), oneForm.m_data[comp],
            AMREX_D_PICK(twoForm.m_data[comp], tempPrimalTwoForm[comp], twoForm.m_data[comp])));
                     , (apply_1d_hodge<yDir, hodgeDegree>(
                           geom, (comp == yDir ? stencilCellToNode : stencilNodeToCell),
                           AMREX_D_PICK((void), tempPrimalTwoForm[comp], twoForm.m_data[comp]),
                           AMREX_D_PICK((void), twoForm.m_data[comp], tempPrimalTwoForm[comp])));
                     , (apply_1d_hodge<zDir, hodgeDegree>(
                           geom, (comp == zDir ? stencilCellToNode : stencilNodeToCell),
                           tempPrimalTwoForm[comp], twoForm.m_data[comp])););

        if (comp == xDir)
        {
            twoForm.m_data[comp].mult((GEMPIC_D_MULT(1, dr[yDir], dr[zDir])) * weight /
                                      GEMPIC_D_MULT(dr[xDir], 1, 1));
        }
        if (comp == yDir)
        {
            twoForm.m_data[comp].mult((GEMPIC_D_MULT(dr[xDir], 1, dr[zDir])) * weight /
                                      GEMPIC_D_MULT(1, dr[yDir], 1));
        }
        if (comp == zDir)
        {
            twoForm.m_data[comp].mult((GEMPIC_D_MULT(dr[xDir], dr[yDir], 1)) * weight /
                                      GEMPIC_D_MULT(1, 1, dr[zDir]));
        }
    }

    twoForm.fill_boundary();
}

/**
 * @brief Computes the Finite Difference Hodge \f$\mathbb{H}_1\f$.
 *
 * @details
 * Computes \f$\mathbb{H}_1\f$ by multiplying the 1D stencils along
 * each direction separately
 * Supports a constant weight function of the form w(x,y,z) = w
 *
 * Boundary conditions are periodic in all directions
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * Parameters:
 * @tparam hodgeDegree int, degree of basis functions used
 * @tparam hodgeScheme HodgeScheme, determines the stencils used for the Hodge operator
 * @param oneForm DRC::Primal1Form, 1-form \f$u^1\f$
 * @param twoForm DRC::Dual2Form, resulting 2-form \f$\tilde{u}^2\f$
 * @param weight Real, constant weight function
 */
template <int hodgeDegree, HodgeScheme hodgeScheme, DeRhamComplexC DRC = DeRhamComplex>
void hodge_operator (typename DRC::Dual2Form& twoForm,
                     typename DRC::Primal1Form const& oneForm,
                     amrex::Real weight = 1.0)
{
    BL_PROFILE("hodge_operator(primal,edge,dual,face,weight)");
    // Select stencil according to degree
    auto [stencilNodeToCell, stencilCellToNode] = get_hodge_stencils<hodgeDegree, hodgeScheme>();

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dr{oneForm.m_deRham->get_dx()};
    amrex::Geometry const geom = oneForm.m_deRham->m_geom;

#if AMREX_SPACEDIM > 1
    amrex::Array<amrex::MultiFab, 3>& tempDualTwoForm = twoForm.m_deRham->get_temp_dual_two_form();
#endif

    for (int comp = 0; comp < 3; ++comp)
    {
        AMREX_D_TERM((apply_1d_hodge<xDir, hodgeDegree>(
            geom, (comp == xDir ? stencilCellToNode : stencilNodeToCell), oneForm.m_data[comp],
            AMREX_D_PICK(twoForm.m_data[comp], tempDualTwoForm[comp], twoForm.m_data[comp])));
                     , (apply_1d_hodge<yDir, hodgeDegree>(
                           geom, (comp == yDir ? stencilCellToNode : stencilNodeToCell),
                           AMREX_D_PICK((void), tempDualTwoForm[comp], twoForm.m_data[comp]),
                           AMREX_D_PICK((void), twoForm.m_data[comp], tempDualTwoForm[comp])));
                     , (apply_1d_hodge<zDir, hodgeDegree>(
                           geom, (comp == zDir ? stencilCellToNode : stencilNodeToCell),
                           tempDualTwoForm[comp], twoForm.m_data[comp])););

        if (comp == xDir)
        {
            twoForm.m_data[comp].mult((GEMPIC_D_MULT(1, dr[yDir], dr[zDir])) * weight /
                                      GEMPIC_D_MULT(dr[xDir], 1, 1));
        }
        if (comp == yDir)
        {
            twoForm.m_data[comp].mult((GEMPIC_D_MULT(dr[xDir], 1, dr[zDir])) * weight /
                                      GEMPIC_D_MULT(1, dr[yDir], 1));
        }
        if (comp == zDir)
        {
            twoForm.m_data[comp].mult((GEMPIC_D_MULT(dr[xDir], dr[yDir], 1)) * weight /
                                      GEMPIC_D_MULT(1, 1, dr[zDir]));
        }
    }

    twoForm.fill_boundary();
}

/**
 * @brief Computes the Finite Difference Hodge \f$\tilde{\mathbb{H}}_0\f$.
 *
 * @details
 * Computes \f$\tilde{\mathbb{H}}_0\f$ by multiplying the 1D stencils along
 * each direction separately
 * Supports a constant weight function of the form w(x,y,z) = w
 *
 * Boundary conditions are periodic in all directions
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * Parameters:
 * @tparam hodgeDegree int, degree of basis functions used
 * @tparam hodgeScheme HodgeScheme, determines the stencils used for the Hodge operator
 * @param zeroForm DRC::Dual0Form, 0-form \f$\tilde{u}^0\f$
 * @param threeForm DRC::Primal3Form, resulting 3-form \f$u^3\f$
 * @param weight Real, constant weight function
 */
template <int hodgeDegree, HodgeScheme hodgeScheme, DeRhamComplexC DRC = DeRhamComplex>
void hodge_operator (typename DRC::Primal3Form& threeForm,
                     typename DRC::Dual0Form const& zeroForm,
                     amrex::Real weight = 1.0)
{
    BL_PROFILE("hodge_operator(dual,node,primal,cell,weight)");
    // Select stencil according to degree
    auto [stencilNodeToCell, stencilCellToNode] = get_hodge_stencils<hodgeDegree, hodgeScheme>();

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dr{zeroForm.m_deRham->get_dx()};
    amrex::Geometry const geom = zeroForm.m_deRham->m_geom;

#if AMREX_SPACEDIM > 1
    amrex::MultiFab& tempPrimalThreeForm = threeForm.m_deRham->get_temp_primal_three_form();
#endif

    AMREX_D_TERM(
        (apply_1d_hodge<xDir, hodgeDegree>(
            geom, stencilNodeToCell, zeroForm.m_data,
            AMREX_D_PICK(threeForm.m_data, tempPrimalThreeForm, threeForm.m_data)));
        , (apply_1d_hodge<yDir, hodgeDegree>(
              geom, stencilNodeToCell, AMREX_D_PICK((void), tempPrimalThreeForm, threeForm.m_data),
              AMREX_D_PICK((void), threeForm.m_data, tempPrimalThreeForm)));
        , (apply_1d_hodge<zDir, hodgeDegree>(geom, stencilNodeToCell, tempPrimalThreeForm,
                                             threeForm.m_data)););

    threeForm.m_data.mult(GEMPIC_D_MULT(dr[xDir], dr[yDir], dr[zDir]) * weight);

    threeForm.fill_boundary();
}

/**
 * @brief Computes the Finite Difference Hodge \f$\mathbb{H}_0\f$.
 *
 * @details
 * Computes \f$\mathbb{H}_0\f$ by multiplying the 1D stencils along
 * each direction separately
 * Supports a constant weight function of the form w(x,y,z) = w
 *
 * Boundary conditions are periodic in all directions
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * Parameters:
 * @tparam hodgeDegree int, degree of basis functions used
 * @tparam hodgeScheme HodgeScheme, determines the stencils used for the Hodge operator
 * @param zeroForm DRC::Primal0Form, 0-form \f$u^0\f$
 * @param threeForm DRC::Dual3Form, resulting 3-form \f$\tilde{u}^3\f$
 * @param weight Real, constant weight function
 */
template <int hodgeDegree, HodgeScheme hodgeScheme, DeRhamComplexC DRC = DeRhamComplex>
void hodge_operator (typename DRC::Dual3Form& threeForm,
                     typename DRC::Primal0Form const& zeroForm,
                     amrex::Real weight = 1.0)
{
    BL_PROFILE("hodge_operator(primal,node,dual,cell,weight)");
    // Select stencil according to degree
    auto [stencilNodeToCell, stencilCellToNode] = get_hodge_stencils<hodgeDegree, hodgeScheme>();

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dr{zeroForm.m_deRham->get_dx()};
    amrex::Geometry const geom = zeroForm.m_deRham->m_geom;

#if AMREX_SPACEDIM > 1
    amrex::MultiFab& tempDualThreeForm = threeForm.m_deRham->get_temp_dual_three_form();
#endif

    AMREX_D_TERM(
        (apply_1d_hodge<xDir, hodgeDegree>(
            geom, stencilNodeToCell, zeroForm.m_data,
            AMREX_D_PICK(threeForm.m_data, tempDualThreeForm, threeForm.m_data)));
        , (apply_1d_hodge<yDir, hodgeDegree>(
              geom, stencilNodeToCell, AMREX_D_PICK((void), tempDualThreeForm, threeForm.m_data),
              AMREX_D_PICK((void), threeForm.m_data, tempDualThreeForm)));
        , (apply_1d_hodge<zDir, hodgeDegree>(geom, stencilNodeToCell, tempDualThreeForm,
                                             threeForm.m_data)););

    threeForm.m_data.mult(GEMPIC_D_MULT(dr[xDir], dr[yDir], dr[zDir]) * weight);

    threeForm.fill_boundary();
}

/**
 * @brief Selects a precompiled Hodge operator from a runtime Hodge degree.
 *
 * Recursively calls itself until the runtime Hodge degree == validDegrees[hodgeDegreeIdx],
 * then calls hodge<hodgeDegree, hodgeScheme>.
 *
 * @tparam FieldRef type of the destination Form (parameter deduction)
 * @tparam FieldConstRef type of the source Form (parameter deduction)
 * @tparam hodgeScheme    Hodge scheme as found by hodge_scheme_selector
 * @tparam hodgeDegreeIdx Not the Hodge degree! The index of the validDegrees array
 * corresponding to the Hodge scheme.
 *
 * @param hodgeSrc const source field
 * @param hodgeDst destination field
 * @param weight optional weight for the hodge
 */
template <HodgeScheme hodgeScheme,
          typename DstFormType,
          typename SrcFormType,
          int hodgeDegreeIdx = 0>
    requires MutualDual<SrcFormType, DstFormType>
void hodge_degree_selector (DstFormType& dstForm, SrcFormType& srcForm, amrex::Real weight)
{
    using DRC = field_to_drc_t<SrcFormType>;
    BL_PROFILE("hodge_degree_selector()")
    AMREX_ASSERT(srcForm.m_deRham->get_hodge_degree() == dstForm.m_deRham->get_hodge_degree());
    constexpr auto validDegreesArray = valid_degrees<hodgeScheme>();
    if constexpr (hodgeDegreeIdx < validDegreesArray.size())
    {
        if (srcForm.m_deRham->get_hodge_degree() == validDegreesArray[hodgeDegreeIdx])
        {
            hodge_operator<validDegreesArray[hodgeDegreeIdx], hodgeScheme, DRC>(dstForm, srcForm,
                                                                                weight);
        }
        else
        {
            hodge_degree_selector<hodgeScheme, DstFormType, SrcFormType, hodgeDegreeIdx + 1>(
                dstForm, srcForm, weight);
        }
    }
    else
    {
        GEMPIC_ERROR("Incompatible Hodge degree and Hodge scheme: No such degree " +
                     std::to_string(static_cast<int>(srcForm.m_deRham->get_hodge_scheme())) +
                     " for selected Hodge scheme");
    }
}

/**
 * @brief Selects a precompiled Hodge operator scheme from a runtime scheme parameter
 *
 * @tparam DstFormType type of the destination form (parameter deduction)
 * @tparam SrcFormType type of the source form (parameter deduction)
 * @param f2 destination form
 * @param f1 const source form
 * @param weight optional weight for the Hodge
 */
template <typename DstFormType, typename SrcFormType>
    requires MutualDual<SrcFormType, DstFormType> && (!HasHodgeScheme<field_to_drc_t<SrcFormType>>)
void hodge(DstFormType& dstForm, SrcFormType const& srcForm, amrex::Real weight = 1.)
{
    BL_PROFILE("hodge()");
    AMREX_ASSERT(dstForm.m_deRham->get_hodge_scheme() == srcForm.m_deRham->get_hodge_scheme());
    // Access m_deRham->m_hodgeScheme safely
    switch (dstForm.m_deRham->get_hodge_scheme())
    {
        case HodgeScheme::FDHodge:
            hodge_degree_selector<HodgeScheme::FDHodge>(dstForm, srcForm, weight);
            break;
        case HodgeScheme::GDECHodge:
            hodge_degree_selector<HodgeScheme::GDECHodge>(dstForm, srcForm, weight);
            break;
        case HodgeScheme::GDECLumpHodge:
            hodge_degree_selector<HodgeScheme::GDECLumpHodge>(dstForm, srcForm, weight);
            break;
        default:
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(false, "Unknown Hodge scheme");
    }
}

/**
 * @brief Computes the \f$formOut=\mathbb{H}_{\boldsymbol M} formIn\f$.
 *
 * @details
 * Computes \f$formOut={\boldsymbol M}*weight*formIn\f$
 * tensor matrix M is the same de Rham field as formOut
 * M is 3 by 3 matrix saved as a DeRhamField with 3 components for each of the three variables
 *
 * @tparam grid primal or dual
 * @tparam space edge or node, corresponding to 1- or 2-forms
 * @param formOut DeRhamField<grid, space>
 * @param formIn DeRhamField<oppositeGrid<grid>, oppositeSpace<space>>
 * @param matrix DeRhamField<grid, space>
 * @param weight weighting factor
 */
template <OneTwoFormC DstFormType, OneTwoFormC SrcFormType>
    requires MutualDual<SrcFormType, DstFormType>
inline void hodge_dk (DstFormType& formOut,
                      SrcFormType& formIn,
                      DstFormType& matrix,
                      amrex::Real weight = 1.)
{
    BL_PROFILE("Gempic::Forms::HodgeDK");
    GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE(formOut.m_data[xDir].nComp() == 1,
                                      "hodge_dk not implemented for multi-component fields");
    // Start by applying the standard Hodge operator. Needs to be stored in temporary multifab
    DstFormType tempFormOut(formOut.m_deRham);
    hodge(tempFormOut, formIn, weight); //get tempFormOut from formIn

    // Update x component assume corresponding multifab has only 1 component
    // y and z components need to be averaged to midpoint of x edge
    for (amrex::MFIter mfi(formOut.m_data[xDir], true); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const& formx = (tempFormOut.m_data[xDir])[mfi].array();
        amrex::Array4<amrex::Real> const& formy = (tempFormOut.m_data[yDir])[mfi].array();
        amrex::Array4<amrex::Real> const& formz = (tempFormOut.m_data[zDir])[mfi].array();
        amrex::Array4<amrex::Real> const& matrixX = matrix.m_data[xDir][mfi].array();
        amrex::Array4<amrex::Real> const& formOutX = (formOut.m_data[xDir])[mfi].array();

        ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        formOutX(i, j, k) = matrixX(i, j, k, xDir) * formx(i, j, k) +
                                            matrixX(i, j, k, yDir) * 0.25 *
                                                (formy(i - 1, j, k) + formy(i, j, k) +
                                                 formy(i - 1, j + 1, k) + formy(i, j + 1, k)) +
                                            matrixX(i, j, k, zDir) * 0.25 *
                                                (formz(i - 1, j, k) + formz(i, j, k) +
                                                 formz(i - 1, j, k) + formz(i, j, k + 1));
                    });
    }
    // Update y component assume corresponding multifab has only 1 component
    // x and z components need to be averaged to midpoint of y edge
    for (amrex::MFIter mfi(formOut.m_data[yDir], true); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const& formx = tempFormOut.m_data[xDir][mfi].array();
        amrex::Array4<amrex::Real> const& formy = tempFormOut.m_data[yDir][mfi].array();
        amrex::Array4<amrex::Real> const& formz = tempFormOut.m_data[zDir][mfi].array();
        amrex::Array4<amrex::Real> const& matrixY = matrix.m_data[yDir][mfi].array();
        amrex::Array4<amrex::Real> const& formOutY = formOut.m_data[yDir][mfi].array();

        ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        formOutY(i, j, k) = matrixY(i, j, k, xDir) * 0.25 *
                                                (formx(i, j - 1, k) + formx(i, j, k) +
                                                 formx(i + 1, j - 1, k) + formx(i + 1, j, k)) +
                                            matrixY(i, j, k, yDir) * formy(i, j, k) +
                                            matrixY(i, j, k, zDir) * 0.25 *
                                                (formz(i, j, k) + formz(i, j - 1, k) +
                                                 formz(i, j, k + 1) + formz(i, j - 1, k + 1));
                    });
    }
    // Update z component assume corresponding multifab has only 1 component
    // x and y components need to be averaged to midpoint of z edge
    for (amrex::MFIter mfi(formOut.m_data[zDir], true); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const& formx = (tempFormOut.m_data[xDir])[mfi].array();
        amrex::Array4<amrex::Real> const& formy = (tempFormOut.m_data[yDir])[mfi].array();
        amrex::Array4<amrex::Real> const& formz = (tempFormOut.m_data[zDir])[mfi].array();
        amrex::Array4<amrex::Real> const& matrixZ = matrix.m_data[zDir][mfi].array();
        amrex::Array4<amrex::Real> const& formOutZ = (formOut.m_data[zDir])[mfi].array();

        ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        formOutZ(i, j, k) = matrixZ(i, j, k, xDir) * 0.25 *
                                                (formx(i, j, k - 1) + formx(i, j, k) +
                                                 formx(i + 1, j, k - 1) + formx(i + 1, j, k)) +
                                            matrixZ(i, j, k, yDir) * 0.25 *
                                                (formy(i, j, k) + formy(i, j, k - 1) +
                                                 formy(i, j + 1, k) + formy(i, j + 1, k - 1)) +
                                            matrixZ(i, j, k, zDir) * formz(i, j, k);
                    });
    }
    formOut.fill_boundary();
}

/**
 * @brief Selects a precompiled eval_form operator from a runtime Hodge degree.
 *
 * Recursively calls itself until the runtime Hodge degree == validDegrees[hodgeDegreeIdx],
 * then calls
 *   eval_form_component<degree, srcFormType> for 0-/3-forms, or
 *   eval_form_component<degree, srcFormType, dir> for 1-/2-forms
 *
 * @tparam dir            Optional direction as found by eval_form_direction_selector
 * @tparam hodgeDegreeIdx Not the Hodge degree! The index of the validDegrees array
 * corresponding to the Hodge scheme. NOTE: Only Finite Difference Hodge schemes work for now.
 *
 * @param field const source field
 * @param evalShift vector of shifts from lower corner of primal cell (between 0 and 1)
 * @param pointValues Multifab in which to store the interpolated values
 */
template <typename SrcFormType, int testDegInd = 0>
void eval_form_degree_selector (
    SrcFormType& field,
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> const& evalShift,
    amrex::MultiFab& pointValues)
{
    BL_PROFILE("eval_form_degree_selector()")
    constexpr auto validDegreesArray = valid_degrees<HodgeScheme::FDHodge>();
    if constexpr (testDegInd < validDegreesArray.size())
    {
        if (field.m_deRham->get_hodge_degree() == validDegreesArray[testDegInd])
        {
            constexpr int degree = validDegreesArray[testDegInd];
            // 0-/3-form call
            eval_form_component<degree, SrcFormType>(
                evalShift, field, pointValues, field.m_deRham->get_dx(), field.m_deRham->m_geom);
        }
        else
        {
            eval_form_degree_selector<SrcFormType, testDegInd + 1>(field, evalShift, pointValues);
        }
    }
    else
    {
        GEMPIC_ERROR("Incompatible Hodge degree and eval_form: No such degree " +
                     std::to_string(field.m_deRham->get_hodge_degree()) + " for eval_form");
    }
}

template <Direction dir, typename SrcFormType, int testDegInd = 0>
void eval_form_degree_selector (
    SrcFormType& field,
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> const& evalShift,
    amrex::MultiFab& pointValues)
{
    BL_PROFILE("eval_form_degree_selector()")
    constexpr auto validDegreesArray = valid_degrees<HodgeScheme::FDHodge>();
    if constexpr (testDegInd < validDegreesArray.size())
    {
        if (field.m_deRham->get_hodge_degree() == validDegreesArray[testDegInd])
        {
            constexpr int degree = validDegreesArray[testDegInd];
            // 1-/2-form call
            eval_form_component<degree, SrcFormType, dir>(
                evalShift, field, pointValues, field.m_deRham->get_dx(), field.m_deRham->m_geom);
        }
        else
        {
            eval_form_degree_selector<dir, SrcFormType, testDegInd + 1>(field, evalShift,
                                                                        pointValues);
        }
    }
    else
    {
        GEMPIC_ERROR("Incompatible Hodge degree and eval_form: No such degree " +
                     std::to_string(field.m_deRham->get_hodge_degree()) + " for eval_form");
    }
}

/**
 * @brief Selects a precompiled direction for eval_form from a runtime direction parameter
 *
 * @param field const source field
 * @param evalShift vector of shifts from lower corner of primal cell (between 0 and 1)
 * @param pointValues Multifab in which to store the interpolated values
 * @param dir Direction in which to evaluate the interpolation(s)
 */

template <typename SrcFormType>
inline void eval_form_direction_selector (
    SrcFormType& field,
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> const& evalShift,
    amrex::MultiFab& pointValues,
    Direction dir)
{
    switch (dir)
    {
        case xDir:
            eval_form_degree_selector<xDir>(field, evalShift, pointValues);
            break;
        case yDir:
            eval_form_degree_selector<yDir>(field, evalShift, pointValues);
            break;
        case zDir:
            eval_form_degree_selector<zDir>(field, evalShift, pointValues);
            break;
        default:
            AMREX_ALWAYS_ASSERT("Unrecognized direction passed to eval_form");
            break;
    }
}

/**
 * @brief Evaluation of a 0-/3-form DeRhamField of nVar variables in
 *        each cell given by a positive normalized shift from the lower corner of the cell
 * @param srcForm     The field to evaluate
 * @param evalShift   The time step at which to evaluate the functions
 * @param pointValues 3D output array of cell-centered MultiFabs each containing nVar * nEvals
 *                    components in the order
 *                    var=0 eval=0,
 *                    var=0 eval=1,
 *                    ...,
 *                    var=0 shift=nEvals,
 *                    var=1 shift=0,
 *                    ...,
 *                    var=nVar shift=nEvals
 */
template <ZeroThreeFormC SrcFormType>
void eval_form (SrcFormType const& srcForm,
                amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> const& evalShift,
                amrex::MultiFab& pointValues)
{
    eval_form_degree_selector(srcForm, evalShift, pointValues);
}

/**
 * @brief Evaluation of the dir-th component of a 1-/2-form DeRhamField of nVar variables in
 *        each cell given by a positive normalized shift from the lower corner of the cell
 * @param srcForm     The field to evaluate
 * @param evalShift   The time step at which to evaluate the functions
 * @param pointValues 3D output array of cell-centered MultiFabs each containing nVar * nEvals
 *                    components in the order
 *                    var=0 eval=0,
 *                    var=0 eval=1,
 *                    ...,
 *                    var=0 shift=nEvals,
 *                    var=1 shift=0,
 *                    ...,
 *                    var=nVar shift=nEvals
 * @param dir         The component to evaluate
 */
template <OneTwoFormC SrcFormType>
void eval_form (SrcFormType const& srcForm,
                amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> const& evalShift,
                amrex::MultiFab& pointValues,
                Direction dir)
{
    eval_form_direction_selector(srcForm, evalShift, pointValues, dir);
}

/**
 * @brief Evaluation of a 1-/2-form DeRhamField of nVar variables (and three components) in
 *        each cell given by a positive normalized shift from the lower corner of the cell
 * @param srcForm     The field to evaluate
 * @param evalShift   The time step at which to evaluate the functions
 * @param pointValues 3D output array of cell-centered MultiFabs each containing nVar * nEvals
 *                    components in the order
 *                    var=0 eval=0,
 *                    var=0 eval=1,
 *                    ...,
 *                    var=0 shift=nEvals,
 *                    var=1 shift=0,
 *                    ...,
 *                    var=nVar shift=nEvals
 */
template <OneTwoFormC SrcFormType>
void eval_form (SrcFormType const& srcForm,
                amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> const& evalShift,
                amrex::Array<amrex::MultiFab*, 3>& pointValues)
{
    eval_form(srcForm, evalShift, *pointValues[xDir], xDir);
    eval_form(srcForm, evalShift, *pointValues[yDir], yDir);
    eval_form(srcForm, evalShift, *pointValues[zDir], zDir);
}

} //namespace Gempic::Forms

#endif