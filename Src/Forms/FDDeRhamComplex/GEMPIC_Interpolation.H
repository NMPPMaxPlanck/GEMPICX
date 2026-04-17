/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef GEMPIC_INTERPOLATION_H_
#define GEMPIC_INTERPOLATION_H_

#include <AMReX_SPACE.H>
#include <AMReX_Vector.H>

#include "GEMPIC_Config.H"
#include "GEMPIC_DeRhamComplex.H"
#include "GEMPIC_FieldTools.H"
#include "GEMPIC_Fields.H"

namespace Gempic::Forms
{
/**
 * @brief Evaluates a 1D Lagrange polynomial
 *
 * @details
 * Evaluates a Lagrange polynomial \f$L_i\f$ on the interval \f$[x_k, x_{k+1}]\f$, where \f$L_0\f$
 * is the Lagrange polynomial associated to node \f$x_k\f$, \f$L_{-1}\f$ to \f$x_{k-1}\f$, \f$L_1\f$
 * to \f$x_{k+1}\f$, and so forth. The whole grid is assumed to be uniform, evaluation is done using
 * Neville's algorithm.
 *
 * @tparam nNodes int, number of interpolation nodes used. (intDegree=nNodes-1: degree of
 * interpolating polynomials)
 * @param polyIndex int, indicates the polynomial w.r.t. \f$x_k\f$
 * @param xk Real, node \f$x_k\f$ which defines the interval of evaluation
 * @param dx (uniform) grid size
 * @param y Real, point in \f$[x_k, x_{k+1}]\f$, where \f$L_i\f$ is to be evaluated
 *
 * @return \f$L_{polyIndex} (y)\f$
 */
template <int nNodes>
AMREX_GPU_HOST_DEVICE amrex::Real eval_lagrange (int const polyIndex,
                                                 amrex::Real const xk,
                                                 amrex::Real const dx,
                                                 amrex::Real const y)
{
    constexpr int sigma{nNodes / 2 - 1}; // memo: (hodgeDegree = nIntNodes = 2 sigma + 2 )
    static_assert(nNodes % 2 == 0, "nNodes must be EVEN");
    AMREX_ASSERT_WITH_MESSAGE(polyIndex >= -sigma && polyIndex <= sigma + 1,
                              "polyIndex must be in [-n, n+1]");
    AMREX_ASSERT_WITH_MESSAGE(y >= xk && y <= xk + dx, "y must be in [xk, xk+dx]");
    amrex::GpuArray<amrex::Real, nNodes> vals;
    amrex::GpuArray<amrex::Real, nNodes> nodes;
    for (int j = 0; j < nNodes; ++j)
    {
        if (j == polyIndex + sigma)
        {
            vals[j] = 1;
        }
        else
        {
            vals[j] = 0;
        }
        nodes[j] = xk + (-sigma + j) * dx;
    }
    for (int i = 1; i <= nNodes - 1; ++i)
    {
        for (int j = 0; j < nNodes - i; ++j)
        {
            vals[j] = vals[j + 1] +
                      (y - nodes[j + i]) * (vals[j + 1] - vals[j]) / (nodes[j + i] - nodes[j]);
        }
    }
    return vals[0];
}
/**
 *  the same but on the reference grid xk=0,dx=1;
 *  the stencil nodes are then: nodes[]={-n,...,0,...,n+1}; these values correspond to the
 * respective polyIndex. note that: even number of nodes; odd interpolating polynomial degree.
 */
template <int nNodes>
AMREX_GPU_HOST_DEVICE amrex::Real eval_lagrange (int const polyIndex, amrex::Real const y)
{
    constexpr int sigma{nNodes / 2 - 1}; // memo: (hodgeDegree = nIntNodes = 2 sigma + 2 )
    static_assert(nNodes % 2 == 0, "nNodes must be EVEN");
    AMREX_ASSERT_WITH_MESSAGE(polyIndex >= -sigma && polyIndex <= sigma + 1,
                              "polyIndex must be in [-n, n+1]");
    AMREX_ASSERT_WITH_MESSAGE(y >= 0.0 && y <= 1.0, "y must be in [0, 1]");
    amrex::GpuArray<amrex::Real, nNodes> vals;
    amrex::GpuArray<amrex::Real, nNodes> nodes;
    for (int j = 0; j < nNodes; ++j)
    {
        if (j == polyIndex + sigma)
        {
            vals[j] = 1;
        }
        else
        {
            vals[j] = 0;
        }
        nodes[j] = (-sigma + j);
    }
    for (int i = 1; i <= nNodes - 1; ++i)
    {
        for (int j = 0; j < nNodes - i; ++j)
        {
            vals[j] = vals[j + 1] +
                      (y - nodes[j + i]) * (vals[j + 1] - vals[j]) / (nodes[j + i] - nodes[j]);
        }
    }
    return vals[0];
}

/**
 * @brief get the derivative of the Lagrange interpolating polynomial at point xi in [xk,xk+dx];
 *        the interpolating polynomials is defined on a local stencil: node indexes
 * {-n,...,0,...,n+1}; x_0 = xk;
 * @tparam intDegree
 * @param polyIndex in {-n,...,0,...,n+1} identifies the polynomial interplating 1 at the
 * polyIndex-th node of the stencil.
 * @param xk corresponds to the x_0 node of the stencil
 * @param dx
 * @param xi the evaluation point
 * @return: derivatives of polyIndex-th Lagrange basis functions evaluated at xi
 */
template <int intDegree>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real eval_lagrange_der (int const polyIndex,
                                                                        amrex::Real const xk,
                                                                        amrex::Real const dx,
                                                                        amrex::Real xi)
{
    constexpr int nNodes{intDegree + 1};
    constexpr int sigma{nNodes / 2 - 1}; // memo: (hodgeDegree = nIntNodes = 2 sigma + 2 )
    AMREX_ASSERT_WITH_MESSAGE(((intDegree - 1) % 2) == 0, "intDegree must be ODD");
    AMREX_ASSERT_WITH_MESSAGE(polyIndex >= -sigma && polyIndex <= sigma + 1,
                              "polyIndex must be in [-n, n+1]");
    AMREX_ASSERT_WITH_MESSAGE(xi >= xk && xi <= xk + dx, "xi must be in [xk, xk+dx]");
    amrex::Real phiXi{0.0};
    int n = (intDegree - 1) / 2;
    int m = polyIndex + n;
    amrex::GpuArray<amrex::Real, nNodes> stencilNodes;

    //Compute phiXi[m] : the derivative of the m - th Lagrange polynomial at xi
    for (int i = 0; i < nNodes; ++i)
    {
        stencilNodes[i] = xk + (-n + i) * dx;
    }
    for (int i = 0; i < nNodes; ++i)
    {
        if (i == m)
        {
            continue;
        }
        amrex::Real tmp{1.0};
        for (int j = 0; j < nNodes; ++j)
        {
            if (j == i || j == m)
            {
                continue;
            }
            tmp *= (xi - stencilNodes[j]) / (stencilNodes[m] - stencilNodes[j]);
        }
        phiXi += tmp / (stencilNodes[m] - stencilNodes[i]);
    }
    return phiXi;
}

/**
 * @brief get the derivative of the Lagrange interpolating polynomial at point xi in [0,1];
 *        the interpolating polynomials is defined on a local stencil: node indexes
 * {-n,...,0,...,n+1}; x_0 = 0;
 * @tparam intDegree degree of the Lagrange polyomial used
 * @param polyIndex in {-n,...,0,...,n+1} identifies the polynomial interplating 1 at the
 * polyIndex-th node of the stencil.
 * @param xi amrex::Real, the evaluation point
 * @return: derivatives of polyIndex-th Lagrange basis functions evaluated at xi
 */
template <int intDegree>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real eval_lagrange_der (int const polyIndex,
                                                                        amrex::Real xi)
{
    constexpr int nNodes{intDegree + 1};
    constexpr int sigma{nNodes / 2 - 1}; // memo: (hodgeDegree = nIntNodes = 2 sigma + 2 )
    AMREX_ASSERT_WITH_MESSAGE(((intDegree - 1) % 2) == 0, "intDegree must be ODD");
    AMREX_ASSERT_WITH_MESSAGE(polyIndex >= -sigma && polyIndex <= sigma + 1,
                              "polyIndex must be in [-n, n+1]");
    AMREX_ASSERT_WITH_MESSAGE(xi >= 0.0 && xi <= 1.0, "xi must be in [0, 1]");
    amrex::Real phiXi{0.0};
    int n = (intDegree - 1) / 2;
    int m = polyIndex + n;
    amrex::GpuArray<amrex::Real, nNodes> stencilNodes;

    //Compute phiXi[m] : the derivative of the m - th Lagrange polynomial at xi
    for (int i = 0; i < nNodes; ++i)
    {
        stencilNodes[i] = -n + i;
    }
    for (int i = 0; i < nNodes; ++i)
    {
        if (i == m)
        {
            continue;
        }
        amrex::Real tmp{1.0};
        for (int j = 0; j < nNodes; ++j)
        {
            if (j == i || j == m)
            {
                continue;
            }
            tmp *= (xi - stencilNodes[j]) / (stencilNodes[m] - stencilNodes[j]);
        }
        phiXi += tmp / (stencilNodes[m] - stencilNodes[i]);
    }
    return phiXi;
}

/**
 * @brief Evaluates a 1D edge function
 *
 * @details
 * Evaluates an edge function \f$e_i\f$ on the interval \f$[x_k, x_{k+1}]\f$, where \f$e_0\f$ is
 * the edge function associated to the cell \f$[x_k,  x_{k+1}]\f$, \f$e_{-1}\f$ to \f$[x_{k-1},
 * x_k]\f$, \f$e_1\f$ to \f$[x_{k+1},  x_{k+2}]\f$, and so forth. The whole grid is assumed to be
 * uniform.
 *
 * @tparam nNodes int, number of nodes used to build the stencil (nCells=nNodes-1: number of dofs
 * of the histopolating basis function)
 * @param polyIndex int, indicates the edge function w.r.t. \f$[x_k, x_{k+1}]\f$, in
 * {-n,...,0,...,n}
 * @param xk Real, node \f$x_k\f$ which defines the interval of evaluation
 * @param dx (uniform) grid size
 * @param xi Real, point in \f$[x_k, x_{k+1}]\f$, where \f$e_i\f$ is to be evaluated
 *
 * @return \f$e_{polyIndex} (y)\f$ the histopolating polynomial evaluated at xi
 */
// NOLINTBEGIN(readability-magic-numbers)

template <int intDegree>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real eval_edge_arbitrary_order (
    int const polyIndex, amrex::Real const xk, amrex::Real const dx, amrex::Real xi)
{
    constexpr int nNodes{intDegree + 1};
    constexpr int sigma{nNodes / 2 - 1}; // memo: (hodgeDegree = nIntNodes = 2 sigma + 2 )
    constexpr int polyIdx = 0 - sigma;   // i = polyIndex + n;
    AMREX_ASSERT_WITH_MESSAGE(((intDegree - 1) % 2) == 0, "intDegree must be ODD");
    AMREX_ASSERT_WITH_MESSAGE(polyIndex >= -sigma && polyIndex <= sigma,
                              "polyIndex must be in [-n, n]");
    AMREX_ASSERT_WITH_MESSAGE(xi >= xk && xi <= xk + dx, "y must be in [xk, xk+dx]");

    amrex::Real phiEdge{eval_lagrange_der<intDegree>(polyIdx, xk, dx, xi)};

    phiEdge *= -1.0;
    for (int polyIdxLoc = 1 - sigma; polyIdxLoc < polyIndex + 1; ++polyIdxLoc)
    {
        amrex::Real phiXiInterp{eval_lagrange_der<intDegree>(polyIdxLoc, xk, dx, xi)};
        phiEdge -= phiXiInterp;
    }

    return phiEdge;
}

/**
 * @brief Evaluates a 1D edge function on unitary grid
 *
 * @details
 * Evaluates an edge function \f$e_i\f$ on the interval \f$[x_k, x_{k+1}]\f$, where \f$e_0\f$ is
 * the edge function associated to the cell \f$[x_k,  x_{k+1}]\f$, \f$e_{-1}\f$ to \f$[x_{k-1},
 * x_k]\f$, \f$e_1\f$ to \f$[x_{k+1},  x_{k+2}]\f$, and so forth. The whole grid is assumed to be
 * uniform.
 *
 * @tparam nNodes int, number of nodes used to build the stencil (nCells=nNodes-1: number of dofs
 * of the histopolating basis function)
 * @param polyIndex int, indicates the edge function w.r.t. \f$[x_k, x_{k+1}]\f$, in
 * {-n,...,0,...,n}
 * @param xi Real, point in \f$[0, 1]\f$, where \f$e_i\f$ is to be evaluated
 *
 * @return \f$e_{polyIndex} (y)\f$ the histopolating polynomial evaluated at xi
 */
// NOLINTBEGIN(readability-magic-numbers)

template <int intDegree>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real eval_edge_arbitrary_order (int const polyIndex,
                                                                                amrex::Real xi)
{
    constexpr int nNodes{intDegree + 1};
    constexpr int sigma{nNodes / 2 - 1}; // memo: (hodgeDegree = nIntNodes = 2 sigma + 2 )
    constexpr int polyIdx = 0 - sigma;   // i = polyIndex + n;

    AMREX_ASSERT_WITH_MESSAGE(((intDegree - 1) % 2) == 0, "intDegree must be ODD");
    AMREX_ASSERT_WITH_MESSAGE(polyIndex >= -sigma && polyIndex <= sigma,
                              "polyIndex must be in [-n, n]");
    AMREX_ASSERT_WITH_MESSAGE(xi >= 0.0 && xi <= 1.0, "xi must be in [0, 1]");

    amrex::Real phiEdge{eval_lagrange_der<intDegree>(polyIdx, xi)};
    phiEdge *= -1.0;
    for (int polyIdxLoc = 1 - sigma; polyIdxLoc < polyIndex + 1; ++polyIdxLoc)
    {
        amrex::Real phiXiInterp{eval_lagrange_der<intDegree>(polyIdxLoc, xi)};
        phiEdge -= phiXiInterp;
    }

    return phiEdge;
}

/**
 * @brief Evaluates a 1D edge function
 *
 * @details
 * Evaluates an edge function \f$e_i\f$ on the interval \f$[x_k, x_{k+1}]\f$, where \f$e_0\f$ is
 * the edge function associated to the cell \f$[x_k,  x_{k+1}]\f$, \f$e_{-1}\f$ to \f$[x_{k-1},
 * x_k]\f$, \f$e_1\f$ to \f$[x_{k+1},  x_{k+2}]\f$, and so forth. The whole grid is assumed to be
 * uniform.
 *
 * @tparam nNodes int, number of nodes used to build the stencil (nCells=nNodes-1: number of dofs
 * of the histopolating basis function)
 * @param polyIndex int, indicates the edge function w.r.t. \f$[x_k, x_{k+1}]\f$, in
 * {-n,...,0,...,n}
 * @param xk Real, node \f$x_k\f$ which defines the interval of evaluation
 * @param dx (uniform) grid size
 * @param y Real, point in \f$[x_k, x_{k+1}]\f$, where \f$e_i\f$ is to be evaluated
 *
 * @return \f$e_{polyIndex} (y)\f$ the histopolating polynomial evaluated at xi
 */
// NOLINTBEGIN(readability-magic-numbers)

template <int nNodes>
AMREX_GPU_HOST_DEVICE amrex::Real eval_edge (int const polyIndex,
                                             amrex::Real const xk,
                                             amrex::Real const dx,
                                             amrex::Real const y)
{
    constexpr int histDegree{nNodes - 2};
    constexpr int intDegree{histDegree + 1};
    constexpr int sigma{nNodes / 2 - 1}; // memo: (hodgeDegree = nIntNodes = 2 sigma + 2 )

    AMREX_ASSERT_WITH_MESSAGE((nNodes % 2) == 0, "nNodes must be EVEN");
    AMREX_ASSERT_WITH_MESSAGE(polyIndex >= -sigma && polyIndex <= sigma,
                              "polyIndex must be in [-n, n]");
    AMREX_ASSERT_WITH_MESSAGE(y >= xk && y <= xk + dx, "y must be in [xk, xk+dx]");
    amrex::Real val = 0;
    amrex::Real diff = y - xk;
    switch (nNodes)
    {
        case 2:
            val = 1. / dx;
            break;
        case 4:
            switch (polyIndex)
            {
                case -1:
                    val = ((diff / (2. * dx) - 1.) * diff / dx + 1. / 3.) / dx;
                    break;
                case 0:
                    val = ((-diff / dx + 1.) * diff / dx + 5. / 6.) / dx;
                    break;
                case 1:
                    val = (std::pow(diff, 2) / (2 * std::pow(dx, 2)) - 1. / 6.) / dx;
                    break;
                default:
                    break;
            }
            break;
        case 6:
            switch (polyIndex)
            {
                case -2:
                    val = ((((1. / 24. * diff / dx - 1. / 6.) * diff / dx + 1. / 8.) * diff / dx +
                            1. / 12.) *
                               diff / dx -
                           1. / 20.) /
                          dx;
                    break;
                case -1:
                    val = (((((-1. / 6. * diff) / dx + 1. / 2.) * diff / dx + 1. / 4.) * diff / dx -
                            5. / 4.) *
                               diff / dx +
                           9. / 20.) /
                          dx;
                    break;
                case 0:
                    val = (((((1. / 4. * diff) / dx - 1. / 2.) * diff / dx - 1.) * diff / dx +
                            5. / 4.) *
                               diff / dx +
                           47. / 60.) /
                          dx;
                    break;
                case 1:
                    val = (((((-1. / 6. * diff) / dx + 1. / 6.) * diff / dx + 3. / 4.) * diff / dx -
                            1. / 12.) *
                               diff / dx -
                           13. / 60.) /
                          dx;
                    break;
                case 2:
                    val = ((1. / 24. * std::pow(diff, 2) / std::pow(dx, 2) - 1. / 8.) *
                               std::pow(diff, 2) / std::pow(dx, 2) +
                           1. / 30.) /
                          dx;
                    break;
                default:
                    break;
            }
            break;
        default:
            if (nNodes % 2 != 0) //check very last bit
            {
                AMREX_ASSERT("Impossible error! ODD orders not implemented");
                amrex::Abort("Impossible error! ODD orders not implemented");
            }
            val = eval_edge_arbitrary_order<intDegree>(polyIndex, xk, dx, y);
    }

    return val;
}

/**
 * the same but on the reference grid with xk=0;dx=1;
 *
 */
template <int nNodes>
AMREX_GPU_HOST_DEVICE amrex::Real eval_edge (int const polyIndex, amrex::Real const y)
{
    constexpr int intDegree{nNodes - 1};
    constexpr int sigma{nNodes / 2 - 1}; // memo: (hodgeDegree = nIntNodes = 2 sigma + 2 )
    AMREX_ASSERT_WITH_MESSAGE((nNodes % 2) == 0, "nNodes must be EVEN");
    AMREX_ASSERT_WITH_MESSAGE(polyIndex >= -sigma && polyIndex <= sigma,
                              "polyIndex must be in [-n, n]");
    AMREX_ASSERT_WITH_MESSAGE(y >= 0.0 && y <= 1.0, "y must be in [0, 1]");

    amrex::Real val = 0;
    amrex::Real diff = y;
    switch (nNodes)
    {
        case 2:
            val = 1.;
            break;
        case 4:
            switch (polyIndex)
            {
                case -1:
                    val = ((diff / 2 - 1.) * diff + 1. / 3.);
                    break;
                case 0:
                    val = ((-diff + 1.) * diff + 5. / 6.);
                    break;
                case 1:
                    val = (std::pow(diff, 2) / 2 - 1. / 6.);
                    break;
                default:
                    break;
            }
            break;
        case 6:
            switch (polyIndex)
            {
                case -2:
                    val =
                        ((((1. / 24. * diff - 1. / 6.) * diff + 1. / 8.) * diff + 1. / 12.) * diff -
                         1. / 20.);
                    break;
                case -1:
                    val = (((((-1. / 6. * diff) + 1. / 2.) * diff + 1. / 4.) * diff - 5. / 4.) *
                               diff +
                           9. / 20.);
                    break;
                case 0:
                    val = (((((1. / 4. * diff) - 1. / 2.) * diff - 1.) * diff + 5. / 4.) * diff +
                           47. / 60.);
                    break;
                case 1:
                    val = (((((-1. / 6. * diff) + 1. / 6.) * diff + 3. / 4.) * diff - 1. / 12.) *
                               diff -
                           13. / 60.);
                    break;
                case 2:
                    val = ((1. / 24. * std::pow(diff, 2) - 1. / 8.) * std::pow(diff, 2) + 1. / 30.);
                    break;
                default:
                    break;
            }
            break;
        default:
            if (nNodes % 2 != 0) //check very last bit
            {
                AMREX_ASSERT("Impossible error! ODD orders not implemented");
                amrex::Abort("Impossible error! ODD orders not implemented");
            }
            val = eval_edge_arbitrary_order<intDegree>(polyIndex, y);
    }

    return val;
}

template <int nNodes, OneTwoFormC SrcFormType, Direction dir, Direction coordinate>
AMREX_GPU_DEVICE inline amrex::Real choose_eval (int const polyIndex,
                                                 amrex::Real const xk,
                                                 amrex::Real const dx,
                                                 amrex::Real const y)
{
    if constexpr ((SrcFormType::s_space == Space::edge && dir != coordinate) ||
                  (SrcFormType::s_space == Space::face && dir == coordinate))
    {
        return eval_lagrange<nNodes>(polyIndex, xk, dx, y);
    }
    else
    {
        return eval_edge<nNodes>(polyIndex, xk, dx, y);
    }
}

template <int nNodes, ZeroThreeFormC SrcFormType>
AMREX_GPU_DEVICE inline amrex::Real choose_eval (int const polyIndex,
                                                 amrex::Real const xk,
                                                 amrex::Real const dx,
                                                 amrex::Real const y)
{
    if constexpr (SrcFormType::s_space == Space::node)
    {
        return eval_lagrange<nNodes>(polyIndex, xk, dx, y);
    }
    else // if constexpr (space == Space::edge)
    {
        return eval_edge<nNodes>(polyIndex, xk, dx, y);
    }
}

/**
the same but assuming xk=0, dx = 1
*/
template <int nNodes, OneTwoFormC SrcFormType, Direction dir, Direction coordinate>
AMREX_GPU_DEVICE inline amrex::Real choose_eval (int const polyIndex, amrex::Real const y)
{
    if constexpr ((SrcFormType::s_space == Space::edge && dir != coordinate) ||
                  (SrcFormType::s_space == Space::face && dir == coordinate))
    {
        return eval_lagrange<nNodes>(polyIndex, y);
    }
    else
    {
        return eval_edge<nNodes>(polyIndex, y);
    }
}
/**
the same but assuming dx = 1
*/
template <int nNodes, ZeroThreeFormC SrcFormType>
AMREX_GPU_DEVICE inline amrex::Real choose_eval (int const polyIndex, amrex::Real const y)
{
    if constexpr (SrcFormType::s_space == Space::node)
    {
        return eval_lagrange<nNodes>(polyIndex, y);
    }
    else // if constexpr (space == Space::edge)
    {
        return eval_edge<nNodes>(polyIndex, y);
    }
}

/**
 * @brief Function that computes the interpolant for a given component of a 0-/3-form at a given
 * point.
 * @tparam srcFormType type of the Form, here constrained to be Zero/Three SpaceGrid-type Form
 * @tparam nNodes order or Lagrange interpolation (only even order implemented)
 * @param funcMF degrees of freedom of the input Form
 * @param r poisition to evaluate the fields
 * @param rr poisition to left grid point
 * @param dr mesh size
 * @param i
 * @param j
 * @param k
 * @param n
 */
template <int nNodes, ZeroThreeFormC SrcFormType>
GEMPIC_FORCE_INLINE AMREX_GPU_HOST_DEVICE amrex::Real eval_form_component_local (
    amrex::Array4<amrex::Real const> const& funcMF,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> rr,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dr,
    int i,
    int j,
    int k,
    int n = 0)
{
    constexpr int sigma{nNodes / 2 - 1};
    constexpr std::array<int, 3> dirShift = make_dir_shift<SrcFormType>();

    amrex::Real fval = 0.0;
    GEMPIC_D_EXCL(int b{0};, int c{0};, )

    GEMPIC_D_LOOP_BEGIN(for (int a = -sigma; a < sigma + dirShift[xDir]; ++a),
                        for (int b = -sigma; b < sigma + dirShift[yDir]; ++b),
                        for (int c = -sigma; c < sigma + dirShift[zDir]; ++c))
        fval += funcMF(i + a, j + b, k + c, n) *
                GEMPIC_D_MULT((choose_eval<nNodes, SrcFormType>(a, rr[xDir], dr[xDir], r[xDir])),
                              (choose_eval<nNodes, SrcFormType>(b, rr[yDir], dr[yDir], r[yDir])),
                              (choose_eval<nNodes, SrcFormType>(c, rr[zDir], dr[zDir], r[zDir])));
    GEMPIC_D_LOOP_END
    return fval;
}

/**
 * @brief
 * @tparam srcFormType type of the Form, here constrained to be Zero/Three SpaceGrid-type Form
 * @tparam nNodes order or Lagrange interpolation (only even order implemented)
 * @param funcMF degrees of freedom of the input Form
 * @param r poisition to evaluate the fields on reference stencil [...,0,1,...], [0,1] the central
 * cell.
 * @param iIn spatial indices of the reference (primal or dual) cell where we want to evaluate the
 * field.
 * @param jIn spatial indices of the reference (primal or dual) cell where we want to evaluate the
 * field.
 * @param kIn spatial indices of the reference (primal or dual) cell where we want to evaluate the
 * field.
 * @param n
 * @return
 */
template <int nNodes, ZeroThreeFormC SrcFormType>
GEMPIC_FORCE_INLINE AMREX_GPU_HOST_DEVICE amrex::Real eval_form_component_local (
    amrex::Array4<amrex::Real const> const& funcMF,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r,
    int iIn,
    int jIn,
    int kIn,
    int n = 0)
{
    constexpr int sigma{nNodes / 2 - 1};
    static constexpr std::array<int, 3> dirShift{StaggeredDofMapping<SrcFormType>::s_mDirShift};

    StaggeredDofMapping<SrcFormType> stagMap(r);
    //m_shiftDualGridIdx add a shift in the indices for dual to primal evaluation
    int i{iIn + stagMap.m_shiftDualGridIdx[xDir]};
    int j{jIn + stagMap.m_shiftDualGridIdx[yDir]};
    int k{kIn + stagMap.m_shiftDualGridIdx[zDir]};

    amrex::Real fval = 0.0;
    GEMPIC_D_EXCL(int b{0};, int c{0};, )
    // m_dirShift add eventually a degree of freedom to the reconsctruction stencil
    GEMPIC_D_LOOP_BEGIN(for (int a = -sigma; a < sigma + 1 + dirShift[xDir]; ++a),
                        for (int b = -sigma; b < sigma + 1 + dirShift[yDir]; ++b),
                        for (int c = -sigma; c < sigma + 1 + dirShift[zDir]; ++c))
        fval +=
            funcMF(i + a, j + b, k + c, n) *
            GEMPIC_D_MULT((Gempic::Forms::choose_eval<nNodes, SrcFormType>(a, stagMap.m_r[xDir])),
                          (Gempic::Forms::choose_eval<nNodes, SrcFormType>(b, stagMap.m_r[yDir])),
                          (Gempic::Forms::choose_eval<nNodes, SrcFormType>(c, stagMap.m_r[zDir])));
    GEMPIC_D_LOOP_END
    return fval;
}

/**
 * @brief Function that computes the interpolant for a given component of a 1-/2-form at a given
 * point.
 * @tparam srcFormType type of the Form, here constrained to be Zero/Three SpaceGrid-type Form
 * @tparam nNodes order or Lagrange interpolation (only even order implemented)
 * @param funcMF degrees of freedom of the input Form
 * @param r poisition to evaluate the fields
 * @param rr poisition to left grid point
 * @param dr mesh size
 * @param i
 * @param j
 * @param k
 * @param n
 */
template <int nNodes, OneTwoFormC SrcFormType, Direction dir>
GEMPIC_FORCE_INLINE AMREX_GPU_HOST_DEVICE amrex::Real eval_form_component_local (
    amrex::Array4<amrex::Real const> const& funcMF,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> rr,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dr,
    int i,
    int j,
    int k,
    int n = 0)
{
    constexpr int sigma{nNodes / 2 - 1};
    constexpr std::array<int, 3> dirShift = make_dir_shift<SrcFormType, dir>();

    amrex::Real fval = 0.0;
    GEMPIC_D_EXCL(int b{0};, int c{0};, )

    GEMPIC_D_LOOP_BEGIN(for (int a = -sigma; a < sigma + 1 + dirShift[xDir]; ++a),
                        for (int b = -sigma; b < sigma + 1 + dirShift[yDir]; ++b),
                        for (int c = -sigma; c < sigma + 1 + dirShift[zDir]; ++c))
        fval += funcMF(i + a, j + b, k + c, n) *
                GEMPIC_D_MULT(
                    (choose_eval<nNodes, SrcFormType, dir, xDir>(a, rr[xDir], dr[xDir], r[xDir])),
                    (choose_eval<nNodes, SrcFormType, dir, yDir>(b, rr[yDir], dr[yDir], r[yDir])),
                    (choose_eval<nNodes, SrcFormType, dir, zDir>(c, rr[zDir], dr[zDir], r[zDir])));
    GEMPIC_D_LOOP_END
    return fval;
}
// the same, but we assume rr = 0.0; dr = 1.0;
template <int nNodes, OneTwoFormC SrcFormType, Direction dir>
GEMPIC_FORCE_INLINE AMREX_GPU_HOST_DEVICE amrex::Real eval_form_component_local (
    amrex::Array4<amrex::Real const> const& funcMF,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r,
    int iIn,
    int jIn,
    int kIn,
    int n = 0)
{
    constexpr int sigma{nNodes / 2 - 1};
    static constexpr std::array<int, 3> dirShift{
        StaggeredDofMapping<SrcFormType, dir>::s_mDirShift};

    StaggeredDofMapping<SrcFormType, dir> stagMap(r);

    int i{iIn + stagMap.m_shiftDualGridIdx[xDir]};
    int j{jIn + stagMap.m_shiftDualGridIdx[yDir]};
    int k{kIn + stagMap.m_shiftDualGridIdx[zDir]};

    amrex::Real fval = 0.0;
    GEMPIC_D_EXCL(int b{0};, int c{0};, )

    GEMPIC_D_LOOP_BEGIN(for (int a = -sigma; a < sigma + 1 + dirShift[xDir]; ++a),
                        for (int b = -sigma; b < sigma + 1 + dirShift[yDir]; ++b),
                        for (int c = -sigma; c < sigma + 1 + dirShift[zDir]; ++c))
        fval += funcMF(i + a, j + b, k + c, n) *
                GEMPIC_D_MULT((choose_eval<nNodes, SrcFormType, dir, xDir>(a, stagMap.m_r[xDir])),
                              (choose_eval<nNodes, SrcFormType, dir, yDir>(b, stagMap.m_r[yDir])),
                              (choose_eval<nNodes, SrcFormType, dir, zDir>(c, stagMap.m_r[zDir])));
    GEMPIC_D_LOOP_END
    return fval;
}

/**
 * @brief Function that computes the interpolant for a given component of a 0-/3-form.
 *
 * @tparam degree order or Lagrange interpolation (only even order implemented)
 * @tparam SrcFormType either a zero- or a three-form field
 * @param evalShift shift from lower corner of primal cell (between 0 and 1)
 * @param srcForm field containing degrees of freedom of form to be evaluated
 * @param values cell centered multifab returning the values at points defined by evalShift
 */
template <int degree, ZeroThreeFormC SrcFormType>
void eval_form_component (
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> const& evalShift,
    SrcFormType const& srcForm,
    amrex::MultiFab& values)
{
    int nEvals{static_cast<int>(evalShift.size())};
    int const nComp{srcForm.get_ncomp()};
    int outputNComp{values.nComp()};
    AMREX_ASSERT_WITH_MESSAGE(nEvals * nComp == outputNComp,
                              "Output MultiFab of eval_form must have enough components to hold "
                              "all evaluations for all field components.");

    // Copy evalShift to GPU array
    amrex::Gpu::DeviceVector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> evalShiftGpu{
        static_cast<size_t>(nEvals)};
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, evalShift.begin(), evalShift.end(),
                          evalShiftGpu.begin());
    auto* const evalShiftGpuPtr = evalShiftGpu.dataPtr();

    for (amrex::MFIter mfi(values); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.validbox();
        amrex::Array4<amrex::Real const> const& funcMF = (srcForm.m_data)[mfi].array();
        amrex::Array4<amrex::Real> const& valuesMF = values[mfi].array();
        amrex::Real scaling = srcForm.m_scaling;
        /// @todo: Reversing the order of for-loops would obviate the need for a DeviceVector.
        ///        It would, however, start nEvals ParallelFor loops. Is it a good idea?
        ParallelFor(
            bx, nComp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
            {
                for (unsigned int evalInd{0}; evalInd < nEvals; ++evalInd)
                {
                    // evalshift is always given with respect to primal grid
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> rr = {
                        AMREX_D_DECL(evalShiftGpuPtr[evalInd][xDir], evalShiftGpuPtr[evalInd][yDir],
                                     evalShiftGpuPtr[evalInd][zDir])};
                    valuesMF(i, j, k, n * nEvals + evalInd) =
                        eval_form_component_local<degree, SrcFormType>(funcMF, rr, i, j, k, n);
                    valuesMF(i, j, k, n * nEvals + evalInd) *= scaling;
                }
            });
    }
}
/**
 * @brief Function that computes the interpolant for a given component of a 1-/2-form.
 *
 * @tparam degree order or Lagrange interpolation (only even order implemented)
 * @tparam SrcFormType either a one- or a two-form field
 * @tparam dir direction of the field (xDir, yDir, zDir)
 * @param evalShift shift from lower corner of primal cell (between 0 and 1)
 * @param srcForm multifab containing degrees of freedom of form to be evaluated
 * @param values cell centered multifab returning the values at points defined by evalShift
 */
template <int degree, OneTwoFormC SrcFormType, Direction dir>
void eval_form_component (
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> const& evalShift,
    SrcFormType const& srcForm,
    amrex::MultiFab& values)
{
    auto nEvals{static_cast<size_t>(evalShift.size())};
    int nComp{srcForm.get_ncomp(dir)};
    int outputNComp{values.nComp()};
    AMREX_ASSERT_WITH_MESSAGE(nEvals * size_t(nComp) == size_t(outputNComp),
                              "Output MultiFab of eval_form must have enough components to hold "
                              "all evaluations for all field components (in one direction).");

    amrex::Gpu::DeviceVector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> evalShiftGpu{nEvals};
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, evalShift.begin(), evalShift.end(),
                          evalShiftGpu.begin());
    auto* const evalShiftGpuPtr = evalShiftGpu.dataPtr();

    for (amrex::MFIter mfi(values); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.validbox();
        amrex::Array4<amrex::Real const> const& funcMF = (srcForm.m_data[dir])[mfi].array();
        amrex::Array4<amrex::Real> const& valuesMF = values[mfi].array();
        amrex::Real scaling{srcForm.m_scaling[dir]};
        /// @todo: Reversing the order of for-loops would obviate the need for a DeviceVector.
        ///        It would, however, start nEvals ParallelFor loops. Is it a good idea?
        ParallelFor(
            bx, nComp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
            {
                for (unsigned int evalInd{0}; evalInd < nEvals; ++evalInd)
                {
                    // dual) evalshift is always given with respect to primal grid
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> rr = {
                        AMREX_D_DECL(evalShiftGpuPtr[evalInd][xDir], evalShiftGpuPtr[evalInd][yDir],
                                     evalShiftGpuPtr[evalInd][zDir])};
                    // TO DO: should we really evaluate component by component? for Multifab yes
                    // (components are non aligned).
                    valuesMF(i, j, k, n * nEvals + evalInd) =
                        eval_form_component_local<degree, SrcFormType, dir>(funcMF, rr, i, j, k, n);
                    valuesMF(i, j, k, n * nEvals + evalInd) *= scaling;
                }
            });
    }
}

/**
 * @brief Computes the maximum norm w.r.t. the values at the cell midpoints
 *
 * @details
 * Interpolates the field, evaluates it a the cell midpoint and computes the max error
 *
 * @tparam degree int, degree of basis functions used
 * @param geom Geometry, needed for physical coordinates
 * @param func ParserExecutor, provides the exact function to be evaluated
 * @param funcH MultiFab, contains the DOF values of the numeric approximation
 * @param dr RealVect, contains the spatial grid sizes
 * @param form int, specifies k with func_h in C_k
 * @param dual bual, true if func_h is defined on dual grid
 * @param comp int, specifies which component func_h belongs to
 * @param t Real, time point at which func is to be evaluated
 */
template <int degree>
amrex::Real max_error_midpoint (amrex::Geometry geom,
                                amrex::ParserExecutor<AMREX_SPACEDIM + 1>& func,
                                amrex::MultiFab& funcH,
                                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dr,
                                int form,
                                bool dual,
                                int comp = 0,
                                amrex::Real t = 0.)
{
    BL_PROFILE("Gempic::Forms::max_error_midpoint()");
    amrex::BoxArray const& ba = funcH.boxArray();
    amrex::DistributionMapping const& dm = funcH.DistributionMap();
    int nghost = funcH.nGrow();
    amrex::MultiFab values(ba, dm, 1, nghost);

    for (amrex::MFIter mfi(funcH); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.validbox();
        amrex::Array4<amrex::Real> const& funcMF = funcH[mfi].array();
        amrex::Array4<amrex::Real> const& valuesMF = values[mfi].array();

        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const r0 = geom.ProbLoArray();

        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const drHalf = {
            AMREX_D_DECL(dr[xDir] / 2, dr[yDir] / 2, dr[zDir] / 2)};

        ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                // Compute the position of the point i, j, k
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> rr = {
                    AMREX_D_DECL(r0[xDir] + i * dr[xDir] - (dual ? drHalf[xDir] : 0),
                                 r0[yDir] + j * dr[yDir] - (dual ? drHalf[yDir] : 0),
                                 r0[zDir] + k * dr[zDir] - (dual ? drHalf[zDir] : 0))};

                amrex::Real error = 0.0;

                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r = {AMREX_D_DECL(
                    rr[xDir] + drHalf[xDir], rr[yDir] + drHalf[yDir], rr[zDir] + drHalf[zDir])};

                amrex::Real fval = 0;
                GEMPIC_D_EXCL(int b{0};, int c{0};, )
                switch (form)
                {
                    case 0:
                        GEMPIC_D_LOOP_BEGIN(for (int a = 1 - degree / 2; a < degree / 2 + 1; ++a),
                                            for (int b = 1 - degree / 2; b < degree / 2 + 1; ++b),
                                            for (int c = 1 - degree / 2; c < degree / 2 + 1; ++c))
                            fval += funcMF(i + a, j + b, k + c) *
                                    GEMPIC_D_MULT(eval_lagrange<degree>(
                                                      a, rr[xDir] + (dual ? dr[xDir] : 0), dr[xDir],
                                                      r[xDir] + (dual ? dr[xDir] : 0)),
                                                  eval_lagrange<degree>(
                                                      b, rr[yDir] + (dual ? dr[yDir] : 0), dr[yDir],
                                                      r[yDir] + (dual ? dr[yDir] : 0)),
                                                  eval_lagrange<degree>(
                                                      c, rr[zDir] + (dual ? dr[zDir] : 0), dr[zDir],
                                                      r[zDir] + (dual ? dr[zDir] : 0)));
                        GEMPIC_D_LOOP_END
                        break;
                    case 1:
                        GEMPIC_D_LOOP_BEGIN(for (int a = 1 - degree / 2;
                                                 a < degree / 2 + (comp == xDir ? 0 : 1); ++a),
                                            for (int b = 1 - degree / 2;
                                                 b < degree / 2 + (comp == yDir ? 0 : 1); ++b),
                                            for (int c = 1 - degree / 2;
                                                 c < degree / 2 + (comp == zDir ? 0 : 1); ++c))
                            fval +=
                                funcMF(i + a, j + b, k + c) *
                                GEMPIC_D_MULT(
                                    comp == xDir
                                        ? eval_edge<degree>(a, rr[xDir], dr[xDir], r[xDir])
                                        : eval_lagrange<degree>(a, rr[xDir], dr[xDir], r[xDir]),
                                    comp == yDir
                                        ? eval_edge<degree>(b, rr[yDir], dr[yDir], r[yDir])
                                        : eval_lagrange<degree>(b, rr[yDir], dr[yDir], r[yDir]),
                                    comp == zDir
                                        ? eval_edge<degree>(c, rr[zDir], dr[zDir], r[zDir])
                                        : eval_lagrange<degree>(c, rr[zDir], dr[zDir], r[zDir]));
                        GEMPIC_D_LOOP_END
                        break;
                    case 2:
                        GEMPIC_D_LOOP_BEGIN(for (int a = 1 - degree / 2;
                                                 a < degree / 2 + (comp == xDir ? 1 : 0); ++a),
                                            for (int b = 1 - degree / 2;
                                                 b < degree / 2 + (comp == yDir ? 1 : 0); ++b),
                                            for (int c = 1 - degree / 2;
                                                 c < degree / 2 + (comp == zDir ? 1 : 0); ++c))
                            fval += funcMF(i + a, j + b, k + c) *
                                    GEMPIC_D_MULT(
                                        comp == xDir
                                            ? eval_lagrange<degree>(a, rr[xDir], dr[xDir], r[xDir])
                                            : eval_edge<degree>(a, rr[xDir], dr[xDir], r[xDir]),
                                        comp == yDir
                                            ? eval_lagrange<degree>(b, rr[yDir], dr[yDir], r[yDir])
                                            : eval_edge<degree>(b, rr[yDir], dr[yDir], r[yDir]),
                                        comp == zDir
                                            ? eval_lagrange<degree>(c, rr[zDir], dr[zDir], r[zDir])
                                            : eval_edge<degree>(c, rr[zDir], dr[zDir], r[zDir]));
                        GEMPIC_D_LOOP_END
                        break;
                    case 3:
                        GEMPIC_D_LOOP_BEGIN(for (int a = 1 - degree / 2; a < degree / 2; ++a),
                                            for (int b = 1 - degree / 2; b < degree / 2; ++b),
                                            for (int c = 1 - degree / 2; c < degree / 2; ++c))
                            fval +=
                                funcMF(i + a, j + b, k + c) *
                                GEMPIC_D_MULT(eval_edge<degree>(a, rr[xDir], dr[xDir], r[xDir]),
                                              eval_edge<degree>(b, rr[yDir], dr[yDir], r[yDir]),
                                              eval_edge<degree>(c, rr[zDir], dr[zDir], r[zDir]));
                        GEMPIC_D_LOOP_END
                        break;

                    default:
                        break;
                }

                switch (form)
                {
                    case 0:
                        error += func(AMREX_D_DECL(r[xDir] + (dual ? dr[xDir] : 0),
                                                   r[yDir] + (dual ? dr[yDir] : 0),
                                                   r[zDir] + (dual ? dr[zDir] : 0)),
                                      t) -
                                 fval;
                        break;
                    case 1:
                        error += func(AMREX_D_DECL(r[xDir] + (dual && comp != xDir ? dr[xDir] : 0),
                                                   r[yDir] + (dual && comp != yDir ? dr[yDir] : 0),
                                                   r[zDir] + (dual && comp != zDir ? dr[zDir] : 0)),
                                      t) -
                                 fval;
                        break;
                    case 2:
                        error += func(AMREX_D_DECL(r[xDir] + (dual && comp == xDir ? dr[xDir] : 0),
                                                   r[yDir] + (dual && comp == yDir ? dr[yDir] : 0),
                                                   r[zDir] + (dual && comp == zDir ? dr[zDir] : 0)),
                                      t) -
                                 fval;
                        break;
                    case 3:
                        error += func(AMREX_D_DECL(r[xDir], r[yDir], r[zDir]), t) - fval;
                        break;
                    default:
                        break;
                }

                valuesMF(i, j, k) = error;
            });
    }
    return values.norm0();
}

// one-, two- form interpolation
template <OneTwoFormC SrcFormType, int degree, Direction dir>
void interpolate_field_to_multifab (amrex::MultiFab& mfDst,
                                    SrcFormType const& srcForm,
                                    amrex::Real scaling,
                                    int dcomp)
{
    for (amrex::MFIter mfiSrc(srcForm.multi_fab(dir)); mfiSrc.isValid(); // don't use tiling!!
         ++mfiSrc)
    {
        std::array<int, 3> shiftCell{0, 0, 0};
        // change box upper limits to compute only the cell-centered output multifab:
        amrex::IntVect shiftCellIVect{AMREX_D_DECL(shiftCell[0], shiftCell[1], shiftCell[2])};
        int nComp = srcForm.get_ncomp(dir);
        GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE(mfDst.nComp() >= dcomp + nComp,
                                          "Insufficient space in destination multifab");
        amrex::Box const& bxSrc = mfiSrc.tilebox(shiftCellIVect);

        amrex::Array4 dst = mfDst[mfiSrc].array();
        amrex::Array4 const src = srcForm.multi_fab(dir).const_array(mfiSrc);

        // Trying to get the refinement ratio: from mfDst and srcForm
        // Number of cells in each direction
        amrex::IntVect ncellsDst = mfDst.boxArray().minimalBox().size();
        amrex::IntVect ncellsSrc = srcForm.get_ncells();
        std::array<int, 3> sNSubCellLoc{GEMPIC_D_PAD_ONE(ncellsDst[xDir] / ncellsSrc[xDir],
                                                         ncellsDst[yDir] / ncellsSrc[yDir],
                                                         ncellsDst[zDir] / ncellsSrc[zDir])};
        GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE(
            GEMPIC_D_AND(sNSubCellLoc[xDir] > 0, sNSubCellLoc[yDir] > 0, sNSubCellLoc[zDir] > 0),
            "Warning: aspect ratio in one or more dimensions are less than or equal to 0");

        amrex::Box const& bxDst = mfDst[mfiSrc].box(); // the Box corresponding to mfDst's tile
        amrex::IntVect ncellsDst2 = bxDst.size();
        amrex::IntVect ncellsSrc2 = bxSrc.size();
        std::array<int, 3> sNSubCellLoc2{GEMPIC_D_PAD_ONE(ncellsDst2[xDir] / ncellsSrc2[xDir],
                                                          ncellsDst2[yDir] / ncellsSrc2[yDir],
                                                          ncellsDst2[zDir] / ncellsSrc2[zDir])};
        GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE(GEMPIC_D_AND(sNSubCellLoc[xDir] == sNSubCellLoc2[xDir],
                                                       sNSubCellLoc[yDir] == sNSubCellLoc2[yDir],
                                                       sNSubCellLoc[zDir] == sNSubCellLoc2[zDir]),
                                          "Warning: aspect ratios don't match");
        ParallelFor(bxSrc, nComp,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        int const iloc = sNSubCellLoc[xDir] * i;
                        int const jloc = sNSubCellLoc[yDir] * j;
                        int const kloc = sNSubCellLoc[zDir] * k;
                        for (int iSub = 0; iSub < sNSubCellLoc[xDir]; ++iSub)
                        {
                            for (int jSub = 0; jSub < sNSubCellLoc[yDir]; ++jSub)
                            {
                                for (int kSub = 0; kSub < sNSubCellLoc[zDir]; ++kSub)
                                {
                                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r = {
                                        AMREX_D_DECL((iSub + 0.5) / sNSubCellLoc[xDir],
                                                     (jSub + 0.5) / sNSubCellLoc[yDir],
                                                     (kSub + 0.5) / sNSubCellLoc[zDir])};
                                    dst(iloc + iSub, jloc + jSub, kloc + kSub, dcomp + n) =
                                        eval_form_component_local<degree, SrcFormType, dir>(
                                            src, r, i, j, k, n);
                                    dst(iloc + iSub, jloc + jSub, kloc + kSub, dcomp + n) *=
                                        scaling;
                                }
                            }
                        }
                    });
    }
}

// zero-, three- form interpolation
template <ZeroThreeFormC SrcFormType, int degree>
void interpolate_field_to_multifab (amrex::MultiFab& mfDst,
                                    SrcFormType const& srcForm,
                                    amrex::Real scaling,
                                    int dcomp)
{
    for (amrex::MFIter mfiSrc(srcForm.multi_fab()); mfiSrc.isValid(); ++mfiSrc)
    {
        std::array<int, 3> shiftCell{0, 0, 0};
        // change box upper limits to compute only the cell-centered output multifab:
        amrex::IntVect shiftCellIVect{AMREX_D_DECL(shiftCell[0], shiftCell[1], shiftCell[2])};
        int nComp = srcForm.get_ncomp();
        GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE(mfDst.nComp() >= dcomp + nComp,
                                          "Insufficient space in destination multifab");
        amrex::Box const& bxSrc = mfiSrc.tilebox(shiftCellIVect);

        amrex::Array4 dst = mfDst[mfiSrc].array();
        amrex::Array4 const src = srcForm.multi_fab().const_array(mfiSrc);

        // Trying to get the refinement ratio: from mfDst and srcForm
        // Number of cells in each direction
        amrex::IntVect ncellsDst = mfDst.boxArray().minimalBox().size();
        amrex::IntVect ncellsSrc = srcForm.get_ncells();
        std::array<int, 3> sNSubCellLoc{GEMPIC_D_PAD_ONE(ncellsDst[xDir] / ncellsSrc[xDir],
                                                         ncellsDst[yDir] / ncellsSrc[yDir],
                                                         ncellsDst[zDir] / ncellsSrc[zDir])};

        GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE(
            GEMPIC_D_AND(sNSubCellLoc[xDir] > 0, sNSubCellLoc[yDir] > 0, sNSubCellLoc[zDir] > 0),
            "Warning: aspect ratio in one or more dimensions are less than or equal to 0");

        amrex::Box const& bxDst = mfDst[mfiSrc].box(); // the Box corresponding to mfDst's tile
        amrex::IntVect ncellsDst2 = bxDst.size();
        amrex::IntVect ncellsSrc2 = bxSrc.size();
        std::array<int, 3> sNSubCellLoc2{GEMPIC_D_PAD_ONE(ncellsDst2[xDir] / ncellsSrc2[xDir],
                                                          ncellsDst2[yDir] / ncellsSrc2[yDir],
                                                          ncellsDst2[zDir] / ncellsSrc2[zDir])};
        GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE(GEMPIC_D_AND(sNSubCellLoc[xDir] == sNSubCellLoc2[xDir],
                                                       sNSubCellLoc[yDir] == sNSubCellLoc2[yDir],
                                                       sNSubCellLoc[zDir] == sNSubCellLoc2[zDir]),
                                          "Warning: aspect ratios don't match");

        ParallelFor(
            bxSrc, nComp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
            {
                int const iloc = sNSubCellLoc[xDir] * i;
                int const jloc = sNSubCellLoc[yDir] * j;
                int const kloc = sNSubCellLoc[zDir] * k;

                for (int iSub = 0; iSub < sNSubCellLoc[xDir]; ++iSub)
                {
                    for (int jSub = 0; jSub < sNSubCellLoc[yDir]; ++jSub)
                    {
                        for (int kSub = 0; kSub < sNSubCellLoc[zDir]; ++kSub)
                        {
                            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r = {
                                AMREX_D_DECL((iSub + 0.5) / sNSubCellLoc[xDir],
                                             (jSub + 0.5) / sNSubCellLoc[yDir],
                                             (kSub + 0.5) / sNSubCellLoc[zDir])};
                            dst(iloc + iSub, jloc + jSub, kloc + kSub, dcomp + n) =
                                eval_form_component_local<degree, SrcFormType>(src, r, i, j, k, n);
                            dst(iloc + iSub, jloc + jSub, kloc + kSub, dcomp + n) *= scaling;
                        }
                    }
                }
            });
    }
}

template <OneTwoFormC SrcFormType, int degree>
void interpolate_field_to_multifab (
    amrex::MultiFab& mfDst, SrcFormType const& srcForm, amrex::Real scaling, int dcomp, int dir)
{
    switch (dir)
    {
        case xDir:
            interpolate_field_to_multifab<SrcFormType, degree, xDir>(mfDst, srcForm, scaling,
                                                                     dcomp);
            break;
        case yDir:
            interpolate_field_to_multifab<SrcFormType, degree, yDir>(mfDst, srcForm, scaling,
                                                                     dcomp);
            break;
        case zDir:
            interpolate_field_to_multifab<SrcFormType, degree, zDir>(mfDst, srcForm, scaling,
                                                                     dcomp);
            break;
        default:
            GEMPIC_ERROR("Unknown direction.");
    }
}

template <Forms::ZeroThreeFormC SrcFormType, int degree>
void interpolate_field_to_multifab (
    amrex::MultiFab& mfDst, SrcFormType const& srcForm, amrex::Real scaling, int dcomp, int /*dir*/)
{
    interpolate_field_to_multifab<SrcFormType, degree>(mfDst, srcForm, scaling, dcomp);
}

template <typename SrcFormType, int testDegInd = 0>
    requires(!HasHodgeScheme<field_to_drc_t<SrcFormType>>)
void interpolate_field_to_multifab_degree_selector (
    amrex::MultiFab& mfDst, SrcFormType const& srcForm, amrex::Real scaling, int dcomp, int dir)
{
    BL_PROFILE("interpolate_field_to_multifab_degree_selector()")
    constexpr auto validDegreesArray = valid_degrees<HodgeScheme::FDHodge>(); // FDHodge hard coded!
    if constexpr (testDegInd < validDegreesArray.size())
    {
        if (srcForm.get_hodge_degree() == validDegreesArray[testDegInd])
        {
            constexpr int degree = validDegreesArray[testDegInd];
            interpolate_field_to_multifab<SrcFormType, degree>(mfDst, srcForm, scaling, dcomp, dir);
        }
        else
        {
            interpolate_field_to_multifab_degree_selector<SrcFormType, testDegInd + 1>(
                mfDst, srcForm, scaling, dcomp, dir);
        }
    }
    else
    {
        GEMPIC_ERROR(
            "Incompatible Hodge degree and interpolate_field_to_multifab_degree_selector: No such "
            "degree " +
            std::to_string(srcForm.get_hodge_degree()) +
            " for interpolate_field_to_multifab_degree_selector");
    }
}

template <typename SrcTypeWithHodge, int testDegreeIndex = 0>
    requires HasHodgeScheme<field_to_drc_t<SrcTypeWithHodge>>
void interpolate_field_to_multifab_degree_selector (amrex::MultiFab& mfDst,
                                                    SrcTypeWithHodge const& srcForm,
                                                    amrex::Real scaling,
                                                    int dcomp,
                                                    int dir)
{
    BL_PROFILE("interpolate_field_to_multifab_degree_selector()")
    constexpr auto validDegreesArray =
        valid_degrees<field_to_drc_t<SrcTypeWithHodge>::s_hodgeScheme>();
    if constexpr (testDegreeIndex < validDegreesArray.size())
    {
        if (srcForm.get_hodge_degree() == validDegreesArray[testDegreeIndex])
        {
            constexpr int degree = validDegreesArray[testDegreeIndex];
            interpolate_field_to_multifab<SrcTypeWithHodge, degree>(mfDst, srcForm, scaling, dcomp,
                                                                    dir);
        }
        else
        {
            interpolate_field_to_multifab_degree_selector<SrcTypeWithHodge, testDegreeIndex + 1>(
                mfDst, srcForm, scaling, dcomp, dir);
        }
    }
    else
    {
        GEMPIC_ERROR(
            "Incompatible Hodge degree and interpolate_field_to_multifab_degree_selector: No such "
            "degree " +
            std::to_string(srcForm.get_hodge_degree()) +
            " for interpolate_field_to_multifab_degree_selector");
    }
}

template <typename SrcFormType>
void rawdata_field_to_multifabs (amrex::MultiFab& /*mfDst*/,
                                 SrcFormType const& /*srcForm*/,
                                 int /*dcomp*/,
                                 Direction /*idir*/)
{
    static_assert(gempic_always_false<SrcFormType>(),
                  "rawdata_field_to_multifabs is not implemented for this srcFormType!");
}
template <typename SrcFormType>
void rawdata_field_to_multifabs (std::vector<amrex::MultiFab>& /*mfDst*/,
                                 SrcFormType const& /*srcForm*/,
                                 int /*dcomp*/,
                                 Direction /*idir*/)
{
    static_assert(gempic_always_false<SrcFormType>(),
                  "rawdata_field_to_multifabs is not implemented for this srcFormType!");
}

template <Forms::ZeroThreeFormC SrcFormType>
void rawdata_field_to_multifabs (amrex::MultiFab& mfDst,
                                 SrcFormType const& srcForm,
                                 int dcomp,
                                 Direction /*idir*/)
{
    // Copying raw MultiFab. You might think this a waste, but amrex's plot function does the same
    // if the MultiFab passed to it has ghost cells.
    amrex::Copy(mfDst, srcForm.multi_fab(), 0, dcomp, mfDst.nComp(), 0);
    // mf_dst.ParallelCopy( *m_mf_src, 0, dcomp, m_mf_src->nComp() );
}

template <Forms::OneTwoFormC SrcFormType>
void rawdata_field_to_multifabs (amrex::MultiFab& mfDst,
                                 SrcFormType const& srcForm,
                                 int dcomp,
                                 Direction idir = noDir)
{
    // Copying raw MultiFab. You might think this a waste, but amrex's plot function does the same
    // if the MultiFab passed to it has ghost cells.
    amrex::Copy(mfDst, srcForm.multi_fab(idir), 0, dcomp, mfDst.nComp(), 0);
    // mf_dst.ParallelCopy( *m_mf_src, 0, dcomp, m_mf_src->nComp() );
}

template <typename SrcFormType>
void field_to_cellcentered_multifabs (amrex::MultiFab& /*mfDst*/,
                                      SrcFormType const& /*srcForm*/,
                                      amrex::Real /*scaling*/,
                                      int /*dcomp*/,
                                      Direction /*idir*/)
{
    static_assert(gempic_always_false<SrcFormType>(),
                  "field_to_cellcentered_multifabs is not implemented for this srcFormType!");
}

template <Forms::ZeroThreeFormC SrcFormType>
void field_to_cellcentered_multifabs (amrex::MultiFab& mfDst,
                                      SrcFormType const& srcForm,
                                      amrex::Real scaling,
                                      int dcomp,
                                      Direction /*idir*/)
{
    // In cartesian geometry, interpolate from simulation MultiFab, m_dataSrc,
    // to output diagnostic MultiFab, mfDst.
    int nComps = srcForm.get_ncomp();

    for (amrex::MFIter mfi(mfDst); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.tilebox();
        amrex::Array4 dst = mfDst[mfi].array();
        amrex::Array4 const src = srcForm.multi_fab()[mfi].array();

        amrex::IntVect const indexTypeSrc = srcForm.multi_fab().boxArray().ixType().toIntVect();

        std::array<amrex::Real, 3> shift{GEMPIC_D_PAD(static_cast<double>(indexTypeSrc[xDir]),
                                                      static_cast<double>(indexTypeSrc[yDir]),
                                                      static_cast<double>(indexTypeSrc[zDir]))};

        amrex::Real ishift{shift[xDir]};
        amrex::Real jshift{shift[yDir]};
        amrex::Real kshift{shift[zDir]};

        ParallelFor(
            bx, nComps,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int srcComp)
            {
                dst(i, j, k, dcomp + srcComp) =
                    0.125 * scaling *
                    (src(i + ishift, j + jshift, k + kshift, srcComp) + src(i, j, k, srcComp) +
                     src(i + ishift, j + jshift, k, srcComp) + src(i, j, k + kshift, srcComp) +
                     src(i + ishift, j, k + kshift, srcComp) + src(i, j + jshift, k, srcComp) +
                     src(i, j + jshift, k + kshift, srcComp) + src(i + ishift, j, k, srcComp));
            });
    }
}

template <OneTwoFormC SrcFormType>
void field_to_cellcentered_multifabs (amrex::MultiFab& mfDst,
                                      SrcFormType const& srcForm,
                                      amrex::Real scaling,
                                      int dcomp,
                                      Direction idir = noDir)
{
    // In cartesian geometry, interpolate from simulation MultiFab, m_dataSrc,
    // to output diagnostic MultiFab, mfDst.
    int nComps = srcForm.get_ncomp(idir);

    for (amrex::MFIter mfi(mfDst); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.tilebox();
        amrex::Array4 dst = mfDst[mfi].array();
        amrex::Array4 const src = srcForm.multi_fab(idir)[mfi].array();

        amrex::IntVect const indexTypeSrc = srcForm.multi_fab(idir).boxArray().ixType().toIntVect();

        std::array<amrex::Real, 3> shift{GEMPIC_D_PAD(static_cast<double>(indexTypeSrc[xDir]),
                                                      static_cast<double>(indexTypeSrc[yDir]),
                                                      static_cast<double>(indexTypeSrc[zDir]))};

        amrex::Real ishift{shift[xDir]};
        amrex::Real jshift{shift[yDir]};
        amrex::Real kshift{shift[zDir]};

        ParallelFor(
            bx, nComps,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int srcComp)
            {
                dst(i, j, k, dcomp + srcComp) =
                    0.125 * scaling *
                    (src(i + ishift, j + jshift, k + kshift, srcComp) + src(i, j, k, srcComp) +
                     src(i + ishift, j + jshift, k, srcComp) + src(i, j, k + kshift, srcComp) +
                     src(i + ishift, j, k + kshift, srcComp) + src(i, j + jshift, k, srcComp) +
                     src(i, j + jshift, k + kshift, srcComp) + src(i + ishift, j, k, srcComp));
            });
    }
}

} //namespace Gempic::Forms

#endif
