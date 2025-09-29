#ifndef GEMPIC_INTERPOLATION_H_
#define GEMPIC_INTERPOLATION_H_

#include <AMReX_SPACE.H>
#include <AMReX_Vector.H>

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
 * @param xi, amrex::Real, the evaluation point
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
 * @tparam nNodes int, number of nodes used to build the stencil (nCells=nNodes-1: #dofs of the
 * histopolating basis function)
 * @param polyIndex int, indicates the edge function w.r.t. \f$[x_k, x_{k+1}]\f$, in
 * {-n,...,0,...,n}
 * @param xk Real, node \f$x_k\f$ which defines the interval of evaluation
 * @param dx (uniform) grid size
 * @param y Real, point in \f$[x_k, x_{k+1}]\f$, where \f$e_i\f$ is to be evaluated
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
 * @tparam nNodes int, number of nodes used to build the stencil (nCells=nNodes-1: #dofs of the
 * histopolating basis function)
 * @param polyIndex int, indicates the edge function w.r.t. \f$[x_k, x_{k+1}]\f$, in
 * {-n,...,0,...,n}
 * @param xk Real, node \f$x_k\f$ which defines the interval of evaluation
 * @param dx (uniform) grid size
 * @param y Real, point in \f$[x_k, x_{k+1}]\f$, where \f$e_i\f$ is to be evaluated
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
 * @tparam nNodes int, number of nodes used to build the stencil (nCells=nNodes-1: #dofs of the
 * histopolating basis function)
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

template <int degree, Grid grid>
AMREX_GPU_DEVICE inline amrex::Real eval_lagrange_primal_dual (int const polyIndex,
                                                               amrex::Real const xk,
                                                               amrex::Real const dx,
                                                               amrex::Real const y)
{
    if constexpr (grid == Grid::primal)
    {
        return eval_lagrange<degree>(polyIndex, xk, dx, y);
    }
    else
    {
        return eval_lagrange<degree>(polyIndex, xk + dx, dx, y);
    }
}
/**
the same but with xk=0,  dx = 1;
*/
template <int degree, Grid grid>
AMREX_GPU_DEVICE inline amrex::Real eval_lagrange_primal_dual (int const polyIndex,
                                                               amrex::Real const y)
{
    if constexpr (grid == Grid::primal)
    {
        return eval_lagrange<degree>(polyIndex, y);
    }
    else
    {
        return eval_lagrange<degree>(polyIndex, y - 1.0);
    }
}

template <int degree, Grid grid, Space space, Direction dir, Direction coordinate>
AMREX_GPU_DEVICE inline amrex::Real choose_eval (int const polyIndex,
                                                 amrex::Real const xk,
                                                 amrex::Real const dx,
                                                 amrex::Real const y)
{
    static_assert(space == Space::face || space == Space::edge,
                  "Tried to call 1-/2-form choose_eval with a 0-/3-form.\n\tThe 0-/3-form "
                  "choose_eval function takes no direction template argument.");
    if constexpr ((space == Space::edge && dir != coordinate) ||
                  (space == Space::face && dir == coordinate))
    {
        return eval_lagrange_primal_dual<degree, grid>(polyIndex, xk, dx, y);
    }
    else
    {
        return eval_edge<degree>(polyIndex, xk, dx, y);
    }
}

template <int degree, Grid grid, Space space>
AMREX_GPU_DEVICE inline amrex::Real choose_eval (int const polyIndex,
                                                 amrex::Real const xk,
                                                 amrex::Real const dx,
                                                 amrex::Real const y)
{
    static_assert(space == Space::node || space == Space::cell,
                  "Tried to call 0-/3-form choose_eval with a 1-/2-form.\n\tThe 1-/2-form "
                  "choose_eval function takes a direction template argument.");
    if constexpr (space == Space::node)
    {
        return eval_lagrange_primal_dual<degree, grid>(polyIndex, xk, dx, y);
    }
    else // if constexpr (space == Space::edge)
    {
        return eval_edge<degree>(polyIndex, xk, dx, y);
    }
}

/**
the same but assuming xk=0, dx = 1
*/
template <int degree, Grid grid, Space space, Direction dir, Direction coordinate>
AMREX_GPU_DEVICE inline amrex::Real choose_eval (int const polyIndex, amrex::Real const y)
{
    static_assert(space == Space::face || space == Space::edge,
                  "Tried to call 1-/2-form choose_eval with a 0-/3-form.\n\tThe 0-/3-form "
                  "choose_eval function takes no direction template argument.");
    if constexpr ((space == Space::edge && dir != coordinate) ||
                  (space == Space::face && dir == coordinate))
    {
        return eval_lagrange_primal_dual<degree, grid>(polyIndex, y);
    }
    else
    {
        return eval_edge<degree>(polyIndex, y);
    }
}
/**
the same but assuming dx = 1
*/
template <int degree, Grid grid, Space space>
AMREX_GPU_DEVICE inline amrex::Real choose_eval (int const polyIndex, amrex::Real const y)
{
    static_assert(space == Space::node || space == Space::cell,
                  "Tried to call 0-/3-form choose_eval with a 1-/2-form.\n\tThe 1-/2-form "
                  "choose_eval function takes a direction template argument.");
    if constexpr (space == Space::node)
    {
        return eval_lagrange_primal_dual<degree, grid>(polyIndex, y);
    }
    else // if constexpr (space == Space::edge)
    {
        return eval_edge<degree>(polyIndex, y);
    }
}

/**
 * @brief Function that computes the interpolant for a given component of a 0-/3-form.
 *
 * @tparam degree order or Lagrange interpolation (only even order implemented)
 * @tparam grid primal or dual
 * @tparam space node or cell, corresponding to 0- or 3-forms
 * @param evalShift shift from lower corner of primal cell (between 0 and 1)
 * @param data multifab containing degrees of freedom of form to be evaluated
 * @param values cell centered multifab returning the values at points defined by evalShift
 * @param dr cell size
 * @param geom
 */
template <int degree, Grid grid, Space space>
void eval_form_component (
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> const& evalShift,
    amrex::MultiFab const& data,
    amrex::MultiFab& values,
    amrex::RealVect const& dr,
    amrex::Geometry const& geom)
{
    static_assert(space == Space::cell || space == Space::node,
                  "Tried to call 0-/3-form eval_form_component with a 1-/2-form.\n\tThe 1-/2-form "
                  "eval_form_component function takes a direction template argument.");
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const r0 = geom.ProbLoArray();

    auto nEvals{static_cast<size_t>(evalShift.size())};
    int nComp{data.nComp()};
    int outputNComp{values.nComp()};
    AMREX_ASSERT_WITH_MESSAGE(nEvals * nComp == outputNComp,
                              "Output MultiFab of eval_form must have enough components to hold "
                              "all evaluations for all field components.");

    int abcTop{degree / 2};

    if constexpr (space == Space::node)
    {
        abcTop += 1;
    }

    // Copy evalShift to GPU array
    amrex::Gpu::DeviceVector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> evalShiftGpu{nEvals};
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, evalShift.begin(), evalShift.end(),
                          evalShiftGpu.begin());
    auto* const evalShiftGpuPtr = evalShiftGpu.dataPtr();

    for (amrex::MFIter mfi(values); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.validbox();
        amrex::Array4<amrex::Real const> const& funcMF = data[mfi].array();
        amrex::Array4<amrex::Real> const& valuesMF = values[mfi].array();

        /// @todo: Reversing the order of for-loops would obviate the need for a DeviceVector.
        ///        It would, however, start nEvals ParallelFor loops. Is it a good idea?
        ParallelFor(
            bx, nComp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
            {
                for (int evalInd{0}; evalInd < nEvals; ++evalInd)
                {
                    // position of point to be evaluated with respect to grid (primal or dual)
                    // evalshift is always given with respect to primal grid
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r = {
                        AMREX_D_DECL(r0[xDir] + (i + evalShiftGpuPtr[evalInd][xDir]) * dr[xDir],
                                     r0[yDir] + (j + evalShiftGpuPtr[evalInd][yDir]) * dr[yDir],
                                     r0[zDir] + (k + evalShiftGpuPtr[evalInd][zDir]) * dr[zDir])};

                    // position of the grid point (x_i, y_j, z_k) to the left of evalshift
                    // (different for primal and dual grids)
                    // evalshift is always given with respect to primal grid
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> rr;
                    if constexpr (grid == Grid::primal)
                    {
                        rr = {AMREX_D_DECL(r0[xDir] + i * dr[xDir], r0[yDir] + j * dr[yDir],
                                           r0[zDir] + k * dr[zDir])};
                    }
                    else if constexpr (grid == Grid::dual)
                    {
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> shiftDualGrid = {AMREX_D_DECL(
                            (std::round(evalShiftGpuPtr[evalInd][xDir]) - 0.5) * dr[xDir],
                            (std::round(evalShiftGpuPtr[evalInd][yDir]) - 0.5) * dr[yDir],
                            (std::round(evalShiftGpuPtr[evalInd][zDir]) - 0.5) * dr[zDir])};
                        rr = {AMREX_D_DECL(r0[xDir] + i * dr[xDir] + shiftDualGrid[xDir],
                                           r0[yDir] + j * dr[yDir] + shiftDualGrid[yDir],
                                           r0[zDir] + k * dr[zDir] + shiftDualGrid[zDir])};
                    }

                    amrex::Real fval = 0.0;
                    GEMPIC_D_EXCL(int b{0};, int c{0};, )

                    GEMPIC_D_LOOP_BEGIN(for (int a = 1 - degree / 2; a < abcTop; ++a),
                                        for (int b = 1 - degree / 2; b < abcTop; ++b),
                                        for (int c = 1 - degree / 2; c < abcTop; ++c))
                        fval +=
                            funcMF(i + a, j + b, k + c, n) *
                            GEMPIC_D_MULT(
                                (choose_eval<degree, grid, space>(a, rr[xDir], dr[xDir], r[xDir])),
                                (choose_eval<degree, grid, space>(b, rr[yDir], dr[yDir], r[yDir])),
                                (choose_eval<degree, grid, space>(c, rr[zDir], dr[zDir], r[zDir])));
                    GEMPIC_D_LOOP_END
                    valuesMF(i, j, k, n * nEvals + evalInd) = fval;
                }
            });
    }
}
/**
 * @brief Function that computes the interpolant for a given component of a 1-/2-form.
 *
 * @tparam degree order or Lagrange interpolation (only even order implemented)
 * @tparam grid primal or dual
 * @tparam space edge or face, corresponding to 1- or 2-forms
 * @tparam dir xDir, yDir, zDir
 * @param evalShift shift from lower corner of primal cell (between 0 and 1)
 * @param data multifab containing degrees of freedom of form to be evaluated
 * @param values cell centered multifab returning the values at points defined by evalShift
 * @param dr cell size
 * @param geom
 */
template <int degree, Grid grid, Space space, Direction dir>
void eval_form_component (
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> const& evalShift,
    amrex::MultiFab const& data,
    amrex::MultiFab& values,
    amrex::RealVect const& dr,
    amrex::Geometry const& geom)
{
    static_assert(space == Space::face || space == Space::edge,
                  "Tried to call 1-/2-form "
                  "eval_form_component with a 0-/3-form.\n\tThe 0-/3-form "
                  "eval_form_component function takes no direction template argument.");

    auto nEvals{static_cast<size_t>(evalShift.size())};
    int nComp{data.nComp()};
    int outputNComp{values.nComp()};
    AMREX_ASSERT_WITH_MESSAGE(nEvals * nComp == outputNComp,
                              "Output MultiFab of eval_form must have enough components to hold "
                              "all evaluations for all field components (in one direction).");

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const r0 = geom.ProbLoArray();

    amrex::GpuArray<int, 3> dirShift = {0, 0, 0};

    if constexpr (space == Space::edge)
    {
        dirShift = {1, 1, 1};
        dirShift[dir] = 0;
    }
    else // if constexpr (space == Space::face)
    {
        dirShift[dir] = 1;
    }

    amrex::Gpu::DeviceVector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> evalShiftGpu{nEvals};
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, evalShift.begin(), evalShift.end(),
                          evalShiftGpu.begin());
    auto* const evalShiftGpuPtr = evalShiftGpu.dataPtr();

    for (amrex::MFIter mfi(values); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.validbox();
        amrex::Array4<amrex::Real const> const& funcMF = data[mfi].array();
        amrex::Array4<amrex::Real> const& valuesMF = values[mfi].array();

        /// @todo: Reversing the order of for-loops would obviate the need for a DeviceVector.
        ///        It would, however, start nEvals ParallelFor loops. Is it a good idea?
        ParallelFor(
            bx, nComp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
            {
                for (int evalInd{0}; evalInd < nEvals; ++evalInd)
                {
                    // position of point to be evaluated with respect to grid (primal or
                    // dual) evalshift is always given with respect to primal grid
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r = {
                        AMREX_D_DECL(r0[xDir] + (i + evalShiftGpuPtr[evalInd][xDir]) * dr[xDir],
                                     r0[yDir] + (j + evalShiftGpuPtr[evalInd][yDir]) * dr[yDir],
                                     r0[zDir] + (k + evalShiftGpuPtr[evalInd][zDir]) * dr[zDir])};

                    // position of the grid point (x_i, y_j, z_k) to the left of evalshift
                    // (different for primal and dual grids)
                    // evalshift is always given with respect to primal grid
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> rr;
                    if constexpr (grid == Grid::primal)
                    {
                        rr = {AMREX_D_DECL(r0[xDir] + i * dr[xDir], r0[yDir] + j * dr[yDir],
                                           r0[zDir] + k * dr[zDir])};
                    }
                    else if constexpr (grid == Grid::dual)
                    {
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> shiftDualGrid = {AMREX_D_DECL(
                            (std::round(evalShiftGpuPtr[evalInd][xDir]) - 0.5) * dr[xDir],
                            (std::round(evalShiftGpuPtr[evalInd][yDir]) - 0.5) * dr[yDir],
                            (std::round(evalShiftGpuPtr[evalInd][zDir]) - 0.5) * dr[zDir])};
                        rr = {AMREX_D_DECL(r0[xDir] + i * dr[xDir] + shiftDualGrid[xDir],
                                           r0[yDir] + j * dr[yDir] + shiftDualGrid[yDir],
                                           r0[zDir] + k * dr[zDir] + shiftDualGrid[zDir])};
                    }

                    amrex::Real fval = 0.0;
                    GEMPIC_D_EXCL(int b{0};, int c{0};, )

                    GEMPIC_D_LOOP_BEGIN(
                        for (int a = 1 - degree / 2; a < degree / 2 + dirShift[xDir]; ++a),
                        for (int b = 1 - degree / 2; b < degree / 2 + dirShift[yDir]; ++b),
                        for (int c = 1 - degree / 2; c < degree / 2 + dirShift[zDir]; ++c))
                        fval += funcMF(i + a, j + b, k + c, n) *
                                GEMPIC_D_MULT((choose_eval<degree, grid, space, dir, xDir>(
                                                  a, rr[xDir], dr[xDir], r[xDir])),
                                              (choose_eval<degree, grid, space, dir, yDir>(
                                                  b, rr[yDir], dr[yDir], r[yDir])),
                                              (choose_eval<degree, grid, space, dir, zDir>(
                                                  c, rr[zDir], dr[zDir], r[zDir])));
                    GEMPIC_D_LOOP_END
                    valuesMF(i, j, k, n * nEvals + evalInd) = fval;
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

} // namespace Gempic::Forms

#endif