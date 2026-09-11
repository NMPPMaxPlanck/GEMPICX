/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <cmath>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "GEMPIC_Interpolation.H"
#include "GEMPIC_NumericalIntegrationDifferentiation.H"

using namespace Gempic;
using namespace Forms;

namespace
{

// Smooth test function (infinitely differentiable)
double f (double x) { return std::sin(2.0 * M_PI * x) + 0.5 * std::cos(4.0 * M_PI * x); }

// reconstruct at point x using stencil centered at i0
template <int nNodes>
double reconstruct (
    std::vector<double> const& grid, std::vector<double> const& u, int i0, double x, int shift)
{
    constexpr int sigma = nNodes / 2 - 1;

    double result = 0.0;

    for (int j = 0; j < nNodes; ++j)
    {
        int polyIndex = j - sigma;

        double lj = eval_lagrange<nNodes>(polyIndex, x, shift);
        result += u[i0 + polyIndex] * lj;
    }

    return result;
}

template <int nNodes>
int get_int_stencil_small (int const i)
{
    // i is the left node, and we build a stencil centered in the cell [i,i+1]
    // stencil covers 2n+1 cells and 2n+2 nodes!
    // i-n,..., i-1,i,i+1,...., i+n+1
    int const n = (nNodes - 2) / 2; // pol degree = 2n+1
    return i - n;
}
template <int nNodes>
int get_int_stencil_big (int const i)
{
    // i is the left node, and we build a stencil centered in the cell [i,i+1]
    // stencil covers 2n+1 cells and 2n+2 nodes!
    // i-n,..., i-1,i,i+1,...., i+n+1
    int const n = (nNodes - 2) / 2; // pol degree = 2n+1
    return i + n + 1;
}

template <int nNodes>
int get_hist_stencil_small (int const i)
{
    // i is the left node, and we build a stencil centered in the cell [i,i+1]
    // stencil covers 2n+1 cells and 2n+2 nodes!
    // i-n,..., i-1,i,i+1,...., i+n+1
    int const n = (nNodes - 2) / 2; // pol degree = 2n+1
    return i - n;
}
template <int nNodes>
int get_hist_stencil_big (int const i)
{
    // i is the left node, and we build a stencil centered in the cell [i,i+1]
    // stencil covers 2n+1 cells and 2n+2 nodes!
    // i-n,..., i-1,i,i+1,...., i+n+1
    int const n = (nNodes - 2) / 2; // pol degree = 2n+1
    return i + n;
}

template <int nNodes>
void run_convergence_test ()
{
    using GL = Gempic::GaussQuadratureUnit<nNodes>;

    constexpr int sigma = nNodes / 2 - 1;

    int const nGrids = 2;
    int baseGrid = 80;
    int const refRatio = 2;
    std::vector<int> resolutions;
    for (int iGrid = 0; iGrid < nGrids; ++iGrid)
    {
        resolutions.push_back(baseGrid);
        baseGrid *= refRatio;
    }
    std::vector<double> l2intErrorList;
    std::vector<double> l2histErrorList;

    double prevIntError = 0.0;
    double prevHistError = 0.0;

    for (int nGridNodes : resolutions)
    {
        double dx = 1.0 / nGridNodes;
        int const nGridCells = nGridNodes - 1;

        std::vector<double> x(nGridNodes + 1);
        std::vector<double> u(nGridNodes + 1);
        std::vector<double> xB(nGridNodes);
        std::vector<double> uB(nGridNodes);

        // initialize 1d grid
        for (int i = 0; i < nGridNodes + 1; ++i)
        {
            x[i] = i * dx;
            u[i] = f(x[i]);
        }
        // initialize 1d grid
        for (int i = 0; i < nGridNodes; ++i)
        {
            xB[i] = i * dx + 0.5 * dx;
            uB[i] = 0.0;
            amrex::Real uAve = 0.0;
            for (int iGP = 0; iGP < nNodes; ++iGP)
            {
                amrex::Real xi = x[i] + GL::s_nodes[iGP] * dx;
                uAve += GL::s_weights[iGP] * f(xi);
            }
            uB[i] = uAve * dx;
        }

        // reconstruction error sampled on local shifted stencil
        double l2intError = 0;
        double l2histError = 0;

        for (int i = 0; i < nGridNodes; ++i)
        {
            {
                // use best internal stencil for INTERPOLATING polynomial reconstruction.
                int iL = get_int_stencil_small<nNodes>(i);
                int iR = get_int_stencil_big<nNodes>(i);
                int stencilShift = 0;
                if (iL < 0)
                {
                    stencilShift = -iL; // shift rec. stencil to the right!
                }
                else if (iR > nGridNodes - 1)
                {
                    stencilShift = -(iR - nGridNodes + 1); // shift rec. stencil to the left!
                }
                double l2intErrLoc = 0;
                for (int iGP = 0; iGP < nNodes; ++iGP)
                {
                    double xGP = x[i] + GL::s_nodes[iGP] * dx;
                    double refSol = f(xGP);

                    double recEval = 0;
                    // stencil loop
                    for (int polyIndex = -sigma; polyIndex <= sigma + 1; ++polyIndex)
                    {
                        // always use the same polynomal basis on unit stencil, evaluated at the
                        // (unit!) shifted point
                        double lj = eval_lagrange<nNodes>(
                            polyIndex, GL::s_nodes[iGP] - stencilShift, stencilShift);
                        // remember we are using a shifted stencil, then use the shifted dof!
                        recEval += u[i + polyIndex + stencilShift] * lj;
                    }
                    double err = std::abs(recEval - refSol);
                    l2intErrLoc += GL::s_weights[iGP] * err * err;
                }
                l2intError += l2intErrLoc;
            }

            {
                // use best internal stencil for HISTPOLATING polynomial reconstruction.
                int iL = get_hist_stencil_small<nNodes>(i);
                int iR = get_hist_stencil_big<nNodes>(i);
                int stencilShift = 0;
                if (iL < 0)
                {
                    stencilShift = -iL; // shift rec. stencil to the right!
                }
                else if (iR > nGridCells - 1)
                {
                    stencilShift = -(iR - nGridCells + 1); // shift rec. stencil to the left!
                }
                double l2histErrLoc = 0;
                for (int iGP = 0; iGP < nNodes; ++iGP)
                {
                    double xGP = x[i] + GL::s_nodes[iGP] * dx;
                    double refSol = f(xGP);

                    double recEval = 0;
                    // stencil loop
                    for (int polyIndex = -sigma; polyIndex <= sigma; ++polyIndex)
                    {
                        // always use the same polynomal basis on unit stencil, evaluated at the
                        // (unit!) shifted point
                        double lj = eval_edge<nNodes>(polyIndex, GL::s_nodes[iGP] - stencilShift,
                                                      stencilShift);
                        // remember we are using a shifted stencil, then use the shifted dof!
                        recEval += uB[i + polyIndex + stencilShift] * lj / dx;
                    }
                    double err = std::abs(recEval - refSol);
                    l2histErrLoc += GL::s_weights[iGP] * err * err;
                }
                l2histError += l2histErrLoc;
            }
        }
        l2intError = sqrt(l2intError * dx);
        l2histError = sqrt(l2histError * dx);
        l2intErrorList.push_back(l2intError);
        l2histErrorList.push_back(l2histError);

        // convergence check (order = nNodes)
        if (prevIntError > 0.0)
        {
            double rate = std::log(prevIntError / l2intError) / std::log(refRatio);

            std::cout << "L2intError: " << l2intError << "   Observed order: " << rate << std::endl;

            EXPECT_GT(rate, nNodes - 0.3); // expected >= p+1
        }

        prevIntError = l2intError;

        // convergence check (order = nNodes)
        if (prevHistError > 0.0)
        {
            double rate = std::log(prevHistError / l2histError) / std::log(refRatio);

            std::cout << "L2histError: " << l2histError << "   Observed order: " << rate
                      << std::endl;

            EXPECT_GT(rate, nNodes - 1 - 0.3); // expected >= p
        }

        prevHistError = l2histError;
    }
}

using interpolationOrders = ::testing::Types<std::integral_constant<int, 2>,
                                             std::integral_constant<int, 4>,
                                             std::integral_constant<int, 6>,
                                             std::integral_constant<int, 8>,
                                             std::integral_constant<int, 10>>;
template <typename T>
class LagrangeConvergenceTest : public ::testing::Test
{
public:
    static constexpr int s_nNodes = T::value;
};

TYPED_TEST_SUITE(LagrangeConvergenceTest, interpolationOrders);

TYPED_TEST(LagrangeConvergenceTest, ShiftedInterpolationTest)
{
    constexpr int nNodes = TestFixture::s_nNodes;
    run_convergence_test<nNodes>();
}

} //namespace