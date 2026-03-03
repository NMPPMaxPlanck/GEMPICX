/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <chrono>
#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_Fields.H"
#include "GEMPIC_Interpolation.H"
#include "GEMPIC_NumericalIntegrationDifferentiation.H"
#include "TestUtils/GEMPIC_TestUtils.H"

namespace
{
using namespace Gempic;
using namespace Forms;

template <int deg, int nvar>
using stencilArray2Dnode =
    amrex::Array2D<amrex::Real, 0, nvar - 1, 0, 2 * (deg + 1) - 2 + 1, amrex::Order::C>;
template <int deg, int nvar>
using stencilArray2D =
    amrex::Array2D<amrex::Real, 0, nvar - 1, 0, 2 * (deg + 1) - 2, amrex::Order::C>;
template <int deg, int nvar>
using dofArray2D = amrex::Array2D<amrex::Real, 0, nvar - 1, 0, deg + 1 - 1, amrex::Order::C>;

/**
 * @brief Tests cell2cell and the nodal2cell weno reconstructions against polynomials. Exact
 * reconstruction expected for polynomial degrees < polynomial reconstruction degree.
 */
template <typename HistDegreeWrapper>
class PolynomialReconstructionTest : public testing::Test
{
public:
    static constexpr int s_histDegree{HistDegreeWrapper::value};
    static constexpr int s_intDegree{HistDegreeWrapper::value + 1};
    static constexpr int s_nVarPde{s_histDegree + 2};
    static constexpr amrex::Real s_myconst{3.5};

    static constexpr int s_1dNodeStencilLo{-s_histDegree / 2};
    static constexpr int s_1dNodeStencilHi{s_histDegree / 2 + 1};
    static constexpr int s_1dCellStencilLo{-s_histDegree / 2};
    static constexpr int s_1dCellStencilHi{s_histDegree / 2};

    amrex::GpuArray<amrex::Real, 2 * s_intDegree> m_xMesh;
    amrex::GpuArray<amrex::Real, 2 * s_intDegree - 1> m_xCell;
    amrex::Real m_dx{1.0};

    PolynomialReconstructionTest()
    {
        for (int ix = s_1dNodeStencilLo, iloc = 0; ix < s_1dNodeStencilHi + 1; ++ix, ++iloc)
        {
            m_xMesh[iloc] = static_cast<amrex::Real>(ix) * m_dx;
        }
        for (int ix = s_1dCellStencilLo, iloc = 0; ix < s_1dCellStencilHi + 1; ++ix, ++iloc)
        {
            m_xCell[iloc] = (static_cast<amrex::Real>(ix) + 0.5) * m_dx;
        }
    }

    /*amrex::Real function_test (amrex::Real x) { return std::sin(2 * M_PI * x);
    }*/

    amrex::Real reference_function (amrex::Real x, int ivar)
    {
        return s_myconst * std::pow(x, ivar);
    }

    dofArray2D<s_intDegree, s_nVarPde> nodes_to_gl_reconstruction ()
    {
        // first get stencil nodes
        stencilArray2Dnode<s_histDegree, s_nVarPde> stencilVector1d;
        dofArray2D<s_intDegree, s_nVarPde> interpolatedValues;
        for (int iVar = 0; iVar < s_nVarPde; iVar++)
        {
            for (int ix = s_1dNodeStencilLo, iloc = 0; ix < s_1dNodeStencilHi + 1; ++ix, ++iloc)
            {
                stencilVector1d(iVar, iloc) = reference_function(m_xMesh[iloc], iVar);
            }
        }

        amrex::Real xk = {0.0};
        // now interpolate nodes to local GL quadrature points
        GaussQuadratureUnit<s_intDegree + 1> gL1;
        for (int ivar = 0; ivar < s_nVarPde; ++ivar)
        {
            for (int igl = 0; igl < s_intDegree + 1; ++igl)
            {
                amrex::Real x{xk + gL1.m_nodes[igl]};

                amrex::Real fval = 0.0;
                for (int a = s_1dNodeStencilLo; a < s_1dNodeStencilHi + 1; ++a)
                {
                    int localIdx = -s_1dNodeStencilLo + a;
                    fval += stencilVector1d(ivar, localIdx) *
                            eval_lagrange<s_intDegree + 1>(a, xk, m_dx, x);
                }
                interpolatedValues(ivar, igl) = fval;
            }
        }

        return interpolatedValues;
    }

    dofArray2D<s_histDegree, s_nVarPde> cells_to_gl_reconstruction ()
    {
        stencilArray2D<s_histDegree, s_nVarPde> stencilVector1d;
        dofArray2D<s_histDegree, s_nVarPde> interpolatedValues;
        GaussQuadratureUnit<s_histDegree + 1> gL1;
        for (int iVar = 0; iVar < s_nVarPde; iVar++)
        {
            for (int ix = s_1dCellStencilLo, iloc = 0; ix < s_1dCellStencilHi + 1; ++ix, ++iloc)
            {
                // compute cell average: gauss quadrature
                amrex::Real intVal{0.0};
                for (int igl = 0; igl < s_histDegree + 1; ++igl)
                {
                    amrex::Real locVal{reference_function(m_xMesh[iloc] + gL1.m_nodes[igl], iVar)};
                    intVal += locVal * gL1.m_weights[igl];
                }
                stencilVector1d(iVar, iloc) = intVal * m_dx;
            }
        }

        amrex::Real xk = {0.0};
        // now interpolate cell average to local GL quadrature points using histopolation
        for (int ivar = 0; ivar < s_nVarPde; ++ivar)
        {
            for (int igl = 0; igl < s_histDegree + 1; ++igl)
            {
                amrex::Real x{xk + gL1.m_nodes[igl]};

                amrex::Real fval = 0.0;
                for (int a = s_1dCellStencilLo; a < s_1dCellStencilHi + 1; ++a)
                {
                    int localIdx = -s_1dCellStencilLo + a;
                    fval += stencilVector1d(ivar, localIdx) *
                            eval_edge<s_intDegree + 1>(a, xk, m_dx, x);
                }
                interpolatedValues(ivar, igl) = fval;
            }
        }

        return interpolatedValues;
    }
};

using MyTypes = ::testing::Types<std::integral_constant<int, 0>,
                                 std::integral_constant<int, 2>,
                                 std::integral_constant<int, 4>,
                                 std::integral_constant<int, 6>,
                                 std::integral_constant<int, 8>,
                                 std::integral_constant<int, 10>>;

TYPED_TEST_SUITE(PolynomialReconstructionTest, MyTypes);

TYPED_TEST(PolynomialReconstructionTest, InterpolationTest)
{
    amrex::Real tol = 1e-6;
    constexpr int intDegree{
        TestFixture::s_intDegree}; // clang complains about "this->s_intDegree" not being a const
                                   // expression. Then we need TestFixture::s_intDegree
    constexpr int nVarPde{TestFixture::s_nVarPde};
    dofArray2D<intDegree, nVarPde> intReconstructionGl{this->nodes_to_gl_reconstruction()};
    amrex::Real linfIntError{0.0};
    GaussQuadratureUnit<intDegree + 1> gL1;
    for (int igl = 0; igl < intDegree + 1; ++igl)
    {
        for (int ivar = 0; ivar < intDegree + 1; ++ivar)
        {
            linfIntError =
                std::max(linfIntError,
                         std::abs(intReconstructionGl(ivar, igl) -
                                  this->reference_function(this->m_dx * gL1.m_nodes[igl], ivar)));
        }
    }

    amrex::Print() << "Linf_interpolation_error_test (int polynomial degree " << this->s_intDegree
                   << ')' << linfIntError << "\n";
    EXPECT_NEAR(linfIntError, 0.0, tol);
}

TYPED_TEST(PolynomialReconstructionTest, HistopolationTest)
{
    amrex::Real tol = 1e-6;
    constexpr int histDegree{TestFixture::s_histDegree};
    constexpr int nVarPde{TestFixture::s_nVarPde};

    dofArray2D<histDegree, nVarPde> histReconstructionGl{this->cells_to_gl_reconstruction()};
    amrex::Real linfIntError{0.0};
    // Access the Gauss-Legendre nodes for this degree from fixed arrays
    GaussQuadratureUnit<histDegree + 1> gL1;
    for (int igl = 0; igl < histDegree + 1; ++igl)
    {
        for (int ivar = 0; ivar < histDegree + 1; ++ivar)
        {
            linfIntError =
                std::max(linfIntError,
                         std::abs(histReconstructionGl(ivar, igl) -
                                  this->reference_function(this->m_dx * gL1.m_nodes[igl], ivar)));
        }
    }

    amrex::Print() << "Linf_histopolation_error_test (int polynomial degree " << this->s_histDegree
                   << ')' << linfIntError << "\n";
    EXPECT_NEAR(linfIntError, 0.0, tol);
}

using MyHistDegreeTypes = ::testing::Types<std::integral_constant<int, 0>,
                                           std::integral_constant<int, 2>,
                                           std::integral_constant<int, 4>>;
/**
 * @brief Test matching between eval_edge and eval_edge_arbitrary_order.
 */
template <typename HistDegreeWrapper>
class MatchArbitraryOrderReconstructionTest : public testing::Test
{
public:
    static constexpr int s_histDegree{HistDegreeWrapper::value};
    static constexpr int s_intDegree{HistDegreeWrapper::value + 1};
    static constexpr int s_points{100};
    static constexpr int s_pointsFiner{10000};
    static constexpr int s_1dCellStencilLo{-s_histDegree / 2};
    static constexpr int s_1dCellStencilHi{s_histDegree / 2};

    amrex::GpuArray<amrex::Real, s_points> m_xMesh;
    amrex::GpuArray<amrex::Real, s_points> m_xMeshUnit;
    amrex::GpuArray<amrex::Real, s_pointsFiner> m_xMeshFiner;
    amrex::GpuArray<amrex::Real, s_pointsFiner> m_xMeshUnitFiner;
    amrex::GpuArray<amrex::GpuArray<amrex::Real, s_histDegree + 1>, s_points> m_xValues;
    amrex::GpuArray<amrex::GpuArray<amrex::Real, s_histDegree + 1>, s_pointsFiner> m_xValuesFiner;
    amrex::GpuArray<amrex::GpuArray<amrex::Real, s_histDegree + 1>, s_points> m_xValuesOld;
    amrex::Real m_xLength{2.0};
    amrex::Real m_x0{0.3};
    amrex::Real m_dx{m_xLength / s_points};

    MatchArbitraryOrderReconstructionTest()
    {
        amrex::GpuArray<amrex::Real, s_histDegree + 1> zeroes;
        amrex::Real dxFiner{m_xLength / s_pointsFiner};
        amrex::Real dxUnit{1.0 / s_points};
        amrex::Real dxUnitFiner{1.0 / s_pointsFiner};
        zeroes.fill(0.0);
        m_xValuesOld.fill(zeroes);
        m_xValues.fill(zeroes);
        for (int i = 0; i < s_points; ++i)
        {
            m_xMesh[i] = m_x0 + static_cast<amrex::Real>(i) * m_dx;
            m_xMeshUnit[i] = static_cast<amrex::Real>(i) * dxUnit;
        }
        for (int i = 0; i < s_pointsFiner; ++i)
        {
            m_xMeshFiner[i] = m_x0 + static_cast<amrex::Real>(i) * dxFiner;
            m_xMeshUnitFiner[i] = static_cast<amrex::Real>(i) * dxUnitFiner;
        }
        m_xMesh[0] += 1e-14;
        m_xMesh[s_points - 1] -= 1e-14;
        m_xMeshUnit[0] += 1e-14;
        m_xMeshUnit[s_points - 1] -= 1e-14;
        m_xMeshFiner[0] += 1e-14;
        m_xMeshFiner[s_points - 1] -= 1e-14;
        m_xMeshUnitFiner[0] += 1e-14;
        m_xMeshUnitFiner[s_points - 1] -= 1e-14;
    }

    amrex::Real eval_edge_arbitrary_order_test ()
    {
        for (int i = 0; i < s_points; ++i)
        {
            for (int a = s_1dCellStencilLo; a < s_1dCellStencilHi + 1; ++a)
            {
                int localIdx = -s_1dCellStencilLo + a;
                m_xValues[i][localIdx] = eval_edge<s_intDegree + 1>(a, m_x0, m_xLength, m_xMesh[i]);
            }
            for (int a = s_1dCellStencilLo; a < s_1dCellStencilHi + 1; ++a)
            {
                int localIdx = -s_1dCellStencilLo + a;
                m_xValuesOld[i][localIdx] =
                    eval_edge_arbitrary_order<s_intDegree>(a, m_x0, m_xLength, m_xMesh[i]);
            }
        }
        amrex::Real linfIntError{0.0};
        for (int i = 0; i < s_points; ++i)
        {
            for (int a = s_1dCellStencilLo; a < s_1dCellStencilHi + 1; ++a)
            {
                int localIdx = -s_1dCellStencilLo + a;
                linfIntError = std::max(
                    linfIntError, std::abs(m_xValues[i][localIdx] - m_xValuesOld[i][localIdx]));
            }
        }
        return linfIntError;
    }

    amrex::Real eval_edge_arbitrary_order_unit_grid_test ()
    {
        for (int i = 0; i < s_points; ++i)
        {
            // test the local histopolation functions in the cell on a fine grid
            for (int a = s_1dCellStencilLo; a < s_1dCellStencilHi + 1; ++a)
            {
                int localIdx = -s_1dCellStencilLo + a;
                m_xValues[i][localIdx] = eval_edge<s_intDegree + 1>(a, m_xMeshUnit[i]);
            }
            for (int a = s_1dCellStencilLo; a < s_1dCellStencilHi + 1; ++a)
            {
                int localIdx = -s_1dCellStencilLo + a;
                m_xValuesOld[i][localIdx] =
                    eval_edge_arbitrary_order<s_intDegree>(a, m_xMeshUnit[i]);
            }
        }
        amrex::Real linfIntError{0.0};
        for (int i = 0; i < s_points; ++i)
        {
            for (int a = s_1dCellStencilLo; a < s_1dCellStencilHi + 1; ++a)
            {
                int localIdx = -s_1dCellStencilLo + a;
                linfIntError = std::max(
                    linfIntError, std::abs(m_xValues[i][localIdx] - m_xValuesOld[i][localIdx]));
            }
        }
        return linfIntError;
    }

    amrex::Real measure_comp_time_eval ()
    {
        auto startA = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < s_pointsFiner; ++i)
        {
            amrex::Real fval = 0.0;
            // test the local histopolation functions in the cell on a fine grid
            for (int a = s_1dCellStencilLo; a < s_1dCellStencilHi + 1; ++a)
            {
                int localIdx = -s_1dCellStencilLo + a;
                m_xValuesFiner[i][localIdx] =
                    eval_edge<s_intDegree + 1>(a, m_x0, m_xLength, m_xMeshFiner[i]);
            }
        }
        auto endA = std::chrono::high_resolution_clock::now();
        std::chrono::duration<amrex::Real> durationA = endA - startA;

        return durationA.count();
    }

    amrex::Real measure_comp_time_eval_arbitrary_order ()
    {
        auto startA = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < s_pointsFiner; ++i)
        {
            amrex::Real fval = 0.0;
            // test the local histopolation functions in the cell on a fine grid
            for (int a = s_1dCellStencilLo; a < s_1dCellStencilHi + 1; ++a)
            {
                int localIdx = -s_1dCellStencilLo + a;
                m_xValuesFiner[i][localIdx] =
                    eval_edge_arbitrary_order<s_intDegree>(a, m_x0, m_xLength, m_xMeshFiner[i]);
            }
        }
        auto endA = std::chrono::high_resolution_clock::now();
        std::chrono::duration<amrex::Real> durationA = endA - startA;

        return durationA.count();
    }

    amrex::Real measure_comp_time_eval_unitgrid ()
    {
        auto startA = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < s_pointsFiner; ++i)
        {
            amrex::Real fval = 0.0;
            // test the local histopolation functions in the cell on a fine grid
            for (int a = s_1dCellStencilLo; a < s_1dCellStencilHi + 1; ++a)
            {
                int localIdx = -s_1dCellStencilLo + a;
                m_xValuesFiner[i][localIdx] = eval_edge<s_intDegree + 1>(a, m_xMeshUnitFiner[i]);
            }
        }
        auto endA = std::chrono::high_resolution_clock::now();
        std::chrono::duration<amrex::Real> durationA = endA - startA;

        return durationA.count();
    }

    amrex::Real measure_comp_time_eval_arbitrary_order_unitgrid ()
    {
        auto startA = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < s_pointsFiner; ++i)
        {
            amrex::Real fval = 0.0;
            // test the local histopolation functions in the cell on a fine grid
            for (int a = s_1dCellStencilLo; a < s_1dCellStencilHi + 1; ++a)
            {
                int localIdx = -s_1dCellStencilLo + a;
                m_xValuesFiner[i][localIdx] =
                    eval_edge_arbitrary_order<s_intDegree>(a, m_xMeshUnitFiner[i]);
            }
        }
        auto endA = std::chrono::high_resolution_clock::now();
        std::chrono::duration<amrex::Real> durationA = endA - startA;

        return durationA.count();
    }
};

TYPED_TEST_SUITE(MatchArbitraryOrderReconstructionTest, MyHistDegreeTypes);

TYPED_TEST(MatchArbitraryOrderReconstructionTest, eval_edge_arbitrary_order_EdgeTest)
{
    amrex::Real tol = 1e-11;

    amrex::Real linfIntError{this->eval_edge_arbitrary_order_test()};

    amrex::Print() << "Linf_matching_error_test (int polynomial degree " << this->s_intDegree << ')'
                   << linfIntError << "\n";
    EXPECT_NEAR(linfIntError, 0.0, tol);
}

TYPED_TEST(MatchArbitraryOrderReconstructionTest, EvalEdgeUnitGridTest)
{
    amrex::Real tol = 1e-11;

    amrex::Real linfIntError{this->eval_edge_arbitrary_order_unit_grid_test()};
    amrex::Print() << "Linf_matching_error_test on unit grid (int polynomial degree "
                   << this->s_intDegree << ')' << linfIntError << "\n";
    EXPECT_NEAR(linfIntError, 0.0, tol);
}

//// commented tests. To be used to test new algorithms.
//TYPED_TEST(MatchArbitraryOrderReconstructionTest, ChronoEvalEdgeTest)
//{
//    amrex::Real timeOldEval{this->measure_comp_time_eval()};
//    amrex::Real timeEval{this->measure_comp_time_eval_arbitrary_order()};
//
//    int intDegreeLoc = this->s_intDegree;
//    amrex::Print() << "Time costs for eval_edge (int polynomial degree " << intDegreeLoc
//                   << ").  (t_OLD / t_NEW) => " << timeOldEval / timeEval << "\n";
//    EXPECT_LT(timeOldEval, timeEval);
//}
//
//TYPED_TEST(MatchArbitraryOrderReconstructionTest, ChronoEvalEdgeUnitGridTest)
//{
//    amrex::Real timeOldEval{this->measure_comp_time_eval_unitgrid()};
//    amrex::Real timeEval{this->measure_comp_time_eval_arbitrary_order_unitgrid()};
//
//    int intDegreeLoc = this->s_intDegree;
//    amrex::Print() << "Time costs for eval_edge on unit grid (int polynomial degree "
//                   << intDegreeLoc << ").  (t_OLD / t_NEW) => " << timeOldEval / timeEval
//                   << "\n";
//    EXPECT_LT(timeOldEval, timeEval);
//}

} // namespace
