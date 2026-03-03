/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <chrono>
#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_FDDeRhamComplex.H"
#include "TestUtils/GEMPIC_TestUtils.H"

namespace
{
using namespace Gempic;
using namespace Forms;

/**
 * @brief Tests cell2cell and the nodal2cell weno reconstructions against polynomials. Exact
 * reconstruction expected for polynomial degrees < polynomial reconstruction degree.
 */
template <typename IntDegreeWrapper>
class MatchGDECweightedMassMatrixTest : public testing::Test
{
public:
    static constexpr int s_histDegree{IntDegreeWrapper::value - 1};
    static constexpr int s_intDegree{IntDegreeWrapper::value};
    static constexpr int s_nQuadraturePoints{s_intDegree + 1};
    static constexpr int s_nShift{s_histDegree / 2};
    // int polyIndexes goes from {-n,...,0,...,n+1}, in total 2n+2 values.  s_intDegree +1
    // hist polyIndexes goes from {-n,...,0,...,n}, in total 2n+1 values.
    static constexpr int s_massStencilLength0{2 * s_intDegree + 1};
    static constexpr int s_leftStencilIdx0{-s_intDegree};
    static constexpr int s_massStencilLength1{2 * s_intDegree - 1};
    static constexpr int s_leftStencilIdx1{-s_histDegree};

    std::array<amrex::Real, s_massStencilLength0> m_hstencil0;
    std::array<amrex::Real, s_massStencilLength0> m_hstencilref0;
    std::array<std::array<std::array<amrex::Real, s_intDegree + 1>, s_intDegree + 1>,
               2 * s_intDegree + 1>
        m_wMassV0Stencil;

    std::array<amrex::Real, s_massStencilLength1> m_hstencil1;
    std::array<amrex::Real, s_massStencilLength1> m_hstencilref1;
    std::array<std::array<std::array<amrex::Real, s_intDegree + 1>, s_intDegree>,
               2 * s_intDegree - 1>
        m_wMassV1Stencil;

    MatchGDECweightedMassMatrixTest() :
        m_hstencil0{compute_v0_mass_matrix_1d_stencil_gdec<s_intDegree>()},
        m_hstencilref0{compute_v0_mass_matrix_1d_stencil_gdec<s_intDegree>()},
        m_wMassV0Stencil{compute_v0_weighted_mass_matrix_1d_stencil_gdec<s_intDegree>()},
        m_hstencil1{compute_v1_mass_matrix_1d_stencil_gdec<s_intDegree>()},
        m_hstencilref1{compute_v1_mass_matrix_1d_stencil_gdec<s_intDegree>()},
        m_wMassV1Stencil{compute_v1_weighted_mass_matrix_1d_stencil_gdec<s_intDegree>()}
    {
    }
};

using MyTypes = ::testing::Types<std::integral_constant<int, 1>,
                                 std::integral_constant<int, 3>,
                                 std::integral_constant<int, 5>,
                                 std::integral_constant<int, 7>,
                                 std::integral_constant<int, 9>,
                                 std::integral_constant<int, 11>,
                                 std::integral_constant<int, 13>,
                                 std::integral_constant<int, 15>>;

TYPED_TEST_SUITE(MatchGDECweightedMassMatrixTest, MyTypes);

TYPED_TEST(MatchGDECweightedMassMatrixTest, WeigthedMassV0Test)
{
    constexpr amrex::Real tol = 1e-13;

    for (int a = 0, sigma = this->s_leftStencilIdx0; a < this->s_massStencilLength0; ++a, ++sigma)
    {
        this->m_hstencilref0[a] = 0;
        for (int iCell = -this->s_nShift - 1, cellIdx = 0; iCell <= this->s_nShift;
             ++iCell, ++cellIdx)
        {
            //amrex::Real val{0.0};
            for (int g = 0; g < this->s_nQuadraturePoints; ++g)
            {
                this->m_hstencilref0[a] += this->m_wMassV0Stencil[a][cellIdx][g];
            }
        }
    }

    amrex::Real maxError{0.0};
    for (int a = 0, sigma = this->s_leftStencilIdx0; a < this->s_massStencilLength0; ++a, ++sigma)
    {
        maxError = std::max(maxError, std::abs(this->m_hstencilref0[a] - this->m_hstencil0[a]));
    }
    amrex::Print() << "error_GDEC_weightedMassMatrix_v0_test (int polynomial degree "
                   << this->s_intDegree << ')' << maxError << "\n";
    EXPECT_NEAR(maxError, 0.0, tol);
}

TYPED_TEST(MatchGDECweightedMassMatrixTest, WeigthedMassV1Test)
{
    constexpr amrex::Real tol = 1e-13;

    for (int a = 0, sigma = this->s_leftStencilIdx1; a < this->s_massStencilLength1; ++a, ++sigma)
    {
        this->m_hstencilref1[a] = 0;
        for (int iCell = -this->s_nShift, cellIdx = 0; iCell <= this->s_nShift; ++iCell, ++cellIdx)
        {
            //amrex::Real val{0.0};
            for (int g = 0; g < this->s_nQuadraturePoints; ++g)
            {
                this->m_hstencilref1[a] += this->m_wMassV1Stencil[a][cellIdx][g];
            }
        }
    }

    amrex::Real maxError{0.0};
    for (int a = 0, sigma = this->s_leftStencilIdx1; a < this->s_massStencilLength1; ++a, ++sigma)
    {
        maxError = std::max(maxError, std::abs(this->m_hstencilref1[a] - this->m_hstencil1[a]));
    }
    amrex::Print() << "error_GDEC_weightedMassMatrix_v1_test (hist polynomial degree "
                   << this->s_histDegree << ')' << maxError << "\n";
    EXPECT_NEAR(maxError, 0.0, tol);
}

} // namespace
