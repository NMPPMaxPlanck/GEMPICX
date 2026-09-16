/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <array>
#include <cmath>
#include <iostream>

#include <gtest/gtest.h>
#if defined(GEMPIC_USE_MKL)
#include <mkl.h>
#include <mkl_lapacke.h>
#elif defined(GEMPIC_USE_LAPACK)
#include <lapacke.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <AMReX.H>

#include "GEMPIC_FieldMethods.H"  // Psi0MomentGenerator and LUdcmp
#include "GEMPIC_Interpolation.H" // basis functions
#include "GEMPIC_NumTools.H"      // basis functions
#include "TestUtils/GEMPIC_TestUtils.H"

using namespace Gempic;
using namespace Forms;

namespace
{
template <typename IntType>
void print_int_vector (char const* desc, IntType n, IntType* a)
{
    IntType j;
    printf("\n %s\n", desc);
    for (j = 0; j < n; j++) printf(" %6i", a[j]);
    printf("\n");
}

TEST(LapackTest, SolveSmallSystem)
{
#ifndef GEMPIC_USE_LAPACK_OR_MKL
    GTEST_SKIP() << "No LAPACK tests because the LAPACK package is not available";
#else
    {
        constexpr amrex::Real sTol = 1e-9;

        // Solve Ax = b using LAPACK
        constexpr lapack_int nSize = 3;
        constexpr lapack_int mklN = static_cast<lapack_int>(nSize);

        {
            // Example: 3x3 linear system
            // A = [[4,1,2],[1,3,0],[2,0,1]], b = [4,5,6]
            constexpr lapack_int nrhs = 1;
            constexpr lapack_int ldb = nrhs; // for rwo major, this is the lenght of a row
            std::array<amrex::Real, nSize * nSize> A{4, 1, 2, 1, 3, 0, 2, 0, 1};
            std::array<amrex::Real, nSize * nSize> A_ref{4, 1, 2, 1, 3, 0, 2, 0, 1};
            std::array<std::array<amrex::Real, nSize>, nSize> Mat_ref{
                {{4, 1, 2}, {1, 3, 0}, {2, 0, 1}}};
            std::array<amrex::Real, ldb * nSize> b{4, 5, 6}; // this will be overwritten
            std::array<amrex::Real, ldb * nSize> b_ref{4, 5, 6};

            std::array<lapack_int, mklN> ipiv2; // pivot indices
            A = A_ref;
            b = b_ref;
            lapack_int info2 = LAPACKE_dgesv(LAPACK_ROW_MAJOR, mklN, nrhs, A.data(), mklN,
                                             ipiv2.data(), b.data(), ldb);
            ASSERT_EQ(info2, 0) << "LAPACK solve failed";

            std::array<amrex::Real, nSize> x_ref{29, -8, -52};
            std::array<amrex::Real, ldb * nSize> b_exp;
            std::array<amrex::Real, ldb * nSize> b_exp2;

            for (lapack_int i = 0; i < nSize; ++i)
            {
                b_exp[i] = 0.;
                b_exp2[i] = 0.;
                int offset = i * nSize;
                for (int j = 0; j < nSize; ++j)
                {
                    // row major product
                    b_exp[i] += A_ref[offset + j] * b[j];
                    b_exp2[i] += A_ref[offset + j] * x_ref[j];
                }
                EXPECT_NEAR(b[i], x_ref[i], sTol) << "Mismatch at index " << i;
                EXPECT_NEAR(b_exp[i], b_ref[i], sTol) << "Mismatch at index " << i;
                EXPECT_NEAR(b_exp2[i], b_ref[i], sTol) << "Mismatch at index " << i;
            }

            ///* Print coefficient matrix */
            //print_matrix_RM("Coefficient matrix", mklN, mklN, A_ref.data(), mklN);
            ///* Print solution */
            //print_matrix_RM("Solution", mklN, nrhs, b.data(), ldb);
            ///* Print details of LU factorization */
            //print_matrix_RM("Details of LU factorization", mklN, mklN, A.data(), mklN);
            ///* Print pivot indices */
            //print_int_vector("Pivot indices", mklN, ipiv2.data());
        }
        {
            // Example: 3x3 linear system
            // A = [[4,1,2],[1,3,0],[2,0,1]], b = [4,5,6]
            constexpr lapack_int nrhs = 1;
            constexpr lapack_int ldb = nSize; // for column major, this is the lenght of a column
            std::array<amrex::Real, nSize * nSize> A{4, 1, 2, 1, 3, 0, 2, 0, 1};
            std::array<amrex::Real, nSize * nSize> A_ref{4, 1, 2, 1, 3, 0, 2, 0, 1};
            std::array<std::array<amrex::Real, nSize>, nSize> Mat_ref{
                {{4, 1, 2}, {1, 3, 0}, {2, 0, 1}}};
            std::array<amrex::Real, ldb * nrhs> b{4, 5, 6}; // this will be overwritten
            std::array<amrex::Real, ldb * nrhs> b_ref{4, 5, 6};

            std::array<lapack_int, mklN> ipiv; // pivot indices

            lapack_int info = LAPACKE_dgesv(LAPACK_COL_MAJOR, mklN, nrhs, A.data(), mklN,
                                            ipiv.data(), b.data(), ldb);
            ASSERT_EQ(info, 0) << "LAPACK solve failed";

            // Expected solution computed manually: x ~ [0.2, 1.6, 2.2]
            std::array<amrex::Real, nSize> x_ref{29, -8, -52};
            std::array<amrex::Real, nSize> b_exp;
            std::array<amrex::Real, nSize> b_exp2;

            for (lapack_int i = 0; i < nSize; ++i)
            {
                b_exp[i] = 0.;
                b_exp2[i] = 0.;
                for (int j = 0; j < nSize; ++j)
                {
                    int offset = j * nSize;
                    b_exp[i] += A_ref[offset + i] * b[j];
                    b_exp2[i] += A_ref[offset + i] * x_ref[j];
                }
                EXPECT_NEAR(b[i], x_ref[i], sTol) << "Mismatch at index " << i;
                EXPECT_NEAR(b_exp[i], b_ref[i], sTol) << "Mismatch at index " << i;
                EXPECT_NEAR(b_exp2[i], b_ref[i], sTol) << "Mismatch at index " << i;
            }
        }
    }
#endif
}

constexpr amrex::Real tol = 1e-10;

// ---------------------------------------------------------
// Helpers
// ---------------------------------------------------------

template <int nSize, typename T>
std::array<T, nSize * nSize> make_test_matrix ()
{
    std::array<T, nSize * nSize> A{};

    for (int i = 0; i < nSize; ++i)
    {
        for (int j = 0; j < nSize; ++j)
        {
            A[i * nSize + j] = static_cast<T>((i + 1) * 10 + (j + 1));
        }
    }

    // diagonally dominant
    for (int i = 0; i < nSize; ++i)
    {
        A[i * nSize + i] += 100;
    }

    return A;
}

template <int nSize, typename T>
void expect_matrix_near (std::array<T, nSize * nSize> const& A,
                         std::array<T, nSize * nSize> const& B,
                         T eps = tol)
{
    for (int i = 0; i < nSize; ++i)
    {
        for (int j = 0; j < nSize; ++j)
        {
            EXPECT_NEAR(A[i * nSize + j], B[i * nSize + j], eps)
                << "Mismatch at (" << i << "," << j << ")";
        }
    }
}

template <int nSize, typename T>
std::array<T, nSize * nSize> multiply (std::array<T, nSize * nSize> const& A,
                                       std::array<T, nSize * nSize> const& B)
{
    std::array<T, nSize * nSize> C{};

    for (int i = 0; i < nSize; ++i)
    {
        for (int j = 0; j < nSize; ++j)
        {
            T sum = 0;
            for (int k = 0; k < nSize; ++k)
            {
                sum += A[i * nSize + k] * B[k * nSize + j];
            }
            C[i * nSize + j] = sum;
        }
    }

    return C;
}

template <int nSize, typename T>
std::array<T, nSize * nSize> identity ()
{
    std::array<T, nSize * nSize> I{};

    for (int i = 0; i < nSize; ++i)
    {
        I[i * nSize + i] = 1;
    }

    return I;
}

// ---------------------------------------------------------
// flatten/unflatten tests
// ---------------------------------------------------------

TEST(MatrixUtils1, FlattenUnflattenArrayRoundTrip)
{
    constexpr int nSize = 4;

    std::array<amrex::Real, nSize * nSize> aflat = make_test_matrix<nSize, amrex::Real>();
    std::array<amrex::Real, nSize * nSize> aflat2;
    std::array<std::array<amrex::Real, nSize>, nSize> anonFlat{};

    unflatten_rm(anonFlat, aflat);

    flatten_rm(aflat2, anonFlat);

    expect_matrix_near<nSize>(aflat, aflat2);
}

TEST(MatrixUtils2, FlattenRowMajorOrdering)
{
    constexpr int nSize = 3;

    std::array<std::array<int, nSize>, nSize> A{{{{1, 2, 3}}, {{4, 5, 6}}, {{7, 8, 9}}}};

    std::array<int, nSize * nSize> flat{};

    flatten_rm(flat, A);

    std::array<int, nSize * nSize> expected{1, 2, 3, 4, 5, 6, 7, 8, 9};

    EXPECT_EQ(flat, expected);
}

TEST(MatrixUtils3, FlattenUnflattenVectorRoundTrip)
{
    constexpr int nSize = 3;

    std::vector<std::vector<amrex::Real>> A{{11, 12, 13}, {21, 22, 23}, {31, 32, 33}};

    std::vector<amrex::Real> flat(nSize * nSize);

    flatten_rm(flat, A);

    std::vector<std::vector<amrex::Real>> B(nSize, std::vector<amrex::Real>(nSize));

    unflatten_rm(B, flat);

    for (int i = 0; i < nSize; ++i)
    {
        for (int j = 0; j < nSize; ++j)
        {
            EXPECT_NEAR(A[i][j], B[i][j], tol);
            EXPECT_NEAR(flat[i * nSize + j], B[i][j], tol);
        }
    }
}

// ---------------------------------------------------------
// LU solve tests
// ---------------------------------------------------------

TEST(LUdcmpTest1, SolveSingleRHS)
{
    constexpr int nSize = 3;

    std::array<amrex::Real, nSize * nSize> A{{4, 1, 2, 1, 3, 0, 2, 0, 1}};

    std::array<amrex::Real, nSize> b{{4, 5, 6}};

    LUdcmp<nSize, amrex::Real> lu(A);

    std::array<amrex::Real, nSize> x{};

    lu.solve(x, b);

    std::array<amrex::Real, nSize> expected{{29, -8, -52}};

    for (int i = 0; i < nSize; ++i)
    {
        EXPECT_NEAR(x[i], expected[i], tol);
    }
}

TEST(LUdcmpTest2, InverseProducesIdentity)
{
    constexpr int nSize = 4;

    std::array<amrex::Real, nSize * nSize> A{make_test_matrix<nSize, amrex::Real>()};

    LUdcmp<nSize, amrex::Real> lu(A);

    std::array<amrex::Real, nSize * nSize> ainv{};

    lu.inverse(ainv);

    std::array<amrex::Real, nSize * nSize> I = multiply<nSize>(A, ainv);

    expect_matrix_near<nSize>(I, identity<nSize, amrex::Real>(), 1e-9);
}

TEST(LUdcmpTest3, SolveSystemWrapper)
{
    constexpr int nSize = 3;

    std::array<amrex::Real, nSize * nSize> A = make_test_matrix<nSize, amrex::Real>();

    std::array<amrex::Real, nSize * nSize> B = identity<nSize, amrex::Real>();

    std::array<amrex::Real, nSize * nSize> X;

    solve_system<nSize>(X, A, B);

    std::array<amrex::Real, nSize * nSize> ax = multiply<nSize>(A, X);

    expect_matrix_near<nSize>(ax, B, 1e-9);
}

// ---------------------------------------------------------
// Runtime LU
// ---------------------------------------------------------

TEST(LUdcmpRuntimeTest, InverseMatchesCompiletimeVersion)
{
    constexpr int nSize = 4;

    std::array<amrex::Real, nSize * nSize> aArr = make_test_matrix<nSize, amrex::Real>();

    LUdcmp<nSize, amrex::Real> luStatic(aArr);

    std::array<amrex::Real, nSize * nSize> invStatic{};

    luStatic.inverse(invStatic);

    LUdcmpRuntime<amrex::Real> luRuntime(std::vector<amrex::Real>(aArr.begin(), aArr.end()));

    std::vector<amrex::Real> invRuntimeFlat;

    luRuntime.inverse(invRuntimeFlat);

    std::array<amrex::Real, nSize * nSize> invRuntime{};

    std::copy(invRuntimeFlat.begin(), invRuntimeFlat.end(), invRuntime.begin());

    expect_matrix_near<nSize>(invStatic, invRuntime, 1e-10);
}

// ---------------------------------------------------------
// Gauss-Jordan comparison
// ---------------------------------------------------------

TEST(GaussJordanTest, MatchesLUInverse)
{
    constexpr int nSize = 4;

    std::vector<std::vector<amrex::Real>> A{
        {10, 2, 3, 1}, {1, 11, 2, 3}, {2, 1, 12, 1}, {3, 2, 1, 13}};

    std::vector<std::vector<amrex::Real>> invGj(nSize, std::vector<amrex::Real>(nSize));

    matrix_inverse_gj(invGj, A);

    std::vector<std::vector<amrex::Real>> invLu(nSize, std::vector<amrex::Real>(nSize));

    matrix_inverse(invLu, A);

    for (int i = 0; i < nSize; ++i)
    {
        for (int j = 0; j < nSize; ++j)
        {
            EXPECT_NEAR(invGj[i][j], invLu[i][j], 1e-9);
        }
    }
}

// ---------------------------------------------------------
// Condition number
// ---------------------------------------------------------

TEST(ConditionNumberTest1, IdentityHasConditionOne)
{
    constexpr int nSize = 4;

    std::array<amrex::Real, nSize * nSize> I = identity<nSize, amrex::Real>();

    auto cond = condition_number<nSize>(I);

    EXPECT_NEAR(cond, 1.0, 1e-12);
}

TEST(ConditionNumberTest2, IllConditionedMatrixLarge)
{
    std::array<amrex::Real, 2 * 2> A{1.0, 1.0, 1.0, 1.0000000001};

    auto cond = condition_number<2>(A);

    EXPECT_GT(cond, 1e8);
}

// ---------------------------------------------------------
// Singular matrices
// ---------------------------------------------------------

TEST(LUdcmpTest4, SingularMatrixThrows)
{
    constexpr int nSize = 3;
    using LU = LUdcmp<nSize, amrex::Real>;

    std::array<amrex::Real, nSize * nSize> A{{1, 2, 3, 1, 2, 3, 4, 5, 6}};

    LU lu(A);

    EXPECT_NE(lu.status(), LU::Status::ok);
}

} //namespace