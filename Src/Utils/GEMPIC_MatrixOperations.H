#ifndef GEMPIC_MATRIXOPERATIONS_H_
#define GEMPIC_MATRIXOPERATIONS_H_

#include <math.h>

namespace Gempic
{

/**
 * just do matrix multiplication: aMatrix(a x b ) * bMatrix (b x c) = result(a x c).
 *
 * \param aMatrix
 * \param bMatrix
 * \return
 */
template <typename T>
std::vector<std::vector<T>> matmul (std::vector<std::vector<T>> const& aMatrix,
                                    std::vector<std::vector<T>> const& bMatrix)
{
    BL_PROFILE("GEMPIC::FVreconstruction::myMatmul(a,b) vec");
    int a = static_cast<int>(aMatrix.size());
    int b = static_cast<int>(bMatrix.size());
    int c = static_cast<int>(bMatrix[0].size());
    std::vector<std::vector<T>> result(a, std::vector<T>(c, 0.0));
    for (int i = 0; i < a; ++i)
    {
        for (int j = 0; j < c; ++j)
        {
            T sum = 0.0;
            for (int k = 0; k < b; ++k)
            {
                sum += aMatrix[i][k] * bMatrix[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}

template <int adim, int bdim, int cdim, typename T>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE
    amrex::Array2D<T, 0, adim - 1, 0, cdim - 1, amrex::Order::C>
    matmul (amrex::Array2D<T, 0, adim - 1, 0, bdim - 1, amrex::Order::C> const& aMatrix,
            amrex::Array2D<T, 0, bdim - 1, 0, cdim - 1, amrex::Order::C> const& bMatrix)
{
    BL_PROFILE_HOST_DEVICE("GEMPIC::FVreconstruction::myMatmul(a,b) arr");
    amrex::Array2D<T, 0, adim - 1, 0, cdim - 1, amrex::Order::C> result;
    for (int i = 0; i < adim; i++)
    {
        for (int j = 0; j < cdim; j++)
        {
            result(i, j) = 0.;
        }
    }
    for (int i = 0; i < adim; i++)
    {
        for (int k = 0; k < bdim; k++)
        {
            for (int j = 0; j < cdim; j++)
            {
                result(i, j) += aMatrix(i, k) * bMatrix(k, j);
            }
        }
    }
    return result;
}

template <typename T>
std::vector<std::vector<T>> mattranspose (std::vector<std::vector<T>> const& matrix)
{
    BL_PROFILE("GEMPIC::FVreconstruction::myTranspose() vec");
    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector<std::vector<T>> result(cols, std::vector<T>(rows));
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            result[i][j] = matrix[j][i];
        }
    }
    return result;
}

template <typename T>
std::vector<std::vector<T>> matsum (std::vector<std::vector<T>> const& aMatrix,
                                    std::vector<std::vector<T>> const& bMatrix)
{
    BL_PROFILE("GEMPIC::FVreconstruction::mySum() vec");
    int rows = aMatrix.size();
    int cols = aMatrix[0].size();
    std::vector<std::vector<T>> result(rows, std::vector<T>(cols, 0.0));
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result[i][j] = aMatrix[i][j] + bMatrix[i][j];
        }
    }
    return result;
}

} //namespace Gempic

#endif
