/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef GEMPIC_FV_H
#define GEMPIC_FV_H

#include <cmath>
#include <iostream>
#include <vector>

#include <AMReX_VisMF.H>

#include "GEMPIC_Array.H"
#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_MatrixOperations.H"
#include "GEMPIC_NumTools.H"
#include "GEMPIC_PDE.H"
#include "GEMPIC_Parameters.H"

enum class ReconstructionMode
{
    CharacteristicVars, // do reconstruction in characteristic variables
    PrimitiveVars       // do reconstruction in primitive variables
};

using mylongprecision = amrex::Real; //long double;

template <int dim>
using squareMatrix = amrex::Array2D<amrex::Real, 0, dim - 1, 0, dim - 1, amrex::Order::C>;
template <int length, int dim>
using squareMatrixArray =
    amrex::Array3D<amrex::Real, 0, length - 1, 0, dim - 1, 0, dim - 1, amrex::Order::C>;

template <int deg>
using stencilArray1D = amrex::Array1D<amrex::Real, 0, 2 * deg>; // total 2*deg + 1
template <int deg>
using stencilArray1Dnode = amrex::Array1D<amrex::Real, 0, 2 * deg + 1>; //  total 2*(deg+1)
template <int deg, int nvar>
using stencilArray2D =
    amrex::Array2D<amrex::Real, 0, nvar - 1, 0, 2 * (deg + 1) - 2, amrex::Order::C>;
template <int deg, int nvar>
using stencilArray2Dnode =
    amrex::Array2D<amrex::Real, 0, nvar - 1, 0, 2 * (deg + 1) - 2 + 1, amrex::Order::C>;
template <int deg, int nvar>
using dofArray2D = amrex::Array2D<amrex::Real, 0, nvar - 1, 0, deg + 1 - 1, amrex::Order::C>;
template <int deg>
using dofArray1D = amrex::Array1D<amrex::Real, 0, deg + 1 - 1>;

// Template function to return std::array initialized with GEMPIC_D_PAD
template <int a, int b = a, int c = a>
constexpr int var_pad_0 (int const idx)
{
    std::array<int, 3> varPad{GEMPIC_D_PAD(a, b, c)};
    return varPad[idx];
}

////V3-weno RECONSTRUCTION - local dofs
//template <int deg, int nvar>
//using dofX_stencilYZ_Array3D = Gempic::Array4D<amrex::Real,
//                                               -var_pad_0<deg>(zDir),
//                                               var_pad_0<deg>(zDir),
//                                               -var_pad_0<deg>(yDir),
//                                               var_pad_0<deg>(yDir),
//                                               0,
//                                               var_pad_0<deg>(xDir),
//                                               0,
//                                               nvar - 1,
//                                               amrex::Order::C>;
//template <int deg, int nvar>
//using dofXY_stencilZ_Array3D = Gempic::Array4D<amrex::Real,
//                                               -var_pad_0<deg>(zDir),
//                                               var_pad_0<deg>(zDir),
//                                               0,
//                                               var_pad_0<deg>(yDir),
//                                               0,
//                                               var_pad_0<deg>(xDir),
//                                               0,
//                                               nvar - 1,
//                                               amrex::Order::C>;
//template <int deg, int nvar>
//using dofXYZ_Array3D = Gempic::Array4D<amrex::Real,
//                                       0,
//                                       var_pad_0<deg>(zDir),
//                                       0,
//                                       var_pad_0<deg>(yDir),
//                                       0,
//                                       var_pad_0<deg>(xDir),
//                                       0,
//                                       nvar - 1,
//                                       amrex::Order::C>;
////V2-weno RECONSTRUCTION - local dofs
//// V2x
//// V2y
//// V2z
//template <int deg>
//using dofX_V2x_stencilYZ_Array3D = amrex::Array3D<amrex::Real,
//                                                  -var_pad_0<deg>(zDir),
//                                                  var_pad_0<deg>(zDir),
//                                                  -var_pad_0<deg>(yDir),
//                                                  var_pad_0<deg>(yDir),
//                                                  0,
//                                                  var_pad_0<deg + 1>(xDir),
//                                                  amrex::Order::C>;
//template <int deg>
//using dofXY_V2x_stencilZ_Array3D = amrex::Array3D<amrex::Real,
//                                                  -var_pad_0<deg>(zDir),
//                                                  var_pad_0<deg>(zDir),
//                                                  0,
//                                                  var_pad_0<deg>(yDir),
//                                                  0,
//                                                  var_pad_0<deg + 1>(xDir),
//                                                  amrex::Order::C>;
//template <int deg>
//using dofXYZ_V2x_Array3D = amrex::Array3D<amrex::Real,
//                                          0,
//                                          var_pad_0<deg>(zDir),
//                                          0,
//                                          var_pad_0<deg>(yDir),
//                                          0,
//                                          var_pad_0<deg + 1>(xDir),
//                                          amrex::Order::C>;
//
//template <int deg>
//using dofX_V2y_stencilYZ_Array3D = amrex::Array3D<amrex::Real,
//                                                  -var_pad_0<deg>(zDir),
//                                                  var_pad_0<deg>(zDir),
//                                                  -var_pad_0<deg>(yDir),
//                                                  var_pad_0<deg + 1>(yDir),
//                                                  0,
//                                                  var_pad_0<deg>(xDir),
//                                                  amrex::Order::C>;
//template <int deg>
//using dofXY_V2y_stencilZ_Array3D = amrex::Array3D<amrex::Real,
//                                                  -var_pad_0<deg>(zDir),
//                                                  var_pad_0<deg>(zDir),
//                                                  0,
//                                                  var_pad_0<deg + 1>(yDir),
//                                                  0,
//                                                  var_pad_0<deg>(xDir),
//                                                  amrex::Order::C>;
//template <int deg>
//using dofXYZ_V2y_Array3D = amrex::Array3D<amrex::Real,
//                                          0,
//                                          var_pad_0<deg>(zDir),
//                                          0,
//                                          var_pad_0<deg + 1>(yDir),
//                                          0,
//                                          var_pad_0<deg>(xDir),
//                                          amrex::Order::C>;
//
//template <int deg>
//using dofX_V2z_stencilYZ_Array3D = amrex::Array3D<amrex::Real,
//                                                  -var_pad_0<deg>(zDir),
//                                                  var_pad_0<deg + 1>(zDir),
//                                                  -var_pad_0<deg>(yDir),
//                                                  var_pad_0<deg>(yDir),
//                                                  0,
//                                                  var_pad_0<deg>(xDir),
//                                                  amrex::Order::C>;
//template <int deg>
//using dofXY_V2z_stencilZ_Array3D = amrex::Array3D<amrex::Real,
//                                                  -var_pad_0<deg>(zDir),
//                                                  var_pad_0<deg + 1>(zDir),
//                                                  0,
//                                                  var_pad_0<deg>(yDir),
//                                                  0,
//                                                  var_pad_0<deg>(xDir),
//                                                  amrex::Order::C>;
//template <int deg>
//using dofXYZ_V2z_Array3D = amrex::Array3D<amrex::Real,
//                                          0,
//                                          var_pad_0<deg + 1>(zDir),
//                                          0,
//                                          var_pad_0<deg>(yDir),
//                                          0,
//                                          var_pad_0<deg>(xDir),
//                                          amrex::Order::C>;

//V3-weno RECONSTRUCTION - local dofs
template <int deg, int nvar>
using dofX_stencilYZ_Array3D = Gempic::Array4D<amrex::Real,
                                               -GEMPIC_D_PAD_Z(deg),
                                               GEMPIC_D_PAD_Z(deg),
                                               -GEMPIC_D_PAD_Y(deg),
                                               GEMPIC_D_PAD_Y(deg),
                                               0,
                                               GEMPIC_D_PAD_X(deg),
                                               0,
                                               nvar - 1,
                                               amrex::Order::C>;
template <int deg, int nvar>
using dofXY_stencilZ_Array3D = Gempic::Array4D<amrex::Real,
                                               -GEMPIC_D_PAD_Z(deg),
                                               GEMPIC_D_PAD_Z(deg),
                                               0,
                                               GEMPIC_D_PAD_Y(deg),
                                               0,
                                               GEMPIC_D_PAD_X(deg),
                                               0,
                                               nvar - 1,
                                               amrex::Order::C>;
template <int deg, int nvar>
using dofXYZ_Array3D = Gempic::Array4D<amrex::Real,
                                       0,
                                       GEMPIC_D_PAD_Z(deg),
                                       0,
                                       GEMPIC_D_PAD_Y(deg),
                                       0,
                                       GEMPIC_D_PAD_X(deg),
                                       0,
                                       nvar - 1,
                                       amrex::Order::C>;

//V2-weno RECONSTRUCTION - local dofs
// V2x
// V2y
// V2z
template <int deg>
using dofX_V2x_stencilYZ_Array3D = amrex::Array3D<amrex::Real,
                                                  -GEMPIC_D_PAD_Z(deg),
                                                  GEMPIC_D_PAD_Z(deg),
                                                  -GEMPIC_D_PAD_Y(deg),
                                                  GEMPIC_D_PAD_Y(deg),
                                                  0,
                                                  GEMPIC_D_PAD_X(deg + 1),
                                                  amrex::Order::C>;
template <int deg>
using dofXY_V2x_stencilZ_Array3D = amrex::Array3D<amrex::Real,
                                                  -GEMPIC_D_PAD_Z(deg),
                                                  GEMPIC_D_PAD_Z(deg),
                                                  0,
                                                  GEMPIC_D_PAD_Y(deg),
                                                  0,
                                                  GEMPIC_D_PAD_X(deg + 1),
                                                  amrex::Order::C>;
template <int deg>
using dofXYZ_V2x_Array3D = amrex::Array3D<amrex::Real,
                                          0,
                                          GEMPIC_D_PAD_Z(deg),
                                          0,
                                          GEMPIC_D_PAD_Y(deg),
                                          0,
                                          GEMPIC_D_PAD_X(deg + 1),
                                          amrex::Order::C>;

template <int deg>
using dofX_V2y_stencilYZ_Array3D = amrex::Array3D<amrex::Real,
                                                  -GEMPIC_D_PAD_Z(deg),
                                                  GEMPIC_D_PAD_Z(deg),
                                                  -GEMPIC_D_PAD_Y(deg),
                                                  GEMPIC_D_PAD_Y(deg + 1),
                                                  0,
                                                  GEMPIC_D_PAD_X(deg),
                                                  amrex::Order::C>;
template <int deg>
using dofXY_V2y_stencilZ_Array3D = amrex::Array3D<amrex::Real,
                                                  -GEMPIC_D_PAD_Z(deg),
                                                  GEMPIC_D_PAD_Z(deg),
                                                  0,
                                                  GEMPIC_D_PAD_Y(deg + 1),
                                                  0,
                                                  GEMPIC_D_PAD_X(deg),
                                                  amrex::Order::C>;
template <int deg>
using dofXYZ_V2y_Array3D = amrex::Array3D<amrex::Real,
                                          0,
                                          GEMPIC_D_PAD_Z(deg),
                                          0,
                                          GEMPIC_D_PAD_Y(deg + 1),
                                          0,
                                          GEMPIC_D_PAD_X(deg),
                                          amrex::Order::C>;

template <int deg>
using dofX_V2z_stencilYZ_Array3D = amrex::Array3D<amrex::Real,
                                                  -GEMPIC_D_PAD_Z(deg),
                                                  GEMPIC_D_PAD_Z(deg + 1),
                                                  -GEMPIC_D_PAD_Y(deg),
                                                  GEMPIC_D_PAD_Y(deg),
                                                  0,
                                                  GEMPIC_D_PAD_X(deg),
                                                  amrex::Order::C>;
template <int deg>
using dofXY_V2z_stencilZ_Array3D = amrex::Array3D<amrex::Real,
                                                  -GEMPIC_D_PAD_Z(deg),
                                                  GEMPIC_D_PAD_Z(deg + 1),
                                                  0,
                                                  GEMPIC_D_PAD_Y(deg),
                                                  0,
                                                  GEMPIC_D_PAD_X(deg),
                                                  amrex::Order::C>;
template <int deg>
using dofXYZ_V2z_Array3D = amrex::Array3D<amrex::Real,
                                          0,
                                          GEMPIC_D_PAD_Z(deg + 1),
                                          0,
                                          GEMPIC_D_PAD_Y(deg),
                                          0,
                                          GEMPIC_D_PAD_X(deg),
                                          amrex::Order::C>;

// Define NodesTable holding GpuArrays for degrees via parameter pack
template <int... degrees>
struct NodesTable
{
    std::tuple<amrex::GpuArray<amrex::Real, degrees + 1>...> m_nodes;

    // Initialize from std::vector<std::vector<amrex::Real>>& source
    void initialize_from_vectors (std::vector<std::vector<amrex::Real>> const& source)
    {
        initialize_from_vectors_impl(source, std::index_sequence<degrees...>{});
    }

    // Non-const access to nodes array by index
    template <std::size_t index>
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE auto& get_array ()
    {
        return std::get<index>(m_nodes);
    }

    // Const access overload
    template <std::size_t index>
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE auto const& get_array () const
    {
        return std::get<index>(m_nodes);
    }

private:
    template <std::size_t... is>
    void initialize_from_vectors_impl (std::vector<std::vector<amrex::Real>> const& source,
                                       std::index_sequence<is...>)
    {
        // expand using fold expression (C++17) to call copy one by one:
        (copy_vector_to_gpu_array<degrees>(source), ...);
    }

    template <int degree>
    void copy_vector_to_gpu_array (std::vector<std::vector<amrex::Real>> const& source)
    {
        if (degree >= static_cast<int>(source.size()))
        {
            throw std::runtime_error("Degree out of bounds in source vectors");
        }
        auto const& srcVec = source[degree];
        if (static_cast<int>(srcVec.size()) != degree + 1)
        {
            throw std::runtime_error("Source vector size does not match degree + 1");
        }
        auto& arr = std::get<degree>(m_nodes);
        std::copy(srcVec.begin(), srcVec.end(), arr.begin());
    }
};

// Helper to convert std::integer_sequence<int, ...> to NodesTable<int...>
template <typename Seq>
struct NodesTableFromSeq;

template <int... degrees>
struct NodesTableFromSeq<std::integer_sequence<int, degrees...>>
{
    using Type = NodesTable<degrees...>;
};

// Helper alias template
template <int maxDegree>
using MakeNodesTable =
    typename NodesTableFromSeq<std::make_integer_sequence<int, maxDegree + 1>>::Type;
/**
 * @brief Compute the unnormalized nonlinear WENO weight
 */
template <int nStencil>
class WENOweight
{
    static_assert(nStencil == 3, "WENOweight only supports nStencil == 3 for now.");
};

template <>
class WENOweight<3>
{
private:
    static constexpr int s_nStencil = 3;
    static constexpr int s_wenoR = 8;
    static constexpr amrex::Real s_wenoEpsilon = 1e-14;
    static constexpr amrex::Real s_wenoLambdaC = 1e5;

public:
    static AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE constexpr amrex::Real get_weno_lambda (int ind)
    {
        constexpr std::array<amrex::Real, 3> wenoLambda = {
            s_wenoLambdaC, 1.0, 1.0}; // must be defined here for device code
        return wenoLambda[ind];
    }

    static AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE constexpr amrex::Real
    compute_oscillation_indicator_to_weight (amrex::Real oscillationIndicator, int iStencil)
    {
        AMREX_ASSERT_WITH_MESSAGE(iStencil >= 0 && iStencil < s_nStencil,
                                  "WENO weight evaluation: stencil index out of bounds");
        amrex::Real denominator = amrex::Math::powi<s_wenoR>(oscillationIndicator + s_wenoEpsilon);
        return get_weno_lambda(iStencil) / denominator;
    }
    WENOweight() { return; };
};

template <int nVarFv, int polyrecDegreeFv>
class FVreconstruction
{
public:
    static constexpr int s_wenoNStencil = 3;  // do not change this.
    using WENO3 = WENOweight<s_wenoNStencil>; // do not confuse this with the method WENO3. the 3
                                              // is merely for the number of stencils.
    static constexpr int s_wenoHistNDof = polyrecDegreeFv + 1;
    static constexpr int s_wenoIntNDof = polyrecDegreeFv + 2;
    static constexpr int s_wenoStencilShift{polyrecDegreeFv};

    static constexpr int s_weno1dCellStencilLo{-polyrecDegreeFv};
    static constexpr int s_weno1dCellStencilHi{polyrecDegreeFv};
    static constexpr int s_weno1dNodeStencilLo{-polyrecDegreeFv};
    static constexpr int s_weno1dNodeStencilHi{polyrecDegreeFv + 1};

    static constexpr int s_volumeDof{AMREX_D_PICK(s_wenoHistNDof,
                                                  s_wenoHistNDof* s_wenoHistNDof,
                                                  s_wenoHistNDof* s_wenoHistNDof* s_wenoHistNDof)};
    static constexpr int s_faceDof{AMREX_D_PICK(1, s_wenoHistNDof, s_wenoHistNDof* s_wenoHistNDof)};
    static constexpr int s_edgeDof{AMREX_D_PICK(1, 1, s_wenoHistNDof)};

    // V3 polynomial degree Accessors
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int fv_degree (int iDir)
    {
        if (iDir == 0) return fv_degree_x();
        if (iDir == 1) return fv_degree_y();
        if (iDir == 2) return fv_degree_z();
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int fv_degree_x ()
    {
        return GEMPIC_D_PAD_X(polyrecDegreeFv);
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int fv_degree_y ()
    {
        return GEMPIC_D_PAD_Y(polyrecDegreeFv);
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int fv_degree_z ()
    {
        return GEMPIC_D_PAD_Z(polyrecDegreeFv);
    }

    // V3 nDof Accessors
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v3_n_dof (int iDir)
    {
        if (iDir == 0) return v3_n_dof_x();
        if (iDir == 1) return v3_n_dof_y();
        if (iDir == 2) return v3_n_dof_z();
        return -1; // Default return value for invalid iDir
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v3_n_dof_x ()
    {
        return GEMPIC_D_PAD_ONEX(s_wenoHistNDof);
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v3_n_dof_y ()
    {
        return GEMPIC_D_PAD_ONEY(s_wenoHistNDof);
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v3_n_dof_z ()
    {
        return GEMPIC_D_PAD_ONEZ(s_wenoHistNDof);
    }

    // V2x, V2y, V2z nDof Accessors
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v2x_n_dof (int iDir)
    {
        if (iDir == 0) return v2x_n_dof_x();
        if (iDir == 1) return v2x_n_dof_y();
        if (iDir == 2) return v2x_n_dof_z();
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v2x_n_dof_x ()
    {
        return GEMPIC_D_PAD_ONEX(s_wenoHistNDof + 1);
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v2x_n_dof_y ()
    {
        return GEMPIC_D_PAD_ONEY(s_wenoHistNDof);
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v2x_n_dof_z ()
    {
        return GEMPIC_D_PAD_ONEZ(s_wenoHistNDof);
    }

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v2y_n_dof (int iDir)
    {
        if (iDir == 0) return v2y_n_dof_x();
        if (iDir == 1) return v2y_n_dof_y();
        if (iDir == 2) return v2y_n_dof_z();
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v2y_n_dof_x ()
    {
        return GEMPIC_D_PAD_ONEX(s_wenoHistNDof);
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v2y_n_dof_y ()
    {
        return GEMPIC_D_PAD_ONEY(s_wenoHistNDof + 1);
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v2y_n_dof_z ()
    {
        return GEMPIC_D_PAD_ONEZ(s_wenoHistNDof);
    }

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v2z_n_dof (int iDir)
    {
        if (iDir == 0) return v2z_n_dof_x();
        if (iDir == 1) return v2z_n_dof_y();
        if (iDir == 2) return v2z_n_dof_z();
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v2z_n_dof_x ()
    {
        return GEMPIC_D_PAD_ONEX(s_wenoHistNDof);
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v2z_n_dof_y ()
    {
        return GEMPIC_D_PAD_ONEY(s_wenoHistNDof);
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v2z_n_dof_z ()
    {
        return GEMPIC_D_PAD_ONEZ(s_wenoHistNDof + 1);
    }

    // V3 Stencil Bounds
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v3_stencil_lo_x ()
    {
        return GEMPIC_D_PAD_X(s_weno1dCellStencilLo);
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v3_stencil_lo_y ()
    {
        return GEMPIC_D_PAD_Y(s_weno1dCellStencilLo);
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v3_stencil_lo_z ()
    {
        return GEMPIC_D_PAD_Z(s_weno1dCellStencilLo);
    }

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v3_stencil_hi_x ()
    {
        return GEMPIC_D_PAD_X(s_weno1dCellStencilHi);
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v3_stencil_hi_y ()
    {
        return GEMPIC_D_PAD_Y(s_weno1dCellStencilHi);
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v3_stencil_hi_z ()
    {
        return GEMPIC_D_PAD_Z(s_weno1dCellStencilHi);
    }

    // V2x Stencil Bounds
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v2x_stencil_lo_x ()
    {
        return GEMPIC_D_PAD_X(s_weno1dNodeStencilLo);
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v2x_stencil_lo_y ()
    {
        return GEMPIC_D_PAD_Y(s_weno1dCellStencilLo);
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v2x_stencil_lo_z ()
    {
        return GEMPIC_D_PAD_Z(s_weno1dCellStencilLo);
    }

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v2x_stencil_hi_x ()
    {
        return GEMPIC_D_PAD_X(s_weno1dNodeStencilHi);
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v2x_stencil_hi_y ()
    {
        return GEMPIC_D_PAD_Y(s_weno1dCellStencilHi);
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v2x_stencil_hi_z ()
    {
        return GEMPIC_D_PAD_Z(s_weno1dCellStencilHi);
    }

    // V2y Stencil Bounds
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v2y_stencil_lo_x ()
    {
        return GEMPIC_D_PAD_X(s_weno1dCellStencilLo);
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v2y_stencil_lo_y ()
    {
        return GEMPIC_D_PAD_Y(s_weno1dNodeStencilLo);
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v2y_stencil_lo_z ()
    {
        return GEMPIC_D_PAD_Z(s_weno1dCellStencilLo);
    }

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v2y_stencil_hi_x ()
    {
        return GEMPIC_D_PAD_X(s_weno1dCellStencilHi);
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v2y_stencil_hi_y ()
    {
        return GEMPIC_D_PAD_Y(s_weno1dNodeStencilHi);
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v2y_stencil_hi_z ()
    {
        return GEMPIC_D_PAD_Z(s_weno1dCellStencilHi);
    }

    // V2z Stencil Bounds
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v2z_stencil_lo_x ()
    {
        return GEMPIC_D_PAD_X(s_weno1dCellStencilLo);
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v2z_stencil_lo_y ()
    {
        return GEMPIC_D_PAD_Y(s_weno1dCellStencilLo);
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v2z_stencil_lo_z ()
    {
        return GEMPIC_D_PAD_Z(s_weno1dNodeStencilLo);
    }

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v2z_stencil_hi_x ()
    {
        return GEMPIC_D_PAD_X(s_weno1dCellStencilHi);
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v2z_stencil_hi_y ()
    {
        return GEMPIC_D_PAD_Y(s_weno1dCellStencilHi);
    }
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static constexpr int v2z_stencil_hi_z ()
    {
        return GEMPIC_D_PAD_Z(s_weno1dNodeStencilHi);
    }

    amrex::GpuArray<amrex::Real, s_wenoHistNDof + 1> m_wenoStencilLocNodes;

    static AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE constexpr amrex::Real get_weno_shift (int ind)
    {
        constexpr amrex::GpuArray<amrex::Real, s_wenoNStencil> wenoShift = {
            0.0, static_cast<amrex::Real>(s_wenoHistNDof / 2),
            -static_cast<amrex::Real>(s_wenoHistNDof / 2)};
        return wenoShift[ind];
    }

    /**
     * @brief  considering the large WENO stencil, as the union of the three local CENTRAL (ind=0),
     * LEFT (ind=1) and RIGHT (ind=2) stencils, get left relative index of the ind-th local stencil
     * with respect to the large WENO stencil. One can verify that barycentric or nodal left indexes
     * are the same.
     * @param ind int, labels the CENTRAL (ind=0), LEFT (ind=1) and RIGHT (ind=2) local stencil for
     * the polynomial reconstruction
     * @return int, the left index of the ind-th local stencil with respect to the large WENO
     * stencil.
     */
    static AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE constexpr int get_weno_left (int ind)
    {
        constexpr amrex::GpuArray<int, s_wenoNStencil> wenoLeft = {-(s_wenoHistNDof) / 2,
                                                                   -s_wenoHistNDof + 1, 0};
        return wenoLeft[ind];
    }

    /**
     * @brief Get the index of the integration direction associated with a face normal.
     *
     * Each face in 3D has a normal direction (x=0, y=1, z=2).
     * For integration over that face, two orthogonal directions are needed.
     *
     * The mapping is encoded as:
     *   - Face normal in x (0) -> integral directions y=1, z=2
     *   - Face normal in y (1) -> integral directions x=0, z=2
     *   - Face normal in z (2) -> integral directions x=0, y=1
     *
     * The input @p ind selects which of the two directions to return:
     *   - ind = 2*nDir     -> first integral direction
     *   - ind = 2*nDir + 1 -> second integral direction
     *
     * @param ind Integer in [0, 5], representing one of the two integration directions per face
     * normal.
     * @return The index (0=x, 1=y, 2=z) of the chosen integration direction.
     */
    static AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE constexpr int get_face_integral_idxs (int ind)
    {
        constexpr amrex::GpuArray<int, 6> faceIntegralIdxs = {1, 2, 0, 2, 0, 1};
        return faceIntegralIdxs[ind];
    }

    // For weno_stencil_loc_nodes, transformation not required because it is not constexpr.
    // Similarly for face_integral_idxs, transformation may be skipped if values are mutable.

    // basis conversion matrix
    // here the histopolating polynomial basis is thought to be the one defined over the stencil of
    // more cells; (dof = cell averages of the stencil) while the interpolating polynomial basis is
    // thought to be the one defined locally in the cell; (dof = interpolating Gauss-Legendre
    // quadrature points in the cell)

    // stencil averages-histopolating basis to local-in-cell interpolating basis
    squareMatrixArray<s_wenoNStencil, s_wenoHistNDof> m_wenOhist2LocalGLint;
    // stencil nodal-intepolating basis to local-in-cell interpolating basis
    squareMatrixArray<s_wenoNStencil, s_wenoIntNDof> m_wenOint2LocalGLint;

    squareMatrixArray<s_wenoNStencil, s_wenoHistNDof> m_wenoOiMatrix;
    squareMatrixArray<s_wenoNStencil, s_wenoIntNDof> m_wenoOiMatrix2;
    squareMatrix<s_wenoHistNDof> m_wenoOiMatrixC;
    squareMatrix<s_wenoIntNDof> m_wenoOiMatrix2C;

    //auto m_nodesGLeg_table = MakeNodesTable<polyrecDegreeFv + 2>{};
    MakeNodesTable<polyrecDegreeFv + 1> m_nodesGLegTable;
    MakeNodesTable<polyrecDegreeFv + 1> m_weightsGLegTable;

    amrex::GpuArray<amrex::Real, polyrecDegreeFv + 1> m_weightsGLegP;
    amrex::GpuArray<amrex::Real, polyrecDegreeFv + 1> m_nodesGLegP;

    // N degree for the 1d V1 space
    amrex::GpuArray<amrex::Real, polyrecDegreeFv + 1> m_phiN1;
    amrex::GpuArray<amrex::Real, polyrecDegreeFv + 1> m_phiN0;
    amrex::GpuArray<amrex::Real, polyrecDegreeFv + 1> m_phiNXi1;
    amrex::GpuArray<amrex::Real, polyrecDegreeFv + 1> m_phiNXi0;

    // M=N+1 degree for the 1d V0 space => N + 2 dofs
    amrex::GpuArray<amrex::Real, polyrecDegreeFv + 2> m_phiM1;
    amrex::GpuArray<amrex::Real, polyrecDegreeFv + 2> m_phiM0;
    amrex::GpuArray<amrex::Real, polyrecDegreeFv + 2> m_phiMXi1;
    amrex::GpuArray<amrex::Real, polyrecDegreeFv + 2> m_phiMXi0;

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE static int weno_dof_2dvect_to_array_indexes (int j,
                                                                                           int i)
    {
        constexpr int nDOF1Dx = (polyrecDegreeFv + 1);
        return nDOF1Dx * j + i;
    }

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void compute_cell2cell_reconstruction_1d(
        amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, polyrecDegreeFv, amrex::Order::C>& whWeno,
        amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, 2 * polyrecDegreeFv, amrex::Order::C> const&
            inputVector) const;

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void compute_cell2cell_reconstruction_1d(
        amrex::Array1D<amrex::Real, 0, polyrecDegreeFv>& whWeno,
        amrex::Array1D<amrex::Real, 0, 2 * polyrecDegreeFv> const& inputVector) const;

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void compute_cell2cell_reconstruction_1d(
        amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, polyrecDegreeFv, amrex::Order::C>& whWeno,
        amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, 2 * polyrecDegreeFv, amrex::Order::C> const&
            inputVector,
        squareMatrixArray<3, polyrecDegreeFv + 1> const& wenOhist2LocalGLintIn,
        squareMatrix<polyrecDegreeFv + 1> const& wenoOiMatrixCIn);

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void compute_cell2cell_reconstruction_1d(
        amrex::Array1D<amrex::Real, 0, polyrecDegreeFv>& whWeno,
        amrex::Array1D<amrex::Real, 0, 2 * polyrecDegreeFv> const& inputVector,
        squareMatrixArray<3, polyrecDegreeFv + 1> const& wenOhist2LocalGLintIn,
        squareMatrix<polyrecDegreeFv + 1> const& wenoOiMatrixCIn);

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void compute_node2cell_reconstruction_1d(
        amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, polyrecDegreeFv + 1, amrex::Order::C>& whWeno,
        amrex::Array2D<amrex::Real,
                       0,
                       nVarFv - 1,
                       0,
                       2 * (polyrecDegreeFv + 1) - 1,
                       amrex::Order::C> const& inputVector) const;

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void compute_node2cell_reconstruction_1d(
        amrex::Array1D<amrex::Real, 0, polyrecDegreeFv + 1>& whWeno,
        amrex::Array1D<amrex::Real, 0, 2 * (polyrecDegreeFv + 1) - 1> const& inputVector) const;

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void compute_node2cell_reconstruction_1d (
        amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, polyrecDegreeFv + 1, amrex::Order::C>& whWeno,
        amrex::Array2D<amrex::Real,
                       0,
                       nVarFv - 1,
                       0,
                       2 * (polyrecDegreeFv + 1) - 1,
                       amrex::Order::C> const& inputVector,
        squareMatrixArray<3, polyrecDegreeFv + 2> const& wenOint2LocalGLintIn,
        squareMatrix<polyrecDegreeFv + 2> const& wenoOiMatrix2CIn)
    {
        BL_PROFILE_HOST_DEVICE("GEMPIC::FVreconstruction::compute_node2cell_reconstruction_1d()");
        weno_node2cell_reconstruction_1d(whWeno, inputVector, wenOint2LocalGLintIn,
                                         wenoOiMatrix2CIn);
    }

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void compute_node2cell_reconstruction_1d (
        amrex::Array1D<amrex::Real, 0, polyrecDegreeFv + 1>& whWeno,
        amrex::Array1D<amrex::Real, 0, 2 * (polyrecDegreeFv + 1) - 1> const& inputVector,
        squareMatrixArray<3, polyrecDegreeFv + 2> const& wenOint2LocalGLintIn,
        squareMatrix<polyrecDegreeFv + 2> const& wenoOiMatrix2CIn)
    {
        BL_PROFILE_HOST_DEVICE("GEMPIC::FVreconstruction::compute_node2cell_reconstruction_1d()");
        weno_node2cell_reconstruction_1d(whWeno, inputVector, wenOint2LocalGLintIn,
                                         wenoOiMatrix2CIn);
    }

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void weno_reconstruction_1d(
        amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, polyrecDegreeFv, amrex::Order::C>& whWeno,
        amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, 2 * polyrecDegreeFv, amrex::Order::C> const&
            inputVector) const;

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void weno_reconstruction_1d(
        amrex::Array1D<amrex::Real, 0, polyrecDegreeFv>& whWeno,
        amrex::Array1D<amrex::Real, 0, 2 * polyrecDegreeFv> const& inputVector) const;

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void weno_reconstruction_1d (
        amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, polyrecDegreeFv, amrex::Order::C>& whWeno,
        amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, 2 * polyrecDegreeFv, amrex::Order::C> const&
            inputVector,
        squareMatrixArray<3, polyrecDegreeFv + 1> const& wenOhist2LocalGLintIn,
        squareMatrix<polyrecDegreeFv + 1> const& wenoOiMatrixCIn)
    {
        BL_PROFILE_HOST_DEVICE("GEMPIC::FVreconstruction::weno_reconstruction_1d()");
        // get local transformation matrices: stencil histopolating polynomials to local
        // interpolating polinomials
        amrex::Array3D<amrex::Real, 0, s_wenoNStencil - 1, 0, nVarFv - 1, 0, s_wenoHistNDof - 1,
                       amrex::Order::C>
            whRec;
        for (int iVar = 0; iVar < nVarFv; iVar++)
        {
            for (int i = 0; i < s_wenoHistNDof; i++)
            {
                whWeno(iVar, i) = 0.0;
            }
        }
        amrex::GpuArray<amrex::Real, s_wenoNStencil> wenoOi;
        amrex::GpuArray<amrex::Real, s_wenoNStencil> wenoOmega;

        for (int iVar = 0; iVar < nVarFv; iVar++)
        {
            for (int s = 0; s < s_wenoNStencil; s++)
            {
                wenoOi[s] = 0.0;
                amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, s_wenoHistNDof - 1, amrex::Order::C>
                    whLoc;
                int jSL = s_wenoStencilShift +
                          get_weno_left(s); // weno_left assumes the central cell being the zeroth;
                for (int i = 0; i < s_wenoHistNDof; i++)
                {
                    amrex::Real whSum = 0;
                    for (int j = 0; j < s_wenoHistNDof; j++)
                    {
                        int jloc = j + jSL;
                        whSum += wenOhist2LocalGLintIn(s, i, j) * inputVector(iVar, jloc);
                    }
                    whRec(s, iVar, i) = whSum;
                    whLoc(iVar, i) = whSum;
                }
                for (int i = 0; i < s_wenoHistNDof; i++)
                {
                    amrex::Real w1 = whLoc(iVar, i);
                    for (int j = 0; j < s_wenoHistNDof; j++)
                    {
                        amrex::Real factor = wenoOiMatrixCIn(i, j) * whLoc(iVar, j);
                        wenoOi[s] += w1 * factor;
                    }
                }
                wenoOmega[s] = WENO3::compute_oscillation_indicator_to_weight(wenoOi[s], s);
            }

            amrex::Real omegaSum = 0.0;
            for (int s = 0; s < s_wenoNStencil; s++)
            {
                omegaSum += wenoOmega[s];
            }

            for (int s = 0; s < s_wenoNStencil; s++)
            {
                wenoOmega[s] /= omegaSum;
                amrex::Real omegaLoc = wenoOmega[s];
                for (int i = 0; i < s_wenoHistNDof; i++)
                {
                    whWeno(iVar, i) += omegaLoc * whRec(s, iVar, i);
                }
            }
        }
    }

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void weno_reconstruction_1d (
        amrex::Array1D<amrex::Real, 0, polyrecDegreeFv>& whWeno,
        amrex::Array1D<amrex::Real, 0, 2 * polyrecDegreeFv> const& inputVector,
        squareMatrixArray<3, polyrecDegreeFv + 1> const& wenOhist2LocalGLintIn,
        squareMatrix<polyrecDegreeFv + 1> const& wenoOiMatrixCIn)
    {
        BL_PROFILE_HOST_DEVICE("GEMPIC::FVreconstruction::weno_reconstruction_1d()");
        amrex::Real omegaSum = 0.0;
        amrex::Array3D<amrex::Real, 0, s_wenoNStencil - 1, 0, 0, 0, s_wenoHistNDof - 1,
                       amrex::Order::C>
            whRec;
        for (int i = 0; i < s_wenoHistNDof; i++)
        {
            whWeno(i) = 0.0;
        }

        amrex::GpuArray<amrex::Real, s_wenoNStencil> wenoOi;
        amrex::GpuArray<amrex::Real, s_wenoNStencil> wenoOmega;
        for (int s = 0; s < s_wenoNStencil; s++)
        {
            wenoOi[s] = 0.0;
            amrex::Array1D<amrex::Real, 0, s_wenoHistNDof - 1> whLoc;
            int jSL = s_wenoStencilShift +
                      get_weno_left(s); // weno_left assumes the central cell being the zeroth;
            for (int i = 0; i < s_wenoHistNDof; i++)
            {
                amrex::Real whSum = 0;
                for (int j = 0; j < s_wenoHistNDof; j++)
                {
                    int jloc = j + jSL;
                    whSum += wenOhist2LocalGLintIn(s, i, j) * inputVector(jloc);
                }
                whRec(s, 0, i) = whSum;
                whLoc(i) = whSum;
            }
            for (int i = 0; i < s_wenoHistNDof; i++)
            {
                amrex::Real w1 = whLoc(i);
                for (int j = 0; j < s_wenoHistNDof; j++)
                {
                    amrex::Real factor = wenoOiMatrixCIn(i, j) * whLoc(j);
                    wenoOi[s] += w1 * factor;
                }
            }
            wenoOmega[s] = WENO3::compute_oscillation_indicator_to_weight(wenoOi[s], s);
            omegaSum += wenoOmega[s];
        }

        for (int s = 0; s < s_wenoNStencil; s++)
        {
            wenoOmega[s] /= omegaSum;
            amrex::Real omegaLoc = wenoOmega[s];
            for (int i = 0; i < s_wenoHistNDof; i++)
            {
                whWeno(i) += omegaLoc * whRec(s, 0, i);
            }
        }
    }

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void weno_node2cell_reconstruction_1d(
        amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, polyrecDegreeFv + 1, amrex::Order::C>& whWeno,
        amrex::Array2D<amrex::Real,
                       0,
                       nVarFv - 1,
                       0,
                       2 * (polyrecDegreeFv + 1) - 1,
                       amrex::Order::C> const& inputVector) const;

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void weno_node2cell_reconstruction_1d(
        amrex::Array1D<amrex::Real, 0, polyrecDegreeFv + 1>& whWeno,
        amrex::Array1D<amrex::Real, 0, 2 * (polyrecDegreeFv + 1) - 1> const& inputVector) const;

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void weno_node2cell_reconstruction_1d (
        amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, polyrecDegreeFv + 1, amrex::Order::C>& whWeno,
        amrex::Array2D<amrex::Real,
                       0,
                       nVarFv - 1,
                       0,
                       2 * (polyrecDegreeFv + 1) - 1,
                       amrex::Order::C> const& inputVector,
        squareMatrixArray<3, polyrecDegreeFv + 2> const& wenOint2LocalGLintIn,
        squareMatrix<polyrecDegreeFv + 2> const& wenoOiMatrix2CIn)
    {
        BL_PROFILE_HOST_DEVICE("GEMPIC::FVreconstruction::weno_node2cell_reconstruction_1d()");
        amrex::Real omegaSum = 0.0;
        amrex::Array3D<amrex::Real, 0, s_wenoNStencil - 1, 0, nVarFv - 1, 0, s_wenoIntNDof - 1,
                       amrex::Order::C>
            whRec;
        for (int iVar = 0; iVar < nVarFv; iVar++)
        {
            for (int i = 0; i < s_wenoIntNDof; i++)
            {
                whWeno(iVar, i) = 0.0;
            }
        }

        amrex::GpuArray<amrex::Real, s_wenoNStencil> wenoOi;
        amrex::GpuArray<amrex::Real, s_wenoNStencil> wenoOmega;
        for (int s = 0; s < s_wenoNStencil; s++)
        {
            wenoOi[s] = 0.0;
            amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, s_wenoIntNDof - 1, amrex::Order::C> whLoc;
            int jSL = s_wenoStencilShift +
                      get_weno_left(s); // weno_left assumes the central index being the zeroth,
                                        // which is placed at the weno_stencil_shift2-th position;
            for (int iVar = 0; iVar < nVarFv; iVar++)
            {
                for (int i = 0; i < s_wenoIntNDof; i++)
                {
                    amrex::Real whSum = 0;
                    for (int j = 0; j < s_wenoIntNDof; j++)
                    {
                        int jloc = j + jSL;
                        whSum += wenOint2LocalGLintIn(s, i, j) * inputVector(iVar, jloc);
                    }
                    whRec(s, iVar, i) = whSum; //wh_loc(i, iVar);
                    whLoc(iVar, i) = whSum;    //wh_loc(i, iVar);
                }
            }
            for (int iVar = 0; iVar < nVarFv; iVar++)
            {
                for (int i = 0; i < s_wenoIntNDof; i++)
                {
                    amrex::Real w1 = whLoc(iVar, i);
                    for (int j = 0; j < s_wenoIntNDof; j++)
                    {
                        amrex::Real factor = wenoOiMatrix2CIn(i, j) * whLoc(iVar, j);
                        wenoOi[s] += w1 * factor;
                    }
                }
            }
            wenoOmega[s] = WENO3::compute_oscillation_indicator_to_weight(wenoOi[s], s);
            omegaSum += wenoOmega[s];
        }

        for (int s = 0; s < s_wenoNStencil; s++)
        {
            wenoOmega[s] /= omegaSum;
            amrex::Real omegaLoc = wenoOmega[s];
            for (int i = 0; i < s_wenoIntNDof; i++)
            {
                for (int iVar = 0; iVar < nVarFv; iVar++)
                {
                    whWeno(iVar, i) += omegaLoc * whRec(s, iVar, i);
                }
            }
        }
    }

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void weno_node2cell_reconstruction_1d (
        amrex::Array1D<amrex::Real, 0, polyrecDegreeFv + 1>& whWeno,
        amrex::Array1D<amrex::Real, 0, 2 * (polyrecDegreeFv + 1) - 1> const& inputVector,
        squareMatrixArray<3, polyrecDegreeFv + 2> const& wenOint2LocalGLintIn,
        squareMatrix<polyrecDegreeFv + 2> const& wenoOiMatrix2CIn)
    {
        BL_PROFILE_HOST_DEVICE("GEMPIC::FVreconstruction::weno_node2cell_reconstruction_1d()");
        amrex::Real omegaSum = 0.0;
        amrex::Array3D<amrex::Real, 0, s_wenoNStencil - 1, 0, 0, 0, s_wenoIntNDof - 1,
                       amrex::Order::C>
            whRec;

        for (int i = 0; i < s_wenoIntNDof; i++)
        {
            whWeno(i) = 0.0;
        }

        amrex::GpuArray<amrex::Real, s_wenoNStencil> wenoOi;
        amrex::GpuArray<amrex::Real, s_wenoNStencil> wenoOmega;
        for (int s = 0; s < s_wenoNStencil; s++)
        {
            wenoOi[s] = 0.0;
            amrex::Array1D<amrex::Real, 0, s_wenoIntNDof - 1> whLoc;
            int jSL = s_wenoStencilShift +
                      get_weno_left(s); // weno_left assumes the central index being the zeroth,
                                        // which is placed at the weno_stencil_shift2-th position;

            for (int i = 0; i < s_wenoIntNDof; i++)
            {
                amrex::Real whSum = 0;
                for (int j = 0; j < s_wenoIntNDof; j++)
                {
                    int jloc = j + jSL;
                    whSum += wenOint2LocalGLintIn(s, i, j) * inputVector(jloc);
                }
                whRec(s, 0, i) = whSum;
                whLoc(i) = whSum;
            }
            for (int i = 0; i < s_wenoIntNDof; i++)
            {
                amrex::Real w1 = whLoc(i);
                for (int j = 0; j < s_wenoIntNDof; j++)
                {
                    amrex::Real factor = wenoOiMatrix2CIn(i, j) * whLoc(j);
                    wenoOi[s] += w1 * factor;
                }
            }
            wenoOmega[s] = WENO3::compute_oscillation_indicator_to_weight(wenoOi[s], s);
            omegaSum += wenoOmega[s];
        }

        for (int s = 0; s < s_wenoNStencil; s++)
        {
            wenoOmega[s] /= omegaSum;
            amrex::Real omegaLoc = wenoOmega[s];
            for (int i = 0; i < s_wenoIntNDof; i++)
            {
                whWeno(i) += omegaLoc * whRec(s, 0, i);
            }
        }
    }

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void polynomial_reconstruction_1d(
        amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, polyrecDegreeFv, amrex::Order::C>& whWeno,
        amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, 2 * polyrecDegreeFv, amrex::Order::C> const&
            inputVector) const;

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void tvd_reconstruction_1d(
        amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, polyrecDegreeFv, amrex::Order::C>& whWeno,
        amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, 2 * polyrecDegreeFv, amrex::Order::C> const&
            inputVector) const;
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void linear_reconstruction_1d(
        amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, polyrecDegreeFv, amrex::Order::C>& whWeno,
        amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, 2 * polyrecDegreeFv, amrex::Order::C> const&
            inputVector) const;

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void zero_reconstruction_1d(
        amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, polyrecDegreeFv, amrex::Order::C>& whWeno,
        amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, 2 * polyrecDegreeFv, amrex::Order::C> const&
            inputVector) const;

    template <int degree>
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void legendre_interpolating_weno_stencil_nodes(
        amrex::GpuArray<amrex::Real, degree + 1>& phi,
        amrex::GpuArray<amrex::Real, degree + 1>& phiXi,
        amrex::Real const xi);
    // RECURSIVE TEMPLATE METHOD: recursive_template_test
    template <int degree>
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void legendre_histopolating_weno_stencil_nodes(
        amrex::GpuArray<amrex::Real, degree + 1>& phi, amrex::Real const xi);
    template <int degree>
    void legendre_histopolating_weno_stencil_nodes(std::vector<amrex::Real>& phi,
                                                   amrex::Real const xi,
                                                   int degreeIn);

    template <typename ParamsClass, typename PdeTagType, ReconstructionMode userRecMode>
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void compute_weno_v3_reconstruction(
        dofXYZ_Array3D<polyrecDegreeFv, nVarFv>& whwenoxyz3d,
        amrex::Array4<amrex::Real const> const& w0,
        ParamsClass const& myparams,
        int const i,
        int const j,
        int const k) const;

    template <typename ParamsClass, typename PdeTagType, ReconstructionMode userRecMode>
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void compute_fv_fluxes(
        amrex::Array4<amrex::Real> const& F,
        amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof> const> const& ql,
        amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof> const> const& qr,
        ParamsClass const& myparams,
        int const idir,
        amrex::GpuArray<amrex::Real, 3> const& n,
        amrex::Real const dS,
        int const i,
        int const j,
        int const k) const;

    template <typename ParamsClass, typename PdeTagType>
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void compute_fv_fluxes_charvar(
        amrex::Array4<amrex::Real> const& F,
        amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof> const> const& ql,
        amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof> const> const& qr,
        ParamsClass const& myparams,
        int const idir,
        amrex::GpuArray<amrex::Real, 3> const& n,
        amrex::Real const dS,
        int const i,
        int const j,
        int const k) const;

    template <typename ParamsClass, typename PdeTagType>
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void compute_fv_fluxes_primvar(
        amrex::Array4<amrex::Real> const& F,
        amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof> const> const& ql,
        amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof> const> const& qr,
        ParamsClass const& myparams,
        int const idir,
        amrex::GpuArray<amrex::Real, 3> const& n,
        amrex::Real const dS,
        int const i,
        int const j,
        int const k) const;

    template <typename ParamsClass, typename PdeTagType>
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void compute_weno_v3_reconstruction_primvar(
        dofXYZ_Array3D<polyrecDegreeFv, nVarFv>& whwenoxyz3d,
        amrex::Array4<amrex::Real const> const& w0,
        ParamsClass const& myparams,
        int const i,
        int const j,
        int const k) const;

    template <typename ParamsClass, typename PdeTagType>
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void compute_weno_v3_reconstruction_charvar(
        dofXYZ_Array3D<polyrecDegreeFv, nVarFv>& whwenoxyz3d,
        amrex::Array4<amrex::Real const> const& w0,
        ParamsClass const& myparams,
        int const i,
        int const j,
        int const k) const;

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void compute_weno_v2x_reconstruction(
        dofXYZ_V2x_Array3D<polyrecDegreeFv>& bxhwenoxyz3d,
        amrex::Array4<amrex::Real const> const& bx0,
        amrex::Real const idFace,
        int const i,
        int const j,
        int const k) const;

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void compute_weno_v2y_reconstruction(
        dofXYZ_V2y_Array3D<polyrecDegreeFv>& byhwenoxyz3d,
        amrex::Array4<amrex::Real const> const& by0,
        amrex::Real const idFace,
        int const i,
        int const j,
        int const k) const;

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void compute_weno_v2z_reconstruction(
        dofXYZ_V2z_Array3D<polyrecDegreeFv>& bzhwenoxyz3d,
        amrex::Array4<amrex::Real const> const& bz0,
        amrex::Real const idFace,
        int const i,
        int const j,
        int const k) const;

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void compute_weno_v3_to_x_face_reconstruction(
        amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof>> const& qLx,
        amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof>> const& qRx,
        dofXYZ_Array3D<polyrecDegreeFv, nVarFv> const& whwenoxyz3d,
        int const i,
        int const j,
        int const k) const;

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void compute_weno_v3_to_y_face_reconstruction(
        amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof>> const& qLy,
        amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof>> const& qRy,
        dofXYZ_Array3D<polyrecDegreeFv, nVarFv> const& whwenoxyz3d,
        int const i,
        int const j,
        int const k) const;

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void compute_weno_v3_to_z_face_reconstruction(
        amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof>> const& qLz,
        amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof>> const& qRz,
        dofXYZ_Array3D<polyrecDegreeFv, nVarFv> const& whwenoxyz3d,
        int const i,
        int const j,
        int const k) const;

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void compute_weno_v2x_to_x_face_reconstruction(
        amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof>> const& qLx,
        amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof>> const& qRx,
        dofXYZ_V2x_Array3D<polyrecDegreeFv> const& v2xWenoxyz3d,
        int const nidx,
        int const i,
        int const j,
        int const k) const;

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void compute_weno_v2y_to_y_face_reconstruction(
        amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof>> const& qLy,
        amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof>> const& qRy,
        dofXYZ_V2y_Array3D<polyrecDegreeFv> const& v2yWenoxyz3d,
        int const nidx,
        int const i,
        int const j,
        int const k) const;

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void compute_weno_v2z_to_z_face_reconstruction(
        amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof>> const& qLz,
        amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof>> const& qRz,
        dofXYZ_V2z_Array3D<polyrecDegreeFv> const& v2zWenoxyz3d,
        int const nidx,
        int const i,
        int const j,
        int const k) const;

    template <int degree>
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void legendre_interpolating_g_leg_nodes_mp(
        amrex::GpuArray<amrex::Real, degree + 1>& phi,
        amrex::GpuArray<amrex::Real, degree + 1>& phiXi,
        amrex::Real const xi);

    template <int degree>
    struct LegendreWrapper
    {
        static void call(FVreconstruction& obj,
                         std::vector<amrex::Real>& phi,
                         std::vector<amrex::Real>& phiXi,
                         amrex::Real const xi);
    };

    // Compile-time dispatch table generator
    template <int... degrees>
    static constexpr auto make_table(std::integer_sequence<int, degrees...>);

    // Maximum degree you want to support
    static constexpr int s_maxDegree = polyrecDegreeFv + 1;

    // Runtime dispatcher
    void legendre_interpolating_vec(std::vector<amrex::Real>& phi,
                                    std::vector<amrex::Real>& phiXi,
                                    amrex::Real xi);

    FVreconstruction();
    AMREX_GPU_HOST_DEVICE ~FVreconstruction() = default;
};

/**
 * compute local 1d pw-linear (TVD) reconstruction, ginven the input 1d stencil
 * the local 1d reconstruction is written in terms of the dof of the local int.polynomials (GLeg
 * points)
 */
template <int nVarFv, int polyrecDegreeFv>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::tvd_reconstruction_1d(
    amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, polyrecDegreeFv, amrex::Order::C>& whWeno,
    amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, 2 * polyrecDegreeFv, amrex::Order::C> const&
        inputVector) const
{
    amrex::GpuArray<amrex::Real, nVarFv> slope;
    amrex::GpuArray<amrex::Real, nVarFv> q0;
    for (int iVar = 0; iVar < nVarFv; iVar++)
    {
        amrex::Real slopeL =
            inputVector(iVar, s_wenoStencilShift) - inputVector(iVar, s_wenoStencilShift - 1);
        amrex::Real slopeR =
            inputVector(iVar, s_wenoStencilShift + 1) - inputVector(iVar, s_wenoStencilShift);
        slope[iVar] = minmod(slopeL, slopeR);
        q0[iVar] = inputVector(iVar, s_wenoStencilShift);
    }

    for (int iVar = 0; iVar < nVarFv; iVar++)
    {
        for (int i = 0; i < s_wenoHistNDof; i++)
        {
            amrex::Real dxI = m_nodesGLegP[i] - 0.5;
            whWeno(iVar, i) = q0[iVar] + dxI * slope[iVar];
        }
    }
}

template <int nVarFv, int polyrecDegreeFv>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::linear_reconstruction_1d(
    amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, polyrecDegreeFv, amrex::Order::C>& whWeno,
    amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, 2 * polyrecDegreeFv, amrex::Order::C> const&
        inputVector) const
{
    amrex::GpuArray<amrex::Real, nVarFv> slope;
    amrex::GpuArray<amrex::Real, nVarFv> q0;
    for (int iVar = 0; iVar < nVarFv; iVar++)
    {
        amrex::Real slopeL =
            inputVector(iVar, s_wenoStencilShift) - inputVector(iVar, s_wenoStencilShift - 1);
        amrex::Real slopeR =
            inputVector(iVar, s_wenoStencilShift + 1) - inputVector(iVar, s_wenoStencilShift);
        slope[iVar] = 0.5 * (slopeL + slopeR);
        q0[iVar] = inputVector(iVar, s_wenoStencilShift);
    }

    for (int iVar = 0; iVar < nVarFv; iVar++)
    {
        for (int i = 0; i < s_wenoHistNDof; i++)
        {
            amrex::Real dxI = m_nodesGLegP[i] - 0.5;
            whWeno(iVar, i) = q0[iVar] + dxI * slope[iVar];
        }
    }
}

/**
 * compute local 1d pw-constant reconstruction, ginven the input 1d stencil
 * the local 1d reconstruction is written in terms of the dof of the local int.polynomials (GLeg
 * points)
 */
template <int nVarFv, int polyrecDegreeFv>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::zero_reconstruction_1d(
    amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, polyrecDegreeFv, amrex::Order::C>& whWeno,
    amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, 2 * polyrecDegreeFv, amrex::Order::C> const&
        inputVector) const
{
    amrex::GpuArray<amrex::Real, nVarFv> q0;
    for (int iVar = 0; iVar < nVarFv; iVar++)
    {
        q0[iVar] = inputVector(iVar, s_wenoStencilShift);
    }

    for (int iVar = 0; iVar < nVarFv; iVar++)
    {
        for (int i = 0; i < s_wenoHistNDof; i++)
        {
            whWeno(iVar, i) = q0[iVar];
        }
    }
}

/**  */
template <int nVarFv, int polyrecDegreeFv>
FVreconstruction<nVarFv, polyrecDegreeFv>::FVreconstruction()
{
    BL_PROFILE("GEMPIC::FVreconstruction::FVreconstruction()");
    static_assert((polyrecDegreeFv % 2 == 0), "polyrec_degree_FV must be an even number");

    std::vector<std::vector<amrex::Real>> mWeightsGLeg;
    std::vector<std::vector<amrex::Real>> mNodesGLeg;
    // auxiliary (vector) mass matrices
    std::vector<std::vector<std::vector<amrex::Real>>> mlocVec;
    std::vector<std::vector<std::vector<amrex::Real>>> iMlocVec;

    std::vector<amrex::Real> mlocGLeg[polyrecDegreeFv + 2];
    std::vector<amrex::Real> iMlocGLeg[polyrecDegreeFv + 2];
    // conversion from N+2 interpolating GL polynomials to N+1 interpolating GL polynomials
    std::vector<std::vector<amrex::Real>> intM2intN;
    std::vector<std::vector<amrex::Real>> hist2intLoc;
    std::vector<std::vector<amrex::Real>> int2histLoc;

    std::vector<std::vector<amrex::Real>> wenoRecStencil[s_wenoNStencil];

    std::vector<std::vector<amrex::Real>> phipsi, psiphi;
    std::vector<std::vector<amrex::Real>> mlocHist, iMlocHist;
    // these are for the histopolant weno:
    std::vector<std::vector<amrex::Real>> kloc[polyrecDegreeFv];
    std::vector<std::vector<amrex::Real>> iMKloc[polyrecDegreeFv];
    std::vector<std::vector<amrex::Real>> iMKlocPow[polyrecDegreeFv];
    std::vector<std::vector<amrex::Real>> iMKlocPowT[polyrecDegreeFv];
    // this are for the higher (+1) order interpolant weno:
    std::vector<std::vector<amrex::Real>> klocInt[polyrecDegreeFv + 1];
    std::vector<std::vector<amrex::Real>> iMKlocInt[polyrecDegreeFv + 1];
    std::vector<std::vector<amrex::Real>> iMKlocPowInt[polyrecDegreeFv + 1];
    std::vector<std::vector<amrex::Real>> iMKlocPowIntT[polyrecDegreeFv + 1];

    // define uniform local scaled ( dx = 1 ) stencil nodes of the FV reconstruction: weno3 has
    // [-1,0]U[0,1]U[1,2];
    for (int i = 0; i < s_wenoHistNDof + 1; i++)
    {
        m_wenoStencilLocNodes[i] = -static_cast<amrex::Real>((s_wenoHistNDof) / 2) + i;
    }

    // Initialize the list of vectors
    mlocVec = std::vector<std::vector<std::vector<amrex::Real>>>(polyrecDegreeFv + 2);
    iMlocVec = std::vector<std::vector<std::vector<amrex::Real>>>(polyrecDegreeFv + 2);
    mWeightsGLeg.resize(polyrecDegreeFv + 2);
    mNodesGLeg.resize(polyrecDegreeFv + 2);
    for (int i = 0; i < polyrecDegreeFv + 2; i++)
    {
        int iDOF = i + 1;
        mlocVec[i] = std::vector<std::vector<amrex::Real>>(iDOF, std::vector<amrex::Real>(iDOF));
        iMlocVec[i] = std::vector<std::vector<amrex::Real>>(iDOF, std::vector<amrex::Real>(iDOF));
        if (i < polyrecDegreeFv)
        {
            kloc[i] =
                std::vector<std::vector<amrex::Real>>(iDOF, std::vector<amrex::Real>(iDOF + 1));
            iMKloc[i] =
                std::vector<std::vector<amrex::Real>>(iDOF, std::vector<amrex::Real>(iDOF + 1));
            iMKlocPow[i] = std::vector<std::vector<amrex::Real>>(
                iDOF, std::vector<amrex::Real>(polyrecDegreeFv + 1));
        }
        if (i < polyrecDegreeFv + 1)
        {
            klocInt[i] =
                std::vector<std::vector<amrex::Real>>(iDOF, std::vector<amrex::Real>(iDOF + 1));
            iMKlocInt[i] =
                std::vector<std::vector<amrex::Real>>(iDOF, std::vector<amrex::Real>(iDOF + 1));
            iMKlocPowInt[i] = std::vector<std::vector<amrex::Real>>(
                iDOF, std::vector<amrex::Real>(polyrecDegreeFv + 2));
        }
        mlocGLeg[i] = std::vector<amrex::Real>(iDOF);
        iMlocGLeg[i] = std::vector<amrex::Real>(iDOF);
        mWeightsGLeg[i].resize(iDOF);
        mNodesGLeg[i].resize(iDOF);

        amrex::Real xL, xR;
        xL = 0.0;
        xR = 1.0;

        // compute the Gauss-Legendre quadrature nodes and weights
        std::tie(mNodesGLeg[i], mWeightsGLeg[i]) = gauss_legendre(xL, xR, iDOF);

        // Initialize the elements of the 2D vector
        for (int j = 0; j < iDOF; j++)
        {
            for (int k = 0; k < iDOF; k++)
            {
                mlocVec[i][j][k] = 0.0;
                iMlocVec[i][j][k] = 0.0;
            }
            mlocVec[i][j][j] = mWeightsGLeg[i][j];
            iMlocVec[i][j][j] = 1.0 / mWeightsGLeg[i][j];
            mlocGLeg[i][j] = mWeightsGLeg[i][j];
            iMlocGLeg[i][j] = 1.0 / mWeightsGLeg[i][j];
        }
    }
    for (int i = 0; i < polyrecDegreeFv + 1; i++)
    {
        m_nodesGLegP[i] = mNodesGLeg[polyrecDegreeFv][i];
        m_weightsGLegP[i] = mWeightsGLeg[polyrecDegreeFv][i];
    }

    m_nodesGLegTable.initialize_from_vectors(mNodesGLeg);
    m_weightsGLegTable.initialize_from_vectors(mWeightsGLeg);

    amrex::Real xL, xR;
    xL = 0.0;
    xR = 1.0;
    legendre_interpolating_g_leg_nodes_mp<polyrecDegreeFv>(m_phiN1, m_phiNXi1, xR);
    legendre_interpolating_g_leg_nodes_mp<polyrecDegreeFv>(m_phiN0, m_phiNXi0, xL);
    legendre_interpolating_g_leg_nodes_mp<polyrecDegreeFv + 1>(m_phiM1, m_phiMXi1, xR);
    legendre_interpolating_g_leg_nodes_mp<polyrecDegreeFv + 1>(m_phiM0, m_phiMXi0, xL);

    // operators for the N-histopolating weno reconstruction
    // projection matrix of the derivative of a polynomial to the smaller space:   psi_{i} *
    // dpsi_{i+1}  , i is the polynomial DEGREE. (2i+1) polynomial, i+1 points quadrature
    // formula is enough!
    for (int i = 0; i < polyrecDegreeFv; i++)
    {
        int iDOF = i + 1;
        std::vector<amrex::Real> basisLocalGlInt(iDOF + 1);
        std::vector<amrex::Real> basisLocalGlIntXi(iDOF + 1);
        std::vector<amrex::Real> basisLocalGlLowerInt(iDOF);
        std::vector<amrex::Real> basisLocalGlLowerIntXi(iDOF);
        for (int j = 0; j < iDOF; j++)
        {
            for (int k = 0; k < iDOF + 1; k++)
            {
                kloc[i][j][k] = 0.0;
                iMKloc[i][j][k] = 0.0;
            }
            for (int k = 0; k < polyrecDegreeFv + 1; k++)
            {
                iMKlocPow[i][j][k] = 0.0;
            }
        }
        for (int g = 0; g < iDOF; g++)
        {
            legendre_interpolating_vec(basisLocalGlInt, basisLocalGlIntXi, mNodesGLeg[i][g]);
            legendre_interpolating_vec(basisLocalGlLowerInt, basisLocalGlLowerIntXi,
                                       mNodesGLeg[i][g]);
            for (int j = 0; j < iDOF; j++)
            {
                for (int k = 0; k < iDOF + 1; k++)
                {
                    kloc[i][j][k] +=
                        mWeightsGLeg[i][g] * basisLocalGlLowerInt[j] * basisLocalGlIntXi[k];
                }
            }
        }
    }
    //multiply by inverse Mass matrix
    for (int i = 0; i < polyrecDegreeFv; i++)
    {
        iMKloc[i] = Gempic::matmul(iMlocVec[i], kloc[i]);
    }
    int locDegree = polyrecDegreeFv - 1;
    int locDof = locDegree + 1;
    //matrix for the oscillation indicator
    for (int j = 0; j < locDof; j++)
    {
        for (int k = 0; k < locDof + 1; k++)
        {
            iMKlocPow[locDegree][j][k] = iMKloc[locDegree][j][k];
        }
    }
    for (int i = locDegree - 1; i >= 0; i--)
    {
        iMKlocPow[i] = Gempic::matmul(iMKloc[i], iMKlocPow[i + 1]);
    }
    // operators for the M-interpolating weno reconstruction
    // projection matrix of the derivative of a polynomial to the smaller space:   psi_{i} *
    // dpsi_{i+1}  , i is the polynomial DEGREE. (2i+1) polynomial, i+1 points quadrature
    // formula is enough!
    for (int i = 0; i < polyrecDegreeFv + 1; i++)
    {
        int iDOF = i + 1;
        std::vector<amrex::Real> phiMLoc(iDOF + 1);
        std::vector<amrex::Real> phiMXiLoc(iDOF + 1);
        std::vector<amrex::Real> phiMmLoc(iDOF);
        std::vector<amrex::Real> phiMmXiLoc(iDOF);
        for (int j = 0; j < iDOF; j++)
        {
            for (int k = 0; k < iDOF + 1; k++)
            {
                klocInt[i][j][k] = 0.0;
                iMKlocInt[i][j][k] = 0.0;
            }
            for (int k = 0; k < polyrecDegreeFv + 2; k++)
            {
                iMKlocPowInt[i][j][k] = 0.0;
            }
        }

        for (int g = 0; g < iDOF; g++)
        {
            legendre_interpolating_vec(phiMLoc, phiMXiLoc, mNodesGLeg[i][g]);
            legendre_interpolating_vec(phiMmLoc, phiMmXiLoc, mNodesGLeg[i][g]);
            for (int j = 0; j < iDOF; j++)
            {
                for (int k = 0; k < iDOF + 1; k++)
                {
                    klocInt[i][j][k] += mWeightsGLeg[i][g] * phiMmLoc[j] * phiMXiLoc[k];
                }
            }
        }
    }
    //multiply by inverse Mass matrix
    for (int i = 0; i < polyrecDegreeFv + 1; i++)
    {
        iMKlocInt[i] = Gempic::matmul(iMlocVec[i], klocInt[i]);
    }
    locDegree = polyrecDegreeFv;
    locDof = locDegree + 1;
    //matrix for the oscillation indicator
    for (int j = 0; j < locDof; j++)
    {
        for (int k = 0; k < locDof + 1; k++)
        {
            iMKlocPowInt[locDegree][j][k] = iMKlocInt[locDegree][j][k];
        }
    }
    for (int i = locDegree - 1; i >= 0; i--)
    {
        iMKlocPowInt[i] = Gempic::matmul(iMKlocInt[i], iMKlocPowInt[i + 1]);
    }

    // now compute the conversion matrix histo to interpolating degrees of freedom
    // the histopolating dof are intended as cell-integral values over a local stencil;
    // the interpolating dof are intended as interpolating values on the local (in-cell) quadrature
    // points;
    // mass matrix (evaluated in the cell) and its inverse
    mlocHist = std::vector<std::vector<amrex::Real>>(
        polyrecDegreeFv + 1, std::vector<amrex::Real>(polyrecDegreeFv + 1, 0.0));
    iMlocHist = std::vector<std::vector<amrex::Real>>(
        polyrecDegreeFv + 1, std::vector<amrex::Real>(polyrecDegreeFv + 1, 0.0));
    phipsi = std::vector<std::vector<amrex::Real>>(
        polyrecDegreeFv + 1, std::vector<amrex::Real>(polyrecDegreeFv + 1, 0.0));
    psiphi = std::vector<std::vector<amrex::Real>>(
        polyrecDegreeFv + 1, std::vector<amrex::Real>(polyrecDegreeFv + 1, 0.0));

    for (int g = 0; g < polyrecDegreeFv + 1; g++)
    {
        amrex::Real weightLoc = mWeightsGLeg[polyrecDegreeFv][g];
        std::vector<amrex::Real> basisLocalGlInt(polyrecDegreeFv + 1);
        std::vector<amrex::Real> basisLocalGlIntXi(polyrecDegreeFv + 1);
        amrex::GpuArray<amrex::Real, polyrecDegreeFv + 1> basiswenostencilHist;
        legendre_interpolating_vec(basisLocalGlInt, basisLocalGlIntXi,
                                   mNodesGLeg[polyrecDegreeFv][g]);
        legendre_histopolating_weno_stencil_nodes<polyrecDegreeFv>(basiswenostencilHist,
                                                                   mNodesGLeg[polyrecDegreeFv][g]);

        for (int i = 0; i < polyrecDegreeFv + 1; i++)
        {
            amrex::Real psiIloc =
                basiswenostencilHist[i]; // histopolating basis function related to the weno-stencil
            amrex::Real phiIloc = basisLocalGlInt[i]; // interpolating basis function related to the
                                                      // cell-local GLeg quadrature points
            for (int j = 0; j < polyrecDegreeFv + 1; j++)
            {
                amrex::Real factor = weightLoc * basiswenostencilHist[j];
                mlocHist[i][j] += factor * psiIloc;
                phipsi[i][j] += factor * phiIloc;
            }
        }
    }

    // transpose
    psiphi = Gempic::mattranspose(phipsi);

#ifdef USE_MKL
    // Solve the system M * A = K
    solve_system_mkl(Mloc_hist, Id_pp, iMloc_hist, polyrec_degree_FV + 1);
    // basis conversion between stencil-histopolating basis and local GL quadrature points
    // interpolating basis.
    hist2int_loc = std::vector<std::vector<amrex::Real>>(polyrec_degree_FV + 1,
                                                         std::vector<amrex::Real>(weno_nDof, 0.0));
    int2hist_loc = std::vector<std::vector<amrex::Real>>(
        weno_nDof, std::vector<amrex::Real>(polyrec_degree_FV + 1, 0.0));
    // Solve the system M * A = K
    solve_system_mkl(Mloc_hist, psiphi, int2hist_loc, polyrec_degree_FV + 1);
    solve_system_mkl(Mloc_vec[polyrec_degree_FV], phipsi, hist2int_loc, polyrec_degree_FV + 1);
#else
    // compute IMloc_hist
    matrix_inverse(iMlocHist, mlocHist, polyrecDegreeFv + 1);

    // basis conversion between stencil-histopolating basis and local GL quadrature points
    // interpolating basis.
    hist2intLoc = std::vector<std::vector<amrex::Real>>(
        polyrecDegreeFv + 1, std::vector<amrex::Real>(s_wenoHistNDof, 0.0));
    int2histLoc = std::vector<std::vector<amrex::Real>>(
        s_wenoHistNDof, std::vector<amrex::Real>(polyrecDegreeFv + 1, 0.0));

    int2histLoc = Gempic::matmul(iMlocHist, psiphi);
    hist2intLoc = Gempic::matmul(iMlocVec[polyrecDegreeFv], phipsi);
#endif
    // we also need the transpose
    for (int i = 0; i < polyrecDegreeFv; i++)
    {
        int iDOF = i + 1;
        iMKlocPowT[i] = std::vector<std::vector<amrex::Real>>(polyrecDegreeFv + 1,
                                                              std::vector<amrex::Real>(iDOF));
        // transpose
        iMKlocPowT[i] = Gempic::mattranspose(iMKlocPow[i]);
    }
    for (int i = 0; i < polyrecDegreeFv + 1; i++)
    {
        int iDOF = i + 1;
        iMKlocPowIntT[i] = std::vector<std::vector<amrex::Real>>(polyrecDegreeFv + 2,
                                                                 std::vector<amrex::Real>(iDOF));
        // transpose
        iMKlocPowIntT[i] = Gempic::mattranspose(iMKlocPowInt[i]);
    }

    for (int s = 0; s < s_wenoNStencil; s++)
    {
        amrex::Real stencilShift = get_weno_shift(s);
        std::vector<std::vector<amrex::Real>> kwenohist2LocalGLint(
            s_wenoHistNDof, std::vector<amrex::Real>(s_wenoHistNDof, 0.0));
        std::vector<std::vector<amrex::Real>> kwenoint2LocalGLint(
            polyrecDegreeFv + 2, std::vector<amrex::Real>(polyrecDegreeFv + 2, 0.0));
        for (int g = 0; g < s_wenoHistNDof; g++)
        {
            amrex::Real weightLoc = mWeightsGLeg[polyrecDegreeFv][g];
            amrex::Real xiLoc = mNodesGLeg[polyrecDegreeFv][g] + stencilShift;
            std::vector<amrex::Real> basisLocalGlInt(s_wenoHistNDof);
            std::vector<amrex::Real> basisLocalGlIntXi(s_wenoHistNDof);
            amrex::GpuArray<amrex::Real, s_wenoHistNDof> basiswenostencilHist;
            legendre_interpolating_vec(basisLocalGlInt, basisLocalGlIntXi,
                                       mNodesGLeg[polyrecDegreeFv][g]);
            legendre_histopolating_weno_stencil_nodes<polyrecDegreeFv>(basiswenostencilHist, xiLoc);
            for (int i = 0; i < s_wenoHistNDof; i++)
            {
                amrex::Real phiIloc = basisLocalGlInt[i];
                for (int j = 0; j < s_wenoHistNDof; j++)
                {
                    amrex::Real factor = weightLoc * basiswenostencilHist[j];
                    kwenohist2LocalGLint[i][j] += factor * phiIloc;
                }
            }
        }
        for (int g = 0; g < s_wenoIntNDof; g++)
        {
            amrex::Real weightLoc = mWeightsGLeg[polyrecDegreeFv + 1][g];
            amrex::Real xiLoc = mNodesGLeg[polyrecDegreeFv + 1][g] + stencilShift;
            std::vector<amrex::Real> basisLocalGlInt(s_wenoIntNDof);
            std::vector<amrex::Real> basisLocalGlIntXi(s_wenoIntNDof);
            amrex::GpuArray<amrex::Real, s_wenoIntNDof> basiswenostencilInt;
            amrex::GpuArray<amrex::Real, s_wenoIntNDof> basiswenostencilIntXi;
            legendre_interpolating_vec(basisLocalGlInt, basisLocalGlIntXi,
                                       mNodesGLeg[polyrecDegreeFv + 1][g]);
            legendre_interpolating_weno_stencil_nodes<polyrecDegreeFv + 1>(
                basiswenostencilInt, basiswenostencilIntXi, xiLoc);
            for (int i = 0; i < s_wenoIntNDof; i++)
            {
                amrex::Real phiIloc = basisLocalGlInt[i];
                for (int j = 0; j < s_wenoIntNDof; j++)
                {
                    amrex::Real factor = weightLoc * basiswenostencilInt[j];
                    kwenoint2LocalGLint[i][j] += factor * phiIloc;
                }
            }
        }

        // from the i-th weno stencil hist dof, get local int. polynomial dofs
        std::vector<std::vector<amrex::Real>> wenOhist2LocalGLintLoc(
            s_wenoHistNDof, std::vector<amrex::Real>(s_wenoHistNDof));
        wenOhist2LocalGLintLoc = Gempic::matmul(iMlocVec[polyrecDegreeFv], kwenohist2LocalGLint);
        for (int i = 0; i < s_wenoHistNDof; i++)
        {
            for (int j = 0; j < s_wenoHistNDof; j++)
            {
                m_wenOhist2LocalGLint(s, i, j) = wenOhist2LocalGLintLoc[i][j];
                if (Gempic::Utils::Verbosity::level() > 1)
                {
                    printf("    ---     wenohist_2_localGLint(%d, %d, %d) = %f  \n", s, i, j,
                           m_wenOhist2LocalGLint(s, i, j));
                }
            }
        }
        // from the i-th weno stencil int dof, get local int. polynomial dofs
        std::vector<std::vector<amrex::Real>> wenOint2LocalGLintLoc(
            s_wenoIntNDof, std::vector<amrex::Real>(s_wenoIntNDof));
        wenOint2LocalGLintLoc = Gempic::matmul(iMlocVec[polyrecDegreeFv + 1], kwenoint2LocalGLint);
        for (int i = 0; i < s_wenoIntNDof; i++)
        {
            for (int j = 0; j < s_wenoIntNDof; j++)
            {
                m_wenOint2LocalGLint(s, i, j) = wenOint2LocalGLintLoc[i][j];
            }
        }

        // build HISTOPOLATING weno operator computing the sum of all k-norms for the oscillation
        // indicator (OI)
        std::vector<std::vector<amrex::Real>> wenoOiMatrixLoc(
            s_wenoHistNDof, std::vector<amrex::Real>(s_wenoHistNDof, 0.0));
        for (int n = 0; n < polyrecDegreeFv; n++)
        {
            std::vector<std::vector<amrex::Real>> iMKlocPow2(
                s_wenoHistNDof, std::vector<amrex::Real>(s_wenoHistNDof));
            std::vector<std::vector<amrex::Real>> temp(n + 1,
                                                       std::vector<amrex::Real>(s_wenoHistNDof));
            temp = Gempic::matmul(
                mlocVec[n],
                iMKlocPow[n]); //this account of the quadrature weights of the integral!
            iMKlocPow2 = Gempic::matmul(iMKlocPowT[n], temp);
            //
            std::vector<std::vector<amrex::Real>> atmp(s_wenoHistNDof,
                                                       std::vector<amrex::Real>(s_wenoHistNDof));
            atmp = Gempic::matmul(iMKlocPow2, wenOhist2LocalGLintLoc);
            std::vector<std::vector<amrex::Real>> wenOhist2LocalGLintT(
                s_wenoHistNDof, std::vector<amrex::Real>(s_wenoHistNDof));
            wenOhist2LocalGLintT = Gempic::mattranspose(wenOhist2LocalGLintLoc);
            std::vector<std::vector<amrex::Real>> wenoOiMatrixLocTmp(
                s_wenoHistNDof, std::vector<amrex::Real>(s_wenoHistNDof));
            wenoOiMatrixLocTmp = Gempic::matmul(wenOhist2LocalGLintT, atmp);
            wenoOiMatrixLoc =
                Gempic::matsum(wenoOiMatrixLoc, wenoOiMatrixLocTmp); //, weno_nDof, weno_nDof);
        }

        for (int i = 0; i < s_wenoHistNDof; i++)
        {
            for (int j = 0; j < s_wenoHistNDof; j++)
            {
                m_wenoOiMatrix(s, i, j) = wenoOiMatrixLoc[i][j];
            }
        }

        // build INTERPOLATING weno operator computing the sum of all k-norms for the oscillation
        // indicator (OI)
        //weno_OI_matrix_loc.resize(weno_nDof2, std::vector<amrex::Real>(weno_nDof2, 0.0));  BUG,
        //only resizes rows
        wenoOiMatrixLoc.resize(s_wenoIntNDof);
        for (int i = 0; i < s_wenoIntNDof; ++i)
        {
            wenoOiMatrixLoc[i].resize(s_wenoIntNDof, 0.0);
        }
        for (int n = 0; n < polyrecDegreeFv + 1; n++)
        {
            std::vector<std::vector<amrex::Real>> iMKlocPow2(
                s_wenoIntNDof, std::vector<amrex::Real>(s_wenoIntNDof));
            std::vector<std::vector<amrex::Real>> temp(n + 1,
                                                       std::vector<amrex::Real>(s_wenoIntNDof));
            temp = Gempic::matmul(
                mlocVec[n],
                iMKlocPowInt[n]); //this account of the quadrature weights of the integral!
            iMKlocPow2 = Gempic::matmul(iMKlocPowIntT[n], temp);
            //
            std::vector<std::vector<amrex::Real>> atmp(s_wenoIntNDof,
                                                       std::vector<amrex::Real>(s_wenoIntNDof));
            atmp = Gempic::matmul(iMKlocPow2, wenOint2LocalGLintLoc);
            std::vector<std::vector<amrex::Real>> wenOint2LocalGLintT(
                s_wenoIntNDof, std::vector<amrex::Real>(s_wenoIntNDof));
            wenOint2LocalGLintT = Gempic::mattranspose(wenOint2LocalGLintLoc);
            std::vector<std::vector<amrex::Real>> wenoOiMatrixLocTmp(
                s_wenoIntNDof, std::vector<amrex::Real>(s_wenoIntNDof));
            wenoOiMatrixLocTmp = Gempic::matmul(wenOint2LocalGLintT, atmp);
            wenoOiMatrixLoc = Gempic::matsum(wenoOiMatrixLoc, wenoOiMatrixLocTmp);
        }
        for (int i = 0; i < s_wenoIntNDof; i++)
        {
            for (int j = 0; j < s_wenoIntNDof; j++)
            {
                m_wenoOiMatrix2(s, i, j) = wenoOiMatrixLoc[i][j];
            }
        }
    }

    // build HISTOPOLATING weno operator computing the sum of all k-norms for the oscillation
    // indicator (OI)
    std::vector<std::vector<amrex::Real>> wenoOiMatrixLoc(
        s_wenoHistNDof, std::vector<amrex::Real>(s_wenoHistNDof, 0.0));
    for (int n = 0; n < polyrecDegreeFv; n++)
    {
        std::vector<std::vector<amrex::Real>> wenoOiMatrixLocTmp(
            s_wenoHistNDof, std::vector<amrex::Real>(s_wenoHistNDof));
        std::vector<std::vector<amrex::Real>> temp(n + 1, std::vector<amrex::Real>(s_wenoHistNDof));
        temp =
            Gempic::matmul(mlocVec[n],
                           iMKlocPow[n]); //this account of the quadrature weights of the integral!
        wenoOiMatrixLocTmp = Gempic::matmul(iMKlocPowT[n], temp);
        wenoOiMatrixLoc = Gempic::matsum(wenoOiMatrixLoc, wenoOiMatrixLocTmp);
    }

    for (int i = 0; i < s_wenoHistNDof; i++)
    {
        for (int j = 0; j < s_wenoHistNDof; j++)
        {
            m_wenoOiMatrixC(i, j) = wenoOiMatrixLoc[i][j];
        }
    }

    // build INTERPOLATING weno operator computing the sum of all k-norms for the oscillation
    // indicator (OI)
    wenoOiMatrixLoc.resize(s_wenoIntNDof);
    for (int i = 0; i < s_wenoIntNDof; ++i)
    {
        wenoOiMatrixLoc[i].resize(s_wenoIntNDof, 0.0);
    }
    for (int n = 0; n < polyrecDegreeFv + 1; n++)
    {
        std::vector<std::vector<amrex::Real>> wenoOiMatrixLocTmp(
            s_wenoIntNDof, std::vector<amrex::Real>(s_wenoIntNDof));
        std::vector<std::vector<amrex::Real>> temp(n + 1, std::vector<amrex::Real>(s_wenoIntNDof));
        temp = Gempic::matmul(
            mlocVec[n],
            iMKlocPowInt[n]); //this account of the quadrature weights of the integral!
        wenoOiMatrixLocTmp = Gempic::matmul(iMKlocPowIntT[n], temp);
        wenoOiMatrixLoc = Gempic::matsum(wenoOiMatrixLoc, wenoOiMatrixLocTmp);
    }

    for (int i = 0; i < s_wenoIntNDof; i++)
    {
        for (int j = 0; j < s_wenoIntNDof; j++)
        {
            m_wenoOiMatrix2C(i, j) = wenoOiMatrixLoc[i][j];
        }
    }
    // TO DO: move this to unit tests...
    stencilArray2D<polyrecDegreeFv, nVarFv> stencilVectorX;
    dofArray2D<polyrecDegreeFv, nVarFv> whWenoX;
    /*
        x-RECONSTRUCTION
    */
    for (int iVar = 0; iVar < nVarFv; iVar++)
    {
        for (int ishift = -s_wenoStencilShift, iloc = 0; iloc < 2 * s_wenoHistNDof - 1;
             iloc++, ishift++)
        {
            amrex::Real xLoc = 0.1 * iloc;
            stencilVectorX(iVar, iloc) = 4 * xLoc * xLoc + 3 * xLoc + 700.1;
        }
    }

    // testing
    zero_reconstruction_1d(whWenoX, stencilVectorX);
    tvd_reconstruction_1d(whWenoX, stencilVectorX);
    linear_reconstruction_1d(whWenoX, stencilVectorX);
    weno_reconstruction_1d(whWenoX, stencilVectorX);
    polynomial_reconstruction_1d(whWenoX, stencilVectorX);
}

// MAIN CALL TO THE CHOSEN RECONSTRUCTION
template <int nVarFv, int polyrecDegreeFv>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::compute_cell2cell_reconstruction_1d(
    amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, polyrecDegreeFv, amrex::Order::C>& whWeno,
    amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, 2 * polyrecDegreeFv, amrex::Order::C> const&
        inputVector) const
{
    BL_PROFILE_HOST_DEVICE("GEMPIC::FVreconstruction::compute_cell2cell_reconstruction_1d()");
    //// compute the reconstruction:
    //// 1st order accurate
    //ZERO_reconstruction_1d(wh_weno, InputVector);
    //// 2nd order accurate (without limiter!)
    //linear_reconstruction_1d(wh_weno, InputVector);
    //// 2nd order accurate (1st order at extrema (TVD!))
    //TVD_reconstruction_1d(wh_weno, InputVector);
    //// nth order accurate (without limiter!)
    //polynomial_reconstruction_1d(wh_weno, InputVector);
    //// nth order accurate (weno)
    weno_reconstruction_1d(whWeno, inputVector);
}

template <int nVarFv, int polyrecDegreeFv>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::compute_cell2cell_reconstruction_1d(
    amrex::Array1D<amrex::Real, 0, polyrecDegreeFv>& whWeno,
    amrex::Array1D<amrex::Real, 0, 2 * polyrecDegreeFv> const& inputVector) const
{
    BL_PROFILE_HOST_DEVICE("GEMPIC::FVreconstruction::compute_cell2cell_reconstruction_1d()");
    weno_reconstruction_1d(whWeno, inputVector);
}

template <int nVarFv, int polyrecDegreeFv>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::compute_cell2cell_reconstruction_1d(
    amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, polyrecDegreeFv, amrex::Order::C>& whWeno,
    amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, 2 * polyrecDegreeFv, amrex::Order::C> const&
        inputVector,
    squareMatrixArray<3, polyrecDegreeFv + 1> const& wenOhist2LocalGLintIn,
    squareMatrix<polyrecDegreeFv + 1> const& wenoOiMatrixCIn)
{
    BL_PROFILE_HOST_DEVICE("GEMPIC::FVreconstruction::compute_cell2cell_reconstruction_1d()");
    // compute the reconstruction:
    weno_reconstruction_1d(whWeno, inputVector, wenOhist2LocalGLintIn, wenoOiMatrixCIn);
}

template <int nVarFv, int polyrecDegreeFv>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::compute_cell2cell_reconstruction_1d(
    amrex::Array1D<amrex::Real, 0, polyrecDegreeFv>& whWeno,
    amrex::Array1D<amrex::Real, 0, 2 * polyrecDegreeFv> const& inputVector,
    squareMatrixArray<3, polyrecDegreeFv + 1> const& wenOhist2LocalGLintIn,
    squareMatrix<polyrecDegreeFv + 1> const& wenoOiMatrixCIn)
{
    BL_PROFILE_HOST_DEVICE("GEMPIC::FVreconstruction::compute_cell2cell_reconstruction_1d()");
    // compute the reconstruction:
    weno_reconstruction_1d(whWeno, inputVector, wenOhist2LocalGLintIn, wenoOiMatrixCIn);
}

// MAIN CALL TO THE CHOSEN RECONSTRUCTION
template <int nVarFv, int polyrecDegreeFv>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::compute_node2cell_reconstruction_1d(
    amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, polyrecDegreeFv + 1, amrex::Order::C>& whWeno,
    amrex::Array2D<amrex::Real,
                   0,
                   nVarFv - 1,
                   0,
                   2 * (polyrecDegreeFv + 1) - 1,
                   amrex::Order::C> const& inputVector) const
{
    BL_PROFILE_HOST_DEVICE("GEMPIC::FVreconstruction::compute_node2cell_reconstruction_1d()");
    //// compute the reconstruction:
    //// 1st order accurate
    //ZERO_reconstruction_1d(wh_weno, InputVector);
    //// 2nd order accurate (without limiter!)
    //linear_reconstruction_1d(wh_weno, InputVector);
    //// 2nd order accurate (1st order at extrema (TVD!))
    //TVD_reconstruction_1d(wh_weno, InputVector);
    //// nth order accurate (without limiter!)
    //polynomial_reconstruction_1d(wh_weno, InputVector);
    //// nth order accurate (weno)
    weno_node2cell_reconstruction_1d(whWeno, inputVector);
}

template <int nVarFv, int polyrecDegreeFv>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::compute_node2cell_reconstruction_1d(
    amrex::Array1D<amrex::Real, 0, polyrecDegreeFv + 1>& whWeno,
    amrex::Array1D<amrex::Real, 0, 2 * (polyrecDegreeFv + 1) - 1> const& inputVector) const
{
    BL_PROFILE_HOST_DEVICE("GEMPIC::FVreconstruction::compute_node2cell_reconstruction_1d()");
    weno_node2cell_reconstruction_1d(whWeno, inputVector);
}

template <int nVarFv, int polyrecDegreeFv>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::weno_reconstruction_1d(
    amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, polyrecDegreeFv, amrex::Order::C>& whWeno,
    amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, 2 * polyrecDegreeFv, amrex::Order::C> const&
        inputVector) const
{
    BL_PROFILE_HOST_DEVICE("GEMPIC::FVreconstruction::weno_reconstruction_1d()");
    // get local transformation matrices: stencil histopolating polynomials to local interpolating
    // polinomials
    amrex::Array3D<amrex::Real, 0, s_wenoNStencil - 1, 0, nVarFv - 1, 0, s_wenoHistNDof - 1,
                   amrex::Order::C>
        whRec;
    for (int iVar = 0; iVar < nVarFv; iVar++)
    {
        for (int i = 0; i < s_wenoHistNDof; i++)
        {
            whWeno(iVar, i) = 0.0;
        }
    }
    amrex::GpuArray<amrex::Real, s_wenoNStencil> wenoOi;
    amrex::GpuArray<amrex::Real, s_wenoNStencil> wenoOmega;

    for (int iVar = 0; iVar < nVarFv; iVar++)
    {
        for (int s = 0; s < s_wenoNStencil; s++)
        {
            wenoOi[s] = 0.0;
            amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, s_wenoHistNDof - 1, amrex::Order::C>
                whLoc;
            int jSL = s_wenoStencilShift +
                      get_weno_left(s); // weno_left assumes the central cell being the zeroth;
            for (int i = 0; i < s_wenoHistNDof; i++)
            {
                amrex::Real whSum = 0;
                for (int j = 0; j < s_wenoHistNDof; j++)
                {
                    int jloc = j + jSL;
                    whSum += m_wenOhist2LocalGLint(s, i, j) * inputVector(iVar, jloc);
                }
                whRec(s, iVar, i) = whSum;
                whLoc(iVar, i) = whSum;
            }
            for (int i = 0; i < s_wenoHistNDof; i++)
            {
                amrex::Real w1 = whLoc(iVar, i);
                for (int j = 0; j < s_wenoHistNDof; j++)
                {
                    amrex::Real factor = m_wenoOiMatrixC(i, j) * whLoc(iVar, j);
                    wenoOi[s] += w1 * factor;
                }
            }
            wenoOmega[s] = WENO3::compute_oscillation_indicator_to_weight(wenoOi[s], s);
        }

        amrex::Real omegaSum = 0.0;
        for (int s = 0; s < s_wenoNStencil; s++)
        {
            omegaSum += wenoOmega[s];
        }

        for (int s = 0; s < s_wenoNStencil; s++)
        {
            wenoOmega[s] /= omegaSum;
            amrex::Real omegaLoc = wenoOmega[s];
            for (int i = 0; i < s_wenoHistNDof; i++)
            {
                whWeno(iVar, i) += omegaLoc * whRec(s, iVar, i);
            }
        }
    }
}

template <int nVarFv, int polyrecDegreeFv>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::weno_reconstruction_1d(
    amrex::Array1D<amrex::Real, 0, polyrecDegreeFv>& whWeno,
    amrex::Array1D<amrex::Real, 0, 2 * polyrecDegreeFv> const& inputVector) const
{
    BL_PROFILE_HOST_DEVICE("GEMPIC::FVreconstruction::weno_reconstruction_1d()");
    amrex::Real omegaSum = 0.0;
    amrex::Array2D<amrex::Real, 0, s_wenoNStencil - 1, 0, s_wenoHistNDof - 1, amrex::Order::C>
        whRec;
    for (int i = 0; i < s_wenoHistNDof; i++)
    {
        whWeno(i) = 0.0;
    }

    amrex::GpuArray<amrex::Real, s_wenoNStencil> wenoOi;
    amrex::GpuArray<amrex::Real, s_wenoNStencil> wenoOmega;
    for (int s = 0; s < s_wenoNStencil; s++)
    {
        wenoOi[s] = 0.0;
        amrex::Array1D<amrex::Real, 0, s_wenoHistNDof - 1> whLoc;
        int jSL = s_wenoStencilShift +
                  get_weno_left(s); // weno_left assumes the central cell being the zeroth;
        for (int i = 0; i < s_wenoHistNDof; i++)
        {
            amrex::Real whSum = 0;
            for (int j = 0; j < s_wenoHistNDof; j++)
            {
                int jloc = j + jSL;
                whSum += m_wenOhist2LocalGLint(s, i, j) * inputVector(jloc);
            }
            whRec(s, i) = whSum;
            whLoc(i) = whSum;
        }
        for (int i = 0; i < s_wenoHistNDof; i++)
        {
            amrex::Real w1 = whLoc(i);
            for (int j = 0; j < s_wenoHistNDof; j++)
            {
                amrex::Real factor = m_wenoOiMatrixC(i, j) * whLoc(j);
                wenoOi[s] += w1 * factor;
            }
        }
        wenoOmega[s] = WENO3::compute_oscillation_indicator_to_weight(wenoOi[s], s);
        omegaSum += wenoOmega[s];
    }

    for (int s = 0; s < s_wenoNStencil; s++)
    {
        wenoOmega[s] /= omegaSum;
        amrex::Real omegaLoc = wenoOmega[s];
        for (int i = 0; i < s_wenoHistNDof; i++)
        {
            whWeno(i) += omegaLoc * whRec(s, i);
        }
    }
}

/**
 * compute local 1d polynomial weno reconstruction, ginven the input 1d NODAL stencil
 * the local 1d INTERPOLANT reconstruction is written in terms of the dof of the local
 * int.polynomials (GLeg points)
 */
template <int nVarFv, int polyrecDegreeFv>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::weno_node2cell_reconstruction_1d(
    amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, polyrecDegreeFv + 1, amrex::Order::C>& whWeno,
    amrex::Array2D<amrex::Real,
                   0,
                   nVarFv - 1,
                   0,
                   2 * (polyrecDegreeFv + 1) - 1,
                   amrex::Order::C> const& inputVector) const
{
    BL_PROFILE_HOST_DEVICE("GEMPIC::FVreconstruction::weno_node2cell_reconstruction_1d()");
    amrex::Real omegaSum = 0.0;
    amrex::Array3D<amrex::Real, 0, s_wenoNStencil - 1, 0, nVarFv - 1, 0, s_wenoIntNDof - 1,
                   amrex::Order::C>
        whRec;
    for (int iVar = 0; iVar < nVarFv; iVar++)
    {
        for (int i = 0; i < s_wenoIntNDof; i++)
        {
            whWeno(iVar, i) = 0.0;
        }
    }

    amrex::GpuArray<amrex::Real, s_wenoNStencil> wenoOi;
    amrex::GpuArray<amrex::Real, s_wenoNStencil> wenoOmega;
    for (int s = 0; s < s_wenoNStencil; s++)
    {
        wenoOi[s] = 0.0;
        amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, s_wenoIntNDof - 1, amrex::Order::C> whLoc;
        int jSL = s_wenoStencilShift +
                  get_weno_left(s); // weno_left assumes the central index being the zeroth, which
                                    // is placed at the weno_stencil_shift2-th position;
        for (int iVar = 0; iVar < nVarFv; iVar++)
        {
            for (int i = 0; i < s_wenoIntNDof; i++)
            {
                amrex::Real whSum = 0;
                for (int j = 0; j < s_wenoIntNDof; j++)
                {
                    int jloc = j + jSL;
                    whSum += m_wenOint2LocalGLint(s, i, j) * inputVector(iVar, jloc);
                }
                whRec(s, iVar, i) = whSum;
                whLoc(iVar, i) = whSum;
            }
        }
        for (int iVar = 0; iVar < nVarFv; iVar++)
        {
            for (int i = 0; i < s_wenoIntNDof; i++)
            {
                amrex::Real w1 = whLoc(iVar, i);
                for (int j = 0; j < s_wenoIntNDof; j++)
                {
                    amrex::Real factor = m_wenoOiMatrix2C(i, j) * whLoc(iVar, j);
                    wenoOi[s] += w1 * factor;
                }
            }
        }
        wenoOmega[s] = WENO3::compute_oscillation_indicator_to_weight(wenoOi[s], s);
        omegaSum += wenoOmega[s];
    }

    for (int s = 0; s < s_wenoNStencil; s++)
    {
        wenoOmega[s] /= omegaSum;
        amrex::Real omegaLoc = wenoOmega[s];
        for (int i = 0; i < s_wenoIntNDof; i++)
        {
            for (int iVar = 0; iVar < nVarFv; iVar++)
            {
                whWeno(iVar, i) += omegaLoc * whRec(s, iVar, i);
            }
        }
    }
}

template <int nVarFv, int polyrecDegreeFv>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::weno_node2cell_reconstruction_1d(
    amrex::Array1D<amrex::Real, 0, polyrecDegreeFv + 1>& whWeno,
    amrex::Array1D<amrex::Real, 0, 2 * (polyrecDegreeFv + 1) - 1> const& inputVector) const
{
    BL_PROFILE_HOST_DEVICE("GEMPIC::FVreconstruction::weno_node2cell_reconstruction_1d()");
    amrex::Real omegaSum = 0.0;
    amrex::Array2D<amrex::Real, 0, s_wenoNStencil - 1, 0, s_wenoIntNDof - 1, amrex::Order::C> whRec;

    for (int i = 0; i < s_wenoIntNDof; i++)
    {
        whWeno(i) = 0.0;
    }

    amrex::GpuArray<amrex::Real, s_wenoNStencil> wenoOi;
    amrex::GpuArray<amrex::Real, s_wenoNStencil> wenoOmega;
    for (int s = 0; s < s_wenoNStencil; s++)
    {
        wenoOi[s] = 0.0;
        amrex::Array1D<amrex::Real, 0, s_wenoIntNDof - 1> whLoc;
        int jSL = s_wenoStencilShift +
                  get_weno_left(s); // weno_left assumes the central index being the zeroth, which
                                    // is placed at the weno_stencil_shift2-th position;

        for (int i = 0; i < s_wenoIntNDof; i++)
        {
            amrex::Real whSum = 0;
            for (int j = 0; j < s_wenoIntNDof; j++)
            {
                int jloc = j + jSL;
                whSum += m_wenOint2LocalGLint(s, i, j) * inputVector(jloc);
            }
            whRec(s, i) = whSum;
            whLoc(i) = whSum;
        }
        for (int i = 0; i < s_wenoIntNDof; i++)
        {
            amrex::Real w1 = whLoc(i);
            for (int j = 0; j < s_wenoIntNDof; j++)
            {
                amrex::Real factor = m_wenoOiMatrix2C(i, j) * whLoc(j);
                wenoOi[s] += w1 * factor;
            }
        }
        wenoOmega[s] = WENO3::compute_oscillation_indicator_to_weight(wenoOi[s], s);
        omegaSum += wenoOmega[s];
    }

    for (int s = 0; s < s_wenoNStencil; s++)
    {
        wenoOmega[s] /= omegaSum;
        amrex::Real omegaLoc = wenoOmega[s];
        for (int i = 0; i < s_wenoIntNDof; i++)
        {
            whWeno(i) += omegaLoc * whRec(s, i);
        }
    }
}

template <int nVarFv, int polyrecDegreeFv>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::polynomial_reconstruction_1d(
    amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, polyrecDegreeFv, amrex::Order::C>& whWeno,
    amrex::Array2D<amrex::Real, 0, nVarFv - 1, 0, 2 * polyrecDegreeFv, amrex::Order::C> const&
        inputVector) const
{
    BL_PROFILE_HOST_DEVICE("GEMPIC::FVreconstruction::polynomial_reconstruction_1d()");
    for (int iVar = 0; iVar < nVarFv; iVar++)
    {
        for (int i = 0; i < s_wenoHistNDof; i++)
        {
            whWeno(iVar, i) = 0.0;
        }
    }
    int s = 0;
    int jSL = s_wenoStencilShift +
              get_weno_left(s); // weno_left assumes the central cell being the zeroth;
    for (int iVar = 0; iVar < nVarFv; iVar++)
    {
        for (int i = 0; i < s_wenoHistNDof; i++)
        {
            amrex::Real whSum = 0;
            for (int j = 0; j < s_wenoHistNDof; j++)
            {
                int jloc = j + jSL;
                whSum += m_wenOhist2LocalGLint(s, i, j) * inputVector(iVar, jloc);
            }
            whWeno(iVar, i) = whSum;
        }
    }
}

template <int nVarFv, int polyrecDegreeFv>
template <int degree>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::legendre_interpolating_g_leg_nodes_mp(
    amrex::GpuArray<amrex::Real, degree + 1>& phi,
    amrex::GpuArray<amrex::Real, degree + 1>& phiXi,
    amrex::Real const xi)
{
    BL_PROFILE_HOST_DEVICE("GEMPIC::FVreconstruction::legendre_interpolating_g_leg_nodes_MP()");
    constexpr int nDOF = degree + 1;
    // Access the Gauss-Legendre nodes for this degree from fixed arrays
    auto const& nodesGLeg = m_nodesGLegTable.template get_array<degree>();
    // Initialize variables
    for (int i = 0; i < nDOF; i++)
    {
        phi[i] = 1.;
        phiXi[i] = 0.;
    }

    // Lagrange interpolating polynomial passing through the Gauss-Legendre quadrature points
    // and its derivative
    for (int m = 0; m < nDOF; m++)
    {
        amrex::Real product = 1.0;
        for (int j = 0; j < nDOF; j++)
        {
            if (j == m)
            {
                continue;
            }
            product *= (xi - nodesGLeg[j]) / (nodesGLeg[m] - nodesGLeg[j]);
        }
        phi[m] = product;
        amrex::Real sum = 0.0;
        for (int i = 0; i < nDOF; i++)
        {
            if (i == m)
            {
                continue;
            }
            amrex::Real tmp = 1.0;
            for (int j = 0; j < nDOF; j++)
            {
                if (j == i || j == m)
                {
                    continue;
                }
                tmp *= (xi - nodesGLeg[j]) / (nodesGLeg[m] - nodesGLeg[j]);
            }
            sum += tmp / (nodesGLeg[m] - nodesGLeg[i]);
        }
        phiXi[m] = sum;
    }
}

// Wrapper struct to call the templated function with vectors
template <int nVarFv, int polyrecDegreeFv>
template <int degree>
void FVreconstruction<nVarFv, polyrecDegreeFv>::LegendreWrapper<degree>::call (
    FVreconstruction<nVarFv, polyrecDegreeFv>& obj,
    std::vector<amrex::Real>& phi,
    std::vector<amrex::Real>& phiXi,
    amrex::Real const xi)
{
    // Check vector sizes at runtime
    if (static_cast<int>(phi.size()) != degree + 1 || static_cast<int>(phiXi.size()) != degree + 1)
    {
        throw std::runtime_error("Vector size does not match degree + 1");
    }

    AMREX_ASSERT_WITH_MESSAGE(static_cast<int>(phi.size()) == degree + 1,
                              "Error: phi size does not match expected size.");
    AMREX_ASSERT_WITH_MESSAGE(static_cast<int>(phiXi.size()) == degree + 1,
                              "Error: phiXi size does not match expected size.");
    // This is the right size, so we copy the contents and call the actual function
    amrex::GpuArray<amrex::Real, degree + 1> phiArr;
    amrex::GpuArray<amrex::Real, degree + 1> phiXiArr;
    std::copy(phi.begin(), phi.end(), phiArr.begin());
    std::copy(phiXi.begin(), phiXi.end(), phiXiArr.begin());

    // to avoid "error : call to non-static member function without an object argument" we need the
    // instance obj. since legendre_interpolating_g_leg_nodes_MP is a non-static member function:
    // Call the actual templated function
    obj.template legendre_interpolating_g_leg_nodes_mp<degree>(phiArr, phiXiArr, xi);

    // Copy back array -> vector
    std::copy(phiArr.begin(), phiArr.end(), phi.begin());
    std::copy(phiXiArr.begin(), phiXiArr.end(), phiXi.begin());
}

template <int nVarFv, int polyrecDegreeFv>
template <int... degrees>
constexpr auto FVreconstruction<nVarFv, polyrecDegreeFv>::make_table(
    std::integer_sequence<int, degrees...>)
{
    return std::array{&LegendreWrapper<degrees>::call...};
}

// if you get "undefined symbol" this could be required
//// Declare the static constexpr dispatch table (outside class, in source file)
//template <int nVarFv, int polyrecDegreeFv>
//constexpr int FVreconstruction<nVarFv, polyrecDegreeFv>::maxDegree;

template <int nVarFv, int polyrecDegreeFv>
void FVreconstruction<nVarFv, polyrecDegreeFv>::legendre_interpolating_vec (
    std::vector<amrex::Real>& phi, std::vector<amrex::Real>& phiXi, amrex::Real const xi)
{
    int degreeIn{static_cast<int>(phi.size()) - 1};
    AMREX_ASSERT_WITH_MESSAGE(static_cast<int>(phi.size()) == static_cast<int>(phiXi.size()),
                              "Error: phi size does not match phiXi size.");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(degreeIn >= 0 && degreeIn <= s_maxDegree,
                                     "Unsupported degree.");
    // Generate dispatch table as static so constructed once
    static constexpr auto legendreTable =
        make_table(std::make_integer_sequence<int, s_maxDegree + 1>{});

    legendreTable[degreeIn](*this, phi, phiXi, xi);
}

/**
 * Legendre intepolating polynomial phi of degree polyrec_degree_FV and its derivative (phi_xi),
 * passing through the weno_nDof2  local weno-stencil nodes  weno_stencil_loc_nodes[:]),
 * evaluated at point xi.
 *
 * \param phi
 * \param phi_xi
 * \param xi
 */
template <int nVarFv, int polyrecDegreeFv>
template <int degree>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::legendre_interpolating_weno_stencil_nodes(
    amrex::GpuArray<amrex::Real, degree + 1>& phi,
    amrex::GpuArray<amrex::Real, degree + 1>& phiXi,
    amrex::Real const xi)
{
    BL_PROFILE_HOST_DEVICE("GEMPIC::FVreconstruction::legendre_interpolating_weno_stencil_nodes()");
    constexpr int nDOF = degree + 1;
    // Initialize variables
    for (int i = 0; i < nDOF; i++)
    {
        phi[i] = 1.0;
        phiXi[i] = 0.0;
    }
    // Lagrange interpolating polynomial passing through the Gauss-Legendre quadrature points
    // and its derivative
    for (int m = 0; m < nDOF; m++)
    {
        amrex::Real product = 1.0;
        for (int j = 0; j < nDOF; j++)
        {
            if (j == m)
            {
                continue;
            }
            product *= (xi - m_wenoStencilLocNodes[j]) /
                       (m_wenoStencilLocNodes[m] - m_wenoStencilLocNodes[j]);
        }
        phi[m] = product;
        amrex::Real sum = 0.0;
        for (int i = 0; i < nDOF; i++)
        {
            if (i == m)
            {
                continue;
            }
            amrex::Real tmp = 1.0;
            for (int j = 0; j < nDOF; j++)
            {
                if (j == i || j == m)
                {
                    continue;
                }
                tmp *= (xi - m_wenoStencilLocNodes[j]) /
                       (m_wenoStencilLocNodes[m] - m_wenoStencilLocNodes[j]);
            }
            sum += tmp / (m_wenoStencilLocNodes[m] - m_wenoStencilLocNodes[i]);
        }
        phiXi[m] = sum;
    }
}

/**
 * Histopolating 1d polynomial with reference to the local stencil nodes of the weno reconstruction.
 * It's built with respect to the Legendre interpolating polynomials for genederal polynomial
 * degrees. Here "degree" is assumed to be the degree of the histopolating polynomial. Then, degree
 * +1 is the degree of the associated interpolaying polynomials.
 *
 * \param phi
 * \param xi
 */
template <int nVarFv, int polyrecDegreeFv>
template <int degree>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::legendre_histopolating_weno_stencil_nodes(
    amrex::GpuArray<amrex::Real, degree + 1>& phi, amrex::Real const xi)
{
    BL_PROFILE_HOST_DEVICE("GEMPIC::FVreconstruction::legendre_histopolating_weno_stencil_nodes()");
    // Initialize variables
    constexpr int nDofHistop = degree + 1;
    constexpr int nDofInterp = degree + 2;
    amrex::GpuArray<amrex::Real, degree + 2> phitmp;
    amrex::GpuArray<amrex::Real, degree + 2> phitmpXi;

    for (int i = 0; i < nDofInterp; i++)
    {
        phitmp[i] = 0.0;
        phitmpXi[i] = 0.0;
    }
    legendre_interpolating_weno_stencil_nodes<degree + 1>(phitmp, phitmpXi, xi);

    for (int i = 0; i < nDofHistop; i++)
    {
        phi[i] = 0.0;
    }

    phi[0] = -phitmpXi[0];
    for (int i = 1; i < nDofHistop; i++)
    {
        phi[i] = phi[i - 1] - phitmpXi[i];
    }
    phi[nDofHistop - 1] = phitmpXi[nDofInterp - 1]; // do we overwrite the last?
}

template <int nVarFv, int polyrecDegreeFv>
template <typename ParamsClass, typename PdeTagType, ReconstructionMode userRecMode>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::compute_weno_v3_reconstruction(
    dofXYZ_Array3D<polyrecDegreeFv, nVarFv>& whwenoxyz3d,
    amrex::Array4<amrex::Real const> const& w0,
    ParamsClass const& myparams,
    int const i,
    int const j,
    int const k) const
{
    if constexpr (userRecMode == ReconstructionMode::CharacteristicVars)
    {
        compute_weno_v3_reconstruction_charvar<ParamsClass, PdeTagType>(whwenoxyz3d, w0, myparams,
                                                                        i, j, k);
    }
    else
    {
        //default
        compute_weno_v3_reconstruction_primvar<ParamsClass, PdeTagType>(whwenoxyz3d, w0, myparams,
                                                                        i, j, k);
    }
    return;
}

template <int nVarFv, int polyrecDegreeFv>
template <typename ParamsClass, typename PdeTagType, ReconstructionMode userRecMode>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::compute_fv_fluxes(
    amrex::Array4<amrex::Real> const& F,
    amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof> const> const& ql,
    amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof> const> const& qr,
    ParamsClass const& myparams,
    int const idir,
    amrex::GpuArray<amrex::Real, 3> const& n,
    amrex::Real const dS,
    int const i,
    int const j,
    int const k) const
{
    if constexpr (userRecMode == ReconstructionMode::CharacteristicVars)
    {
        compute_fv_fluxes_charvar<ParamsClass, PdeTagType>(F, ql, qr, myparams, idir, n, dS, i, j,
                                                           k);
    }
    else
    {
        //default
        compute_fv_fluxes_primvar<ParamsClass, PdeTagType>(F, ql, qr, myparams, idir, n, dS, i, j,
                                                           k);
    }
    return;
}

template <int nVarFv, int polyrecDegreeFv>
template <typename ParamsClass, typename PdeTagType>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::compute_fv_fluxes_charvar(
    amrex::Array4<amrex::Real> const& F,
    amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof> const> const& ql,
    amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof> const> const& qr,
    ParamsClass const& myparams,
    int const idir,
    amrex::GpuArray<amrex::Real, 3> const& n,
    amrex::Real const dS,
    int const i,
    int const j,
    int const k) const
{
    /**
        here we must compute the surface integral!
        - for loop over the 2d quadrature points
        - solve the nodal Riemann problem
        - add quadrature contribution to the surface integral
    */
    int iDir1 = get_face_integral_idxs(2 * idir);     // faster index
    int iDir2 = get_face_integral_idxs(2 * idir + 1); //
    varArray<nVarFv> nFlux;
    nFlux.fill(0.0);
    for (int iDof2 = 0; iDof2 < v3_n_dof(iDir2); ++iDof2)
    {
        for (int iDof1 = 0; iDof1 < v3_n_dof(iDir1); ++iDof1)
        {
            varArray<nVarFv> fRGl;
            varArray<nVarFv> fLGl;
            varArray<nVarFv> qrGl;
            varArray<nVarFv> qlGl;

            int iDof2DlLoc = weno_dof_2dvect_to_array_indexes(iDof2, iDof1);
            for (int iVar = 0; iVar < nVarFv; iVar++)
            {
                qlGl[iVar] = ql(i, j, k, iVar)[iDof2DlLoc];
                qrGl[iVar] = qr(i, j, k, iVar)[iDof2DlLoc];
            }
            pde_numerical_flux<nVarFv, ParamsClass, PdeTagType>(
                qlGl, qrGl, n, fLGl, fRGl,
                myparams); // use "this ->" for derived
                           // template member functions
            amrex::Real quadratureWeights =
                GEMPIC_D_MULT(1.0, m_weightsGLegP[iDof1], m_weightsGLegP[iDof2]);
            for (int iVar = 0; iVar < nVarFv; iVar++)
            {
                nFlux[iVar] += quadratureWeights * fLGl[iVar];
            }
        }
    }
    for (int iVar = 0; iVar < nVarFv; iVar++)
    {
        F(i, j, k, iVar) = dS * nFlux[iVar];
    }
}

template <int nVarFv, int polyrecDegreeFv>
template <typename ParamsClass, typename PdeTagType>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::compute_fv_fluxes_primvar(
    amrex::Array4<amrex::Real> const& F,
    amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof> const> const& ql,
    amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof> const> const& qr,
    ParamsClass const& myparams,
    int const idir,
    amrex::GpuArray<amrex::Real, 3> const& n,
    amrex::Real const dS,
    int const i,
    int const j,
    int const k) const
{
    /**
        here we must compute the surface integral!
        - for loop over the 2d quadrature points
        - solve the nodal Riemann problem
        - add quadrature contribution to the surface integral
    */
    // xface
    int iDir1 = get_face_integral_idxs(2 * idir);
    int iDir2 = get_face_integral_idxs(2 * idir + 1);
    varArray<nVarFv> nFlux;
    nFlux.fill(0.0);
    for (int iDof2 = 0; iDof2 < v3_n_dof(iDir2); ++iDof2)
    {
        for (int iDof1 = 0; iDof1 < v3_n_dof(iDir1); ++iDof1)
        {
            varArray<nVarFv> fRGl;
            varArray<nVarFv> fLGl;
            varArray<nVarFv> vrGl;
            varArray<nVarFv> vlGl;
            varArray<nVarFv> qrGl;
            varArray<nVarFv> qlGl;

            int iDof2DlLoc = weno_dof_2dvect_to_array_indexes(iDof2, iDof1);
            for (int iVar = 0; iVar < nVarFv; iVar++)
            {
                vlGl[iVar] = ql(i, j, k, iVar)[iDof2DlLoc];
                vrGl[iVar] = qr(i, j, k, iVar)[iDof2DlLoc];
            }
            pde_prim_2_cons<nVarFv, ParamsClass, PdeTagType>(vlGl, qlGl, myparams);
            pde_prim_2_cons<nVarFv, ParamsClass, PdeTagType>(vrGl, qrGl, myparams);
            pde_numerical_flux<nVarFv, ParamsClass, PdeTagType>(
                qlGl, qrGl, n, fLGl, fRGl,
                myparams); // use "this ->" for derived
                           // template member functions
            amrex::Real quadratureWeights =
                GEMPIC_D_MULT(1.0, m_weightsGLegP[iDof1], m_weightsGLegP[iDof2]);
            for (int iVar = 0; iVar < nVarFv; iVar++)
            {
                nFlux[iVar] += quadratureWeights * fLGl[iVar];
            }
        }
    }
    for (int iVar = 0; iVar < nVarFv; iVar++)
    {
        F(i, j, k, iVar) = dS * nFlux[iVar];
    }
}

template <int nVarFv, int polyrecDegreeFv>
template <typename ParamsClass, typename PdeTagType>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::compute_weno_v3_reconstruction_primvar(
    dofXYZ_Array3D<polyrecDegreeFv, nVarFv>& whwenoxyz3d,
    amrex::Array4<amrex::Real const> const& w0,
    ParamsClass const& myparams,
    int const i,
    int const j,
    int const k) const
{
    //build weno stencil:
    dofX_stencilYZ_Array3D<polyrecDegreeFv, nVarFv> whwenox3d;
    dofXY_stencilZ_Array3D<polyrecDegreeFv, nVarFv> whwenoxy3d;

    // x-polynomial reconstruction
    for (int iz = v3_stencil_lo_z(); iz < v3_stencil_hi_z() + 1; ++iz)
    {
        for (int iy = v3_stencil_lo_y(); iy < v3_stencil_hi_y() + 1; ++iy)
        {
            stencilArray2D<fv_degree_x(), nVarFv> stencilVector1d;
            dofArray2D<polyrecDegreeFv, nVarFv> whWeno1d;
            // get the 1d stencil
            for (int iVar = 0; iVar < nVarFv; iVar++)
            {
                for (int ix = v3_stencil_lo_x(), iloc = 0; ix < v3_stencil_hi_x() + 1; ++ix, ++iloc)
                {
                    stencilVector1d(iVar, iloc) = w0(i + ix, j + iy, k + iz, iVar);
                }
            }
            compute_cell2cell_reconstruction_1d(whWeno1d, stencilVector1d);
            // store the 1d reconstruction
            for (int iDofx = 0; iDofx < v3_n_dof_x(); ++iDofx)
            {
                for (int iVar = 0; iVar < nVarFv; iVar++)
                {
                    whwenox3d(iz, iy, iDofx, iVar) = whWeno1d(iVar, iDofx);
                }
            }
        }
    }
#if AMREX_SPACEDIM > 1
    // y-polynomial reconstruction
    for (int iz = v3_stencil_lo_z(); iz < v3_stencil_hi_z() + 1; ++iz)
    {
        for (int iDofx = 0; iDofx < v3_n_dof_x(); ++iDofx)
        {
            stencilArray2D<fv_degree_y(), nVarFv> stencilVector1d;
            dofArray2D<polyrecDegreeFv, nVarFv> whWeno1d;
            // get the 1d stencil
            for (int iVar = 0; iVar < nVarFv; iVar++)
            {
                for (int iy = v3_stencil_lo_y(), iloc = 0; iy < v3_stencil_hi_y() + 1; ++iy, ++iloc)
                {
                    stencilVector1d(iVar, iloc) = whwenox3d(iz, iy, iDofx, iVar);
                }
            }
            compute_cell2cell_reconstruction_1d(whWeno1d, stencilVector1d);
            // store the 1d reconstruction
            for (int iDofy = 0; iDofy < v3_n_dof_y(); ++iDofy)
            {
                for (int iVar = 0; iVar < nVarFv; iVar++)
                {
                    whwenoxy3d(iz, iDofy, iDofx, iVar) = whWeno1d(iVar, iDofy);
                }
            }
        }
    }
#else
    for (int iDofz = 0; iDofz < v3_n_dof_z(); ++iDofz)
    {
        for (int iDofy = 0; iDofy < v3_n_dof_y(); ++iDofy)
        {
            for (int iDofx = 0; iDofx < v3_n_dof_x(); ++iDofx)
            {
                for (int iVar = 0; iVar < nVarFv; iVar++)
                {
                    whwenoxy3d(iDofz, iDofy, iDofx, iVar) = whwenox3d(iDofz, iDofy, iDofx, iVar);
                }
            }
        }
    }
#endif

#if AMREX_SPACEDIM > 2
    // z-polynomial reconstruction
    for (int iDofy = 0; iDofy < v3_n_dof_y(); ++iDofy)
    {
        for (int iDofx = 0; iDofx < v3_n_dof_x(); ++iDofx)
        {
            stencilArray2D<fv_degree_z(), nVarFv> stencilVector1d;
            dofArray2D<polyrecDegreeFv, nVarFv> whWeno1d;
            // get the stencil
            for (int iVar = 0; iVar < nVarFv; iVar++)
            {
                for (int iz = v3_stencil_lo_z(), iloc = 0; iz < v3_stencil_hi_z() + 1; ++iz, ++iloc)
                {
                    stencilVector1d(iVar, iloc) = whwenoxy3d(iz, iDofy, iDofx, iVar);
                }
            }
            compute_cell2cell_reconstruction_1d(whWeno1d, stencilVector1d);
            // store the reconstruction
            for (int iDofz = 0; iDofz < v3_n_dof_z(); ++iDofz)
            {
                for (int iVar = 0; iVar < nVarFv; iVar++)
                {
                    whwenoxyz3d(iDofz, iDofy, iDofx, iVar) = whWeno1d(iVar, iDofz);
                }
            }
        }
    }
#else
    for (int iDofz = 0; iDofz < v3_n_dof_z(); ++iDofz)
    {
        for (int iDofy = 0; iDofy < v3_n_dof_y(); ++iDofy)
        {
            for (int iDofx = 0; iDofx < v3_n_dof_x(); ++iDofx)
            {
                for (int iVar = 0; iVar < nVarFv; iVar++)
                {
                    whwenoxyz3d(iDofz, iDofy, iDofx, iVar) = whwenoxy3d(iDofz, iDofy, iDofx, iVar);
                }
            }
        }
    }
#endif
}

template <int nVarFv, int polyrecDegreeFv>
template <typename ParamsClass, typename PdeTagType>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::compute_weno_v3_reconstruction_charvar(
    dofXYZ_Array3D<polyrecDegreeFv, nVarFv>& whwenoxyz3d,
    amrex::Array4<amrex::Real const> const& w0,
    ParamsClass const& myparams,
    int const i,
    int const j,
    int const k) const
{
    //build weno stencil:
    dofX_stencilYZ_Array3D<polyrecDegreeFv, nVarFv> whwenox3d;
    dofXY_stencilZ_Array3D<polyrecDegreeFv, nVarFv> whwenoxy3d;

    // compute eigenvector matrices (for characteristic variables)
    varArray2D<nVarFv> R;
    varArray2D<nVarFv> L;
    varArray2D<nVarFv> iR;
    varArray<nVarFv> qcGpu;
    amrex::GpuArray<amrex::Real, 3> nx{1.0, 0.0, 0.0};
    for (int iVar = 0; iVar < nVarFv; iVar++)
    {
        qcGpu[iVar] = w0(i, j, k, iVar);
    }
    pde_get_eigenvectors<nVarFv, ParamsClass, PdeTagType>(qcGpu, nx, R, L, iR, myparams);
    // x-polynomial reconstruction
    for (int iz = v3_stencil_lo_z(); iz < v3_stencil_hi_z() + 1; ++iz)
    {
        for (int iy = v3_stencil_lo_y(); iy < v3_stencil_hi_y() + 1; ++iy)
        {
            stencilArray2D<fv_degree_x(), nVarFv> stencilVector1d;
            dofArray2D<polyrecDegreeFv, nVarFv> whWeno1d;
            // get the 1d stencil
            for (int iVar = 0; iVar < nVarFv; iVar++)
            {
                for (int ix = v3_stencil_lo_x(), iloc = 0; ix < v3_stencil_hi_x() + 1; ++ix, ++iloc)
                {
                    stencilVector1d(iVar, iloc) = w0(i + ix, j + iy, k + iz, iVar);
                }
            }
            // do the cons2char transformation (only use the central average)
            stencilVector1d =
                Gempic::matmul<nVarFv, nVarFv, 2 * polyrecDegreeFv + 1>(iR, stencilVector1d);
            // compute the 1d reconstruction
            compute_cell2cell_reconstruction_1d(whWeno1d, stencilVector1d);
            // do the char2const transformation (can be done directly after the face evaluation)
            whWeno1d = Gempic::matmul<nVarFv, nVarFv, polyrecDegreeFv + 1>(R, whWeno1d);
            // store the 1d reconstruction
            for (int iDofx = 0; iDofx < v3_n_dof_x(); ++iDofx)
            {
                for (int iVar = 0; iVar < nVarFv; iVar++)
                {
                    whwenox3d(iz, iy, iDofx, iVar) = whWeno1d(iVar, iDofx);
                }
            }
        }
    }
#if AMREX_SPACEDIM > 1
    amrex::GpuArray<amrex::Real, 3> ny{0.0, 1.0, 0.0};
    pde_get_eigenvectors<nVarFv, ParamsClass, PdeTagType>(qcGpu, ny, R, L, iR, myparams);
    // y-polynomial reconstruction
    for (int iz = v3_stencil_lo_z(); iz < v3_stencil_hi_z() + 1; ++iz)
    {
        for (int iDofx = 0; iDofx < v3_n_dof_x(); ++iDofx)
        {
            stencilArray2D<fv_degree_y(), nVarFv> stencilVector1d;
            dofArray2D<polyrecDegreeFv, nVarFv> whWeno1d;
            // get the 1d stencil
            for (int iVar = 0; iVar < nVarFv; iVar++)
            {
                for (int iy = v3_stencil_lo_y(), iloc = 0; iy < v3_stencil_hi_y() + 1; ++iy, ++iloc)
                {
                    stencilVector1d(iVar, iloc) = whwenox3d(iz, iy, iDofx, iVar);
                }
            }
            // do the cons2char transformation (only use the central average)
            stencilVector1d =
                Gempic::matmul<nVarFv, nVarFv, 2 * polyrecDegreeFv + 1>(iR, stencilVector1d);
            // compute the 1d reconstruction
            compute_cell2cell_reconstruction_1d(whWeno1d, stencilVector1d);
            // do the char2const transformation (can be done
            // directly after the face evaluation)
            whWeno1d = Gempic::matmul<nVarFv, nVarFv, polyrecDegreeFv + 1>(R, whWeno1d);
            // store the 1d reconstruction
            for (int iDofy = 0; iDofy < v3_n_dof_y(); ++iDofy)
            {
                for (int iVar = 0; iVar < nVarFv; iVar++)
                {
                    whwenoxy3d(iz, iDofy, iDofx, iVar) = whWeno1d(iVar, iDofy);
                }
            }
        }
    }
#else
    for (int iDofz = 0; iDofz < v3_n_dof_z(); ++iDofz)
    {
        for (int iDofy = 0; iDofy < v3_n_dof_y(); ++iDofy)
        {
            for (int iDofx = 0; iDofx < v3_n_dof_x(); ++iDofx)
            {
                for (int iVar = 0; iVar < nVarFv; iVar++)
                {
                    whwenoxy3d(iDofz, iDofy, iDofx, iVar) = whwenox3d(iDofz, iDofy, iDofx, iVar);
                }
            }
        }
    }
#endif

#if AMREX_SPACEDIM > 2
    amrex::GpuArray<amrex::Real, 3> nz{0.0, 0.0, 1.0};
    pde_get_eigenvectors<nVarFv, ParamsClass, PdeTagType>(qcGpu, nz, R, L, iR, myparams);
    // z-polynomial reconstruction
    for (int iDofy = 0; iDofy < v3_n_dof_y(); ++iDofy)
    {
        for (int iDofx = 0; iDofx < v3_n_dof_x(); ++iDofx)
        {
            stencilArray2D<fv_degree_z(), nVarFv> stencilVector1d;
            dofArray2D<polyrecDegreeFv, nVarFv> whWeno1d;
            // get the stencil
            for (int iVar = 0; iVar < nVarFv; iVar++)
            {
                for (int iz = v3_stencil_lo_z(), iloc = 0; iz < v3_stencil_hi_z() + 1; ++iz, ++iloc)
                {
                    stencilVector1d(iVar, iloc) = whwenoxy3d(iz, iDofy, iDofx, iVar);
                }
            }
            // do the cons2char transformation (only use the central average)
            stencilVector1d =
                Gempic::matmul<nVarFv, nVarFv, 2 * polyrecDegreeFv + 1>(iR, stencilVector1d);
            // compute the 1d reconstruction
            compute_cell2cell_reconstruction_1d(whWeno1d, stencilVector1d);
            // do the char2const transformation (can be done directly after the face evaluation)
            whWeno1d = Gempic::matmul<nVarFv, nVarFv, polyrecDegreeFv + 1>(R, whWeno1d);
            // store the reconstruction
            for (int iDofz = 0; iDofz < v3_n_dof_z(); ++iDofz)
            {
                for (int iVar = 0; iVar < nVarFv; iVar++)
                {
                    whwenoxyz3d(iDofz, iDofy, iDofx, iVar) = whWeno1d(iVar, iDofz);
                }
            }
        }
    }
#else
    for (int iDofz = 0; iDofz < v3_n_dof_z(); ++iDofz)
    {
        for (int iDofy = 0; iDofy < v3_n_dof_y(); ++iDofy)
        {
            for (int iDofx = 0; iDofx < v3_n_dof_x(); ++iDofx)
            {
                for (int iVar = 0; iVar < nVarFv; iVar++)
                {
                    whwenoxyz3d(iDofz, iDofy, iDofx, iVar) = whwenoxy3d(iDofz, iDofy, iDofx, iVar);
                }
            }
        }
    }
#endif
}

template <int nVarFv, int polyrecDegreeFv>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::compute_weno_v2x_reconstruction(
    dofXYZ_V2x_Array3D<polyrecDegreeFv>& bxhwenoxyz3d,
    amrex::Array4<amrex::Real const> const& bx0,
    amrex::Real const idFace,
    int const i,
    int const j,
    int const k) const
{
    dofX_V2x_stencilYZ_Array3D<polyrecDegreeFv> bxhwenox3d;
    dofXY_V2x_stencilZ_Array3D<polyrecDegreeFv> bxhwenoxy3d;

    // weno V2 to V3 reconstruction
    // Bx
    // x-polynomial reconstruction
    for (int iz = v2x_stencil_lo_z(); iz < v2x_stencil_hi_z() + 1; ++iz)
    {
        for (int iy = v2x_stencil_lo_y(); iy < v2x_stencil_hi_y() + 1; ++iy)
        {
            stencilArray1Dnode<fv_degree_x()> stencilVector1d;
            dofArray1D<polyrecDegreeFv + 1> bhWeno1d;
            // get the 1d stencil
            for (int ix = v2x_stencil_lo_x(), iloc = 0; ix < v2x_stencil_hi_x() + 1; ++ix, ++iloc)
            {
                stencilVector1d(iloc) = bx0(i + ix, j + iy, k + iz) * idFace;
            }
            compute_node2cell_reconstruction_1d(bhWeno1d, stencilVector1d);
            // store the 1d reconstruction
            for (int iDofx = 0; iDofx < v2x_n_dof_x(); ++iDofx)
            {
                bxhwenox3d(iz, iy, iDofx) = bhWeno1d(iDofx);
            }
        }
    }
#if AMREX_SPACEDIM > 1
    // y-polynomial reconstruction
    for (int iz = v2x_stencil_lo_z(); iz < v2x_stencil_hi_z() + 1; ++iz)
    {
        for (int iDofx = 0; iDofx < v2x_n_dof_x(); ++iDofx)
        {
            stencilArray1D<fv_degree_y()> stencilVector1d;
            dofArray1D<fv_degree_y()> bhWeno1d;
            // get the 1d stencil
            for (int iy = v2x_stencil_lo_y(), iloc = 0; iy < v2x_stencil_hi_y() + 1; ++iy, ++iloc)
            {
                stencilVector1d(iloc) = bxhwenox3d(iz, iy, iDofx);
            }
            compute_cell2cell_reconstruction_1d(bhWeno1d, stencilVector1d);
            // store the 1d reconstruction
            for (int iDofy = 0; iDofy < v2x_n_dof_y(); ++iDofy)
            {
                bxhwenoxy3d(iz, iDofy, iDofx) = bhWeno1d(iDofy);
            }
        }
    }
#else
    for (int iDofz = 0; iDofz < v2x_n_dof_z(); ++iDofz)
    {
        for (int iDofy = 0; iDofy < v2x_n_dof_y(); ++iDofy)
        {
            for (int iDofx = 0; iDofx < v2x_n_dof_x(); ++iDofx)
            {
                bxhwenoxy3d(iDofz, iDofy, iDofx) = bxhwenox3d(iDofz, iDofy, iDofx);
            }
        }
    }
#endif

#if AMREX_SPACEDIM > 2
    // z-polynomial reconstruction
    for (int iDofy = 0; iDofy < v2x_n_dof_y(); ++iDofy)
    {
        for (int iDofx = 0; iDofx < v2x_n_dof_x(); ++iDofx)
        {
            stencilArray1D<fv_degree_z()> stencilVector1d;
            dofArray1D<fv_degree_z()> bhWeno1d;
            // get the 1d stencil
            for (int iz = v2x_stencil_lo_z(), iloc = 0; iz < v2x_stencil_hi_z() + 1; ++iz, ++iloc)
            {
                stencilVector1d(iloc) = bxhwenoxy3d(iz, iDofy, iDofx);
            }
            compute_cell2cell_reconstruction_1d(bhWeno1d, stencilVector1d);
            // store the 1d recontruction
            for (int iDofz = 0; iDofz < v2x_n_dof_z(); ++iDofz)
            {
                bxhwenoxyz3d(iDofz, iDofy, iDofx) = bhWeno1d(iDofz);
            }
        }
    }
#else
    for (int iDofz = 0; iDofz < v2x_n_dof_z(); ++iDofz)
    {
        for (int iDofy = 0; iDofy < v2x_n_dof_y(); ++iDofy)
        {
            for (int iDofx = 0; iDofx < v2x_n_dof_x(); ++iDofx)
            {
                bxhwenoxyz3d(iDofz, iDofy, iDofx) = bxhwenoxy3d(iDofz, iDofy, iDofx);
            }
        }
    }
#endif
}

template <int nVarFv, int polyrecDegreeFv>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::compute_weno_v2y_reconstruction(
    dofXYZ_V2y_Array3D<polyrecDegreeFv>& byhwenoxyz3d,
    amrex::Array4<amrex::Real const> const& by0,
    amrex::Real const idFace,
    int const i,
    int const j,
    int const k) const
{
    dofX_V2y_stencilYZ_Array3D<polyrecDegreeFv> byhwenox3d;
    dofXY_V2y_stencilZ_Array3D<polyrecDegreeFv> byhwenoxy3d;

    // weno V2 to V3 reconstruction
    // By
    // x-polynomial reconstruction
    for (int iz = v2y_stencil_lo_z(); iz < v2y_stencil_hi_z() + 1; ++iz)
    {
        for (int iy = v2y_stencil_lo_y(); iy < v2y_stencil_hi_y() + 1; ++iy)
        {
            stencilArray1D<fv_degree_x()> stencilVector1d;
            dofArray1D<fv_degree_x()> bhWeno1d;
            // get the 1d stencil
            for (int ix = v2y_stencil_lo_x(), iloc = 0; ix < v2y_stencil_hi_x() + 1; ++ix, ++iloc)
            {
                stencilVector1d(iloc) = by0(i + ix, j + iy, k + iz) * idFace;
            }
            compute_cell2cell_reconstruction_1d(bhWeno1d, stencilVector1d);
            // store the 1d recontruction
            for (int iDofx = 0; iDofx < v2y_n_dof_x(); ++iDofx)
            {
                byhwenox3d(iz, iy, iDofx) = bhWeno1d(iDofx);
            }
        }
    }
#if AMREX_SPACEDIM > 1
    // y-polynomial reconstruction
    for (int iz = v2y_stencil_lo_z(); iz < v2y_stencil_hi_z() + 1; ++iz)
    {
        for (int iDofx = 0; iDofx < v2y_n_dof_x(); ++iDofx)
        {
            stencilArray1Dnode<fv_degree_y()> stencilVector1d;
            dofArray1D<polyrecDegreeFv + 1> bhWeno1d;
            // get the 1d stencil
            for (int iy = v2y_stencil_lo_y(), iloc = 0; iy < v2y_stencil_hi_y() + 1; ++iy, ++iloc)
            {
                stencilVector1d(iloc) = byhwenox3d(iz, iy, iDofx);
            }
            compute_node2cell_reconstruction_1d(bhWeno1d, stencilVector1d);
            // store the 1d reconstruction
            for (int iDofy = 0; iDofy < v2y_n_dof_y(); ++iDofy)
            {
                byhwenoxy3d(iz, iDofy, iDofx) = bhWeno1d(iDofy);
            }
        }
    }
#else
    for (int iDofz = 0; iDofz < v2y_n_dof_z(); ++iDofz)
    {
        for (int iDofy = 0; iDofy < v2y_n_dof_y(); ++iDofy)
        {
            for (int iDofx = 0; iDofx < v2y_n_dof_x(); ++iDofx)
            {
                byhwenoxy3d(iDofz, iDofy, iDofx) = byhwenox3d(iDofz, iDofy, iDofx);
            }
        }
    }
#endif

#if AMREX_SPACEDIM > 2
    // z-polynomial reconstruction
    for (int iDofy = 0; iDofy < v2y_n_dof_y(); ++iDofy)
    {
        for (int iDofx = 0; iDofx < v2y_n_dof_x(); ++iDofx)
        {
            stencilArray1D<fv_degree_z()> stencilVector1d;
            dofArray1D<fv_degree_z()> bhWeno1d;
            // get the 1d stencil
            for (int iz = v2y_stencil_lo_z(), iloc = 0; iz < v2y_stencil_hi_z() + 1; ++iz, ++iloc)
            {
                stencilVector1d(iloc) = byhwenoxy3d(iz, iDofy, iDofx);
            }
            compute_cell2cell_reconstruction_1d(bhWeno1d, stencilVector1d);
            // store the reconstruction
            for (int iDofz = 0; iDofz < v2y_n_dof_z(); ++iDofz)
            {
                byhwenoxyz3d(iDofz, iDofy, iDofx) = bhWeno1d(iDofz);
            }
        }
    }
#else
    for (int iDofz = 0; iDofz < v2y_n_dof_z(); ++iDofz)
    {
        for (int iDofy = 0; iDofy < v2y_n_dof_y(); ++iDofy)
        {
            for (int iDofx = 0; iDofx < v2y_n_dof_x(); ++iDofx)
            {
                byhwenoxyz3d(iDofz, iDofy, iDofx) = byhwenoxy3d(iDofz, iDofy, iDofx);
            }
        }
    }
#endif
}

template <int nVarFv, int polyrecDegreeFv>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::compute_weno_v2z_reconstruction(
    dofXYZ_V2z_Array3D<polyrecDegreeFv>& bzhwenoxyz3d,
    amrex::Array4<amrex::Real const> const& bz0,
    amrex::Real const idFace,
    int const i,
    int const j,
    int const k) const
{
    dofX_V2z_stencilYZ_Array3D<polyrecDegreeFv> bzhwenox3d;
    dofXY_V2z_stencilZ_Array3D<polyrecDegreeFv> bzhwenoxy3d;

    // weno V2 to V3 reconstruction
    // Bz
    // x-polynomial reconstruction
    for (int iz = v2z_stencil_lo_z(); iz < v2z_stencil_hi_z() + 1; ++iz)
    {
        for (int iy = v2z_stencil_lo_y(); iy < v2z_stencil_hi_y() + 1; ++iy)
        {
            stencilArray1D<fv_degree_x()> stencilVector1d;
            dofArray1D<fv_degree_x()> bhWeno1d;
            // get the 1d stencil
            for (int ix = v2z_stencil_lo_x(), iloc = 0; ix < v2z_stencil_hi_x() + 1; ++ix, ++iloc)
            {
                stencilVector1d(iloc) = bz0(i + ix, j + iy, k + iz) * idFace;
            }
            compute_cell2cell_reconstruction_1d(bhWeno1d, stencilVector1d);
            // store the 1d reconstruction
            for (int iDofx = 0; iDofx < v2z_n_dof_x(); ++iDofx)
            {
                bzhwenox3d(iz, iy, iDofx) = bhWeno1d(iDofx);
            }
        }
    }
#if AMREX_SPACEDIM > 1
    // y-polynomial reconstruction
    for (int iz = v2z_stencil_lo_z(); iz < v2z_stencil_hi_z() + 1; ++iz)
    {
        for (int iDofx = 0; iDofx < v2z_n_dof_x(); ++iDofx)
        {
            stencilArray1D<fv_degree_y()> stencilVector1d;
            dofArray1D<fv_degree_y()> bhWeno1d;
            // get the 1d stencil
            for (int iy = v2z_stencil_lo_y(), iloc = 0; iy < v2z_stencil_hi_y() + 1; ++iy, ++iloc)
            {
                stencilVector1d(iloc) = bzhwenox3d(iz, iy, iDofx);
            }
            compute_cell2cell_reconstruction_1d(bhWeno1d, stencilVector1d);
            // store the 1d reconstruction
            for (int iDofy = 0; iDofy < v2z_n_dof_y(); ++iDofy)
            {
                bzhwenoxy3d(iz, iDofy, iDofx) = bhWeno1d(iDofy);
            }
        }
    }
#else
    for (int iDofz = 0; iDofz < v2z_n_dof_z(); ++iDofz)
    {
        for (int iDofy = 0; iDofy < v2z_n_dof_y(); ++iDofy)
        {
            for (int iDofx = 0; iDofx < v2z_n_dof_x(); ++iDofx)
            {
                bzhwenoxy3d(iDofz, iDofy, iDofx) = bzhwenox3d(iDofz, iDofy, iDofx);
            }
        }
    }
#endif

#if AMREX_SPACEDIM > 2
    // z-polynomial reconstruction
    for (int iDofy = 0; iDofy < v2z_n_dof_y(); ++iDofy)
    {
        for (int iDofx = 0; iDofx < v2z_n_dof_x(); ++iDofx)
        {
            stencilArray1Dnode<fv_degree_z()> stencilVector1d;
            dofArray1D<polyrecDegreeFv + 1> bhWeno1d;
            // get the 1d stencil
            for (int iz = v2z_stencil_lo_z(), iloc = 0; iz < v2z_stencil_hi_z() + 1; ++iz, ++iloc)
            {
                stencilVector1d(iloc) = bzhwenoxy3d(iz, iDofy, iDofx);
            }
            compute_node2cell_reconstruction_1d(bhWeno1d, stencilVector1d);
            //// store the 1d reconstruction
            for (int iDofz = 0; iDofz < v2z_n_dof_z(); ++iDofz)
            {
                bzhwenoxyz3d(iDofz, iDofy, iDofx) = bhWeno1d(iDofz);
            }
        }
    }
#else
    for (int iDofz = 0; iDofz < v2z_n_dof_z(); ++iDofz)
    {
        for (int iDofy = 0; iDofy < v2z_n_dof_y(); ++iDofy)
        {
            for (int iDofx = 0; iDofx < v2z_n_dof_x(); ++iDofx)
            {
                bzhwenoxyz3d(iDofz, iDofy, iDofx) = bzhwenoxy3d(iDofz, iDofy, iDofx);
            }
        }
    }
#endif
}

template <int nVarFv, int polyrecDegreeFv>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::compute_weno_v3_to_x_face_reconstruction(
    amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof>> const& qLx,
    amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof>> const& qRx,
    dofXYZ_Array3D<polyrecDegreeFv, nVarFv> const& whwenoxyz3d,
    int const i,
    int const j,
    int const k) const
{
    /*
        x-face 2d V3 face-reconstructed quadrature points
    */
    for (int iVar = 0; iVar < nVarFv; iVar++) // iVar3D  the x- y- z- array components
                                              // are stored one next to the other
    {
        for (int iDofz = 0; iDofz < v3_n_dof_z(); ++iDofz)
        {
            for (int iDofy = 0; iDofy < v3_n_dof_y(); ++iDofy)
            {
                int iDof2DLoc = weno_dof_2dvect_to_array_indexes(iDofz, iDofy);
                amrex::Real qlLoc{0.0};
                amrex::Real qrLoc{0.0};
                for (int iDofx = 0; iDofx < v3_n_dof_x(); ++iDofx)
                {
                    qrLoc +=
                        m_phiN1[iDofx] * whwenoxyz3d(iDofz, iDofy, iDofx,
                                                     iVar); // whWeno1d(iVar, n);  //note: this is
                                                            // the LEFT state of the i+1/2 face;
                    qlLoc += m_phiN0[iDofx] * whwenoxyz3d(iDofz, iDofy, iDofx,
                                                          iVar); //note: this is the RIGHT
                                                                 //state of the i-1/2 face;
                }
                qRx(i, j, k, iVar)[iDof2DLoc] = qlLoc;
                qLx(i + 1, j, k, iVar)[iDof2DLoc] = qrLoc;
            }
        }
    }
}

template <int nVarFv, int polyrecDegreeFv>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::compute_weno_v3_to_y_face_reconstruction(
    amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof>> const& qLy,
    amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof>> const& qRy,
    dofXYZ_Array3D<polyrecDegreeFv, nVarFv> const& whwenoxyz3d,
    int const i,
    int const j,
    int const k) const
{
    /*
                y-face 2d V3 face-reconstructed quadrature points
    */
    for (int iVar = 0; iVar < nVarFv; iVar++) // iVar3D  the x- y- z- array components
                                              // are stored one next to the other
    {
        for (int iDofz = 0; iDofz < v3_n_dof_z(); ++iDofz)
        {
            for (int iDofx = 0; iDofx < v3_n_dof_x(); ++iDofx)
            {
                int iDof2DLoc = weno_dof_2dvect_to_array_indexes(iDofz, iDofx);
                amrex::Real qlLoc{0.0};
                amrex::Real qrLoc{0.0};
                for (int iDofy = 0; iDofy < v3_n_dof_y(); ++iDofy)
                {
                    qrLoc +=
                        m_phiN1[iDofy] * whwenoxyz3d(iDofz, iDofy, iDofx,
                                                     iVar); // whWeno1d(iVar, n);  //note: this is
                                                            // the LEFT state of the i+1/2 face;
                    qlLoc += m_phiN0[iDofy] *
                             whwenoxyz3d(iDofz, iDofy, iDofx,
                                         iVar); //note: this is the RIGHT state of the i-1/2 face;
                }
                qRy(i, j, k, iVar)[iDof2DLoc] = qlLoc;
                qLy(i, j + 1, k, iVar)[iDof2DLoc] = qrLoc;
            }
        }
    }
}

template <int nVarFv, int polyrecDegreeFv>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::compute_weno_v3_to_z_face_reconstruction(
    amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof>> const& qLz,
    amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof>> const& qRz,
    dofXYZ_Array3D<polyrecDegreeFv, nVarFv> const& whwenoxyz3d,
    int const i,
    int const j,
    int const k) const
{
    /*
        z-face 2d V3 face-reconstructed quadrature points
    */
    for (int iVar = 0; iVar < nVarFv; iVar++) // iVar3D  the x- y- z- array components
                                              // are stored one next to the other
    {
        for (int iDofy = 0; iDofy < v3_n_dof_y(); ++iDofy)
        {
            for (int iDofx = 0; iDofx < v3_n_dof_x(); ++iDofx)
            {
                int iDof2DLoc = weno_dof_2dvect_to_array_indexes(iDofy, iDofx);
                amrex::Real qlLoc{0.0};
                amrex::Real qrLoc{0.0};
                for (int iDofz = 0; iDofz < v3_n_dof_z(); ++iDofz)
                {
                    qrLoc +=
                        m_phiN1[iDofz] * whwenoxyz3d(iDofz, iDofy, iDofx,
                                                     iVar); // whWeno1d(iVar, n);  //note: this is
                                                            // the LEFT state of the i+1/2 face;
                    qlLoc += m_phiN0[iDofz] *
                             whwenoxyz3d(iDofz, iDofy, iDofx,
                                         iVar); //note: this is the RIGHT state of the i-1/2 face;
                }
                qRz(i, j, k, iVar)[iDof2DLoc] = qlLoc;
                qLz(i, j, k + 1, iVar)[iDof2DLoc] = qrLoc;
            }
        }
    }
}

template <int nVarFv, int polyrecDegreeFv>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::compute_weno_v2x_to_x_face_reconstruction(
    amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof>> const& qLx,
    amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof>> const& qRx,
    dofXYZ_V2x_Array3D<polyrecDegreeFv> const& v2xWenoxyz3d,
    int const nidx,
    int const i,
    int const j,
    int const k) const
{
    /*
        x-face V2x face-reconstructed quadrature points
    */
    for (int iDofz = 0; iDofz < v2x_n_dof_z(); ++iDofz)
    {
        for (int iDofy = 0; iDofy < v2x_n_dof_y(); ++iDofy)
        {
            int iDof2DLoc = weno_dof_2dvect_to_array_indexes(iDofz, iDofy);
            amrex::Real qlLoc{0.0};
            amrex::Real qrLoc{0.0};
            for (int iDofx = 0; iDofx < v2x_n_dof_x(); ++iDofx)
            {
                qrLoc +=
                    m_phiM1[iDofx] * v2xWenoxyz3d(iDofz, iDofy,
                                                  iDofx); // whWeno1d(iVar, n);  //note: this is
                                                          // the LEFT state of the i+1/2 face;
                qlLoc += m_phiM0[iDofx] * v2xWenoxyz3d(iDofz, iDofy,
                                                       iDofx); //note: this is the RIGHT
                                                               //state of the i-1/2 face;
            }
            qRx(i, j, k, nidx)[iDof2DLoc] = qlLoc;
            qLx(i + 1, j, k, nidx)[iDof2DLoc] = qrLoc;
        }
    }
}

template <int nVarFv, int polyrecDegreeFv>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::compute_weno_v2y_to_y_face_reconstruction(
    amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof>> const& qLy,
    amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof>> const& qRy,
    dofXYZ_V2y_Array3D<polyrecDegreeFv> const& v2yWenoxyz3d,
    int const nidx,
    int const i,
    int const j,
    int const k) const
{
    /*
        y-face 2d V2 face-reconstructed quadrature points
    */
    // By
    for (int iDofz = 0; iDofz < v2y_n_dof_z(); ++iDofz)
    {
        for (int iDofx = 0; iDofx < v2y_n_dof_x(); ++iDofx)
        {
            int iDof2DLoc = weno_dof_2dvect_to_array_indexes(iDofz, iDofx);
            amrex::Real qlLoc{0.0};
            amrex::Real qrLoc{0.0};
            for (int iDofy = 0; iDofy < v2y_n_dof_y(); ++iDofy)
            {
                qrLoc +=
                    m_phiM1[iDofy] * v2yWenoxyz3d(iDofz, iDofy,
                                                  iDofx); // whWeno1d(iVar, n);  //note: this is
                                                          // the LEFT state of the i+1/2 face;
                qlLoc += m_phiM0[iDofy] *
                         v2yWenoxyz3d(iDofz, iDofy,
                                      iDofx); //note: this is the RIGHT state of the i-1/2 face;
            }
            qRy(i, j, k, nidx)[iDof2DLoc] = qlLoc;
            qLy(i, j + 1, k, nidx)[iDof2DLoc] = qrLoc;
        }
    }
}

template <int nVarFv, int polyrecDegreeFv>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
FVreconstruction<nVarFv, polyrecDegreeFv>::compute_weno_v2z_to_z_face_reconstruction(
    amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof>> const& qLz,
    amrex::Array4<amrex::GpuArray<amrex::Real, s_faceDof>> const& qRz,
    dofXYZ_V2z_Array3D<polyrecDegreeFv> const& v2zWenoxyz3d,
    int const nidx,
    int const i,
    int const j,
    int const k) const
{
    /*
        z-face 2d V2 face-reconstructed quadrature points
    */
    for (int iDofy = 0; iDofy < v2z_n_dof_y(); ++iDofy)
    {
        for (int iDofx = 0; iDofx < v2z_n_dof_x(); ++iDofx)
        {
            int iDof2DLoc = weno_dof_2dvect_to_array_indexes(iDofy, iDofx);
            amrex::Real qlLoc{0.0};
            amrex::Real qrLoc{0.0};
            for (int iDofz = 0; iDofz < v2z_n_dof_z(); ++iDofz)
            {
                qrLoc +=
                    m_phiM1[iDofz] * v2zWenoxyz3d(iDofz, iDofy,
                                                  iDofx); // whWeno1d(iVar, n);  //note: this is
                                                          // the LEFT state of the i+1/2 face;
                qlLoc += m_phiM0[iDofz] *
                         v2zWenoxyz3d(iDofz, iDofy,
                                      iDofx); //note: this is the RIGHT state of the i-1/2 face;
            }
            qRz(i, j, k, nidx)[iDof2DLoc] = qlLoc;
            qLz(i, j, k + 1, nidx)[iDof2DLoc] = qrLoc;
        }
    }
}

#endif
