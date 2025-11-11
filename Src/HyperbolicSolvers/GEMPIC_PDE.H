#ifndef GEMPIC_PDE_H_
#define GEMPIC_PDE_H_

#include <iostream>

#include <AMReX.H>

#include "GEMPIC_Config.H"

template <int nvar>
using varArray = amrex::GpuArray<amrex::Real, nvar>;

template <int nvar>
using varArray2D = amrex::Array2D<amrex::Real, 0, nvar - 1, 0, nvar - 1, amrex::Order::C>;

template <int nvar>
using fluxArray = amrex::GpuArray<amrex::GpuArray<amrex::Real, nvar>, AMREX_SPACEDIM>;

template <int nVarPde, typename ParamsClass, typename PdeTagType>
AMREX_GPU_HOST_DEVICE void pde_cons_2_prim (varArray<nVarPde> const& Q,
                                            varArray<nVarPde>& V,
                                            ParamsClass const& params)
{
    static_assert(gempic_always_false<PdeTagType>(),
                  "pde_cons_2_prim is not implemented for this TagType!");
}

template <int nVarPde, typename ParamsClass, typename PdeTagType>
AMREX_GPU_HOST_DEVICE void pde_prim_2_cons (varArray<nVarPde> const& V,
                                            varArray<nVarPde>& Q,
                                            ParamsClass const& params)
{
    static_assert(gempic_always_false<PdeTagType>(),
                  "pde_prim_2_cons is not implemented for this TagType!");
}

template <int nVarPde, typename ParamsClass, typename PdeTagType>
AMREX_GPU_HOST_DEVICE void pde_get_fluxes (varArray<nVarPde> const& Q,
                                           fluxArray<nVarPde>& f,
                                           ParamsClass const& params)
{
    static_assert(gempic_always_false<PdeTagType>(),
                  "pde_get_fluxes is not implemented for this TagType!");
}

template <int nVarPde, typename ParamsClass, typename PdeTagType>
AMREX_GPU_HOST_DEVICE void pde_get_jacobian (varArray<nVarPde> const& Q,
                                             AMREX_D_DECL(varArray2D<nVarPde>& Ax,
                                                          varArray2D<nVarPde>& Ay,
                                                          varArray2D<nVarPde>& Az),
                                             ParamsClass const& params)
{
    static_assert(gempic_always_false<PdeTagType>(),
                  "pde_get_jacobian is not implemented for this TagType!");
}

template <int nVarPde, typename ParamsClass, typename PdeTagType>
AMREX_GPU_HOST_DEVICE void pde_get_eigenvalues (varArray<nVarPde> const& Q,
                                                amrex::GpuArray<amrex::Real, 3> const& n,
                                                varArray<nVarPde>& mask,
                                                varArray<nVarPde>& L,
                                                ParamsClass const& params)
{
    static_assert(gempic_always_false<PdeTagType>(),
                  "pde_get_eigenvalues is not implemented for this TagType!");
}

template <int nVarPde, typename ParamsClass, typename PdeTagType>
AMREX_GPU_HOST_DEVICE void pde_get_eigenvectors (varArray<nVarPde> const& Q,
                                                 amrex::GpuArray<amrex::Real, 3> const& n,
                                                 varArray2D<nVarPde>& R,
                                                 varArray2D<nVarPde>& L,
                                                 varArray2D<nVarPde>& iR,
                                                 ParamsClass const& params)
{
    static_assert(gempic_always_false<PdeTagType>(),
                  "pde_get_eigenvectors is not implemented for this TagType!");
}

template <int nVarPde, typename ParamsClass, typename PdeTagType>
AMREX_GPU_HOST_DEVICE void pde_numerical_flux (varArray<nVarPde>& ql,
                                               varArray<nVarPde>& qr,
                                               amrex::GpuArray<amrex::Real, 3> n,
                                               varArray<nVarPde>& flux,
                                               varArray<nVarPde>& flux1,
                                               ParamsClass const& params)
{
    static_assert(gempic_always_false<PdeTagType>(),
                  "pde_numerical_flux is not implemented for this TagType!");
}

#endif