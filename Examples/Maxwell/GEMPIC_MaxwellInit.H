#ifndef GEMPIC_MAXWELLINIT_H_
#define GEMPIC_MAXWELLINIT_H_

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_FabArray.H>
#include <AMReX_MultiFab.H>

#include "GEMPIC_Config.H"
#include "GEMPIC_MaxwellStructDefinitions.H"

AMREX_GPU_HOST_DEVICE inline amrex::GpuArray<amrex::Real, 3> initial_e(
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& r, const amrex::Real time);

AMREX_GPU_HOST_DEVICE inline amrex::GpuArray<amrex::Real, 3> initial_b(
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& r, const amrex::Real time);

void maxwell_initial_condition(MaxwellDiscretization& disc);

#endif