#ifndef MAG2dEB_K_H_
#define MAG2dEB_K_H_

#include <AMReX_FArrayBox.H>

using namespace amrex;

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void actual_init_ebpoisson(int i, int j,
			   Array4<Real > const& rhs,
			   Array4<Real > const& exact,
			   Array4<Real > const& alpha,
			   Array4<Real > const& beta,
               GpuArray<Real,AMREX_SPACEDIM> const& prob_lo,
               GpuArray<Real,AMREX_SPACEDIM> const& dx) noexcept
{
  Real x = prob_lo[0] + dx[0]*(i + 0.5);
  Real y = prob_lo[1] + dx[1]*(j + 0.5);
  Real r2 = x*x + y*y;
  alpha(i,j,0) = 0.;
  beta(i,j,0) = 1.;
  if(r2 > 16.0) {
    exact(i,j,0) = 0.;
    rhs(i,j,0) = 0.;
  }
  else {
    exact(i,j,0) = (r2 - 16.0)*(r2 - 0.25)*x*y*y;
    rhs(i,j,0) = -32.0*x*y*y*(r2 - 8.125) - 8.0*x*y*y*r2
      - 2.0*x*(r2 - 16.0)*(r2 - 0.25);
  }
}
    
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void actual_init_ebpolar(int i, int j,
			 Array4<Real > const& rhs,
			 Array4<Real > const& exact,
			 Array4<Real > const& alpha,
			 Array4<Real > const& beta,
             GpuArray<Real,AMREX_SPACEDIM> const& prob_lo,
             GpuArray<Real,AMREX_SPACEDIM> const& dx) noexcept
{
  Real x = prob_lo[0] + dx[0]*(i + 0.5);
  Real y = prob_lo[1] + dx[1]*(j + 0.5);
  Real r2 = x*x + y*y;
  alpha(i,j,0) = 0.;
  beta(i,j,0) = 1.;
  if(r2 > 16.0) {
    exact(i,j,0) = 0.;
    rhs(i,j,0) = 0.;
  }
  else if(r2 < 0.25) {
    exact(i,j,0) = 0.;
    rhs(i,j,0) = 0.;
  }
  else {
    exact(i,j,0) = (r2 - 16.0)*(r2 - 0.25)*x*y*y*y/(r2*r2*r2*r2);
    Real rtmp = (r2 - 16.0)*(r2 - 0.25)*x*y;
    rtmp *= 8.0*x*x*x*x + 14.0*x*x*y*y + 6.0*y*y*y*y;
    rtmp /= -r2*r2*r2*r2*r2*r2;
    rtmp += 12.0*x*y*y*y*(2.0*r2 - 16.25)/(r2*r2*r2*r2); 
    rhs(i,j,0) = rtmp - 8.0*x*y*y*y/(r2*r2*r2);
  }
}

#endif
