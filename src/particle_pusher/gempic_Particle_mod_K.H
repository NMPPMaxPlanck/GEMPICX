//------------------------------------------------------------------------------
// functions copied from the AMReX library to make small changes
// AMReX file containing original functions: AMReX_Particle_mod_K.H
// (amrex/Src/Particle/AMReX_Particle_mod_K.H) pulled on 07.10.2019
//------------------------------------------------------------------------------

#ifndef GEMPIC_PARTICLE_MOD_K_H_
#define GEMPIC_PARTICLE_MOD_K_H_

#include <AMReX_Particles.H>
#include <AMReX_FArrayBox.H>
#include <cmath>

template <typename P>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void gempic_deposit_cic (P const& p, amrex::Real charge,
			 amrex::Array4<amrex::Real> const& rho,
			 amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
			 amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi)
{
    amrex::Real lx = (p.pos(0) - plo[0]) * dxi[0]; // don't need to displace because rho is nodal
    amrex::Real ly = (p.pos(1) - plo[1]) * dxi[1];
    amrex::Real lz = (p.pos(2) - plo[2]) * dxi[2];

    int i = std::floor(lx);
    int j = std::floor(ly);
    int k = std::floor(lz);

    amrex::Real xint = lx - i;
    amrex::Real yint = ly - j;
    amrex::Real zint = lz - k;

    amrex::Real sx[] = {1.-xint, xint};
    amrex::Real sy[] = {1.-yint, yint};
    amrex::Real sz[] = {1.-zint, zint};
    
    for (int kk = 0; kk <= 1; ++kk) { 
        for (int jj = 0; jj <= 1; ++jj) { 
            for (int ii = 0; ii <= 1; ++ii) {
                amrex::Gpu::Atomic::Add(&rho(i+ii, j+jj, k+kk, 0),
                                        sx[ii]*sy[jj]*sz[kk]*charge);
            }
        }
    }
}

template <typename P>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void gempic_deposit_J_cic (P const& p, amrex::Real charge, int cmp,
			   amrex::Array4<amrex::Real> const& j_comp,
			   amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
			   amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi,
                            amrex::IndexType fieldIndex)
{
  // J is on the same grid as E
    amrex::Real lx = (p.pos(0) - plo[0]) * dxi[0] - 0.5*(1-fieldIndex[0]);
    amrex::Real ly = (p.pos(1) - plo[1]) * dxi[1] - 0.5*(1-fieldIndex[0]);
    amrex::Real lz = (p.pos(2) - plo[2]) * dxi[2] - 0.5*(1-fieldIndex[0]);

    int i = std::floor(lx);
    int j = std::floor(ly);
    int k = std::floor(lz);

    amrex::Real xint = lx - i;
    amrex::Real yint = ly - j;
    amrex::Real zint = lz - k;

    amrex::Real sx[] = {1.-xint, xint};
    amrex::Real sy[] = {1.-yint, yint};
    amrex::Real sz[] = {1.-zint, zint};
    
    for (int kk = 0; kk <= 1; ++kk) { 
        for (int jj = 0; jj <= 1; ++jj) { 
            for (int ii = 0; ii <= 1; ++ii) {
	      // p.rdata(cmp): velocity-component cmp
                amrex::Gpu::Atomic::Add(&j_comp(i+ii, j+jj, k+kk, 0),
					sx[ii]*sy[jj]*sz[kk]*charge*p.rdata(cmp));
            }
        }
    }
}

template <typename P>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real gempic_interpolate_cic (P const& p, amrex::Array4<amrex::Real const> const& acc,
                            amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
                            amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi,
                            amrex::IndexType fieldIndex)
{
    amrex::Real acceleration = 0.0;
    amrex::Real lx = (p.pos(0) - plo[0]) * dxi[0] - 0.5*(1-fieldIndex[0]);
    amrex::Real ly = (p.pos(1) - plo[1]) * dxi[1] - 0.5*(1-fieldIndex[1]);
    amrex::Real lz = (p.pos(2) - plo[2]) * dxi[2] - 0.5*(1-fieldIndex[2]);
    
    int i = std::floor(lx);
    int j = std::floor(ly);
    int k = std::floor(lz);
    
    amrex::Real xint = lx - i;
    amrex::Real yint = ly - j;
    amrex::Real zint = lz - k;
    
    amrex::Real sx[] = {1.-xint, xint};
    amrex::Real sy[] = {1.-yint, yint};
    amrex::Real sz[] = {1.-zint, zint};

    for (int kk = 0; kk <= 1; ++kk) { 
            for (int jj = 0; jj <= 1; ++jj) { 
                for (int ii = 0; ii <= 1; ++ii) {
                    acceleration += sx[ii]*sy[jj]*sz[kk]*acc(i+ii,j+jj,k+kk,0);
                }
            }
        }
    
 return(acceleration);
}

template <typename P>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
std::array<amrex::Real,6> push_particle(P const& p,
					amrex::Real dt,
					amrex::Real chargemass, //charge/mass
					std::array<amrex::Real,3> Ep,
					std::array<amrex::Real,3> Bp,
					infrastructure infra){

  std::array<amrex::Real,6> newPos;
  // new positions (modulo the upper bound of the box)
  newPos[0] = fmod(p.pos(0) + p.rdata(0)*dt, infra.geom.ProbLo()[0]);
  newPos[1] = fmod(p.pos(1) + p.rdata(1)*dt, infra.geom.ProbLo()[0]);
  newPos[2] = fmod(p.pos(2) + p.rdata(2)*dt, infra.geom.ProbLo()[0]);

  newPos[3] = p.rdata(0) + chargemass*(Ep[0]+Bp[0]*p.rdata(0));
  newPos[4] = p.rdata(1) + chargemass*(Ep[1]+Bp[1]*p.rdata(1));
  newPos[5] = p.rdata(2) + chargemass*(Ep[2]+Bp[2]*p.rdata(2));

  return(newPos);
}

#endif
