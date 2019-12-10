//------------------------------------------------------------------------------
// functions copied from the AMReX library to make small changes
// AMReX file containing original functions: AMReX_Particle_mod_K.H
// (amrex/Src/Particle/AMReX_Particle_mod_K.H) pulled on 07.10.2019
//------------------------------------------------------------------------------

#ifndef PARTICLE_MESH_COUPLING_H_
#define PARTICLE_MESH_COUPLING_H_

#include <AMReX_Particles.H>
#include <AMReX_FArrayBox.H>
#include <cmath>

template <typename P>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void gempic_deposit_cic (P const& p, amrex::Real charge,
			 amrex::Array4<amrex::Real> const& rho,
			 amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
			 amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi,
             std::array<int,3> deg = {1,1,1})
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

    // ---- interpolation in x-dir
    std::vector<amrex::Real> sx;
    switch (deg[0]) {
        case 1:
            sx = {1.-xint, xint};
        case 3:
            sx = {-1./6.*xint*(xint-1)*(xint-2), 1./2.*(xint+1)*(xint-1)*(xint-2),
                    -1./2.*(xint+1)*xint*(xint-2), 1./6.*(xint+1)*xint*(xint-1)};
    }

    // ---- interpolation in y-dir
    std::vector<amrex::Real> sy;
    switch (deg[1]) {
        case 1:
            sy = {1.-yint, yint};
        case 3:
            sy = {-1./6.*yint*(yint-1)*(yint-2), 1./2.*(yint+1)*(yint-1)*(yint-2),
                  -1./2.*(yint+1)*yint*(yint-2), 1./6.*(yint+1)*yint*(yint-1)};
    }

    // ---- interpolation in z-dir
    std::vector<amrex::Real> sz;
    switch (deg[2]) {
        case 1:
            sz = {1.-zint, zint};
        case 3:
            sz = {-1./6.*zint*(zint-1)*(zint-2), 1./2.*(zint+1)*(zint-1)*(zint-2),
                  -1./2.*(zint+1)*zint*(zint-2), 1./6.*(zint+1)*zint*(zint-1)};
    }
    
    for (int kk = 0-(deg[2]-1)/2; kk <= (deg[2]+1)/2; ++kk) {
        for (int jj = 0-(deg[1]-1)/2; jj <= (deg[1]+1)/2; ++jj) {
            for (int ii = 0-(deg[0]-1)/2; ii <= (deg[0]+1)/2; ++ii) {
                amrex::Gpu::Atomic::Add(&rho(i+ii, j+jj, k+kk, 0),
                                        sx[ii]*sy[jj]*sz[kk]*charge*p.rdata(3));
            }
        }
    }
}

template <typename P>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void gempic_deposit_J_cic (P const& p, amrex::Real charge, int cmp,
			   amrex::Array4<amrex::Real> const& j_comp,
			   std::array<amrex::Real,3> ploF,
			   amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi,
                           amrex::Real coefficient = 1.0,
               std::array<int,3> deg = {1,1,1})
{

  amrex::Real lx = p.pos(0)*dxi[0] - ploF[0];
  amrex::Real ly = p.pos(1)*dxi[1] - ploF[1];
  amrex::Real lz = p.pos(2)*dxi[2] - ploF[2];

    int i = std::floor(lx);
    int j = std::floor(ly);
    int k = std::floor(lz);

    amrex::Real xint = lx - i;
    amrex::Real yint = ly - j;
    amrex::Real zint = lz - k;

    // ---- interpolation in x-dir
    std::vector<amrex::Real> sx;
    switch (deg[0]) {
        case 1:
            sx = {1.-xint, xint};
        case 3:
            sx = {-1./6.*xint*(xint-1)*(xint-2), 1./2.*(xint+1)*(xint-1)*(xint-2),
                    -1./2.*(xint+1)*xint*(xint-2), 1./6.*(xint+1)*xint*(xint-1)};
    }

    // ---- interpolation in y-dir
    std::vector<amrex::Real> sy;
    switch (deg[1]) {
        case 1:
            sy = {1.-yint, yint};
        case 3:
            sy = {-1./6.*yint*(yint-1)*(yint-2), 1./2.*(yint+1)*(yint-1)*(yint-2),
                  -1./2.*(yint+1)*yint*(yint-2), 1./6.*(yint+1)*yint*(yint-1)};
    }

    // ---- interpolation in z-dir
    std::vector<amrex::Real> sz;
    switch (deg[2]) {
        case 1:
            sz = {1.-zint, zint};
        case 3:
            sz = {-1./6.*zint*(zint-1)*(zint-2), 1./2.*(zint+1)*(zint-1)*(zint-2),
                  -1./2.*(zint+1)*zint*(zint-2), 1./6.*(zint+1)*zint*(zint-1)};
    }
    
    for (int kk = 0-(deg[2]-1)/2; kk <= (deg[2]+1)/2; ++kk) {
        for (int jj = 0-(deg[1]-1)/2; jj <= (deg[1]+1)/2; ++jj) {
            for (int ii = 0-(deg[0]-1)/2; ii <= (deg[0]+1)/2; ++ii) {
	      // p.rdata(cmp): velocity-component cmp, p.rdata(3): weight
                amrex::Gpu::Atomic::Add(&j_comp(i+ii, j+jj, k+kk, 0),
                    coefficient*sx[ii]*sy[jj]*sz[kk]*charge*p.rdata(cmp)*p.rdata(3));
            }
        }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void gempic_deposit_J_cic_global (std::array<amrex::Real,6> partInfo, amrex::Real part_weight, amrex::Real charge, int cmp,
               maxwell_yee *mw_yee,
               std::array<amrex::Real,3> ploF,
               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi,
               std::array<int,3> deg = {1,1,1})
{

  amrex::Real lx = partInfo[0]*dxi[0] - ploF[0];
  amrex::Real ly = partInfo[1]*dxi[1] - ploF[1];
  amrex::Real lz = partInfo[2]*dxi[2] - ploF[2];

    int i = std::floor(lx);
    int j = std::floor(ly);
    int k = std::floor(lz);

    amrex::Real xint = lx - i;
    amrex::Real yint = ly - j;
    amrex::Real zint = lz - k;

    // ---- interpolation in x-dir
    std::vector<amrex::Real> sx;
    switch (deg[0]) {
        case 1:
            sx = {1.-xint, xint};
        case 3:
            sx = {-1./6.*xint*(xint-1)*(xint-2), 1./2.*(xint+1)*(xint-1)*(xint-2),
                    -1./2.*(xint+1)*xint*(xint-2), 1./6.*(xint+1)*xint*(xint-1)};
    }

    // ---- interpolation in y-dir
    std::vector<amrex::Real> sy;
    switch (deg[1]) {
        case 1:
            sy = {1.-yint, yint};
        case 3:
            sy = {-1./6.*yint*(yint-1)*(yint-2), 1./2.*(yint+1)*(yint-1)*(yint-2),
                  -1./2.*(yint+1)*yint*(yint-2), 1./6.*(yint+1)*yint*(yint-1)};
    }

    // ---- interpolation in z-dir
    std::vector<amrex::Real> sz;
    switch (deg[2]) {
        case 1:
            sz = {1.-zint, zint};
        case 3:
            sz = {-1./6.*zint*(zint-1)*(zint-2), 1./2.*(zint+1)*(zint-1)*(zint-2),
                  -1./2.*(zint+1)*zint*(zint-2), 1./6.*(zint+1)*zint*(zint-1)};
    }

    for (int kk = 0-(deg[2]-1)/2; kk <= (deg[2]+1)/2; ++kk) {
        for (int jj = 0-(deg[1]-1)/2; jj <= (deg[1]+1)/2; ++jj) {
            for (int ii = 0-(deg[0]-1)/2; ii <= (deg[0]+1)/2; ++ii) {
                amrex::Box cc({i+ii,j+jj,k+kk}, {i+ii,j+jj,k+kk}, (*(*mw_yee).E_Index[cmp]));
                (*(*mw_yee).J_Array[cmp]).plus(sx[ii]*sy[jj]*sz[kk]*charge*partInfo[cmp+3]*part_weight, cc, 0, 1);
            }
        }
    }
}

template <typename P>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real gempic_interpolate_cic (P const& p, amrex::Array4<amrex::Real const> const& acc,
                                    std::array<amrex::Real,3> ploF,
                            amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi,
                    std::array<int,3> deg = {3,3,3})
{
    amrex::Real acceleration = 0.0;

    amrex::Real lx = p.pos(0)*dxi[0] - ploF[0];
    amrex::Real ly = p.pos(1)*dxi[1] - ploF[1];
    amrex::Real lz = p.pos(2)*dxi[2] - ploF[2];
    
    int i = std::floor(lx);
    int j = std::floor(ly);
    int k = std::floor(lz);
    
    amrex::Real xint = lx - i;
    amrex::Real yint = ly - j;
    amrex::Real zint = lz - k;
    
    // ---- interpolation in x-dir
    std::vector<amrex::Real> sx;
    switch (deg[0]) {
        case 1:
            sx = {1.-xint, xint};
        case 3:
            sx = {-1./6.*xint*(xint-1)*(xint-2), 1./2.*(xint+1)*(xint-1)*(xint-2),
                    -1./2.*(xint+1)*xint*(xint-2), 1./6.*(xint+1)*xint*(xint-1)};
    }

    // ---- interpolation in y-dir
    std::vector<amrex::Real> sy;
    switch (deg[1]) {
        case 1:
            sy = {1.-yint, yint};
        case 3:
            sy = {-1./6.*yint*(yint-1)*(yint-2), 1./2.*(yint+1)*(yint-1)*(yint-2),
                  -1./2.*(yint+1)*yint*(yint-2), 1./6.*(yint+1)*yint*(yint-1)};
    }

    // ---- interpolation in z-dir
    std::vector<amrex::Real> sz;
    switch (deg[2]) {
        case 1:
            sz = {1.-zint, zint};
        case 3:
            sz = {-1./6.*zint*(zint-1)*(zint-2), 1./2.*(zint+1)*(zint-1)*(zint-2),
                  -1./2.*(zint+1)*zint*(zint-2), 1./6.*(zint+1)*zint*(zint-1)};
    }

    for (int kk = 0-(deg[2]-1)/2; kk <= (deg[2]+1)/2; ++kk) {
        for (int jj = 0-(deg[1]-1)/2; jj <= (deg[1]+1)/2; ++jj) {
            for (int ii = 0-(deg[0]-1)/2; ii <= (deg[0]+1)/2; ++ii) {
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
  newPos[0] = fmod(p.pos(0) + p.rdata(0)*dt, infra.geom.ProbHi()[0]);
  newPos[1] = fmod(p.pos(1) + p.rdata(1)*dt, infra.geom.ProbHi()[1]);
  newPos[2] = fmod(p.pos(2) + p.rdata(2)*dt, infra.geom.ProbHi()[2]);

  newPos[3] = p.rdata(0) + dt*chargemass*(Ep[0]+(p.rdata(1)*Bp[2]-p.rdata(2)*Bp[1]));
  newPos[4] = p.rdata(1) + dt*chargemass*(Ep[1]+(p.rdata(2)*Bp[0]-p.rdata(0)*Bp[2]));
  newPos[5] = p.rdata(2) + dt*chargemass*(Ep[2]+(p.rdata(0)*Bp[1]-p.rdata(1)*Bp[0]));

  return(newPos);
}

template <typename P>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
std::array<amrex::Real,6> push_particle_boris(P const& p,
                    amrex::Real dt,
                    amrex::Real chargemass, //charge/mass
                    std::array<amrex::Real,3> Ep,
                    std::array<amrex::Real,3> Bp,
                    infrastructure infra){

  std::array<amrex::Real,6> newPos;
  std::array<amrex::Real,3> vm;
  std::array<amrex::Real,3> vt;
  std::array<amrex::Real,3> vp;
  amrex::Real S;

  // v-
  vm[0] = p.rdata(0) + dt*chargemass/2*Ep[0];
  vm[1] = p.rdata(1) + dt*chargemass/2*Ep[1];
  vm[2] = p.rdata(2) + dt*chargemass/2*Ep[2];

  // v'
  vt[0] = vm[0] + (vm[1]*dt*chargemass/2*Bp[2]-vm[2]*dt*chargemass/2*Bp[1]);
  vt[1] = vm[1] + (vm[2]*dt*chargemass/2*Bp[0]-vm[0]*dt*chargemass/2*Bp[2]);
  vt[2] = vm[2] + (vm[0]*dt*chargemass/2*Bp[1]-vm[1]*dt*chargemass/2*Bp[0]);

  S = pow(dt*chargemass/2*Bp[0], 2.)+pow(dt*chargemass/2*Bp[1], 2.)+pow(dt*chargemass/2*Bp[2], 2.);

  // v+
  vp[0] = vm[0] + (vt[1]*dt*chargemass/(1.+S)*Bp[2]-vt[2]*dt*chargemass/2*Bp[1]);
  vp[1] = vm[1] + (vt[2]*dt*chargemass/(1.+S)*Bp[0]-vt[0]*dt*chargemass/2*Bp[2]);
  vp[2] = vm[2] + (vt[0]*dt*chargemass/(1.+S)*Bp[1]-vt[1]*dt*chargemass/2*Bp[0]);

  // v
  newPos[3] = vp[0] + dt*chargemass/2*Ep[0];
  newPos[4] = vp[1] + dt*chargemass/2*Ep[1];
  newPos[5] = vp[2] + dt*chargemass/2*Ep[2];

  // new positions (modulo the upper bound of the box)
  newPos[0] = fmod(p.pos(0) + newPos[0]*dt, infra.geom.ProbHi()[0]);
  newPos[1] = fmod(p.pos(1) + newPos[1]*dt, infra.geom.ProbHi()[1]);
  newPos[2] = fmod(p.pos(2) + newPos[2]*dt, infra.geom.ProbHi()[2]);



  return(newPos);
}

#endif
