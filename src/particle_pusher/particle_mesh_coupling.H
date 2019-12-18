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
#include <gempic_Config.H>

template <typename P>
void gempic_deposit_cic (P const& p, amrex::Real charge,
             amrex::Array4<amrex::Real> const& rho,
             amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> const& plo,
             amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> const& dxi,
             amrex::IntVect deg = amrex::IntVect{AMREX_D_DECL(1,1,1)})

{
    amrex::Real lx = (p.pos(0) - plo[0]) * dxi[0]; // don't need to displace because rho is nodal
    int i = std::floor(lx);
    amrex::Real xint = lx - i;

    // ---- interpolation in x-dir
    std::vector<amrex::Real> sx;
    switch (deg[0]) {
        case 1:
            sx = {1.-xint, xint};
        break;
        case 3:
            sx = {-1./6.*xint*(xint-1)*(xint-2), 1./2.*(xint+1)*(xint-1)*(xint-2),
                    -1./2.*(xint+1)*xint*(xint-2), 1./6.*(xint+1)*xint*(xint-1)};
        break;
    }

#if (GEMPIC_SPACEDIM > 1)
    amrex::Real ly = (p.pos(1) - plo[1]) * dxi[1];
    int j = std::floor(ly);
    amrex::Real yint = ly - j;

    // ---- interpolation in y-dir
    std::vector<amrex::Real> sy;
    switch (deg[1]) {
        case 1:
            sy = {1.-yint, yint};
        break;
        case 3:
            sy = {-1./6.*yint*(yint-1)*(yint-2), 1./2.*(yint+1)*(yint-1)*(yint-2),
                  -1./2.*(yint+1)*yint*(yint-2), 1./6.*(yint+1)*yint*(yint-1)};
        break;
    }

#endif
#if (GEMPIC_SPACEDIM > 2)
    amrex::Real lz = (p.pos(2) - plo[2]) * dxi[2];
    int k = std::floor(lz);
    amrex::Real zint = lz - k;

    // ---- interpolation in z-dir
    std::vector<amrex::Real> sz;
    switch (deg[2]) {
        case 1:
            sz = {1.-zint, zint};
        break;
        case 3:
            sz = {-1./6.*zint*(zint-1)*(zint-2), 1./2.*(zint+1)*(zint-1)*(zint-2),
                  -1./2.*(zint+1)*zint*(zint-2), 1./6.*(zint+1)*zint*(zint-1)};
        break;
    }

#endif

#if (GEMPIC_SPACEDIM > 2)
    for (int kk = 0-(deg[2]-1)/2; kk <= (deg[2]+1)/2; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0-(deg[1]-1)/2; jj <= (deg[1]+1)/2; ++jj) {
#endif
            for (int ii = 0-(deg[0]-1)/2; ii <= (deg[0]+1)/2; ++ii) {
                double s_coeff = sx[ii]
        #if(GEMPIC_SPACEDIM > 1)
                        * sy[jj]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                        * sz[kk]
        #endif
                        ;
                //TODO: try exchanging for amrex::Add()
                amrex::Gpu::Atomic::Add(&rho(i+ii
                             #if (GEMPIC_SPACEDIM > 1)
                                             , j+jj
                             #endif
                             #if (GEMPIC_SPACEDIM > 2)
                                             , k+kk
                             #endif
                                             , 0), s_coeff*charge*p.rdata(GEMPIC_SPACEDIM));


            }
#if (GEMPIC_SPACEDIM > 1)
        }
#endif
#if (GEMPIC_SPACEDIM > 2)
    }
#endif
}

template <typename P>
void gempic_deposit_J_cic (P const& p, amrex::Real charge, int cmp,
                           amrex::Array4<amrex::Real> const& j_comp,
               std::array<amrex::Real,GEMPIC_SPACEDIM> ploF,
               amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> const& dxi,
                           amrex::Real coefficient = 1.0,
               amrex::IntVect deg = amrex::IntVect{AMREX_D_DECL(1,1,1)})
{
  amrex::Real lx = p.pos(0)*dxi[0] - ploF[0];
  int i = std::floor(lx);
  amrex::Real xint = lx - i;

  // ---- interpolation in x-dir
  std::vector<amrex::Real> sx;
  switch (deg[0]) {
      case 1:
          sx = {1.-xint, xint};
      break;
      case 3:
          sx = {-1./6.*xint*(xint-1)*(xint-2), 1./2.*(xint+1)*(xint-1)*(xint-2),
                  -1./2.*(xint+1)*xint*(xint-2), 1./6.*(xint+1)*xint*(xint-1)};
      break;
  }

#if (GEMPIC_SPACEDIM > 1)
  amrex::Real ly = p.pos(1)*dxi[1] - ploF[1];
  int j = std::floor(ly);
  amrex::Real yint = ly - j;

  // ---- interpolation in y-dir
  std::vector<amrex::Real> sy;
  switch (deg[1]) {
      case 1:
          sy = {1.-yint, yint};
      break;
      case 3:
          sy = {-1./6.*yint*(yint-1)*(yint-2), 1./2.*(yint+1)*(yint-1)*(yint-2),
                -1./2.*(yint+1)*yint*(yint-2), 1./6.*(yint+1)*yint*(yint-1)};
      break;
  }

#endif
#if (GEMPIC_SPACEDIM > 2)
  amrex::Real lz = p.pos(2)*dxi[2] - ploF[2];
  int k = std::floor(lz);
  amrex::Real zint = lz - k;

  // ---- interpolation in z-dir
  std::vector<amrex::Real> sz;
  switch (deg[2]) {
      case 1:
          sz = {1.-zint, zint};
      break;
      case 3:
          sz = {-1./6.*zint*(zint-1)*(zint-2), 1./2.*(zint+1)*(zint-1)*(zint-2),
                -1./2.*(zint+1)*zint*(zint-2), 1./6.*(zint+1)*zint*(zint-1)};
      break;
  }

#endif

#if (GEMPIC_SPACEDIM > 2)
    for (int kk = 0-(deg[2]-1)/2; kk <= (deg[2]+1)/2; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0-(deg[1]-1)/2; jj <= (deg[1]+1)/2; ++jj) {
#endif
            for (int ii = 0-(deg[0]-1)/2; ii <= (deg[0]+1)/2; ++ii) {
                double s_coeff = sx[ii]
        #if(GEMPIC_SPACEDIM > 1)
                        * sy[jj]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                        * sz[kk]
        #endif
                        ;
          // p.rdata(cmp): velocity-component cmp, p.rdata(GEMPIC_SPACEDIM): weight
                amrex::Gpu::Atomic::Add(&j_comp(i+ii
                             #if (GEMPIC_SPACEDIM > 1)
                                             , j+jj
                             #endif
                             #if (GEMPIC_SPACEDIM > 2)
                                             , k+kk
                             #endif
                                             , 0), coefficient*s_coeff*charge*p.rdata(cmp)*p.rdata(GEMPIC_SPACEDIM));

            }
#if (GEMPIC_SPACEDIM > 1)
        }
#endif
#if (GEMPIC_SPACEDIM > 2)
    }
#endif
}

void gempic_deposit_J_cic_global (std::array<amrex::Real,2*GEMPIC_SPACEDIM> partInfo, amrex::Real part_weight, amrex::Real charge, int cmp,
                                  amrex::Array4<amrex::Real> const& j_comp,
               std::array<amrex::Real,GEMPIC_SPACEDIM> ploF,
               amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> const& dxi,
               amrex::IntVect deg = amrex::IntVect{AMREX_D_DECL(1,1,1)})
{
  amrex::Real lx = partInfo[0]*dxi[0] - ploF[0];
  int i = std::floor(lx);
  amrex::Real xint = lx - i;

  // ---- interpolation in x-dir
  std::vector<amrex::Real> sx;
  switch (deg[0]) {
      case 1:
          sx = {1.-xint, xint};
      break;
      case 3:
          sx = {-1./6.*xint*(xint-1)*(xint-2), 1./2.*(xint+1)*(xint-1)*(xint-2),
                  -1./2.*(xint+1)*xint*(xint-2), 1./6.*(xint+1)*xint*(xint-1)};
      break;
  }

#if (GEMPIC_SPACEDIM > 1)
  amrex::Real ly = partInfo[1]*dxi[1] - ploF[1];
  int j = std::floor(ly);
  amrex::Real yint = ly - j;

  // ---- interpolation in y-dir
  std::vector<amrex::Real> sy;
  switch (deg[1]) {
      case 1:
          sy = {1.-yint, yint};
      break;
      case 3:
          sy = {-1./6.*yint*(yint-1)*(yint-2), 1./2.*(yint+1)*(yint-1)*(yint-2),
                -1./2.*(yint+1)*yint*(yint-2), 1./6.*(yint+1)*yint*(yint-1)};
      break;
  }

#endif
#if (GEMPIC_SPACEDIM > 2)
  amrex::Real lz = partInfo[2]*dxi[2] - ploF[2];
  int k = std::floor(lz);
  amrex::Real zint = lz - k;

  // ---- interpolation in z-dir
  std::vector<amrex::Real> sz;
  switch (deg[2]) {
      case 1:
          sz = {1.-zint, zint};
      break;
      case 3:
          sz = {-1./6.*zint*(zint-1)*(zint-2), 1./2.*(zint+1)*(zint-1)*(zint-2),
                -1./2.*(zint+1)*zint*(zint-2), 1./6.*(zint+1)*zint*(zint-1)};
      break;
  }

#endif

    for (int kk = 0-(deg[2]-1)/2; kk <= (deg[2]+1)/2; ++kk) {
        for (int jj = 0-(deg[1]-1)/2; jj <= (deg[1]+1)/2; ++jj) {
            for (int ii = 0-(deg[0]-1)/2; ii <= (deg[0]+1)/2; ++ii) {
                double s_coeff = sx[ii]
        #if(GEMPIC_SPACEDIM > 1)
                        * sy[jj]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                        * sz[kk]
        #endif
                        ;
                amrex::Gpu::Atomic::Add(&j_comp(i+ii
                             #if (GEMPIC_SPACEDIM > 1)
                                             , j+jj
                             #endif
                             #if (GEMPIC_SPACEDIM > 2)
                                             , k+kk
                             #endif
                                             , 0), s_coeff*charge*partInfo[cmp+GEMPIC_SPACEDIM]*part_weight);
            }
        }
    }
}

template <typename P>
amrex::Real gempic_interpolate_cic (P const& p, amrex::Array4<amrex::Real const> const& acc,
                                    std::array<amrex::Real,GEMPIC_SPACEDIM> ploF,
                            amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> const& dxi,
                    amrex::IntVect deg = amrex::IntVect{AMREX_D_DECL(1,1,1)})
{
    amrex::Real acceleration = 0.0;

    amrex::Real lx = p.pos(0)*dxi[0] - ploF[0];
    int i = std::floor(lx);
    amrex::Real xint = lx - i;

    // ---- interpolation in x-dir
    std::vector<amrex::Real> sx;
    switch (deg[0]) {
        case 1:
            sx = {1.-xint, xint};
        break;
        case 3:
            sx = {-1./6.*xint*(xint-1)*(xint-2), 1./2.*(xint+1)*(xint-1)*(xint-2),
                    -1./2.*(xint+1)*xint*(xint-2), 1./6.*(xint+1)*xint*(xint-1)};
        break;
    }

#if (GEMPIC_SPACEDIM > 1)
    amrex::Real ly = p.pos(1)*dxi[1] - ploF[1];
    int j = std::floor(ly);
    amrex::Real yint = ly - j;

    // ---- interpolation in y-dir
    std::vector<amrex::Real> sy;
    switch (deg[1]) {
        case 1:
            sy = {1.-yint, yint};
        break;
        case 3:
            sy = {-1./6.*yint*(yint-1)*(yint-2), 1./2.*(yint+1)*(yint-1)*(yint-2),
                  -1./2.*(yint+1)*yint*(yint-2), 1./6.*(yint+1)*yint*(yint-1)};
        break;
    }

#endif
#if (GEMPIC_SPACEDIM > 2)
    amrex::Real lz = p.pos(2)*dxi[2] - ploF[2];
    int k = std::floor(lz);
    amrex::Real zint = lz - k;

    // ---- interpolation in z-dir
    std::vector<amrex::Real> sz;
    switch (deg[2]) {
        case 1:
            sz = {1.-zint, zint};
        break;
        case 3:
            sz = {-1./6.*zint*(zint-1)*(zint-2), 1./2.*(zint+1)*(zint-1)*(zint-2),
                  -1./2.*(zint+1)*zint*(zint-2), 1./6.*(zint+1)*zint*(zint-1)};
        break;
    }

#endif

#if (GEMPIC_SPACEDIM > 2)
    for (int kk = 0-(deg[2]-1)/2; kk <= (deg[2]+1)/2; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0-(deg[1]-1)/2; jj <= (deg[1]+1)/2; ++jj) {
#endif
            for (int ii = 0-(deg[0]-1)/2; ii <= (deg[0]+1)/2; ++ii) {
                double s_coeff = sx[ii]
        #if(GEMPIC_SPACEDIM > 1)
                        * sy[jj]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                        * sz[kk]
        #endif
                        ;
                    acceleration += s_coeff*acc(i+ii
                            #if(GEMPIC_SPACEDIM > 1)
                                                ,j+jj
                            #endif
                            #if(GEMPIC_SPACEDIM > 2)
                                                ,k+kk
                            #endif
                                                ,0);
                }
#if (GEMPIC_SPACEDIM > 1)
            }
#endif
#if (GEMPIC_SPACEDIM > 2)
        }
#endif
    
 return(acceleration);
}

template <typename P>
std::array<amrex::Real,2*GEMPIC_SPACEDIM> push_particle(P const& p,
					amrex::Real dt,
					amrex::Real chargemass, //charge/mass
                    std::array<amrex::Real,GEMPIC_SPACEDIM> Ep,
                    std::array<amrex::Real,GEMPIC_SPACEDIM> Bp,
					infrastructure infra){

  std::array<amrex::Real,2*GEMPIC_SPACEDIM> newPos;
  // new positions (modulo the upper bound of the box)
  newPos[0] = fmod(p.pos(0) + p.rdata(0)*dt, infra.geom.ProbHi()[0]);
#if (GEMPIC_SPACEDIM > 1)
  newPos[1] = fmod(p.pos(1) + p.rdata(1)*dt, infra.geom.ProbHi()[1]);
#endif
#if (GEMPIC_SPACEDIM > 2)
  newPos[2] = fmod(p.pos(2) + p.rdata(2)*dt, infra.geom.ProbHi()[2]);
#endif

#if (GEMPIC_SPACEDIM == 1)
  newPos[1] = p.rdata(0) + dt*chargemass*(Ep[0]);
#elif (GEMPIC_SPACEDIM == 2)
#elif (GEMPIC_SPACEDIM == 3)
  newPos[3] = p.rdata(0) + dt*chargemass*(Ep[0]+(p.rdata(1)*Bp[2]-p.rdata(2)*Bp[1]));
  newPos[4] = p.rdata(1) + dt*chargemass*(Ep[1]+(p.rdata(2)*Bp[0]-p.rdata(0)*Bp[2]));
  newPos[5] = p.rdata(2) + dt*chargemass*(Ep[2]+(p.rdata(0)*Bp[1]-p.rdata(1)*Bp[0]));
#endif
  return(newPos);
}

template <typename P>
std::array<amrex::Real,2*GEMPIC_SPACEDIM> push_particle_pos(P const& p,
                    amrex::Real dt,
                    infrastructure infra,
                    amrex::Real coefficient){

  std::array<amrex::Real,2*GEMPIC_SPACEDIM> newPos;
  // new positions (modulo the upper bound of the box)
  newPos[0] = fmod(p.pos(0) + p.rdata(0)*coefficient*dt, infra.geom.ProbHi()[0]);
  newPos[GEMPIC_SPACEDIM] = p.rdata(0);
#if (GEMPIC_SPACEDIM > 1)
  newPos[1] = fmod(p.pos(1) + p.rdata(1)*coefficient*dt, infra.geom.ProbHi()[1]);
  newPos[GEMPIC_SPACEDIM+1] = p.rdata(1);
#endif
#if (GEMPIC_SPACEDIM > 2)
  newPos[2] = fmod(p.pos(2) + p.rdata(2)*coefficient*dt, infra.geom.ProbHi()[2]);
  newPos[GEMPIC_SPACEDIM+2] = p.rdata(2);
#endif
  return(newPos);
}

template <typename P>
std::array<amrex::Real,2*GEMPIC_SPACEDIM> push_particle_boris(P const& p,
                    amrex::Real dt,
                    amrex::Real chargemass, //charge/mass
                    std::array<amrex::Real,GEMPIC_SPACEDIM> Ep,
                    std::array<amrex::Real,GEMPIC_SPACEDIM> Bp,
                    infrastructure infra){

  std::array<amrex::Real,2*GEMPIC_SPACEDIM> newPos;
  std::array<amrex::Real,GEMPIC_SPACEDIM> vm;
  std::array<amrex::Real,GEMPIC_SPACEDIM> vt;
  std::array<amrex::Real,GEMPIC_SPACEDIM> vp;
  amrex::Real S;

  // v-
  vm[0] = p.rdata(0) + dt*chargemass/2*Ep[0];
#if (GEMPIC_SPACEDIM > 1)
  vm[1] = p.rdata(1) + dt*chargemass/2*Ep[1];
#endif
#if (GEMPIC_SPACEDIM > 2)
  vm[2] = p.rdata(2) + dt*chargemass/2*Ep[2];
#endif

#if (GEMPIC_SPACEDIM == 1)
  vp[0] = vm[0];
#elif (GEMPIC_SPACEDIM == 2)

#elif (GEMPIC_SPACEDIM == 3)
  // v'
  amrex::Real coeff_t = dt*chargemass/2;
  vt[0] = vm[0] + coeff_t*(vm[1]*Bp[2]-vm[2]*Bp[1]);
  vt[1] = vm[1] + coeff_t*(vm[2]*Bp[0]-vm[0]*Bp[2]);
  vt[2] = vm[2] + coeff_t*(vm[0]*Bp[1]-vm[1]*Bp[0]);

  S = pow(dt*chargemass/2*Bp[0], 2.)+pow(dt*chargemass/2*Bp[1], 2.)+pow(dt*chargemass/2*Bp[2], 2.);

  // v+
  amrex::Real coeff = dt*chargemass/(1.+S);
  vp[0] = vm[0] + coeff*(vt[1]*Bp[2]-vt[2]*Bp[1]);
  vp[1] = vm[1] + coeff*(vt[2]*Bp[0]-vt[0]*Bp[2]);
  vp[2] = vm[2] + coeff*(vt[0]*Bp[1]-vt[1]*Bp[0]);
#endif

  // v
  newPos[GEMPIC_SPACEDIM] = vp[0] + dt*chargemass/2*Ep[0];
#if (GEMPIC_SPACEDIM > 1)
  newPos[GEMPIC_SPACEDIM+1] = vp[1] + dt*chargemass/2*Ep[1];
#endif
#if (GEMPIC_SPACEDIM > 2)
  newPos[GEMPIC_SPACEDIM+2] = vp[2] + dt*chargemass/2*Ep[2];
#endif

  // new positions (modulo the upper bound of the box)
  newPos[0] = fmod(p.pos(0) + newPos[GEMPIC_SPACEDIM]*dt, infra.geom.ProbHi()[0]);
#if (GEMPIC_SPACEDIM > 1)
  newPos[1] = fmod(p.pos(1) + newPos[GEMPIC_SPACEDIM+1]*dt, infra.geom.ProbHi()[1]);
#endif
#if (GEMPIC_SPACEDIM > 2)
  newPos[2] = fmod(p.pos(2) + newPos[GEMPIC_SPACEDIM+2]*dt, infra.geom.ProbHi()[2]);
#endif



  return(newPos);
}

#endif
