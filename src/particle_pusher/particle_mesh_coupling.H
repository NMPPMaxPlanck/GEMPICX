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
#include <splines.H>

template <typename P>
void gempic_deposit_rho (P const& p, amrex::Real charge,
                         amrex::Array4<amrex::Real> const& rho,
                         std::array<amrex::Real,GEMPIC_SPACEDIM> const& plo,
                         std::array<amrex::Real,GEMPIC_SPACEDIM+1> const& dxi)

{
    splines spl({AMREX_D_DECL(false, false, false)}, {AMREX_D_DECL(false, false, false)});
    amrex::Real lx = (p.pos(0) - plo[0]) * dxi[0]; // don't need to displace because rho is nodal
    int i = std::floor(lx);
    amrex::Real xint = lx - i;
    std::vector<amrex::Real> sx_node = spl.spline_eval_vec (spl.shifted_x(xint, 0, false), GEMPIC_DEG_X);
#if (GEMPIC_SPACEDIM > 1)
    amrex::Real ly = (p.pos(1) - plo[1]) * dxi[1];
    int j = std::floor(ly);
    amrex::Real yint = ly - j;
    std::vector<amrex::Real> sy_node = spl.spline_eval_vec (spl.shifted_x(yint, 1, false), GEMPIC_DEG_Y);
#endif
#if (GEMPIC_SPACEDIM > 2)
    amrex::Real lz = (p.pos(2) - plo[2]) * dxi[2];
    int k = std::floor(lz);
    amrex::Real zint = lz - k;
    std::vector<amrex::Real> sz_node = spl.spline_eval_vec (spl.shifted_x(zint, 2, false), GEMPIC_DEG_Z);
#endif
    amrex::Real particle_coeff = dxi[GEMPIC_SPACEDIM]*charge*p.rdata(GEMPIC_VDIM);

    //TODOrefine
#if (GEMPIC_SPACEDIM > 2)
    for (int kk = 0; kk <= GEMPIC_DEG_Z; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0; jj <= GEMPIC_DEG_Y; ++jj) {
#endif
            for (int ii = 0; ii <= GEMPIC_DEG_X; ++ii) {
                double s_coeff = sx_node[ii]
                                   #if(GEMPIC_SPACEDIM > 1)
                                                       *sy_node[jj]
                                               #endif
                                               #if(GEMPIC_SPACEDIM > 2)
                                                                   *sz_node[kk]
                                                           #endif
                                                                               ;
                std::array<int,GEMPIC_VDIM> cell_index = {AMREX_D_DECL(spl.compute_cell_index(i+ii, 0, xint, false),
                                                          spl.compute_cell_index(j+jj, 1, yint, false),
                                                          spl.compute_cell_index(k+kk, 2, zint, false))};

                amrex::Gpu::Atomic::Add(&rho(cell_index[0]
                             #if (GEMPIC_SPACEDIM > 1)
                                             , cell_index[1]
                             #endif
                             #if (GEMPIC_SPACEDIM > 2)
                                             , cell_index[2]
                             #endif
                                             , 0), s_coeff*particle_coeff);
            }
#if (GEMPIC_SPACEDIM > 1)
        }
#endif
#if (GEMPIC_SPACEDIM > 2)
    }
#endif
}

template <typename P>
std::array<amrex::Real, GEMPIC_VDIM+GEMPIC_BDIM> gempic_deposit_J_and_interpolate (P const& p, amrex::Real charge,
                                                                                   std::array<amrex::Real,GEMPIC_SPACEDIM+1> const& dxi,
                                                                                   std::array<amrex::Real,GEMPIC_SPACEDIM> const& plo,
                                                                                   std::array<amrex::Array4<amrex::Real>,GEMPIC_VDIM> jArray,
                                                                                   std::array<amrex::Array4<amrex::Real>,GEMPIC_VDIM> eArray,
                                                                                   std::array<amrex::Array4<amrex::Real>,GEMPIC_BDIM> bArray,
                                                                                   amrex::Real coefficient = 1.,
                                                                                   bool field_int = false)

{
    std::array<amrex::Real, GEMPIC_VDIM+GEMPIC_BDIM> fields = {AMREX_D_DECL(0.,0.,0.)};

    splines spl({AMREX_D_DECL(false, false, false)}, {AMREX_D_DECL(false, false, false)});
    amrex::Real lx = (p.pos(0)-plo[0])*dxi[0];
    int i = std::floor(lx);
    amrex::Real xint = lx - i;
    std::vector<amrex::Real> sx_cell = spl.spline_eval_vec (spl.shifted_x(xint, 0, true), GEMPIC_DEG_X);
    std::vector<amrex::Real> sx_node = spl.spline_eval_vec (spl.shifted_x(xint, 0, false), GEMPIC_DEG_X);
    std::array<std::vector<amrex::Real>,2> sx = {sx_cell, sx_node};
#if (GEMPIC_SPACEDIM > 1)
    amrex::Real ly = (p.pos(1)-plo[1])*dxi[1];
    int j = std::floor(ly);
    amrex::Real yint = ly - j;
    std::vector<amrex::Real> sy_cell = spl.spline_eval_vec (spl.shifted_x(yint, 1, true), GEMPIC_DEG_Y);
    std::vector<amrex::Real> sy_node = spl.spline_eval_vec (spl.shifted_x(yint, 1, false), GEMPIC_DEG_Y);
    std::array<std::vector<amrex::Real>,2> sy = {sy_cell, sy_node};
#endif
#if (GEMPIC_SPACEDIM > 2)
    amrex::Real lz = (p.pos(2)-plo[2])*dxi[2];
    int k = std::floor(lz);
    amrex::Real zint = lz - k;
    std::vector<amrex::Real> sz_cell = spl.spline_eval_vec (spl.shifted_x(zint, 2, true), GEMPIC_DEG_Z);
    std::vector<amrex::Real> sz_node = spl.spline_eval_vec (spl.shifted_x(zint, 2, false), GEMPIC_DEG_Z);
    std::array<std::vector<amrex::Real>,2> sz = {sz_cell, sz_node};
#endif
    amrex::Real particle_coeff = dxi[GEMPIC_SPACEDIM]*charge*coefficient*p.rdata(GEMPIC_VDIM);

#if (GEMPIC_SPACEDIM > 2)
    for (int kk = 0; kk <= GEMPIC_DEG_Z; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0; jj <= GEMPIC_DEG_Y; ++jj) {
#endif
            for (int ii = 0; ii <= GEMPIC_DEG_X; ++ii) {

                // deposit in J and interpolate in E
                for (int cc=0;cc<GEMPIC_VDIM;cc++) {
                    double s_coeff = sx[cc%3==0?0:1][ii]
                            #if(GEMPIC_SPACEDIM > 1)
                                                *sy[cc%3==1?0:1][jj]
                                        #endif
                                        #if(GEMPIC_SPACEDIM > 2)
                                                            *sz[cc%3==2?0:1][kk]
                                                    #endif
                                                                        ;
                    std::array<int,GEMPIC_VDIM> cell_index = {AMREX_D_DECL(spl.compute_cell_index(i+ii, 0, xint, cc%3==0?true:false),
                                                              spl.compute_cell_index(j+jj, 1, yint, cc%3==1?true:false),
                                                              spl.compute_cell_index(k+kk, 2, zint, cc%3==2?true:false))};

                    amrex::Gpu::Atomic::Add(&(jArray[cc])(cell_index[0]
                        #if (GEMPIC_SPACEDIM > 1)
                                            , cell_index[1]
        #endif
        #if (GEMPIC_SPACEDIM > 2)
                            , cell_index[2]
        #endif
                            , 0), s_coeff*particle_coeff*p.rdata(cc));

                    if (field_int) {
                        fields[cc] += s_coeff*(eArray[cc])(cell_index[0]
            #if(GEMPIC_SPACEDIM > 1)
                                ,cell_index[1]
            #endif
            #if(GEMPIC_SPACEDIM > 2)
                                ,cell_index[2]
            #endif
                                ,0);
                    }
                }

                if (field_int) {
                    // deposit in B
                    for (int cc=0;cc<GEMPIC_VDIM;cc++) {
                        double s_coeff = sx[cc%3==0?1:0][ii]
                                #if(GEMPIC_SPACEDIM > 1)
                                                    *sy[cc%3==1?1:0][jj]
                                            #endif
                                            #if(GEMPIC_SPACEDIM > 2)
                                                                *sz[cc%3==2?1:0][kk]
                                                        #endif
                                                                            ;
                        std::array<int,GEMPIC_VDIM> cell_index = {AMREX_D_DECL(spl.compute_cell_index(i+ii, 0, xint, cc%3==0?false:true),
                                                                  spl.compute_cell_index(j+jj, 1, yint, cc%3==1?false:true),
                                                                  spl.compute_cell_index(k+kk, 2, zint, cc%3==2?false:true))};

                        fields[cc+GEMPIC_VDIM] += s_coeff*(bArray[cc])(cell_index[0]
            #if(GEMPIC_SPACEDIM > 1)
                                ,cell_index[1]
            #endif
            #if(GEMPIC_SPACEDIM > 2)
                                ,cell_index[2]
            #endif
                                ,0);
                    }
                }

            }
#if (GEMPIC_SPACEDIM > 1)
        }
#endif
#if (GEMPIC_SPACEDIM > 2)
    }
#endif
    return(fields);
}

template <typename P>
std::array<amrex::Real,GEMPIC_SPACEDIM+GEMPIC_VDIM> push_particle(P const& p,
                                                                  amrex::Real dt,
                                                                  amrex::Real chargemass, //charge/mass
                                                                  std::array<amrex::Real,GEMPIC_VDIM> Ep,
                                                                  std::array<amrex::Real,GEMPIC_BDIM> Bp,
                                                                  infrastructure infra){

    std::array<amrex::Real,GEMPIC_SPACEDIM+GEMPIC_VDIM> newPos;
    // new positions (modulo the upper bound of the box)
    newPos[0] = fmod(p.pos(0) + p.rdata(0)*dt, infra.geom.ProbHi()[0]);
#if (GEMPIC_SPACEDIM > 1)
    newPos[1] = fmod(p.pos(1) + p.rdata(1)*dt, infra.geom.ProbHi()[1]);
#endif
#if (GEMPIC_SPACEDIM > 2)
    newPos[2] = fmod(p.pos(2) + p.rdata(2)*dt, infra.geom.ProbHi()[2]);
#endif

#if (GEMPIC_VDIM == 1)
    newPos[1] = p.rdata(0) + dt*chargemass*(Ep[0]);
#elif (GEMPIC_VDIM == 2)
    newPos[2] = p.rdata(0) + dt*chargemass*(Ep[0]+p.rdata(1)*Bp[0]);
    newPos[3] = p.rdata(1) + dt*chargemass*(Ep[1]+p.rdata(0)*Bp[0]);
#elif (GEMPIC_VDIM == 3)
    newPos[3] = p.rdata(0) + dt*chargemass*(Ep[0]+(p.rdata(1)*Bp[2]-p.rdata(2)*Bp[1]));
    newPos[4] = p.rdata(1) + dt*chargemass*(Ep[1]+(p.rdata(2)*Bp[0]-p.rdata(0)*Bp[2]));
    newPos[5] = p.rdata(2) + dt*chargemass*(Ep[2]+(p.rdata(0)*Bp[1]-p.rdata(1)*Bp[0]));
#endif
    return(newPos);
}

template <typename P>
std::array<amrex::Real,GEMPIC_SPACEDIM+GEMPIC_VDIM> push_particle_pos(P const& p,
                                                                      amrex::Real dt,
                                                                      infrastructure infra){

    std::array<amrex::Real,GEMPIC_SPACEDIM+GEMPIC_VDIM> newPos;
    // new positions (modulo the upper bound of the box)
    newPos[0] = fmod(p.pos(0) + p.rdata(0)*dt, infra.geom.ProbHi()[0]);
    newPos[GEMPIC_SPACEDIM] = p.rdata(0);
#if (GEMPIC_SPACEDIM > 1)
    newPos[1] = fmod(p.pos(1) + p.rdata(1)*dt, infra.geom.ProbHi()[1]);
#endif
#if (GEMPIC_SPACEDIM > 2)
    newPos[2] = fmod(p.pos(2) + p.rdata(2)*dt, infra.geom.ProbHi()[2]);
#endif
#if (GEMPIC_VDIM > 1)
    newPos[GEMPIC_SPACEDIM+1] = p.rdata(1);
#endif
#if (GEMPIC_VDIM > 2)
    newPos[GEMPIC_SPACEDIM+2] = p.rdata(2);
#endif
    return(newPos);
}

template <typename P>
std::array<amrex::Real,GEMPIC_SPACEDIM+GEMPIC_VDIM> push_particle_boris(P const& p,
                                                                        amrex::Real dt,
                                                                        amrex::Real chargemass, //charge/mass
                                                                        std::array<amrex::Real,GEMPIC_VDIM> Ep,
                                                                        std::array<amrex::Real,GEMPIC_BDIM> Bp,
                                                                        infrastructure infra){

    std::array<amrex::Real,GEMPIC_SPACEDIM+GEMPIC_VDIM> newPos;
    std::array<amrex::Real,GEMPIC_VDIM> vm;
    std::array<amrex::Real,GEMPIC_VDIM> vt;
    std::array<amrex::Real,GEMPIC_VDIM> vp;
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
    amrex::Real alpha = chargemass*dt/2*Bp[0];
    vp[0] = (1-pow(alpha,2))/(1+pow(alpha,2))*vm[0] + 2*alpha/(1+pow(alpha,2))*vm[1];
    vp[1] = -2*alpha/(1+pow(alpha,2))*vm[0] + (1-pow(alpha,2))/(1+pow(alpha,2))*vm[1];
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
