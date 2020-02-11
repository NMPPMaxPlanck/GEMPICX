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
amrex::Real disp = -0.5;

template <typename P>
void gempic_deposit_cic (P const& p, amrex::Real charge,
                         amrex::Array4<amrex::Real> const& rho,
                         std::array<amrex::Real,GEMPIC_SPACEDIM> const& plo,
                         std::array<amrex::Real,GEMPIC_SPACEDIM> const& dxi,
                         amrex::IntVect deg = amrex::IntVect{AMREX_D_DECL(3,3,3)})

{
    splines spl(deg, {AMREX_D_DECL(false, false, false)}, {AMREX_D_DECL(false, false, false)});
    amrex::Real lx = (p.pos(0) - plo[0]) * dxi[0]; // don't need to displace because rho is nodal
    int i = std::floor(lx);
    amrex::Real xint = lx - i;

#if (GEMPIC_SPACEDIM > 1)
    amrex::Real ly = (p.pos(1) - plo[1]) * dxi[1];
    int j = std::floor(ly);
    amrex::Real yint = ly - j;
#endif
#if (GEMPIC_SPACEDIM > 2)
    amrex::Real lz = (p.pos(2) - plo[2]) * dxi[2];
    int k = std::floor(lz);
    amrex::Real zint = lz - k;
#endif

    //TODOrefine
#if (GEMPIC_SPACEDIM > 2)
    for (int kk = 0; kk <= deg[2]; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0; jj <= deg[1]; ++jj) {
#endif
            for (int ii = 0; ii <= deg[0]; ++ii) {                
                double s_coeff = spl.deposition_coefficient ({AMREX_D_DECL(xint, yint, zint)},{AMREX_D_DECL(ii, jj, kk)});
                s_coeff *= dxi[0]
        #if(GEMPIC_SPACEDIM > 1)
                        *dxi[1]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                        *dxi[2]
        #endif
                        ;

                amrex::Gpu::Atomic::Add(&rho(spl.compute_cell_index(i+ii, 0, xint)
                             #if (GEMPIC_SPACEDIM > 1)
                                             , spl.compute_cell_index(j+jj, 0, yint)
                             #endif
                             #if (GEMPIC_SPACEDIM > 2)
                                             , spl.compute_cell_index(k+kk, 0, zint)
                             #endif
                                             , 0), s_coeff*charge*p.rdata(GEMPIC_VDIM));

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
                           std::array<amrex::Real,GEMPIC_SPACEDIM> const& dxi,
                           amrex::Real coefficient = 1.0,
                           amrex::IntVect deg = amrex::IntVect{AMREX_D_DECL(3,3,3)})
{
    amrex::Real lx = p.pos(0)*dxi[0] - ploF[0];
    int i = std::floor(lx);
    amrex::Real xint = lx - i;
    std::array<int,GEMPIC_SPACEDIM> low_idx, high_idx;

    // ---- interpolation in x-dir
    std::vector<amrex::Real> sx;
    switch (deg[0]) {
    case 1:
        sx = {1.-xint, xint};
        low_idx[0] = 0;
        high_idx[0] = 0;
        break;
    case 2:
        if (xint < 1./2.) {
            sx = {pow(1./2.-xint,2)/2., 1./2.+(1./2.-xint)-pow(1./2.-xint,2), 1./2.-(1./2.-xint)+pow(1./2.-xint,2)/2};
            low_idx[0] = -1;
            high_idx[0] = 1;
        } else {
            sx = {pow(3./2.-xint,2)/2., 1./2.+(3./2.-xint)-pow(3./2.-xint,2), 1./2.-(3./2.-xint)+pow(3./2.-xint,2)/2};
            low_idx[0] = 0;
            high_idx[0] = 2;
        }
        break;
    case 3:
        sx = {1./6.*pow((1.-xint),3), 2./3.-pow(xint,2.)+pow(xint,3)/2,
              2./3.-pow(xint-1,2.)-pow(xint-1,3)/2, 1./6.*pow(xint,3)};
        low_idx[0] = -1;
        high_idx[0] = 2;
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
        low_idx[1] = 0;
        high_idx[1] = 0;
        break;
    case 2:
        if (yint < 1./2.) {
            sy = {pow(1./2.-yint,2)/2., 1./2.+(1./2.-yint)-pow(1./2.-yint,2), 1./2.-(1./2.-yint)+pow(1./2.-yint,2)/2};
            low_idx[1] = -1;
            high_idx[1] = 1;
        } else {
            sy = {pow(3./2.-yint,2)/2., 1./2.+(3./2.-yint)-pow(3./2.-yint,2), 1./2.-(3./2.-yint)+pow(3./2.-yint,2)/2};
            low_idx[1] = 0;
            high_idx[1] = 2;
        }
        break;
    case 3:
        sy = {1./6.*pow((1.-yint),3), 2./3.-pow(yint,2.)+pow(yint,3)/2,
              2./3.-pow(yint-1,2.)-pow(yint-1,3)/2, 1./6.*pow(yint,3)};
        low_idx[1] = -1;
        high_idx[1] = 2;
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
        low_idx[2] = 0;
        high_idx[2] = 0;
        break;
    case 2:
        if (zint < 1./2.) {
            sz = {pow(1./2.-zint,2)/2., 1./2.+(1./2.-zint)-pow(1./2.-zint,2), 1./2.-(1./2.-zint)+pow(1./2.-zint,2)/2};
            low_idx[2] = -1;
            high_idx[2] = 1;
        } else {
            sz = {pow(3./2.-zint,2)/2., 1./2.+(3./2.-zint)-pow(3./2.-zint,2), 1./2.-(3./2.-zint)+pow(3./2.-zint,2)/2};
            low_idx[2] = 0;
            high_idx[2] = 2;
        }
        break;
    case 3:
        sz = {1./6.*pow((1.-zint),3), 2./3.-pow(zint,2.)+pow(zint,3)/2,
              2./3.-pow(zint-1,2.)-pow(zint-1,3)/2, 1./6.*pow(zint,3)};
        low_idx[2] = -1;
        high_idx[2] = 2;
        break;
    }

#endif

#if (GEMPIC_SPACEDIM > 2)
    for (int kk = low_idx[2]; kk <= high_idx[2]; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = low_idx[1]; jj <= high_idx[1]; ++jj) {
#endif
            for (int ii = low_idx[0]; ii <= high_idx[0]; ++ii) {
                double s_coeff = sx[ii-low_idx[0]]*dxi[0]
        #if(GEMPIC_SPACEDIM > 1)
                        * sy[jj-low_idx[1]]*dxi[1]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                        * sz[kk-low_idx[2]]*dxi[2]
        #endif
                        ;

                // p.rdata(cmp): velocity-component cmp, p.rdata(GEMPIC_VDIM): weight
                amrex::Gpu::Atomic::Add(&j_comp(i+ii
                                #if (GEMPIC_SPACEDIM > 1)
                                                , j+jj
                                #endif
                                #if (GEMPIC_SPACEDIM > 2)
                                                , k+kk
                                #endif
                                                , 0), coefficient*s_coeff*charge*p.rdata(cmp)*p.rdata(GEMPIC_VDIM));

            }
#if (GEMPIC_SPACEDIM > 1)
        }
#endif
#if (GEMPIC_SPACEDIM > 2)
    }
#endif
}

template <typename P>
void gempic_deposit_J_cic_class (P const& p, amrex::Real charge, int cmp,
                                 amrex::Array4<amrex::Real> const& j_comp,
                                 std::array<amrex::Real,GEMPIC_SPACEDIM> ploF,
                                 std::array<amrex::Real,GEMPIC_SPACEDIM> const& dxi,
                                 amrex::IndexType indType,
                                 amrex::Real coefficient = 1.0,
                                 amrex::IntVect deg = amrex::IntVect{AMREX_D_DECL(2,2,2)})

{
    splines spl(deg, {AMREX_D_DECL(indType[0] == 0, indType[1] == 0, indType[2] == 0)}, {AMREX_D_DECL(false, false, false)});
    amrex::Real lx = p.pos(0)*dxi[0] - ploF[0]; // don't need to displace because rho is nodal
    int i = std::floor(lx);
    amrex::Real xint = lx - i;
#if (GEMPIC_SPACEDIM > 1)
    amrex::Real ly = p.pos(1)*dxi[1] - ploF[1];
    int j = std::floor(ly);
    amrex::Real yint = ly - j;
#endif
#if (GEMPIC_SPACEDIM > 2)
    amrex::Real lz = p.pos(2)*dxi[2] - ploF[2];
    int k = std::floor(lz);
    amrex::Real zint = lz - k;
#endif

#if (GEMPIC_SPACEDIM > 2)
    for (int kk = 0; kk <= deg[2]; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0; jj <= deg[1]; ++jj) {
#endif
            for (int ii = 0; ii <= deg[0]; ++ii) {
                double s_coeff = spl.deposition_coefficient ({AMREX_D_DECL(xint, yint, zint)},{AMREX_D_DECL(ii, jj, kk)});
                s_coeff *= dxi[0]
        #if(GEMPIC_SPACEDIM > 1)
                        *dxi[1]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                        *dxi[2]
        #endif
                        ;

                amrex::Gpu::Atomic::Add(&j_comp(spl.compute_cell_index(i+ii, 0, xint)
                                #if (GEMPIC_SPACEDIM > 1)
                                                , spl.compute_cell_index(j+jj, 0, yint)
                                #endif
                                #if (GEMPIC_SPACEDIM > 2)
                                                , spl.compute_cell_index(k+kk, 0, zint)
                                #endif
                                                , 0), coefficient*s_coeff*charge*p.rdata(cmp)*p.rdata(GEMPIC_VDIM));

            }
#if (GEMPIC_SPACEDIM > 1)
        }
#endif
#if (GEMPIC_SPACEDIM > 2)
    }
#endif
}

void gempic_deposit_J_cic_global (std::array<amrex::Real,GEMPIC_SPACEDIM+GEMPIC_VDIM> partInfo, amrex::Real part_weight, amrex::Real charge, int cmp,
                                  amrex::Array4<amrex::Real> const& j_comp,
                                  std::array<amrex::Real,GEMPIC_SPACEDIM> ploF,
                                  std::array<amrex::Real,GEMPIC_SPACEDIM> const& dxi,
                                  amrex::IntVect deg = amrex::IntVect{AMREX_D_DECL(1,1,1)})
{
    amrex::Real lx = partInfo[0]*dxi[0] - ploF[0];
    int i = std::floor(lx);
    amrex::Real xint = lx - i;
    std::array<int,GEMPIC_SPACEDIM> low_idx, high_idx;

    // ---- interpolation in x-dir
    std::vector<amrex::Real> sx;
    switch (deg[0]) {
    case 1:
        sx = {1.-xint, xint};
        low_idx[0] = 0;
        high_idx[0] = 0;
        break;
    case 2:
        if (xint < 1./2.) {
            sx = {pow(1./2.-xint,2)/2., 1./2.+(1./2.-xint)-pow(1./2.-xint,2), 1./2.-(1./2.-xint)+pow(1./2.-xint,2)/2};
            low_idx[0] = -1;
            high_idx[0] = 1;
        } else {
            sx = {pow(3./2.-xint,2)/2., 1./2.+(3./2.-xint)-pow(3./2.-xint,2), 1./2.-(3./2.-xint)+pow(3./2.-xint,2)/2};
            low_idx[0] = 0;
            high_idx[0] = 2;
        }
        break;
    case 3:
        sx = {1./6.*pow((1.-xint),3), 2./3.-pow(xint,2.)+pow(xint,3)/2,
              2./3.-pow(xint-1,2.)-pow(xint-1,3)/2, 1./6.*pow(xint,3)};
        low_idx[0] = -1;
        high_idx[0] = 2;
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
        low_idx[1] = 0;
        high_idx[1] = 0;
        break;
    case 2:
        if (yint < 1./2.) {
            sy = {pow(1./2.-yint,2)/2., 1./2.+(1./2.-yint)-pow(1./2.-yint,2), 1./2.-(1./2.-yint)+pow(1./2.-yint,2)/2};
            low_idx[1] = -1;
            high_idx[1] = 1;
        } else {
            sy = {pow(3./2.-yint,2)/2., 1./2.+(3./2.-yint)-pow(3./2.-yint,2), 1./2.-(3./2.-yint)+pow(3./2.-yint,2)/2};
            low_idx[1] = 0;
            high_idx[1] = 2;
        }
        break;
    case 3:
        sy = {1./6.*pow((1.-yint),3), 2./3.-pow(yint,2.)+pow(yint,3)/2,
              2./3.-pow(yint-1,2.)-pow(yint-1,3)/2, 1./6.*pow(yint,3)};
        low_idx[1] = -1;
        high_idx[1] = 2;
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
        low_idx[2] = 0;
        high_idx[2] = 0;
        break;
    case 2:
        if (zint < 1./2.) {
            sz = {pow(1./2.-zint,2)/2., 1./2.+(1./2.-zint)-pow(1./2.-zint,2), 1./2.-(1./2.-zint)+pow(1./2.-zint,2)/2};
            low_idx[2] = -1;
            high_idx[2] = 1;
        } else {
            sx = {pow(3./2.-zint,2)/2., 1./2.+(3./2.-zint)-pow(3./2.-zint,2), 1./2.-(3./2.-zint)+pow(3./2.-zint,2)/2};
            low_idx[2] = 0;
            high_idx[2] = 2;
        }
        break;
    case 3:
        sz = {1./6.*pow((1.-zint),3), 2./3.-pow(zint,2.)+pow(zint,3)/2,
              2./3.-pow(zint-1,2.)-pow(zint-1,3)/2, 1./6.*pow(zint,3)};
        low_idx[2] = -1;
        high_idx[2] = 2;
        break;
    }

#endif

    for (int kk = low_idx[2]; kk <= high_idx[2]; ++kk) {
        for (int jj = low_idx[1]; jj <= high_idx[1]; ++jj) {
            for (int ii = low_idx[0]; ii <= high_idx[0]; ++ii) {
                double s_coeff = sx[ii-low_idx[0]]*dxi[0]
        #if(GEMPIC_SPACEDIM > 1)
                        * sy[jj-low_idx[1]]*dxi[0]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                        * sz[kk-low_idx[2]]*dxi[0]
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
                                    std::array<amrex::Real,GEMPIC_SPACEDIM> const& dxi,
                                    amrex::IntVect deg = amrex::IntVect{AMREX_D_DECL(3,3,3)})
{
    amrex::Real acceleration = 0.0;

    amrex::Real lx = p.pos(0)*dxi[0] - ploF[0];
    int i = std::floor(lx);
    amrex::Real xint = lx - i;
    std::array<int,GEMPIC_SPACEDIM> low_idx, high_idx;

    // ---- interpolation in x-dir
    std::vector<amrex::Real> sx;
    switch (deg[0]) {
    case 1:
        sx = {1.-xint, xint};
        low_idx[0] = 0;
        high_idx[0] = 0;
        break;
    case 2:
        if (xint < 1./2.) {
            sx = {pow(1./2.-xint,2)/2., 1./2.+(1./2.-xint)-pow(1./2.-xint,2), 1./2.-(1./2.-xint)+pow(1./2.-xint,2)/2};
            low_idx[0] = -1;
            high_idx[0] = 1;
        } else {
            sx = {pow(3./2.-xint,2)/2., 1./2.+(3./2.-xint)-pow(3./2.-xint,2), 1./2.-(3./2.-xint)+pow(3./2.-xint,2)/2};
            low_idx[0] = 0;
            high_idx[0] = 2;
        }
        break;
    case 3:
        sx = {1./6.*pow((1.-xint),3), 2./3.-pow(xint,2.)+pow(xint,3)/2,
              2./3.-pow(xint-1,2.)-pow(xint-1,3)/2, 1./6.*pow(xint,3)};
        low_idx[0] = -1;
        high_idx[0] = 2;
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
        low_idx[1] = 0;
        high_idx[1] = 0;
        break;
    case 2:
        if (yint < 1./2.) {
            sy = {pow(1./2.-yint,2)/2., 1./2.+(1./2.-yint)-pow(1./2.-yint,2), 1./2.-(1./2.-yint)+pow(1./2.-yint,2)/2};
            low_idx[1] = -1;
            high_idx[1] = 1;
        } else {
            sy = {pow(3./2.-yint,2)/2., 1./2.+(3./2.-yint)-pow(3./2.-yint,2), 1./2.-(3./2.-yint)+pow(3./2.-yint,2)/2};
            low_idx[1] = 0;
            high_idx[1] = 2;
        }
        break;
    case 3:
        sy = {1./6.*pow((1.-yint),3), 2./3.-pow(yint,2.)+pow(yint,3)/2,
              2./3.-pow(yint-1,2.)-pow(yint-1,3)/2, 1./6.*pow(yint,3)};
        low_idx[1] = -1;
        high_idx[1] = 2;
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
        low_idx[2] = 0;
        high_idx[2] = 0;
        break;
    case 2:
        if (zint < 1./2.) {
            sz = {pow(1./2.-zint,2)/2., 1./2.+(1./2.-zint)-pow(1./2.-zint,2), 1./2.-(1./2.-zint)+pow(1./2.-zint,2)/2};
            low_idx[2] = -1;
            high_idx[2] = 1;
        } else {
            sz = {pow(3./2.-zint,2)/2., 1./2.+(3./2.-zint)-pow(3./2.-zint,2), 1./2.-(3./2.-zint)+pow(3./2.-zint,2)/2};
            low_idx[2] = 0;
            high_idx[2] = 2;
        }
        break;
    case 3:
        sz = {1./6.*pow((1.-zint),3), 2./3.-pow(zint,2.)+pow(zint,3)/2,
              2./3.-pow(zint-1,2.)-pow(zint-1,3)/2, 1./6.*pow(zint,3)};
        low_idx[2] = -1;
        high_idx[2] = 2;
        break;
    }

#endif

#if (GEMPIC_SPACEDIM > 2)
    for (int kk = low_idx[2]; kk <= high_idx[2]; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = low_idx[1]; jj <= high_idx[1]; ++jj) {
#endif
            for (int ii = low_idx[0]; ii <= high_idx[0]; ++ii) {
                double s_coeff = sx[ii-low_idx[0]]
        #if(GEMPIC_SPACEDIM > 1)
                        * sy[jj-low_idx[1]]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                        * sz[kk-low_idx[2]]
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
amrex::Real gempic_interpolate_cic_class (P const& p, amrex::Array4<amrex::Real const> const& acc,
                                          std::array<amrex::Real,GEMPIC_SPACEDIM> ploF,
                                          std::array<amrex::Real,GEMPIC_SPACEDIM> const& dxi,
                                          amrex::IndexType indType,
                                          amrex::IntVect deg = amrex::IntVect{AMREX_D_DECL(3,3,3)})
{
    splines spl(deg, {AMREX_D_DECL(indType[0] == 0, indType[1] == 0, indType[2] == 0)}, {AMREX_D_DECL(false, false, false)});
    amrex::Real acceleration = 0.0;

    amrex::Real lx = p.pos(0)*dxi[0] - ploF[0];
    int i = std::floor(lx);
    amrex::Real xint = lx - i;
#if (GEMPIC_SPACEDIM > 1)
    amrex::Real ly = p.pos(1)*dxi[1] - ploF[1];
    int j = std::floor(ly);
    amrex::Real yint = ly - j;
#endif
#if (GEMPIC_SPACEDIM > 2)
    amrex::Real lz = p.pos(2)*dxi[2] - ploF[2];
    int k = std::floor(lz);
    amrex::Real zint = lz - k;
#endif

#if (GEMPIC_SPACEDIM > 2)
    for (int kk = 0; kk <= deg[2]; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0; jj <= deg[1]; ++jj) {
#endif
            for (int ii = 0; ii <= deg[0]; ++ii) {
                double s_coeff = spl.deposition_coefficient ({AMREX_D_DECL(xint, yint, zint)},{AMREX_D_DECL(ii, jj, kk)});
                s_coeff *= dxi[0]
        #if(GEMPIC_SPACEDIM > 1)
                        * dxi[1]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                        * dxi[2]
        #endif
                        ;
                acceleration += s_coeff*acc(spl.compute_cell_index(i+ii, 0, xint)
                            #if(GEMPIC_SPACEDIM > 1)
                                            ,spl.compute_cell_index(j+jj, 0, yint)
                            #endif
                            #if(GEMPIC_SPACEDIM > 2)
                                            ,spl.compute_cell_index(k+kk, 0, zint)
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
                                                                      infrastructure infra,
                                                                      amrex::Real coefficient){

    std::array<amrex::Real,GEMPIC_SPACEDIM+GEMPIC_VDIM> newPos;
    // new positions (modulo the upper bound of the box)
    newPos[0] = fmod(p.pos(0) + p.rdata(0)*coefficient*dt, infra.geom.ProbHi()[0]);
    newPos[GEMPIC_SPACEDIM] = p.rdata(0);
#if (GEMPIC_SPACEDIM > 1)
    newPos[1] = fmod(p.pos(1) + p.rdata(1)*coefficient*dt, infra.geom.ProbHi()[1]);
#endif
#if (GEMPIC_SPACEDIM > 2)
    newPos[2] = fmod(p.pos(2) + p.rdata(2)*coefficient*dt, infra.geom.ProbHi()[2]);
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
