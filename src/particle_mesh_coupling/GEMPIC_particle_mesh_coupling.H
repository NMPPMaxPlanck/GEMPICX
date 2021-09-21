/*------------------------------------------------------------------------------
 functions copied from the AMReX library to make small changes
 AMReX file containing original functions: AMReX_Particle_mod_K.H
 (amrex/Src/Particle/AMReX_Particle_mod_K.H) pulled on 07.10.2019
------------------------------------------------------------------------------*/

#ifndef PARTICLE_MESH_COUPLING_H_
#define PARTICLE_MESH_COUPLING_H_

#include <cmath>

#include <AMReX_Array.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Particles.H>
#include <AMReX_Print.H>

#include <GEMPIC_Config.H>
#include <GEMPIC_splines.H>

namespace Gempic {
namespace Particles {

// we don't use fmod, because fmod(-0.2,1)=-0.2
amrex::Real mod_dom(amrex::Real val,
                    amrex::Real upper,
                    amrex::Real lower)
{
    return(val-(upper-lower)*floor((val-lower)/(upper-lower)));
}

template <typename P, int vdim, int degx, int degy, int degz>
AMREX_GPU_HOST_DEVICE  AMREX_NO_INLINE void gempic_deposit_rho (P const& p, amrex::Real charge,
                         amrex::Array4<amrex::Real> const& rho,
                         amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> const& plo,
                         amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM+1> const& dxi)

{
    splines spl({AMREX_D_DECL(false, false, false)}, {AMREX_D_DECL(false, false, false)});
    amrex::Real lx = (p.pos(0) - plo[0]) * dxi[0]; // don't need to displace because rho is nodal
    int i = std::floor(lx);
    amrex::Real xint = lx - i;
    std::vector<amrex::Real> sx_node = spl.spline_eval_vec (spl.shifted_x(xint, degx, false), degx);
#if (GEMPIC_SPACEDIM > 1)
    amrex::Real ly = (p.pos(1) - plo[1]) * dxi[1];
    int j = std::floor(ly);
    amrex::Real yint = ly - j;
    std::vector<amrex::Real> sy_node = spl.spline_eval_vec (spl.shifted_x(yint, degy, false), degy);
#endif
#if (GEMPIC_SPACEDIM > 2)
    amrex::Real lz = (p.pos(2) - plo[2]) * dxi[2];
    int k = std::floor(lz);
    amrex::Real zint = lz - k;
    std::vector<amrex::Real> sz_node = spl.spline_eval_vec (spl.shifted_x(zint, degz, false), degz);
#endif
    amrex::Real particle_coeff = dxi[GEMPIC_SPACEDIM]*charge*p.rdata(vdim);

    //TODOrefine
#if (GEMPIC_SPACEDIM > 2)
    for (int kk = 0; kk <= degz; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0; jj <= degy; ++jj) {
#endif
            for (int ii = 0; ii <= degx; ++ii) {
                double s_coeff = sx_node[ii]
        #if(GEMPIC_SPACEDIM > 1)
                        *sy_node[jj]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                        *sz_node[kk]
        #endif
                        ;
                amrex::GpuArray<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(spl.compute_cell_index(i+ii, xint, degx, false),
                                                              spl.compute_cell_index(j+jj, yint, degy, false),
                                                              spl.compute_cell_index(k+kk, zint,  degz, false))};
#if (GEMPIC_SPACEDIM == 1)
                amrex::HostDevice::Atomic::Add(rho.ptr(cell_index[0], 0, 0, 0), s_coeff*particle_coeff);
#else
                amrex::HostDevice::Atomic::Add(rho.ptr(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0), s_coeff*particle_coeff);
#endif
            }
#if (GEMPIC_SPACEDIM > 1)
        }
#endif
#if (GEMPIC_SPACEDIM > 2)
    }
#endif
}


/*!
     *  Evaluation of a FEM one form
     */
template<typename P, int vdim, int degx, int degy, int degz>
amrex::GpuArray<amrex::Real, vdim> evaluate_fem_efield(P const& p,
                                                  amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM+1> const& dxi,
                                                  amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> const& plo,
                                                  amrex::GpuArray<amrex::Array4<amrex::Real>,vdim> eArray)
{
    amrex::GpuArray<amrex::Real,vdim> fields;
    for (int comp = 0; comp < vdim; comp++) {
        fields[comp] = 0.;
    }
    splines spl({AMREX_D_DECL(false, false, false)}, {AMREX_D_DECL(false, false, false)});

    amrex::Real lx = (p.pos(0)-plo[0])*dxi[0];
    int i = std::floor(lx);
    amrex::Real xint = lx - i;
    std::vector<amrex::Real> sx_cell = spl.spline_eval_vec_back<degx-1> ( xint );
    std::vector<amrex::Real> sx_node = spl.spline_eval_vec_back<degx> ( xint );
#if (GEMPIC_SPACEDIM > 1)
    amrex::Real ly = (p.pos(1)-plo[1])*dxi[1];
    int j = std::floor(ly);
    amrex::Real yint = ly - j;
    std::vector<amrex::Real> sy_cell = spl.spline_eval_vec_back<degy-1> (yint);
    std::vector<amrex::Real> sy_node = spl.spline_eval_vec_back<degy> (yint);
#endif
#if (GEMPIC_SPACEDIM > 2)
    amrex::Real lz = (p.pos(2)-plo[2])*dxi[2];
    int k = std::floor(lz);
    amrex::Real zint = lz - k;
    std::vector<amrex::Real> sz_cell = spl.spline_eval_vec_back<degz-1> (zint);
    std::vector<amrex::Real> sz_node = spl.spline_eval_vec_back<degz> (zint);
#endif

#if (GEMPIC_SPACEDIM > 2)
    for (int kk = 0; kk <= degz; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0; jj <= degy; ++jj) {
#endif
            for (int ii = 0; ii <= degx; ++ii) {
                // Compute cell index. TODO: Make this work for order higher than 1 when the index needs to be shifted somehow
                amrex::GpuArray<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(i+ii, j+jj, k+kk)};
                if ( ii < degx ) {
                    double s_coeff = sx_cell[ii]
        #if(GEMPIC_SPACEDIM > 1)
                            *sy_node[jj]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                            *sz_node[kk]
        #endif
                            ;
#if (GEMPIC_SPACEDIM == 1)
                fields[0] += s_coeff*(eArray[0])(cell_index[0], 0, 0 ,0);
#else
                fields[0] += s_coeff*(eArray[0])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]) ,0);
#endif
                }
#if (GEMPIC_SPACEDIM > 1)
                if(vdim > 1) {
                    if ( jj < degy ) {
                        double s_coeff = sx_node[ii]
                                *sy_cell[jj]
        #if(GEMPIC_SPACEDIM > 2)
                                *sz_node[kk]
        #endif
                                ;
                        fields[1] += s_coeff*(eArray[1])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]) ,0);
                    }
                }
#endif
#if(GEMPIC_SPACEDIM > 2)
                if ( kk < degz ) {
                    double s_coeff = sx_node[ii]
                            *sy_node[jj]
                            *sz_cell[kk]
                            ;
                    fields[2] += s_coeff*(eArray[2])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]) ,0);
                }
#endif
            }
#if (GEMPIC_SPACEDIM > 1)
        }
#endif
#if (GEMPIC_SPACEDIM > 2)
    }
#endif
    return(fields);
}


/*!
     * Accumulation: contribution for one point
     * TODO: This function should better combine all points from one interval to reduce the number of loops
     */
template <int vdim, int degx, int degy, int degz>
void accumulate_j_fem_integral_point(amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> position_box,
                                     amrex::GpuArray<int,GEMPIC_SPACEDIM> box,
                                     amrex::GpuArray<amrex::Real, vdim>  weight,
                                     amrex::GpuArray<amrex::Array4<amrex::Real>,vdim> jArray)
{
    splines spl({AMREX_D_DECL(false, false, false)}, {AMREX_D_DECL(false, false, false)});

    std::vector<amrex::Real> sx_cell = spl.spline_eval_vec_back<degx-1> (position_box[0]);
    std::vector<amrex::Real> sx_node = spl.spline_eval_vec_back<degx> (position_box[0]);
#if (GEMPIC_SPACEDIM > 1)
    std::vector<amrex::Real> sy_cell = spl.spline_eval_vec_back<degy-1> (position_box[1]);
    std::vector<amrex::Real> sy_node = spl.spline_eval_vec_back<degy> (position_box[1]);
#endif
#if (GEMPIC_SPACEDIM > 2)
    std::vector<amrex::Real> sz_cell = spl.spline_eval_vec_back<degz-1> (position_box[2]);
    std::vector<amrex::Real> sz_node = spl.spline_eval_vec_back<degz> (position_box[2]);
#endif


#if (GEMPIC_SPACEDIM > 2)
    for (int kk = 0; kk <= degz; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0; jj <= degy; ++jj) {
#endif
            for (int ii = 0; ii <= degx; ++ii) {
                // Compute cell index. TODO: Make this work for order higher than 1 when the index needs to be shifted somehow
                amrex::GpuArray<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(box[0]+ii, box[1]+jj, box[2]+kk)};
                if ( ii < degx ) {
                    double s_coeff = sx_cell[ii]
        #if(GEMPIC_SPACEDIM > 1)
                            *sy_node[jj]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                            *sz_node[kk]
        #endif
                            ;
#if (GEMPIC_SPACEDIM == 1)
                amrex::HostDevice::Atomic::Add(jArray[0].ptr(cell_index[0], 0, 0 , 0), s_coeff*weight[0]);
#else
                amrex::HostDevice::Atomic::Add(jArray[0].ptr(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]) , 0), s_coeff*weight[0]);
#endif
                }
                if(vdim > 1) {
#if (GEMPIC_SPACEDIM > 1)
                    if ( jj < degy ) {
                        double s_coeff = sx_node[ii]
                                *sy_cell[jj]
        #if(GEMPIC_SPACEDIM > 2)
                                *sz_node[kk]
        #endif
                                ;
#else
                    double s_coeff = sx_node[ii];
#endif
#if (GEMPIC_SPACEDIM == 1)
                amrex::HostDevice::Atomic::Add(jArray[1].ptr(cell_index[0], 0, 0, 0), s_coeff*weight[1]);
#else
                amrex::HostDevice::Atomic::Add(jArray[1].ptr(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0), s_coeff*weight[1]);
#endif
#if (GEMPIC_SPACEDIM > 1)
                }
#endif
            }
            if(vdim > 2) {

#if(GEMPIC_SPACEDIM > 2)
                if ( kk < degz ) {
                    double s_coeff = sx_node[ii]
                            *sy_node[jj]
                            *sz_cell[kk];
#else
                double s_coeff = sx_node[ii]
        #if (GEMPIC_SPACEDIM > 1)
                        *sy_node[jj]
        #endif
                        ;
#endif
#if (GEMPIC_SPACEDIM == 1)
                amrex::HostDevice::Atomic::Add(jArray[2].ptr(cell_index[0], 0, 0, 0), s_coeff*weight[2]);
#else
                amrex::HostDevice::Atomic::Add(jArray[2].ptr(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0), s_coeff*weight[2]);
#endif
            }
        }
#if (GEMPIC_SPACEDIM > 1)
    }
#endif
#if (GEMPIC_SPACEDIM > 2)
} }
#endif
}



/*!
     *  Accumulation of the current for a FEM solver with integral over time (line integral)
     *  Currently this is implemented for degree one with two quadrature points in the line
     * integral for each box
     */
template <int vdim, int degx, int degy, int degz>
void accumulate_j_fem_integral(amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> position_old,
                               amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> position_new,
                               amrex::GpuArray<amrex::Real, vdim> velocity,
                               amrex::Real charge, // q*p.rdata(vdim)
                               amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM+1> const& dxi,
                               amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> const& plo,
                               amrex::GpuArray<amrex::Array4<amrex::Real>,vdim> jArray)
{
    // The quadrature formula TODO: Make this part of the class and generic for variable degree
    const int n_quad_points = 2;
    amrex::GpuArray<amrex::Real, n_quad_points> q_points = {(1.-1./std::sqrt(3.))*0.5, (1.+1./std::sqrt(3.))*0.5};
    amrex::GpuArray<amrex::Real, n_quad_points> q_weights = {0.5, 0.5};

    // 1a. Identify the box and relative position of position old
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position_box_old;
    amrex::GpuArray<int, GEMPIC_SPACEDIM> box_old;
    for (unsigned int d=0; d<GEMPIC_SPACEDIM;++d)
    {
        amrex::Real pos_norm = (position_old[d]-plo[d])*dxi[d];
        box_old[d] = std::floor(pos_norm);
        position_box_old[d] = pos_norm - box_old[d];
    }
    // 1b. Identify the box and relative position of position new
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position_box_new;
    amrex::GpuArray<int, GEMPIC_SPACEDIM> box_new;
    for (unsigned int d=0; d<GEMPIC_SPACEDIM;++d)
    {
        amrex::Real pos_norm = (position_new[d]-plo[d])*dxi[d];
        box_new[d] = std::floor(pos_norm);
        position_box_new[d] = pos_norm - box_new[d];
    }

    amrex::GpuArray<int, GEMPIC_SPACEDIM> boxdiff;
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> xnewtilde;
    int max_box = 0;
    for (unsigned int d=0;d<GEMPIC_SPACEDIM;++d)
    {
        boxdiff[d] = box_new[d]-box_old[d];
        xnewtilde[d] = position_box_new[d]+boxdiff[d];
        if (std::abs(boxdiff[d])> max_box)
            max_box = std::abs(boxdiff[d]);
    }

    max_box = max_box+1;

    amrex::GpuArray<int,GEMPIC_SPACEDIM> box = box_old; // We start in the old box

    amrex::GpuArray<int,GEMPIC_SPACEDIM>  sigma_counter = {AMREX_D_DECL(0, 0, 0)};

    amrex::GpuArray<std::vector<amrex::Real>, GEMPIC_SPACEDIM> sigmas;
    amrex::GpuArray<int, GEMPIC_SPACEDIM> increment;
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> sigma_next;

    for (unsigned int d=0;d<GEMPIC_SPACEDIM; ++d)
    {
        if (boxdiff[d] > 0 )
        {
            sigmas[d].resize(boxdiff[d]+1);
            for (unsigned int bl=0; bl<boxdiff[d];++bl)
                sigmas[d][bl] = (bl+1  - position_box_old[d])/(xnewtilde[d]-position_box_old[d]) ;

            sigmas[d][boxdiff[d]] = 1.0;
            sigma_next[d] = sigmas[d][0];
            increment[d] = 1;
        }
        else if (boxdiff[d] < 0 )
        {
            sigmas[d].resize(-boxdiff[d]+1);
            for (unsigned int bl=boxdiff[d]+1; bl<=0; --bl )
                sigmas[d][-bl] = (bl  - position_box_old[d])/(xnewtilde[d]-position_box_old[d]);

            sigmas[d][-boxdiff[d]] = 1.0;
            sigma_next[d] = sigmas[d][0];
            increment[d] = -1;
        }
        else
        {
            sigma_next[d] = 1.0;
            increment[d] = 0;
        }
    }


    amrex::Real sigma_l = 0.0;
    amrex::Real sigma_r = 0.0;
    while ( sigma_r < 1.0 )
    {
        // Identify index of next intersection
        auto next = std::min_element(sigma_next.begin(), sigma_next.end());
        int index_is = next-sigma_next.begin();
        sigma_r = *next; //sigma_next[index_is];
        for ( unsigned int  q = 0; q<n_quad_points; ++q )
        {
            amrex::Real sigma = sigma_l + (sigma_r-sigma_l) * q_points[q];
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position_box;
            for (unsigned int d=0;d<GEMPIC_SPACEDIM;++d)
            {
                position_box[d]= position_box_old[d]* (1.0 - sigma) + xnewtilde[d] * sigma  - sigma_counter[d]*increment[d];
            }
            AMREX_ASSERT_WITH_MESSAGE( (*(std::max_element(position_box.begin(), position_box.end())) < 1.0) &
                                       (*(std::min_element(position_box.begin(), position_box.end())) > 0.0), "Position is outside the interval [0,1].");

            amrex::Real weight = q_weights[q]*(sigma_r-sigma_l)*charge*dxi[GEMPIC_SPACEDIM];
            amrex::GpuArray<amrex::Real, vdim> weight_vec;
            for (unsigned int d=0;d<vdim;++d)
                weight_vec[d] = velocity[d]*weight;
            accumulate_j_fem_integral_point<vdim,degx,degy,degz>(position_box, box, weight_vec, jArray);
        }
        if (sigma_r < 1.0 )
        {
            // Update the
            sigma_counter[index_is] = sigma_counter[index_is]+1;
            sigma_next[index_is] = sigmas[index_is][sigma_counter[index_is]];
            box[index_is] = box[index_is] + increment[index_is];
            sigma_l = sigma_r;
        }
    }
}

/*!
     *  Accumulation of the integrated current and integrated b field for v update along dimension pdim.
     *  This routine is for the zigzagging Hamiltonian splitting scheme.
     *  Note this is currently only implemented for 3d3v.
     *
     */

template <int vdim, int degp, int degp1, int degp2, int pdim>
void accumulate_j_update_v_fem_integral(amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> position_old,
                               amrex::Real position_new,
                               amrex::Real charge, // q*p.rdata(vdim)
                               const amrex::Real* dx,
                               amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM+1> const& dxi,
                               amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> const& plo,
                               amrex::GpuArray<amrex::Array4<amrex::Real>,int(vdim/2.5)*2+1> bArray,
                               amrex::GpuArray<amrex::Array4<amrex::Real>,vdim> jArray,
                               amrex::GpuArray<amrex::Real, 2> &fields         )
{
  int pdim1 = (pdim+1)%3;
  int pdim2 = (pdim+2)%3;

  fields = {0.,0.};


    // 1a. Identify the box and relative position of position old
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position_box;
    amrex::GpuArray<int, GEMPIC_SPACEDIM> box;
    for (unsigned int d=0; d<GEMPIC_SPACEDIM;++d)
    {
        amrex::Real pos_norm = (position_old[d]-plo[d])*dxi[d];
        box[d] = std::floor(pos_norm);
        position_box[d] = pos_norm - box[d];
    }

    // 1b. Identify the box and relative position of position new
    amrex::Real position_box_new;
    int box_new;
    amrex::Real pos_norm = (position_new-plo[pdim])*dxi[pdim];
    box_new = std::floor(pos_norm);
    position_box_new = pos_norm - box_new;

    int local_size = std::abs(box_new-box[pdim])+degp;
    int box_start = box[pdim];
    if (box_new<box[pdim])
      box_start = box_new;

    // 2a. Evaluate the spline for the dimensions other than pdim
    splines spl({AMREX_D_DECL(false, false, false)}, {AMREX_D_DECL(false, false, false)});

    std::vector<amrex::Real> sy_cell = spl.spline_eval_vec_back<degp1-1> (position_box[pdim1]);
    std::vector<amrex::Real> sy_node = spl.spline_eval_vec_back<degp1> (position_box[pdim1]);

    std::vector<amrex::Real> sz_cell = spl.spline_eval_vec_back<degp2-1> (position_box[pdim2]);
    std::vector<amrex::Real> sz_node = spl.spline_eval_vec_back<degp2> (position_box[pdim2]);

    // 2b. Evaluate the primitive for dimension pdim (this is hard-coded for order one)
    std::vector<amrex::Real> sx_prim_old = spl.spline_eval_prim_vec<degp-1> (position_box[pdim]);
    std::vector<amrex::Real> sx_prim_new = spl.spline_eval_prim_vec<degp-1> (position_box_new);
    std::vector<amrex::Real> primitive(local_size+degp);
    if ( box[pdim] < box_new ) {
        for (unsigned int j =0;j<degp; j++) {
            primitive[local_size-degp+j] = sx_prim_new[j]*dx[pdim];
          }
        for(unsigned int j=0;j<local_size-degp;j++) {
            primitive[j] = dx[pdim];
          }
        for (unsigned int j=0;j<degp;j++) {
            primitive[j] -= sx_prim_old[j]*dx[pdim];
          }
      }
    else {
        for (unsigned int j =0;j<degp; j++) {
            primitive[local_size-degp+j] = -sx_prim_old[j]*dx[pdim];
          }
        for(unsigned int j=0;j<local_size-degp;j++) {
            primitive[j] = -dx[pdim];
          }
        for (unsigned int j=0;j<degp;j++) {
            primitive[j] += sx_prim_new[j]*dx[pdim];
          }
      }


    for (int kk = 0; kk <=degp2; ++kk)
      {
        for (int jj = 0; jj <= degp1; ++jj)
          {
            for (int ii = 0; ii <= local_size; ++ii)
              {
                amrex::GpuArray<int,GEMPIC_SPACEDIM> cell_index;
                if ( pdim == 0 )
                  cell_index = {AMREX_D_DECL(box_start+ii, box[1]+jj, box[2]+kk)};
                else if (pdim == 1 )
                  cell_index = {AMREX_D_DECL( box[0]+kk, box_start+ii, box[2]+jj)};
                else if (pdim == 2)
                  cell_index = {AMREX_D_DECL( box[0]+jj, box[1]+kk, box_start+ii)};
                double s_coeff = primitive[ii]
                        *sy_node[jj]
                        *sz_node[kk];
#if (GEMPIC_SPACEDIM == 1)
                amrex::HostDevice::Atomic::Add(jArray[pdim].ptr(cell_index[0], 0, 0 , 0), s_coeff*charge);
#else
                amrex::HostDevice::Atomic::Add(jArray[pdim].ptr(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]) , 0), s_coeff*charge);
#endif

                    if ( jj < degp1 )
                      {
                        double s_coeff = primitive[ii]
                              *sy_cell[jj]
                              *sz_node[kk];

#if (GEMPIC_SPACEDIM == 1)
                fields[0] += s_coeff*(bArray[pdim2])(cell_index[0], 0, 0, 0);
#else
                fields[0] += s_coeff*(bArray[pdim2])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0);
#endif
                          }
                  if ( kk < degp2 ){
                      double s_coeff = primitive[ii]
                              *sy_node[jj]
                              *sz_cell[kk];
#if (GEMPIC_SPACEDIM == 1)
                fields[1] += s_coeff*(bArray[pdim1])(cell_index[0], 0, 0, 0);
#else
                fields[1] += s_coeff*(bArray[pdim1])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0);
#endif
                          }
                    }
                    }
                    }
}


/*!
     *  Accumulation of the current for a FEM solver with integral over time (line integral)
     *  Currently this is implemented for degree one with two quadrature points in the line
     * integral for each box
     */
template <int vdim, int degx, int degy, int degz>
void accumulate_j_update_v_x1_fem_integral(amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> position_old,
                               amrex::Real position_new,
                               //amrex::GpuArray<amrex::Real, vdim> velocity,
                               amrex::Real charge, // q*p.rdata(vdim)
                               //amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM+1> const& dx,
                               const amrex::Real* dx,
                               amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM+1> const& dxi,
                               amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> const& plo,
                               amrex::GpuArray<amrex::Array4<amrex::Real>,int(vdim/2.5)*2+1> bArray,
                               amrex::GpuArray<amrex::Array4<amrex::Real>,vdim> jArray,
                               amrex::GpuArray<amrex::Real, 2> &fields         )
{
  int pdim = 0;
  fields = {0.,0.};
    // Identify the degree in the direction pdim
    int degpdim;
    if (pdim == 0)
       degpdim = degx;
     else if (pdim == 1)
       degpdim = degy;
     else if (pdim == 2)
       degpdim = degz;

    // 1a. Identify the box and relative position of position old
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position_box;
    amrex::GpuArray<int, GEMPIC_SPACEDIM> box;
    for (unsigned int d=0; d<GEMPIC_SPACEDIM;++d)
    {
        amrex::Real pos_norm = (position_old[d]-plo[d])*dxi[d];
        box[d] = std::floor(pos_norm);
        position_box[d] = pos_norm - box[d];
    }

    // 1b. Identify the box and relative position of position new
    amrex::Real position_box_new;
    int box_new;
    amrex::Real pos_norm = (position_new-plo[pdim])*dxi[pdim];
    box_new = std::floor(pos_norm);
    position_box_new = pos_norm - box_new;

    int local_size = std::abs(box_new-box[pdim])+degpdim;
    int box_start = box[pdim];
    if (box_new<box[pdim])
      box_start = box_new;

    // 2a. Evaluate the spline for the dimensions other than pdim
    splines spl({AMREX_D_DECL(false, false, false)}, {AMREX_D_DECL(false, false, false)});

#if (GEMPIC_SPACEDIM > 1)
    std::vector<amrex::Real> sy_cell = spl.spline_eval_vec_back<degy-1> (position_box[1]);
    std::vector<amrex::Real> sy_node = spl.spline_eval_vec_back<degy> (position_box[1]);
#endif
#if (GEMPIC_SPACEDIM > 2)
    std::vector<amrex::Real> sz_cell = spl.spline_eval_vec_back<degz-1> (position_box[2]);
    std::vector<amrex::Real> sz_node = spl.spline_eval_vec_back<degz> (position_box[2]);
#endif

    // 2b. Evaluate the primitive for dimension pdim (this is hard-coded for order one)
    std::vector<amrex::Real> sx_prim_old = spl.spline_eval_prim_vec<degx-1> (position_box[pdim]);
    std::vector<amrex::Real> sx_prim_new = spl.spline_eval_prim_vec<degx-1> (position_box_new);
    std::vector<amrex::Real> primitive(local_size+degpdim);
    if ( box[pdim] < box_new ) {
        for (unsigned int j =0;j<degpdim; j++) {
            primitive[local_size-degpdim+j] = sx_prim_new[j]*dx[pdim];
          }
        for(unsigned int j=0;j<local_size-degpdim;j++) {
            primitive[j] = dx[pdim];
          }
        for (unsigned int j=0;j<degpdim;j++) {
            primitive[j] -= sx_prim_old[j]*dx[pdim];
          }
      }
    else {
        for (unsigned int j =0;j<degpdim; j++) {
            primitive[local_size-degpdim+j] = -sx_prim_old[j]*dx[pdim];
          }
        for(unsigned int j=0;j<local_size-degpdim;j++) {
            primitive[j] = -dx[pdim];
          }
        for (unsigned int j=0;j<degpdim;j++) {
            primitive[j] += sx_prim_new[j]*dx[pdim];
          }
      }


#if (GEMPIC_SPACEDIM > 2 )
    for (int kk = 0; kk <=degz; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0; jj <= degy; ++jj) {
#endif
            for (int ii = 0; ii <= local_size; ++ii) {
                amrex::GpuArray<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(box_start+ii, box[1]+jj, box[2]+kk)};
                double s_coeff = primitive[ii]
    #if(GEMPIC_SPACEDIM > 1)
                        *sy_node[jj]
    #endif
    #if(GEMPIC_SPACEDIM > 2)
                        *sz_node[kk]
    #endif
                    ;
    #if (GEMPIC_SPACEDIM == 1)
                    amrex::HostDevice::Atomic::Add(jArray[0].ptr(cell_index[0], 0, 0 , 0), s_coeff*charge);
    #else
                    amrex::HostDevice::Atomic::Add(jArray[0].ptr(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]) , 0), s_coeff*charge);
    #endif

#if(GEMPIC_SPACEDIM > 1)
    if ( jj < degy ){
#endif
        s_coeff = primitive[ii]
#if(GEMPIC_SPACEDIM > 1)
                *sy_cell[jj]
#endif
#if(GEMPIC_SPACEDIM > 2)
                *sz_node[kk]
#endif
            ;
#if (GEMPIC_SPACEDIM == 1)
            fields[0] += s_coeff*(bArray[2])(cell_index[0], 0, 0, 0);
#else
            fields[0] += s_coeff*(bArray[2])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0);
#endif
#if (GEMPIC_SPACEDIM > 1)
            }
#endif
#if(GEMPIC_SPACEDIM > 2)
    if ( kk < degz ){
#endif
        s_coeff = primitive[ii]
#if(GEMPIC_SPACEDIM > 1)
                *sy_node[jj]
#endif
#if(GEMPIC_SPACEDIM > 2)
                *sz_cell[kk]
#endif
            ;
#if (GEMPIC_SPACEDIM == 1)
            fields[1] += s_coeff*(bArray[1])(cell_index[0], 0, 0, 0);
#else
            fields[1] += s_coeff*(bArray[1])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0);
#endif

#if(GEMPIC_SPACEDIM > 2)
            }
#endif
      }
#if (GEMPIC_SPACEDIM > 1 )
      }
#endif
#if (GEMPIC_SPACEDIM > 2 )
      }
#endif


}


/*!
     *  Accumulation of the current for a FEM solver with integral over time (line integral)
     *  Currently this is implemented for degree one with two quadrature points in the line
     * integral for each box
     */
template <int vdim, int degx, int degy, int degz>
void accumulate_j_update_v_x2_fem_integral(amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> position_old,
                               amrex::Real position_new,
                               //amrex::GpuArray<amrex::Real, vdim> velocity,
                               amrex::Real charge, // q*p.rdata(vdim)
                               //amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM+1> const& dx,
                               const amrex::Real* dx,
                               amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM+1> const& dxi,
                               amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> const& plo,
                               amrex::GpuArray<amrex::Array4<amrex::Real>,int(vdim/2.5)*2+1> bArray,
                               amrex::GpuArray<amrex::Array4<amrex::Real>,vdim> jArray,
                               amrex::GpuArray<amrex::Real, 2> &fields         )
{
  int pdim = 1;
  fields = {0.,0.};
  // Identify the degree in the direction pdim
  int degpdim;
  if (pdim == 0)
     degpdim = degx;
   else if (pdim == 1)
     degpdim = degy;
   else if (pdim == 2)
     degpdim = degz;

    // 1a. Identify the box and relative position of position old
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position_box;
    amrex::GpuArray<int, GEMPIC_SPACEDIM> box;
    for (unsigned int d=0; d<GEMPIC_SPACEDIM;++d)
    {
        amrex::Real pos_norm = (position_old[d]-plo[d])*dxi[d];
        box[d] = std::floor(pos_norm);
        position_box[d] = pos_norm - box[d];
    }

    // 1b. Identify the box and relative position of position new
    amrex::Real position_box_new;
    int box_new;
    amrex::Real pos_norm = (position_new-plo[pdim])*dxi[pdim];
    box_new = std::floor(pos_norm);
    position_box_new = pos_norm - box_new;

    int local_size = std::abs(box_new-box[pdim])+degpdim;
    int box_start = box[pdim];
    if (box_new<box[pdim])
      box_start = box_new;

    // 2a. Evaluate the spline for the dimensions other than pdim
    splines spl({AMREX_D_DECL(false, false, false)}, {AMREX_D_DECL(false, false, false)});

    std::vector<amrex::Real> sx_cell = spl.spline_eval_vec_back<degx-1> (position_box[0]);
    std::vector<amrex::Real> sx_node = spl.spline_eval_vec_back<degx> (position_box[0]);
#if (GEMPIC_SPACEDIM > 2)
    std::vector<amrex::Real> sz_cell = spl.spline_eval_vec_back<degz-1> (position_box[2]);
    std::vector<amrex::Real> sz_node = spl.spline_eval_vec_back<degz> (position_box[2]);
#endif

    // 2b. Evaluate the primitive for dimension pdim (this is hard-coded for order one)
    std::vector<amrex::Real> sx_prim_old = spl.spline_eval_prim_vec<degy-1> (position_box[pdim]);
    std::vector<amrex::Real> sx_prim_new = spl.spline_eval_prim_vec<degy-1> (position_box_new);
    std::vector<amrex::Real> primitive(local_size+degpdim);
    if ( box[pdim] < box_new ) {
        for (unsigned int j =0;j<degpdim; j++) {
            primitive[local_size-degpdim+j] = sx_prim_new[j]*dx[pdim];
          }
        for(unsigned int j=0;j<local_size-degpdim;j++) {
            primitive[j] = dx[pdim];
          }
        for (unsigned int j=0;j<degpdim;j++) {
            primitive[j] -= sx_prim_old[j]*dx[pdim];
          }
      }
    else {
        for (unsigned int j =0;j<degpdim; j++) {
            primitive[local_size-degpdim+j] = -sx_prim_old[j]*dx[pdim];
          }
        for(unsigned int j=0;j<local_size-degpdim;j++) {
            primitive[j] = -dx[pdim];
          }
        for (unsigned int j=0;j<degpdim;j++) {
            primitive[j] += sx_prim_new[j]*dx[pdim];
          }
      }


#if (GEMPIC_SPACEDIM > 2 )
    for (int kk = 0; kk <=degz; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0; jj <= local_size; ++jj) {
#endif
            for (int ii = 0; ii <= degx; ++ii) {
                amrex::GpuArray<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(box[0]+ii, box_start+jj, box[2]+kk)};
                double s_coeff = sx_node[ii]
    #if(GEMPIC_SPACEDIM > 1)
                        *primitive[jj]
    #endif
    #if(GEMPIC_SPACEDIM > 2)
                        *sz_node[kk]
    #endif
                    ;
    #if (GEMPIC_SPACEDIM == 1)
                    amrex::HostDevice::Atomic::Add(jArray[1].ptr(cell_index[0], 0, 0 , 0), s_coeff*charge);
    #else
                    amrex::HostDevice::Atomic::Add(jArray[1].ptr(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]) , 0), s_coeff*charge);
    #endif

    if ( ii < degx ){
        double s_coeff = sx_cell[ii]
#if(GEMPIC_SPACEDIM > 1)
                *primitive[jj]
#endif
#if(GEMPIC_SPACEDIM > 2)
                *sz_node[kk]
#endif
            ;
#if (GEMPIC_SPACEDIM == 1)
            fields[1] += s_coeff*(bArray[2])(cell_index[0], 0, 0, 0);
#else
            fields[1] += s_coeff*(bArray[2])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0);
#endif
#if (GEMPIC_SPACEDIM > 1)
            }
#endif
#if(GEMPIC_SPACEDIM > 2)
    if ( kk < degz ){
#endif
        s_coeff = sx_node[ii]
#if(GEMPIC_SPACEDIM > 1)
                *primitive[jj]
#endif
#if(GEMPIC_SPACEDIM > 2)
                *sz_cell[kk]
#endif
            ;
#if (GEMPIC_SPACEDIM == 1)
            fields[0] += s_coeff*(bArray[0])(cell_index[0], 0, 0, 0);
#else
            fields[0] += s_coeff*(bArray[0])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0);
#endif

#if(GEMPIC_SPACEDIM > 2)
            }
#endif
      }
      }
#if (GEMPIC_SPACEDIM > 2 )
      }
#endif
}


/*!
     *  Accumulation of the current for a FEM solver with integral over time (line integral)
     *  Currently this is implemented for degree one with two quadrature points in the line
     * integral for each box
     */
template <int vdim, int degx, int degy, int degz>
void accumulate_j_update_v_x3_fem_integral(amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> position_old,
                               amrex::Real position_new,
                               //amrex::GpuArray<amrex::Real, vdim> velocity,
                               amrex::Real charge, // q*p.rdata(vdim)
                               //amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM+1> const& dx,
                               const amrex::Real* dx,
                               amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM+1> const& dxi,
                               amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> const& plo,
                               amrex::GpuArray<amrex::Array4<amrex::Real>,int(vdim/2.5)*2+1> bArray,
                               amrex::GpuArray<amrex::Array4<amrex::Real>,vdim> jArray,
                               amrex::GpuArray<amrex::Real, 2> &fields         )
{
  int pdim = 2;
  fields = {0.,0.};
    // Identify the degree in the direction pdim
    int degpdim;
    if (pdim == 0)
       degpdim = degx;
     else if (pdim == 1)
       degpdim = degy;
     else if (pdim == 2)
       degpdim = degz;

    // 1a. Identify the box and relative position of position old
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position_box;
    amrex::GpuArray<int, GEMPIC_SPACEDIM> box;
    for (unsigned int d=0; d<GEMPIC_SPACEDIM;++d)
    {
        amrex::Real pos_norm = (position_old[d]-plo[d])*dxi[d];
        box[d] = std::floor(pos_norm);
        position_box[d] = pos_norm - box[d];
    }

    // 1b. Identify the box and relative position of position new
    amrex::Real position_box_new;
    int box_new;
    amrex::Real pos_norm = (position_new-plo[pdim])*dxi[pdim];
    box_new = std::floor(pos_norm);
    position_box_new = pos_norm - box_new;

    int local_size = std::abs(box_new-box[pdim])+degpdim;
    int box_start = box[pdim];
    if (box_new<box[pdim])
      box_start = box_new;

    // 2a. Evaluate the spline for the dimensions other than pdim
    splines spl({AMREX_D_DECL(false, false, false)}, {AMREX_D_DECL(false, false, false)});

    std::vector<amrex::Real> sx_cell = spl.spline_eval_vec_back<degx-1> (position_box[0]);
    std::vector<amrex::Real> sx_node = spl.spline_eval_vec_back<degx> (position_box[0]);
#if (GEMPIC_SPACEDIM > 1)
    std::vector<amrex::Real> sy_cell = spl.spline_eval_vec_back<degy-1> (position_box[1]);
    std::vector<amrex::Real> sy_node = spl.spline_eval_vec_back<degy> (position_box[1]);
#endif


    // 2b. Evaluate the primitive for dimension pdim (this is hard-coded for order one)
    std::vector<amrex::Real> sx_prim_old = spl.spline_eval_prim_vec<degz-1> (position_box[pdim]);
    std::vector<amrex::Real> sx_prim_new = spl.spline_eval_prim_vec<degz-1> (position_box_new);
    std::vector<amrex::Real> primitive(local_size+degpdim);
    if ( box[pdim] < box_new ) {
        for (unsigned int j =0;j<degpdim; j++) {
            primitive[local_size-degpdim+j] = sx_prim_new[j]*dx[pdim];
          }
        for(unsigned int j=0;j<local_size-degpdim;j++) {
            primitive[j] = dx[pdim];
          }
        for (unsigned int j=0;j<degpdim;j++) {
            primitive[j] -= sx_prim_old[j]*dx[pdim];
          }
      }
    else {
        for (unsigned int j =0;j<degpdim; j++) {
            primitive[local_size-degpdim+j] = -sx_prim_old[j]*dx[pdim];
          }
        for(unsigned int j=0;j<local_size-degpdim;j++) {
            primitive[j] = -dx[pdim];
          }
        for (unsigned int j=0;j<degpdim;j++) {
            primitive[j] += sx_prim_new[j]*dx[pdim];
          }
      }


#if (GEMPIC_SPACEDIM > 2 )
    for (int kk = 0; kk <=local_size; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0; jj <= degy; ++jj) {
#endif
            for (int ii = 0; ii <= degx; ++ii) {
                amrex::GpuArray<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(box[0]+ii, box[1]+jj, box_start+kk)};
                double s_coeff = sx_node[ii]
    #if(GEMPIC_SPACEDIM > 1)
                        *sy_node[jj]
    #endif
    #if(GEMPIC_SPACEDIM > 2)
                        *primitive[kk]
    #endif
                    ;
    #if (GEMPIC_SPACEDIM == 1)
                    amrex::HostDevice::Atomic::Add(jArray[2].ptr(cell_index[0], 0, 0 , 0), s_coeff*charge);
    #else
                    amrex::HostDevice::Atomic::Add(jArray[2].ptr(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]) , 0), s_coeff*charge);
    #endif

#if(GEMPIC_SPACEDIM > 1)
    if ( jj < degy ){
        double s_coeff = sx_node[ii]
#endif
#if(GEMPIC_SPACEDIM > 1)
                *sy_cell[jj]
#endif
#if(GEMPIC_SPACEDIM > 2)
                *primitive[kk]
#endif
            ;
#if (GEMPIC_SPACEDIM == 1)
            fields[0] += s_coeff*(bArray[1])(cell_index[0], 0, 0, 0);
#else
            fields[0] += s_coeff*(bArray[1])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0);
#endif
#if (GEMPIC_SPACEDIM > 1)
            }
#endif
    if ( ii < degx ){
        double s_coeff = sx_cell[ii]
#if(GEMPIC_SPACEDIM > 1)
                *sy_node[jj]
#endif
#if(GEMPIC_SPACEDIM > 2)
                *primitive[kk]
#endif
            ;
#if (GEMPIC_SPACEDIM == 1)
            fields[1] += s_coeff*(bArray[0])(cell_index[0], 0, 0, 0);
#else
            fields[1] += s_coeff*(bArray[0])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0);
#endif

            }
      }
#if (GEMPIC_SPACEDIM > 1 )
      }
#endif
#if (GEMPIC_SPACEDIM > 2 )
      }
#endif


}

/*!
     *  Accumulation of the current for a FEM solver with integral over time (line integral)
     *  Currently this is implemented for Gauss-Quadrature with only one point
     */
template<int vdim, int degx, int degy, int degz>
void accumulate_j_fem_integral(amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> position,
                               amrex::GpuArray<amrex::Real, vdim> velocity,
                               amrex::Real charge, // q*p.rdata(vdim)
                               amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM+1> const& dxi,
                               amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> const& plo,
                               amrex::GpuArray<amrex::Array4<amrex::Real>,vdim> jArray)
{
    splines spl({AMREX_D_DECL(false, false, false)}, {AMREX_D_DECL(false, false, false)});

    amrex::Real lx = (position[0]-plo[0])*dxi[0];
    int i = std::floor(lx);
    amrex::Real xint = lx - i;
    std::vector<amrex::Real> sx_cell = spl.spline_eval_vec (spl.shifted_x(xint, degx-1, true), degx-1);
    std::vector<amrex::Real> sx_node = spl.spline_eval_vec (spl.shifted_x(xint, degx, false), degx);
#if (GEMPIC_SPACEDIM > 1)
    amrex::Real ly = (position[1]-plo[1])*dxi[1];
    int j = std::floor(ly);
    amrex::Real yint = ly - j;
    std::vector<amrex::Real> sy_cell = spl.spline_eval_vec (spl.shifted_x(yint, degy-1, true), degy-1);
    std::vector<amrex::Real> sy_node = spl.spline_eval_vec (spl.shifted_x(yint, degy, false), degy);
#endif
#if (GEMPIC_SPACEDIM > 2)
    amrex::Real lz = (position[2]-plo[2])*dxi[2];
    int k = std::floor(lz);
    amrex::Real zint = lz - k;
    std::vector<amrex::Real> sz_cell = spl.spline_eval_vec (spl.shifted_x(zint, degz-1, true), degz-1);
    std::vector<amrex::Real> sz_node = spl.spline_eval_vec (spl.shifted_x(zint, degz, false), degz);
#endif
    amrex::Real particle_coeff = dxi[GEMPIC_SPACEDIM]*charge;


#if (GEMPIC_SPACEDIM > 2)
    for (int kk = 0; kk <= degz; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0; jj <= degy; ++jj) {
#endif
            for (int ii = 0; ii <= degx; ++ii) {
                // Compute cell index. TODO: Make this work for order higher than 1 when the index needs to be shifted somehow
                amrex::GpuArray<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(i+ii, j+jj, k+kk)};
                if ( ii < degx ) {
                    double s_coeff = sx_cell[ii]
        #if(GEMPIC_SPACEDIM > 1)
                            *sy_node[jj]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                            *sz_node[kk]
        #endif
                            ;
#if (GEMPIC_SPACEDIM == 1)
                amrex::HostDevice::Atomic::Add(jArray[0].ptr(cell_index[0], 0, 0, 0), s_coeff*particle_coeff*velocity[0]);
#else
                amrex::HostDevice::Atomic::Add(jArray[0].ptr(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0), s_coeff*particle_coeff*velocity[0]);
#endif
                }
#if (GEMPIC_SPACEDIM > 1)
                if(vdim > 1) {
                    if ( jj < degy ) {
                        double s_coeff = sx_node[ii]
                                *sy_cell[jj]
        #if(GEMPIC_SPACEDIM > 2)
                                *sz_node[kk]
        #endif
                                ;
                        amrex::HostDevice::Atomic::Add(jArray[1].ptr(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0), s_coeff*particle_coeff*velocity[1]);
                    }
                }
#endif
#if(GEMPIC_SPACEDIM > 2)
                if ( kk < degz ) {
                    double s_coeff = sx_node[ii]
                            *sy_node[jj]
                            *sz_cell[kk]
                            ;
                    amrex::HostDevice::Atomic::Add(jArray[2].ptr(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0), s_coeff*particle_coeff*velocity[2]);
                }
#endif
            }
#if (GEMPIC_SPACEDIM > 1)
        }
#endif
#if (GEMPIC_SPACEDIM > 2)
    }
#endif
}

/*!
     *  Field evluation of a two form described by finite elements
     */
template <int vdim, int degx, int degy, int degz>
AMREX_GPU_HOST_DEVICE  AMREX_NO_INLINE amrex::GpuArray<amrex::Real, int(vdim/2.5)*2+1> evaluate_fem_bfield(amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> position,
                                                               amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM+1> const& dxi,
                                                               amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> const& plo,
                                                                      amrex::GpuArray<amrex::Array4<amrex::Real>,int(vdim/2.5)*2+1> bArray)
{
    amrex::GpuArray<amrex::Real, int(vdim/2.5)*2+1> fields;
    for (int comp = 0; comp < int(vdim/2.5)*2+1; comp++) {
        fields[comp] = 0.;
    }
    splines spl({AMREX_D_DECL(false, false, false)}, {AMREX_D_DECL(false, false, false)});

    amrex::Real lx = (position[0]-plo[0])*dxi[0];
    int i = std::floor(lx);
    amrex::Real xint = lx - i;
    std::vector<amrex::Real> sx_cell = spl.spline_eval_vec_back<degx-1> ( xint );
    std::vector<amrex::Real> sx_node = spl.spline_eval_vec_back<degx> ( xint );
#if (GEMPIC_SPACEDIM > 1)
    amrex::Real ly = (position[1]-plo[1])*dxi[1];
    int j = std::floor(ly);
    amrex::Real yint = ly - j;
    std::vector<amrex::Real> sy_cell = spl.spline_eval_vec_back<degy-1> ( yint );
    std::vector<amrex::Real> sy_node = spl.spline_eval_vec_back<degy> ( yint );
#endif
#if (GEMPIC_SPACEDIM > 2)
    amrex::Real lz = (position[2]-plo[2])*dxi[2];
    int k = std::floor(lz);
    amrex::Real zint = lz - k;
    std::vector<amrex::Real> sz_cell = spl.spline_eval_vec_back<degz-1> ( zint );
    std::vector<amrex::Real> sz_node = spl.spline_eval_vec_back<degz> ( zint );
#endif

    // First B_0
#if (GEMPIC_SPACEDIM > 2)
    for (int kk = 0; kk <= degz-1; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0; jj <= degy-1; ++jj) {
#endif
            for (int ii = 0; ii <= degx; ++ii) {
                // Compute cell index. TODO: Make this work for order higher than 1 when the index needs to be shifted somehow
                amrex::GpuArray<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(i+ii, j+jj, k+kk)};
                double s_coeff = sx_node[ii]
        #if(GEMPIC_SPACEDIM > 1)
                        *sy_cell[jj]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                        *sz_cell[kk]
        #endif
                        ;
#if (GEMPIC_SPACEDIM == 1)
                fields[0] += s_coeff*(bArray[0])(cell_index[0], 0, 0, 0);
#else
                fields[0] += s_coeff*(bArray[0])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0);
#endif
            }
#if (GEMPIC_SPACEDIM > 1)
        }
#endif
#if (GEMPIC_SPACEDIM > 2)
    }
#endif
    // B_1
    if (int(vdim/2.5)*2+1>1) {
#if (GEMPIC_SPACEDIM > 2)
        for (int kk = 0; kk <= degz-1; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
            for (int jj = 0; jj <= degy; ++jj) {
#endif
                for (int ii = 0; ii <= degx; ++ii) {
                    // Compute cell index. TODO: Make this work for order higher than 1 when the index needs to be shifted somehow
                    amrex::GpuArray<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(i+ii, j+jj, k+kk)};
                    double s_coeff = sx_cell[ii]
        #if(GEMPIC_SPACEDIM > 1)
                            *sy_node[jj]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                            *sz_cell[kk]
        #endif
                            ;
#if (GEMPIC_SPACEDIM == 1)
                fields[1] += s_coeff*(bArray[1])(cell_index[0], 0, 0, 0);
#else
                fields[1] += s_coeff*(bArray[1])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0);
#endif
                }
#if (GEMPIC_SPACEDIM > 1)
            }
#endif
#if (GEMPIC_SPACEDIM > 2)
        }
#endif
    }

    // B_2
    if (int(vdim/2.5)*2+1>2){
#if (GEMPIC_SPACEDIM > 2)
        for (int kk = 0; kk <= degz; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
            for (int jj = 0; jj <= degy-1; ++jj) {
#endif
                for (int ii = 0; ii <= degx-1; ++ii) {
                    // Compute cell index. TODO: Make this work for order higher than 1 when the index needs to be shifted somehow
                    amrex::GpuArray<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(i+ii, j+jj, k+kk)};
                    double s_coeff = sx_cell[ii]
        #if(GEMPIC_SPACEDIM > 1)
                            *sy_cell[jj]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                            *sz_node[kk]
        #endif
                            ;
#if (GEMPIC_SPACEDIM == 1)
                fields[2] += s_coeff*(bArray[2])(cell_index[0], 0, 0, 0);
#else
                fields[2] += s_coeff*(bArray[2])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0);
#endif
                }
#if (GEMPIC_SPACEDIM > 1)
            }
#endif
#if (GEMPIC_SPACEDIM > 2)
        }
#endif
    }
    return(fields);
}



template <typename P, int vdim, int degx, int degy, int degz>
amrex::GpuArray<amrex::Real, vdim+int(vdim/2.5)*2+1> gempic_deposit_J_and_interpolate (P const& p, amrex::Real charge,
                                                                                  amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM+1> const& dxi,
                                                                                  amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> const& plo,
                                                                                         amrex::GpuArray<amrex::Array4<amrex::Real>,vdim> jArray,
                                                                                         amrex::GpuArray<amrex::Array4<amrex::Real>,vdim> eArray,
                                                                                         amrex::GpuArray<amrex::Array4<amrex::Real>,int(vdim/2.5)*2+1> bArray,
                                                                                         amrex::Real coefficient = 1.,
                                                                                         bool field_int = false)

{
    amrex::GpuArray<amrex::Real, vdim+int(vdim/2.5)*2+1> fields = {AMREX_D_DECL(0.,0.,0.)};

    splines spl({AMREX_D_DECL(false, false, false)}, {AMREX_D_DECL(false, false, false)});
    amrex::Real lx = (p.pos(0)-plo[0])*dxi[0];
    int i = std::floor(lx);
    amrex::Real xint = lx - i;
    std::vector<amrex::Real> sx_node = spl.spline_eval_vec (spl.shifted_x(xint, degx, false), degx);
    std::vector<amrex::Real> sx_cell_1 = spl.spline_eval_vec (spl.shifted_x(xint, degx-1, true), degx-1);
    amrex::GpuArray<std::vector<amrex::Real>,2> sx_1 = {sx_cell_1, sx_node};
#if (GEMPIC_SPACEDIM > 1)
    amrex::Real ly = (p.pos(1)-plo[1])*dxi[1];
    int j = std::floor(ly);
    amrex::Real yint = ly - j;
    std::vector<amrex::Real> sy_node = spl.spline_eval_vec (spl.shifted_x(yint, degy, false), degy);
    std::vector<amrex::Real> sy_cell_1 = spl.spline_eval_vec (spl.shifted_x(yint, degy-1, true), degy-1);
    amrex::GpuArray<std::vector<amrex::Real>,2> sy_1 = {sy_cell_1, sy_node};
#endif
#if (GEMPIC_SPACEDIM > 2)
    amrex::Real lz = (p.pos(2)-plo[2])*dxi[2];
    int k = std::floor(lz);
    amrex::Real zint = lz - k;
    std::vector<amrex::Real> sz_node = spl.spline_eval_vec (spl.shifted_x(zint, degz, false), degz);
    std::vector<amrex::Real> sz_cell_1 = spl.spline_eval_vec (spl.shifted_x(zint, degz-1, true), degz-1);
    amrex::GpuArray<std::vector<amrex::Real>,2> sz_1 = {sz_cell_1, sz_node};
#endif
    amrex::Real particle_coeff = dxi[GEMPIC_SPACEDIM]*charge*coefficient*p.rdata(vdim);

#if (GEMPIC_SPACEDIM > 2)
    for (int kk = 0; kk <= degz; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0; jj <= degy; ++jj) {
#endif
            for (int ii = 0; ii <= degx; ++ii) {

                // deposit in J and interpolate in E
                for (int cc=0;cc<vdim;cc++) {
                    if(!(
            #if (GEMPIC_SPACEDIM > 2)
                                ((kk==degz) & (cc%3==2))|
            #endif
            #if (GEMPIC_SPACEDIM > 1)
                                ((jj==degy) & (cc%3==1))|
            #endif
                                ((ii==degx) & (cc%3==0)))) {
                        double s_coeff = sx_1[cc%3==0?0:1][ii]
                              #if(GEMPIC_SPACEDIM > 1)
                                                      *sy_1[cc%3==1?0:1][jj]
                                            #endif
                                            #if(GEMPIC_SPACEDIM > 2)
                                                                    *sz_1[cc%3==2?0:1][kk]
                                                          #endif
                                                                                  ;
                        amrex::GpuArray<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(spl.compute_cell_index(i+ii, xint, degx-(cc%3==0?1:0), cc%3==0?true:false),
                                                                      spl.compute_cell_index(j+jj, yint, degy-(cc%3==1?1:0), cc%3==1?true:false),
                                                                      spl.compute_cell_index(k+kk, zint, degz-(cc%3==2?1:0), cc%3==2?true:false))};

#if (GEMPIC_SPACEDIM == 1)
                amrex::HostDevice::Atomic::Add(jArray[cc].ptr(cell_index[0], 0, 0, 0), s_coeff*particle_coeff*p.rdata(cc));
#else
                amrex::HostDevice::Atomic::Add(jArray[cc].ptr(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0), s_coeff*particle_coeff*p.rdata(cc));
#endif

                        if (field_int) {
#if (GEMPIC_SPACEDIM == 1)
                fields[cc] += s_coeff*(eArray[cc])(cell_index[0], 0, 0, 0);
#else
                fields[cc] += s_coeff*(eArray[cc])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0);
#endif
                        }
                    }
                }
                if (field_int) {
                    // deposit in B
                    for (int cc=0;cc<int(vdim/2.5)*2+1;cc++) {
                        if(!(
            #if (GEMPIC_SPACEDIM > 2)
                                    ((kk==degz) & (cc%3!=2))|
            #endif
            #if (GEMPIC_SPACEDIM > 1)
                                    ((jj==degy) & (cc%3!=1))|
            #endif
                                    ((ii==degx) & (cc%3!=0)))) {
                            double s_coeff = sx_1[cc%3==0?1:0][ii]
                              #if(GEMPIC_SPACEDIM > 1)
                                                          *sy_1[cc%3==1?1:0][jj]
                                            #endif
                                            #if(GEMPIC_SPACEDIM > 2)
                                                                        *sz_1[cc%3==2?1:0][kk]
                                                          #endif
                                                                                      ;
                            amrex::GpuArray<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(spl.compute_cell_index(i+ii, xint, degx-(cc%3==0?0:1), cc%3==0?false:true),
                                                                          spl.compute_cell_index(j+jj, yint, degy-(cc%3==1?0:1), cc%3==1?false:true),
                                                                          spl.compute_cell_index(k+kk, zint, degz-(cc%3==2?0:1), cc%3==2?false:true))};


#if (GEMPIC_SPACEDIM == 1)
                fields[cc+vdim] += s_coeff*(bArray[cc])(cell_index[0], 0, 0, 0);
#else
                fields[cc+vdim] += s_coeff*(bArray[cc])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0);
#endif
                        }
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

template <typename P, int vdim, int degx, int degy, int degz>
void gempic_deposit_charge_indextype (P const& p, amrex::Real charge,
                                      amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM+1> const& dxi,
                                      amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> const& plo,
                                      amrex::Array4<amrex::Real> jArray,
                                      amrex::IndexType IndT)

{

    splines spl({AMREX_D_DECL(false, false, false)}, {AMREX_D_DECL(false, false, false)});
    amrex::Real lx = (p.pos(0)-plo[0])*dxi[0];
    int i = std::floor(lx);
    amrex::Real xint = lx - i;
    std::vector<amrex::Real> sx_node = spl.spline_eval_vec (spl.shifted_x(xint, degx, false), degx);
    std::vector<amrex::Real> sx_cell_1 = spl.spline_eval_vec (spl.shifted_x(xint, degx, true), degx);
    amrex::GpuArray<std::vector<amrex::Real>,2> sx_1 = {sx_cell_1, sx_node};
    amrex::Real ly = (p.pos(1)-plo[1])*dxi[1];
    int j = std::floor(ly);
    amrex::Real yint = ly - j;
    std::vector<amrex::Real> sy_node = spl.spline_eval_vec (spl.shifted_x(yint, degy, false), degy);
    std::vector<amrex::Real> sy_cell_1 = spl.spline_eval_vec (spl.shifted_x(yint, degy, true), degy);
    amrex::GpuArray<std::vector<amrex::Real>,2> sy_1 = {sy_cell_1, sy_node};
    amrex::Real lz = (p.pos(2)-plo[2])*dxi[2];
    int k = std::floor(lz);
    amrex::Real zint = lz - k;
    std::vector<amrex::Real> sz_node = spl.spline_eval_vec (spl.shifted_x(zint, degz, false), degz);
    std::vector<amrex::Real> sz_cell_1 = spl.spline_eval_vec (spl.shifted_x(zint, degz, true), degz);
    amrex::GpuArray<std::vector<amrex::Real>,2> sz_1 = {sz_cell_1, sz_node};
    amrex::Real particle_coeff = dxi[GEMPIC_SPACEDIM]*charge*p.rdata(vdim);

    for (int kk = 0; kk <= degz; ++kk) {
        for (int jj = 0; jj <= degy; ++jj) {
            for (int ii = 0; ii <= degx; ++ii) {

                // deposit in J and interpolate in E
                double s_coeff = sx_1[IndT[0]==0?0:1][ii]*sy_1[IndT[1]==0?0:1][jj]*sz_1[IndT[2]==0?0:1][kk]
                                                                                                   ;
                amrex::GpuArray<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(spl.compute_cell_index(i+ii, xint, degx, IndT[0]==0?true:false),
                                                              spl.compute_cell_index(j+jj, yint, degy, IndT[1]==0?true:false),
                                                              spl.compute_cell_index(k+kk, zint, degz, IndT[2]==0?true:false))};

#if (GEMPIC_SPACEDIM == 1)
                amrex::HostDevice::Atomic::Add(jArray.ptr(cell_index[0], 0, 0, 0), s_coeff*particle_coeff);
#else
                amrex::HostDevice::Atomic::Add(jArray.ptr(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0), s_coeff*particle_coeff);
#endif


            }
        }
    }
}

/*!
     * Push the velocity with a given electric field
     */
template <typename P, int vdim>
AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE amrex::GpuArray<amrex::Real, vdim> push_v_efield(P const& p,
                                            amrex::Real dt,
                                            amrex::Real chargemass, //charge/mass
                                            amrex::GpuArray<amrex::Real,vdim> Ep){
    amrex::GpuArray<amrex::Real,vdim> newPos;

    newPos[0] = p.rdata(0) + dt*chargemass*(Ep[0]);
    if (vdim > 1) {
        newPos[1] = p.rdata(1) + dt*chargemass*(Ep[1]);
    }
    if (vdim > 2) {
        newPos[2] = p.rdata(2) + dt*chargemass*(Ep[2]);
    }
    return(newPos);
}


/*!
     * Push the position and the velocity (only the vxB part) in a nonlinear iteration
     */
template <typename P, int vdim, int degx, int degy, int degz>
amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM+vdim> push_x_v_bfield(P const& p,
                                                              amrex::Real dt,
                                                              amrex::Real chargemass, //charge/mass
                                                              amrex::GpuArray<amrex::Array4<amrex::Real>,int(vdim/2.5)*2+1> bArray,
                                                              CompDom::computational_domain infra){
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> oldX;
    for (unsigned int d=0;d<GEMPIC_SPACEDIM; ++d)
        oldX[d]= p.pos(d);
    amrex::GpuArray<amrex::Real, vdim> oldV;
    for (unsigned int d=0;d<vdim;++d)
        oldV[d] = p.rdata(d);
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> newX = oldX;
    amrex::GpuArray<amrex::Real, vdim> newV = oldV;
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> newXiter;
    amrex::GpuArray<amrex::Real, vdim> newViter;
    amrex::Real tolerance = infra.tolerance_particle;

    amrex::Real diff = tolerance + 1.0;
    int iter_counter = 0;
    int max_iter = 20;
    while ((diff > tolerance) && (iter_counter < max_iter)) {
        iter_counter += 1;
        newXiter = newX;
        newViter = newV;
        // Update the position
        for (unsigned int d=0; d<GEMPIC_SPACEDIM; ++d)
        {
            if (d < vdim)
                newX[d] = oldX[d] + dt*0.5*( newV[d] + oldV[d] );
        }
        // Compute xMean
        amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> xMean;
        for (unsigned int d=0; d<GEMPIC_SPACEDIM; ++d)
        {
            if (d<vdim)
                xMean[d] = mod_dom(0.5*(newX[d]+oldX[d]),infra.geom.ProbHi()[d],infra.geom.ProbLo()[d]);
        }
        // evaluate B at xMean
        amrex::GpuArray<amrex::Real,int(vdim/2.5)*2+1> Bp = evaluate_fem_bfield<vdim,degx,degy,degz>(xMean,
                                                                                                       infra.dxi,
                                                                                                       infra.plo,
                                                                                                       bArray);
        if  (vdim == 2) {
            newV[0] = oldV[0] + dt*chargemass*( 0.5*(oldV[1]+newViter[1])*Bp[0]);
            newV[1] = oldV[1] + dt*chargemass*( 0.5*(oldV[0]+newViter[0])*Bp[0]);
        } else if (vdim == 3) {
            newV[0] = p.rdata(0) + dt*chargemass*(0.5*(oldV[1]+newViter[1])*Bp[2]-0.5*(oldV[2]+newViter[2])*Bp[1]);
            newV[1] = p.rdata(1) + dt*chargemass*(0.5*(oldV[2]+newViter[2])*Bp[0]-0.5*(oldV[0]+newViter[0])*Bp[2]);
            newV[2] = p.rdata(2) + dt*chargemass*(0.5*(oldV[0]+newViter[0])*Bp[1]-0.5*(oldV[1]+newViter[1])*Bp[0]);
        }
        diff = 0.0;
        for (unsigned d=0;d<GEMPIC_SPACEDIM;++d)
            diff = std::max(diff, std::abs(newXiter[d] - newX[d]));
        for (unsigned d=0;d<vdim;++d)
            diff = std::max(diff, std::abs(newViter[d] - newV[d]));
    }
    if (iter_counter == max_iter)
        amrex::Print() << " Number of itertions: " << iter_counter << " " << diff << std::endl;
    amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM+vdim> newPos;
    for (unsigned int d=0;d<GEMPIC_SPACEDIM;++d)
        newPos[d] = newX[d];
    for (unsigned int d=0;d<vdim;++d)
        newPos[GEMPIC_SPACEDIM+d]= newV[d];
    return(newPos);
}



template <typename P, int vdim>
amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM+vdim> push_particle(P const& p,
                                                           amrex::Real dt,
                                                           amrex::Real chargemass, //charge/mass
                                                           amrex::GpuArray<amrex::Real,vdim> Ep,
                                                           amrex::GpuArray<amrex::Real,int(vdim/2.5)*2+1> Bp,
                                                           CompDom::computational_domain infra){

    amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM+vdim> newPos;
    // new positions (modulo the upper bound of the box)
    newPos[0] = mod_dom(p.pos(0) + p.rdata(0)*dt, infra.geom.ProbHi()[0], infra.geom.ProbLo()[0]);
#if (GEMPIC_SPACEDIM > 1)
    newPos[1] = mod_dom(p.pos(1) + p.rdata(1)*dt, infra.geom.ProbHi()[1], infra.geom.ProbLo()[1]);
#endif
#if (GEMPIC_SPACEDIM > 2)
    newPos[2] = mod_dom(p.pos(2) + p.rdata(2)*dt, infra.geom.ProbHi()[2], infra.geom.ProbLo()[2]);
#endif

    if (vdim == 1) {
        newPos[1] = p.rdata(0) + dt*chargemass*(Ep[0]);
    } else if (vdim == 2) {
        newPos[2] = p.rdata(0) + dt*chargemass*(Ep[0]+p.rdata(1)*Bp[0]);
        newPos[3] = p.rdata(1) + dt*chargemass*(Ep[1]+p.rdata(0)*Bp[0]);
    } else if (vdim == 3) {
        newPos[3] = p.rdata(0) + dt*chargemass*(Ep[0]+(p.rdata(1)*Bp[2]-p.rdata(2)*Bp[1]));
        newPos[4] = p.rdata(1) + dt*chargemass*(Ep[1]+(p.rdata(2)*Bp[0]-p.rdata(0)*Bp[2]));
        newPos[5] = p.rdata(2) + dt*chargemass*(Ep[2]+(p.rdata(0)*Bp[1]-p.rdata(1)*Bp[0]));
    }
    return(newPos);
}

template <typename P, int vdim>
amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM+vdim> push_particle_pos(P const& p,
                                                               amrex::Real dt,
                                                               CompDom::computational_domain infra){

    amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM+vdim> newPos;
    // new positions (modulo the upper bound of the box)
    newPos[0] = mod_dom(p.pos(0) + p.rdata(0)*dt, infra.geom.ProbHi()[0], infra.geom.ProbLo()[0]);
    newPos[GEMPIC_SPACEDIM] = p.rdata(0);
#if (GEMPIC_SPACEDIM > 1)
    newPos[1] = mod_dom(p.pos(1) + p.rdata(1)*dt, infra.geom.ProbHi()[1], infra.geom.ProbLo()[1]);
#endif
#if (GEMPIC_SPACEDIM > 2)
    newPos[2] = mod_dom(p.pos(2) + p.rdata(2)*dt, infra.geom.ProbHi()[2], infra.geom.ProbLo()[2]);
#endif
    if (vdim > 1) {
        newPos[GEMPIC_SPACEDIM+1] = p.rdata(1);
    }
    if (vdim > 2) {
        newPos[GEMPIC_SPACEDIM+2] = p.rdata(2);
    }
    return(newPos);
}

template <typename P, int vdim>
amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM+vdim> push_particle_boris(P const& p,
                                                                 amrex::Real dt,
                                                                 amrex::Real chargemass, //charge/mass
                                                                 amrex::GpuArray<amrex::Real,vdim> Ep,
                                                                 amrex::GpuArray<amrex::Real,int(vdim/2.5)*2+1> Bp,
                                                                 CompDom::computational_domain infra){

    amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM+vdim> newPos;
    amrex::GpuArray<amrex::Real,vdim> vm;
    amrex::GpuArray<amrex::Real,vdim> vt;
    amrex::GpuArray<amrex::Real,vdim> vp;
    amrex::Real S;

    // v-
    for (int vd=0;vd<vdim;vd++) {
        vm[vd] = p.rdata(vd) + dt*chargemass/2*Ep[vd];
    }

    if (vdim == 1) {
        vp[0] = vm[0];
    } else if (vdim == 2) {
        amrex::Real alpha = chargemass*dt/2*Bp[0];
        vp[0] = (1-pow(alpha,2))/(1+pow(alpha,2))*vm[0] + 2*alpha/(1+pow(alpha,2))*vm[1];
        vp[1] = -2*alpha/(1+pow(alpha,2))*vm[0] + (1-pow(alpha,2))/(1+pow(alpha,2))*vm[1];
    } else if (vdim == 3) {
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
    }

    // v
    newPos[GEMPIC_SPACEDIM] = vp[0] + dt*chargemass/2*Ep[0];
    if (vdim > 1) {
        newPos[GEMPIC_SPACEDIM+1] = vp[1] + dt*chargemass/2*Ep[1];
    }
    if (vdim > 2) {
        newPos[GEMPIC_SPACEDIM+2] = vp[2] + dt*chargemass/2*Ep[2];
    }

    // new positions (modulo the upper bound of the box)
    newPos[0] = mod_dom(p.pos(0) + newPos[GEMPIC_SPACEDIM]*dt, infra.geom.ProbHi()[0], infra.geom.ProbLo()[0]);
#if (GEMPIC_SPACEDIM > 1)
    newPos[1] = mod_dom(p.pos(1) + newPos[GEMPIC_SPACEDIM+1]*dt, infra.geom.ProbHi()[1], infra.geom.ProbLo()[1]);
#endif
#if (GEMPIC_SPACEDIM > 2)
    newPos[2] = mod_dom(p.pos(2) + newPos[GEMPIC_SPACEDIM+2]*dt, infra.geom.ProbHi()[2], infra.geom.ProbLo()[2]);
#endif


    return(newPos);
}

template<int vdim, int degx, int degy, int degz>
void gempic_deposit_magnetization (amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> position,
                                   amrex::Real mu,
                                   amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM+1> const& dxi,
                                   amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> const& plo,
                                   amrex::GpuArray<amrex::Array4<amrex::Real>,int(vdim/2.5)*2+1> bArray,
                                   amrex::GpuArray<amrex::Array4<amrex::Real>,vdim> mArray)

{
    // 1. Identify the box and relative position of position old
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position_box;
    amrex::GpuArray<int, GEMPIC_SPACEDIM> box;
    for (unsigned int d=0; d<GEMPIC_SPACEDIM;++d)
    {
        amrex::Real pos_norm = (position[d]-plo[d])*dxi[d];
        box[d] = std::floor(pos_norm);
        position_box[d] = pos_norm - box[d];
    }

    // 2. Evaluate the splines at the particle position
    Gempic::Particles::splines spl({AMREX_D_DECL(false, false, false)}, {AMREX_D_DECL(false, false, false)});

    std::vector<amrex::Real> sx_cell = spl.spline_eval_vec (spl.shifted_x(position_box[0], degx-1, true), degx-1);
    std::vector<amrex::Real> sx_node = spl.spline_eval_vec (spl.shifted_x(position_box[0], degx, false), degx);
#if (GEMPIC_SPACEDIM > 1)
    std::vector<amrex::Real> sy_cell = spl.spline_eval_vec (spl.shifted_x(position_box[1], degy-1, true), degy-1);
    std::vector<amrex::Real> sy_node = spl.spline_eval_vec (spl.shifted_x(position_box[1], degy, false), degy);
#endif
#if (GEMPIC_SPACEDIM > 2)
    std::vector<amrex::Real> sz_cell = spl.spline_eval_vec (spl.shifted_x(position_box[2], degz-1, true), degz-1);
    std::vector<amrex::Real> sz_node = spl.spline_eval_vec (spl.shifted_x(position_box[2], degz, false), degz);
#endif

    // 3. Provide the magnetic field and other parts of the weighting factor
    // Evaluate the magnetic field
    amrex::GpuArray<amrex::Real, int(vdim/2.5)*2+1> bfield = Gempic::Particles::evaluate_fem_bfield(position, dxi, plo, bArray);
    // Compute modulus of magnetic field
    amrex::Real bfield_mod = 0.0;
    for (int d =0;d<int(vdim/2.5)*2+1;++d )
        bfield_mod += bfield[d]*bfield[d];
    bfield_mod = std::sqrt( bfield_mod );

    // weight is given by mu devided by the bfield modulus times the inverse volume
    amrex::Real weight = mu*dxi[GEMPIC_SPACEDIM]/bfield_mod;

    // 4. Loop of the contributions to the magnetization vector
#if (GEMPIC_SPACEDIM > 2)
    for (int kk = 0; kk <= degz; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0; jj <= degy; ++jj) {
#endif
            for (int ii = 0; ii <= degz; ++ii) {
                // Compute cell index. TODO: Make this work for order higher than 1 when the index needs to be shifted somehow
                amrex::GpuArray<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(box[0]+ii, box[1]+jj, box[2]+kk)};
                if ( ii < degx ) {
                    double s_coeff = sx_node[ii]
        #if(GEMPIC_SPACEDIM > 1)
                            *sy_cell[jj]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                            *sz_cell[kk]
        #endif
                            ;
#if (GEMPIC_SPACEDIM == 1)
                amrex::HostDevice::Atomic::Add(mArray[0].ptr(cell_index[0], 0, 0, 0), s_coeff*bfield[0]*weight);
#else
                amrex::HostDevice::Atomic::Add(mArray[0].ptr(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0), s_coeff*bfield[0]*weight);
#endif
                }
                if(vdim > 1) {
#if (GEMPIC_SPACEDIM > 1)
                    if ( jj < degy ) {
                        double s_coeff = sx_node[ii]
                                *sy_node[jj]
        #if(GEMPIC_SPACEDIM > 2)
                                *sz_node[kk]
        #endif
                                ;
#else
                    double s_coeff = sx_cell[ii];
#endif

                    amrex::HostDevice::Atomic::Add(mArray[1].ptr(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0), s_coeff*bfield[1]*weight);
#if (GEMPIC_SPACEDIM > 1)
                }
#endif
            }
            if(vdim > 2) {

#if(GEMPIC_SPACEDIM > 2)
                if ( kk < degz ) {
                    double s_coeff = sx_cell[ii]
                            *sy_cell[jj]
                            *sz_node[kk];
#else
                double s_coeff = sx_cell[ii]
        #if (GEMPIC_SPACEDIM > 1)
                        *sy_cell[jj]
        #endif
                        ;
#endif
#if (GEMPIC_SPACEDIM == 1)
                amrex::HostDevice::Atomic::Add(mArray[2].ptr(cell_index[0], 0, 0, 0), s_coeff*bfield[2]*weight);
#else
                amrex::HostDevice::Atomic::Add(mArray[2].ptr(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0), s_coeff*bfield[2]*weight);
#endif

            }
        }
#if (GEMPIC_SPACEDIM > 1)
    }
#endif
#if (GEMPIC_SPACEDIM > 2)
} }
#endif

}


/*!
     *  Field evluation of a two form described by finite elements
     */
template <int vdim, int degx, int degy, int degz>
amrex::GpuArray<amrex::Real, int(vdim/2.5)*2+1> evaluate_grad_mod_bfield(amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> position,
                                                                    amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM+1> const& dxi,
                                                                    amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> const& plo,
                                                                           amrex::GpuArray<amrex::Array4<amrex::Real>, int(vdim/2.5)*2+1> bArray)
{
    amrex::GpuArray<amrex::Real, int(vdim/2.5)*2+1> fields;
    switch (int(vdim/2.5)*2+1) {
    case 1:
        fields = {0.};
        break;
    case 3:
        fields = {0.,0.,0.};
        break;
    }
    splines spl({AMREX_D_DECL(false, false, false)}, {AMREX_D_DECL(false, false, false)});

    amrex::Real lx = (position[0]-plo[0])*dxi[0];
    int i = std::floor(lx);
    amrex::Real xint = lx - i;
    std::vector<amrex::Real> sx_cell = spl.spline_eval_vec (spl.shifted_x(xint, degx-1, true), degx-1);
    std::vector<amrex::Real> sx_node = spl.spline_eval_vec (spl.shifted_x(xint, degx, false), degx);
#if (GEMPIC_SPACEDIM > 1)
    amrex::Real ly = (position[1]-plo[1])*dxi[1];
    int j = std::floor(ly);
    amrex::Real yint = ly - j;
    std::vector<amrex::Real> sy_cell = spl.spline_eval_vec (spl.shifted_x(yint, degy-1, true), degy-1);
    std::vector<amrex::Real> sy_node = spl.spline_eval_vec (spl.shifted_x(yint, degy, false), degy);
#endif
#if (GEMPIC_SPACEDIM > 2)
    amrex::Real lz = (position[2]-plo[2])*dxi[2];
    int k = std::floor(lz);
    amrex::Real zint = lz - k;
    std::vector<amrex::Real> sz_cell = spl.spline_eval_vec (spl.shifted_x(zint, degz-1, true), degz-1);
    std::vector<amrex::Real> sz_node = spl.spline_eval_vec (spl.shifted_x(zint, degz, false), degz);
#endif

    // First B_0
#if (GEMPIC_SPACEDIM > 2)
    for (int kk = 0; kk <= degz-1; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0; jj <= degy-1; ++jj) {
#endif
            for (int ii = 0; ii <= degx; ++ii) {
                // Compute cell index. TODO: Make this work for order higher than 1 when the index needs to be shifted somehow
                amrex::GpuArray<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(i+ii, j+jj, k+kk)};
                double s_coeff = sx_node[ii]
        #if(GEMPIC_SPACEDIM > 1)
                        *sy_cell[jj]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                        *sz_cell[kk]
        #endif
                        ;
#if (GEMPIC_SPACEDIM == 1)
                fields[0] += s_coeff*(bArray[0])(cell_index[0], 0, 0, 0);
#else
                fields[0] += s_coeff*(bArray[0])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0);
#endif
            }
#if (GEMPIC_SPACEDIM > 1)
        }
#endif
#if (GEMPIC_SPACEDIM > 2)
    }
#endif
    // B_1
    if (int(vdim/2.5)*2+1>1) {
#if (GEMPIC_SPACEDIM > 2)
        for (int kk = 0; kk <= degz-1; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
            for (int jj = 0; jj <= degy; ++jj) {
#endif
                for (int ii = 0; ii <= degx-1; ++ii) {
                    // Compute cell index. TODO: Make this work for order higher than 1 when the index needs to be shifted somehow
                    amrex::GpuArray<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(i+ii, j+jj, k+kk)};
                    double s_coeff = sx_cell[ii]
        #if(GEMPIC_SPACEDIM > 1)
                            *sy_node[jj]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                            *sz_cell[kk]
        #endif
                            ;
#if (GEMPIC_SPACEDIM == 1)
                fields[1] += s_coeff*(bArray[1])(cell_index[0], 0, 0, 0);
#else
                fields[1] += s_coeff*(bArray[1])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0);
#endif
                }
#if (GEMPIC_SPACEDIM > 1)
            }
#endif
#if (GEMPIC_SPACEDIM > 2)
        }
#endif
    }

    // B_2
    if (int(vdim/2.5)*2+1>2) {
#if (GEMPIC_SPACEDIM > 2)
        for (int kk = 0; kk <= degz; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
            for (int jj = 0; jj <= degy-1; ++jj) {
#endif
                for (int ii = 0; ii <= degx-1; ++ii) {
                    // Compute cell index. TODO: Make this work for order higher than 1 when the index needs to be shifted somehow
                    amrex::GpuArray<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(i+ii, j+jj, k+kk)};
                    double s_coeff = sx_cell[ii]
        #if(GEMPIC_SPACEDIM > 1)
                            *sy_cell[jj]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                            *sz_node[kk]
        #endif
                            ;
#if (GEMPIC_SPACEDIM == 1)
                fields[2] += s_coeff*(bArray[2])(cell_index[0], 0, 0, 0);
#else
                fields[2] += s_coeff*(bArray[2])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0);
#endif
                }
#if (GEMPIC_SPACEDIM > 1)
            }
#endif
#if (GEMPIC_SPACEDIM > 2)
        }
#endif
    }
    return(fields);
}


} // end namespace particles
} // end namespace GEMPIC

#endif
