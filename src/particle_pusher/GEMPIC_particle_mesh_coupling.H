/*------------------------------------------------------------------------------
 functions copied from the AMReX library to make small changes
 AMReX file containing original functions: AMReX_Particle_mod_K.H
 (amrex/Src/Particle/AMReX_Particle_mod_K.H) pulled on 07.10.2019
------------------------------------------------------------------------------*/

#ifndef PARTICLE_MESH_COUPLING_H_
#define PARTICLE_MESH_COUPLING_H_

#include <cmath>

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
    std::vector<amrex::Real> sx_node = spl.spline_eval_vec (spl.shifted_x(xint, GEMPIC_DEG_X, false), GEMPIC_DEG_X);
#if (GEMPIC_SPACEDIM > 1)
    amrex::Real ly = (p.pos(1) - plo[1]) * dxi[1];
    int j = std::floor(ly);
    amrex::Real yint = ly - j;
    std::vector<amrex::Real> sy_node = spl.spline_eval_vec (spl.shifted_x(yint, GEMPIC_DEG_Y, false), GEMPIC_DEG_Y);
#endif
#if (GEMPIC_SPACEDIM > 2)
    amrex::Real lz = (p.pos(2) - plo[2]) * dxi[2];
    int k = std::floor(lz);
    amrex::Real zint = lz - k;
    std::vector<amrex::Real> sz_node = spl.spline_eval_vec (spl.shifted_x(zint, GEMPIC_DEG_Z, false), GEMPIC_DEG_Z);
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
                std::array<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(spl.compute_cell_index(i+ii, xint, GEMPIC_DEG_X, false),
                                                              spl.compute_cell_index(j+jj, yint, GEMPIC_DEG_Y, false),
                                                              spl.compute_cell_index(k+kk, zint,  GEMPIC_DEG_Z, false))};

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


/*!
     *  Evaluation of a FEM one form
     */
template<typename P>
std::array<amrex::Real, GEMPIC_VDIM> evaluate_fem_efield(P const& p,
                                                         std::array<amrex::Real,GEMPIC_SPACEDIM+1> const& dxi,
                                                         std::array<amrex::Real,GEMPIC_SPACEDIM> const& plo,
                                                         std::array<amrex::Array4<amrex::Real>,GEMPIC_VDIM> eArray)
{
    std::array<amrex::Real, GEMPIC_VDIM> fields = {AMREX_V_DECL(0.,0.,0.)};
    splines spl({AMREX_D_DECL(false, false, false)}, {AMREX_D_DECL(false, false, false)});

    amrex::Real lx = (p.pos(0)-plo[0])*dxi[0];
    int i = std::floor(lx);
    amrex::Real xint = lx - i;
    std::vector<amrex::Real> sx_cell = spl.spline_eval_vec (xint, GEMPIC_DEG_X-1);
    std::vector<amrex::Real> sx_node = spl.spline_eval_vec (xint, GEMPIC_DEG_X);
#if (GEMPIC_SPACEDIM > 1)
    amrex::Real ly = (p.pos(1)-plo[1])*dxi[1];
    int j = std::floor(ly);
    amrex::Real yint = ly - j;
    std::vector<amrex::Real> sy_cell = spl.spline_eval_vec (yint, GEMPIC_DEG_Y-1);
    std::vector<amrex::Real> sy_node = spl.spline_eval_vec (yint, GEMPIC_DEG_Y);
#endif
#if (GEMPIC_SPACEDIM > 2)
    amrex::Real lz = (p.pos(2)-plo[2])*dxi[2];
    int k = std::floor(lz);
    amrex::Real zint = lz - k;
    std::vector<amrex::Real> sz_cell = spl.spline_eval_vec (zint, GEMPIC_DEG_Z-1);
    std::vector<amrex::Real> sz_node = spl.spline_eval_vec (zint, GEMPIC_DEG_Z);
#endif

#if (GEMPIC_SPACEDIM > 2)
    for (int kk = 0; kk <= GEMPIC_DEG_Z; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0; jj <= GEMPIC_DEG_Y; ++jj) {
#endif
            for (int ii = 0; ii <= GEMPIC_DEG_X; ++ii) {
                // Compute cell index. TODO: Make this work for order higher than 1 when the index needs to be shifted somehow
                std::array<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(i+ii, j+jj, k+kk)};
                if ( ii < GEMPIC_DEG_X ) {
                    double s_coeff = sx_cell[ii]
        #if(GEMPIC_SPACEDIM > 1)
                            *sy_node[jj]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                            *sz_node[kk]
        #endif
                            ;
                    fields[0] += s_coeff*(eArray[0])(cell_index[0]
        #if(GEMPIC_SPACEDIM > 1)
                            ,cell_index[1]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                            ,cell_index[2]
        #endif
                            ,0);
                }
#if(GEMPIC_VDIM > 1)
                if ( jj < GEMPIC_DEG_Y ) {
                    double s_coeff = sx_node[ii]
                            *sy_cell[jj]
        #if(GEMPIC_SPACEDIM > 2)
                            *sz_node[kk]
        #endif
                            ;
                    fields[1] += s_coeff*(eArray[1])(cell_index[0]
        #if(GEMPIC_SPACEDIM > 1)
                            ,cell_index[1]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                            ,cell_index[2]
        #endif
                            ,0);
                }
#endif
#if(GEMPIC_VDIM > 2)
                if ( kk < GEMPIC_DEG_Z ) {
                    double s_coeff = sx_node[ii]
                            *sy_node[jj]
                            *sz_cell[kk]
                            ;
                    fields[2] += s_coeff*(eArray[2])(cell_index[0]
        #if(GEMPIC_SPACEDIM > 1)
                            ,cell_index[1]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                            ,cell_index[2]
        #endif
                            ,0);
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
void accumulate_j_fem_integral_point(std::array<amrex::Real,GEMPIC_SPACEDIM> position_box,
                                     std::array<int,GEMPIC_SPACEDIM> box,
                                     std::array<amrex::Real, GEMPIC_VDIM>  weight,
                                     std::array<amrex::Array4<amrex::Real>,GEMPIC_VDIM> jArray)
{
    splines spl({AMREX_D_DECL(false, false, false)}, {AMREX_D_DECL(false, false, false)});

    std::vector<amrex::Real> sx_cell = spl.spline_eval_vec (position_box[0], GEMPIC_DEG_X-1);
    std::vector<amrex::Real> sx_node = spl.spline_eval_vec (position_box[0], GEMPIC_DEG_X);
#if (GEMPIC_SPACEDIM > 1)
    std::vector<amrex::Real> sy_cell = spl.spline_eval_vec (position_box[1], GEMPIC_DEG_Y-1);
    std::vector<amrex::Real> sy_node = spl.spline_eval_vec (position_box[1], GEMPIC_DEG_Y);
#endif
#if (GEMPIC_SPACEDIM > 2)
    std::vector<amrex::Real> sz_cell = spl.spline_eval_vec (position_box[2], GEMPIC_DEG_Z-1);
    std::vector<amrex::Real> sz_node = spl.spline_eval_vec (position_box[2], GEMPIC_DEG_Z);
#endif


#if (GEMPIC_SPACEDIM > 2)
    for (int kk = 0; kk <= GEMPIC_DEG_Z; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0; jj <= GEMPIC_DEG_Y; ++jj) {
#endif
            for (int ii = 0; ii <= GEMPIC_DEG_X; ++ii) {
                // Compute cell index. TODO: Make this work for order higher than 1 when the index needs to be shifted somehow
                std::array<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(box[0]+ii, box[1]+jj, box[2]+kk)};
                if ( ii < GEMPIC_DEG_X ) {
                    double s_coeff = sx_cell[ii]
        #if(GEMPIC_SPACEDIM > 1)
                            *sy_node[jj]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                            *sz_node[kk]
        #endif
                            ;
                    amrex::Gpu::Atomic::Add(&(jArray[0])(cell_index[0]
        #if (GEMPIC_SPACEDIM > 1)
                            , cell_index[1]
        #endif
        #if (GEMPIC_SPACEDIM > 2)
                            , cell_index[2]
        #endif
                            , 0), s_coeff*weight[0]);
                }
#if(GEMPIC_VDIM > 1)
                if ( jj < GEMPIC_DEG_Y ) {
                    double s_coeff = sx_node[ii]
                            *sy_cell[jj]
        #if(GEMPIC_SPACEDIM > 2)
                            *sz_node[kk]
        #endif
                            ;
                    amrex::Gpu::Atomic::Add(&(jArray[1])(cell_index[0]
        #if (GEMPIC_SPACEDIM > 1)
                            , cell_index[1]
        #endif
        #if (GEMPIC_SPACEDIM > 2)
                            , cell_index[2]
        #endif
                            , 0), s_coeff*weight[1]);
                }
#endif
#if(GEMPIC_VDIM > 2)
                if ( kk < GEMPIC_DEG_Z ) {
                    double s_coeff = sx_node[ii]
                            *sy_node[jj]
                            *sz_cell[kk]
                            ;
                    amrex::Gpu::Atomic::Add(&(jArray[2])(cell_index[0]
        #if (GEMPIC_SPACEDIM > 1)
                            , cell_index[1]
        #endif
        #if (GEMPIC_SPACEDIM > 2)
                            , cell_index[2]
        #endif
                            , 0), s_coeff*weight[2]);
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
     *  Accumulation of the current for a FEM solver with integral over time (line integral)
     *  Currently this is implemented for Gauss-Quadrature with only one point
     */
void accumulate_j_fem_integral(std::array<amrex::Real,GEMPIC_SPACEDIM> position_old,
                               std::array<amrex::Real,GEMPIC_SPACEDIM> position_new,
                               std::array<amrex::Real, GEMPIC_VDIM> velocity,
                               amrex::Real charge, // q*p.rdata(GEMPIC_VDIM)
                               std::array<amrex::Real,GEMPIC_SPACEDIM+1> const& dxi,
                               std::array<amrex::Real,GEMPIC_SPACEDIM> const& plo,
                               std::array<amrex::Array4<amrex::Real>,GEMPIC_VDIM> jArray)
{
  // The quadrature formula TODO: Make this part of the class and generic for variable degree
  const int n_quad_points = 2;
  std::array<amrex::Real, n_quad_points> q_points = {(1.-1./std::sqrt(3.))*0.5, (1.+1./std::sqrt(3.))*0.5};
  std::array<amrex::Real, n_quad_points> q_weights = {0.5, 0.5};

  // 1a. Identify the box and relative position of position old
  std::array<amrex::Real, GEMPIC_SPACEDIM> position_box_old;
  std::array<int, GEMPIC_SPACEDIM> box_old;
  for (unsigned int d=0; d<GEMPIC_SPACEDIM;++d)
    {
      amrex::Real pos_norm = (position_old[d]-plo[d])*dxi[d];
      box_old[d] = std::floor(pos_norm);
      position_box_old[d] = pos_norm - box_old[d];
    }
  // 1b. Identify the box and relative position of position new
  std::array<amrex::Real, GEMPIC_SPACEDIM> position_box_new;
  std::array<int, GEMPIC_SPACEDIM> box_new;
  for (unsigned int d=0; d<GEMPIC_SPACEDIM;++d)
    {
      amrex::Real pos_norm = (position_new[d]-plo[d])*dxi[d];
      box_new[d] = std::floor(pos_norm);
      position_box_new[d] = pos_norm - box_new[d];
    }

  std::array<int, GEMPIC_SPACEDIM> boxdiff;
  std::array<amrex::Real, GEMPIC_SPACEDIM> xnewtilde;
  int max_box = 0;
  for (unsigned int d=0;d<GEMPIC_SPACEDIM;++d)
    {
      boxdiff[d] = box_new[d]-box_old[d];
      xnewtilde[d] = position_box_new[d]+boxdiff[d];
      if (std::abs(boxdiff[d])> max_box)
        max_box = std::abs(boxdiff[d]);
    }

  max_box = max_box+1;

  std::array<int,GEMPIC_SPACEDIM> box = box_old; // We start in the old box

  std::array<int,GEMPIC_SPACEDIM>  sigma_counter = {AMREX_D_DECL(0, 0, 0)};

  std::array<std::vector<amrex::Real>, GEMPIC_SPACEDIM> sigmas;
  std::array<int, GEMPIC_SPACEDIM> increment;
  std::array<amrex::Real, GEMPIC_SPACEDIM> sigma_next;

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
        std::array<amrex::Real, GEMPIC_SPACEDIM> position_box;
        for (unsigned int d=0;d<GEMPIC_SPACEDIM;++d)
          {
            position_box[d]= position_box_old[d]* (1.0 - sigma) + xnewtilde[d] * sigma  - sigma_counter[d]*increment[d];
          }
        AMREX_ASSERT_WITH_MESSAGE( (*(std::max_element(position_box.begin(), position_box.end())) < 1.0) &
         (*(std::min_element(position_box.begin(), position_box.end())) > 0.0), "Position is outside the interval [0,1].");

        amrex::Real weight = q_weights[q]*(sigma_r-sigma_l)*charge*dxi[GEMPIC_SPACEDIM];
        std::array<amrex::Real, GEMPIC_VDIM> weight_vec;
        for (unsigned int d=0;d<GEMPIC_VDIM;++d)
          weight_vec[d] = velocity[d]*weight;
        accumulate_j_fem_integral_point(position_box, box, weight_vec, jArray);
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
     *  Accumulation of the current for a FEM solver with integral over time (line integral)
     *  Currently this is implemented for Gauss-Quadrature with only one point
     */
void accumulate_j_fem_integral(std::array<amrex::Real,GEMPIC_SPACEDIM> position,
                               std::array<amrex::Real, GEMPIC_VDIM> velocity,
                               amrex::Real charge, // q*p.rdata(GEMPIC_VDIM)
                               std::array<amrex::Real,GEMPIC_SPACEDIM+1> const& dxi,
                               std::array<amrex::Real,GEMPIC_SPACEDIM> const& plo,
                               std::array<amrex::Array4<amrex::Real>,GEMPIC_VDIM> jArray)
{
    splines spl({AMREX_D_DECL(false, false, false)}, {AMREX_D_DECL(false, false, false)});

    amrex::Real lx = (position[0]-plo[0])*dxi[0];
    int i = std::floor(lx);
    amrex::Real xint = lx - i;
    std::vector<amrex::Real> sx_cell = spl.spline_eval_vec (xint, GEMPIC_DEG_X-1);
    std::vector<amrex::Real> sx_node = spl.spline_eval_vec (xint, GEMPIC_DEG_X);
#if (GEMPIC_SPACEDIM > 1)
    amrex::Real ly = (position[1]-plo[1])*dxi[1];
    int j = std::floor(ly);
    amrex::Real yint = ly - j;
    std::vector<amrex::Real> sy_cell = spl.spline_eval_vec (yint, GEMPIC_DEG_Y-1);
    std::vector<amrex::Real> sy_node = spl.spline_eval_vec (yint, GEMPIC_DEG_Y);
#endif
#if (GEMPIC_SPACEDIM > 2)
    amrex::Real lz = (position[2]-plo[2])*dxi[2];
    int k = std::floor(lz);
    amrex::Real zint = lz - k;
    std::vector<amrex::Real> sz_cell = spl.spline_eval_vec (zint, GEMPIC_DEG_Z-1);
    std::vector<amrex::Real> sz_node = spl.spline_eval_vec (zint, GEMPIC_DEG_Z);
#endif
    amrex::Real particle_coeff = dxi[GEMPIC_SPACEDIM]*charge;


#if (GEMPIC_SPACEDIM > 2)
    for (int kk = 0; kk <= GEMPIC_DEG_Z; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0; jj <= GEMPIC_DEG_Y; ++jj) {
#endif
            for (int ii = 0; ii <= GEMPIC_DEG_X; ++ii) {
                // Compute cell index. TODO: Make this work for order higher than 1 when the index needs to be shifted somehow
                std::array<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(i+ii, j+jj, k+kk)};
                if ( ii < GEMPIC_DEG_X ) {
                    double s_coeff = sx_cell[ii]
        #if(GEMPIC_SPACEDIM > 1)
                            *sy_node[jj]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                            *sz_node[kk]
        #endif
                            ;
                    amrex::Gpu::Atomic::Add(&(jArray[0])(cell_index[0]
        #if (GEMPIC_SPACEDIM > 1)
                            , cell_index[1]
        #endif
        #if (GEMPIC_SPACEDIM > 2)
                            , cell_index[2]
        #endif
                            , 0), s_coeff*particle_coeff*velocity[0]);
                }
#if(GEMPIC_VDIM > 1)
                if ( jj < GEMPIC_DEG_Y ) {
                    double s_coeff = sx_node[ii]
                            *sy_cell[jj]
        #if(GEMPIC_SPACEDIM > 2)
                            *sz_node[kk]
        #endif
                            ;
                    amrex::Gpu::Atomic::Add(&(jArray[1])(cell_index[0]
        #if (GEMPIC_SPACEDIM > 1)
                            , cell_index[1]
        #endif
        #if (GEMPIC_SPACEDIM > 2)
                            , cell_index[2]
        #endif
                            , 0), s_coeff*particle_coeff*velocity[1]);
                }
#endif
#if(GEMPIC_VDIM > 2)
                if ( kk < GEMPIC_DEG_Z ) {
                    double s_coeff = sx_node[ii]
                            *sy_node[jj]
                            *sz_cell[kk]
                            ;
                    amrex::Gpu::Atomic::Add(&(jArray[2])(cell_index[0]
        #if (GEMPIC_SPACEDIM > 1)
                            , cell_index[1]
        #endif
        #if (GEMPIC_SPACEDIM > 2)
                            , cell_index[2]
        #endif
                            , 0), s_coeff*particle_coeff*velocity[2]);
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
std::array<amrex::Real, GEMPIC_BDIM> evaluate_fem_bfield(std::array<amrex::Real,GEMPIC_SPACEDIM> position,
                                                         std::array<amrex::Real,GEMPIC_SPACEDIM+1> const& dxi,
                                                         std::array<amrex::Real,GEMPIC_SPACEDIM> const& plo,
                                                         std::array<amrex::Array4<amrex::Real>,GEMPIC_BDIM> bArray)
{
    std::array<amrex::Real, GEMPIC_BDIM> fields = {AMREX_B_DECL(0.,0.,0.)};
    splines spl({AMREX_D_DECL(false, false, false)}, {AMREX_D_DECL(false, false, false)});

    amrex::Real lx = (position[0]-plo[0])*dxi[0];
    int i = std::floor(lx);
    amrex::Real xint = lx - i;
    std::vector<amrex::Real> sx_cell = spl.spline_eval_vec (xint, GEMPIC_DEG_X-1);
    std::vector<amrex::Real> sx_node = spl.spline_eval_vec (xint, GEMPIC_DEG_X);
#if (GEMPIC_SPACEDIM > 1)
    amrex::Real ly = (position[1]-plo[1])*dxi[1];
    int j = std::floor(ly);
    amrex::Real yint = ly - j;
    std::vector<amrex::Real> sy_cell = spl.spline_eval_vec (yint, GEMPIC_DEG_Y-1);
    std::vector<amrex::Real> sy_node = spl.spline_eval_vec (yint, GEMPIC_DEG_Y);
#endif
#if (GEMPIC_SPACEDIM > 2)
    amrex::Real lz = (position[2]-plo[2])*dxi[2];
    int k = std::floor(lz);
    amrex::Real zint = lz - k;
    std::vector<amrex::Real> sz_cell = spl.spline_eval_vec (zint, GEMPIC_DEG_Z-1);
    std::vector<amrex::Real> sz_node = spl.spline_eval_vec (zint, GEMPIC_DEG_Z);
#endif

    // First B_0
#if (GEMPIC_SPACEDIM > 2)
    for (int kk = 0; kk <= GEMPIC_DEG_Z-1; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0; jj <= GEMPIC_DEG_Y-1; ++jj) {
#endif
            for (int ii = 0; ii <= GEMPIC_DEG_X; ++ii) {
                // Compute cell index. TODO: Make this work for order higher than 1 when the index needs to be shifted somehow
                std::array<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(i+ii, j+jj, k+kk)};
                double s_coeff = sx_node[ii]
        #if(GEMPIC_SPACEDIM > 1)
                        *sy_cell[jj]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                        *sz_cell[kk]
        #endif
                        ;
                fields[0] += s_coeff*(bArray[0])(cell_index[0]
        #if(GEMPIC_SPACEDIM > 1)
                        ,cell_index[1]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                        ,cell_index[2]
        #endif
                        ,0);
            }
#if (GEMPIC_SPACEDIM > 1)
        }
#endif
#if (GEMPIC_SPACEDIM > 2)
    }
#endif
    // B_1
#if (GEMPIC_BDIM>1)
#if (GEMPIC_SPACEDIM > 2)
    for (int kk = 0; kk <= GEMPIC_DEG_Z-1; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0; jj <= GEMPIC_DEG_Y; ++jj) {
#endif
            for (int ii = 0; ii <= GEMPIC_DEG_X-1; ++ii) {
                // Compute cell index. TODO: Make this work for order higher than 1 when the index needs to be shifted somehow
                std::array<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(i+ii, j+jj, k+kk)};
                double s_coeff = sx_cell[ii]
        #if(GEMPIC_SPACEDIM > 1)
                        *sy_node[jj]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                        *sz_cell[kk]
        #endif
                        ;
                fields[1] += s_coeff*(bArray[1])(cell_index[0]
        #if(GEMPIC_SPACEDIM > 1)
                        ,cell_index[1]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                        ,cell_index[2]
        #endif
                        ,0);
            }
#if (GEMPIC_SPACEDIM > 1)
        }
#endif
#if (GEMPIC_SPACEDIM > 2)
    }
#endif
#endif

    // B_2
#if (GEMPIC_BDIM>2)
#if (GEMPIC_SPACEDIM > 2)
    for (int kk = 0; kk <= GEMPIC_DEG_Z; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0; jj <= GEMPIC_DEG_Y-1; ++jj) {
#endif
            for (int ii = 0; ii <= GEMPIC_DEG_X-1; ++ii) {
                // Compute cell index. TODO: Make this work for order higher than 1 when the index needs to be shifted somehow
                std::array<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(i+ii, j+jj, k+kk)};
                double s_coeff = sx_cell[ii]
        #if(GEMPIC_SPACEDIM > 1)
                        *sy_cell[jj]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                        *sz_node[kk]
        #endif
                        ;
                fields[2] += s_coeff*(bArray[2])(cell_index[0]
        #if(GEMPIC_SPACEDIM > 1)
                        ,cell_index[1]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                        ,cell_index[2]
        #endif
                        ,0);
            }
#if (GEMPIC_SPACEDIM > 1)
        }
#endif
#if (GEMPIC_SPACEDIM > 2)
    }
#endif
#endif
    return(fields);
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
    std::vector<amrex::Real> sx_node = spl.spline_eval_vec (spl.shifted_x(xint, GEMPIC_DEG_X, false), GEMPIC_DEG_X);
    std::vector<amrex::Real> sx_cell_1 = spl.spline_eval_vec (spl.shifted_x(xint, GEMPIC_DEG_X-1, true), GEMPIC_DEG_X-1);
    std::array<std::vector<amrex::Real>,2> sx_1 = {sx_cell_1, sx_node};
#if (GEMPIC_SPACEDIM > 1)
    amrex::Real ly = (p.pos(1)-plo[1])*dxi[1];
    int j = std::floor(ly);
    amrex::Real yint = ly - j;
    std::vector<amrex::Real> sy_node = spl.spline_eval_vec (spl.shifted_x(yint, GEMPIC_DEG_Y, false), GEMPIC_DEG_Y);
    std::vector<amrex::Real> sy_cell_1 = spl.spline_eval_vec (spl.shifted_x(yint, GEMPIC_DEG_Y-1, true), GEMPIC_DEG_Y-1);
    std::array<std::vector<amrex::Real>,2> sy_1 = {sy_cell_1, sy_node};
#endif
#if (GEMPIC_SPACEDIM > 2)
    amrex::Real lz = (p.pos(2)-plo[2])*dxi[2];
    int k = std::floor(lz);
    amrex::Real zint = lz - k;
    std::vector<amrex::Real> sz_node = spl.spline_eval_vec (spl.shifted_x(zint, GEMPIC_DEG_Z, false), GEMPIC_DEG_Z);
    std::vector<amrex::Real> sz_cell_1 = spl.spline_eval_vec (spl.shifted_x(zint, GEMPIC_DEG_Z-1, true), GEMPIC_DEG_Z-1);
    std::array<std::vector<amrex::Real>,2> sz_1 = {sz_cell_1, sz_node};
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
                    if(!(((kk==GEMPIC_DEG_Z) & (cc%3==2))|((jj==GEMPIC_DEG_Y) & (cc%3==1))|((ii==GEMPIC_DEG_X) & (cc%3==0)))) {
                        double s_coeff = sx_1[cc%3==0?0:1][ii]
                              #if(GEMPIC_SPACEDIM > 1)
                                                      *sy_1[cc%3==1?0:1][jj]
                                            #endif
                                            #if(GEMPIC_SPACEDIM > 2)
                                                                    *sz_1[cc%3==2?0:1][kk]
                                                          #endif
                                                                                  ;
                        std::array<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(spl.compute_cell_index(i+ii, xint, GEMPIC_DEG_X-(cc%3==0?1:0), cc%3==0?true:false),
                                                                      spl.compute_cell_index(j+jj, yint, GEMPIC_DEG_Y-(cc%3==1?1:0), cc%3==1?true:false),
                                                                      spl.compute_cell_index(k+kk, zint, GEMPIC_DEG_Z-(cc%3==2?1:0), cc%3==2?true:false))};

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
                }
                if (field_int) {
                    // deposit in B
                    for (int cc=0;cc<GEMPIC_BDIM;cc++) {
                        if(!(((kk==GEMPIC_DEG_Z) & (cc%3!=2))|((jj==GEMPIC_DEG_Y) & (cc%3!=1))|((ii==GEMPIC_DEG_X) & (cc%3!=0)))) {
                            double s_coeff = sx_1[cc%3==0?1:0][ii]
                              #if(GEMPIC_SPACEDIM > 1)
                                                          *sy_1[cc%3==1?1:0][jj]
                                            #endif
                                            #if(GEMPIC_SPACEDIM > 2)
                                                                        *sz_1[cc%3==2?1:0][kk]
                                                          #endif
                                                                                      ;
                            std::array<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(spl.compute_cell_index(i+ii, xint, GEMPIC_DEG_X-(cc%3==0?0:1), cc%3==0?false:true),
                                                                          spl.compute_cell_index(j+jj, yint, GEMPIC_DEG_Y-(cc%3==1?0:1), cc%3==1?false:true),
                                                                          spl.compute_cell_index(k+kk, zint, GEMPIC_DEG_Z-(cc%3==2?0:1), cc%3==2?false:true))};


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
void gempic_deposit_charge_indextype (P const& p, amrex::Real charge,
                                      std::array<amrex::Real,GEMPIC_SPACEDIM+1> const& dxi,
                                      std::array<amrex::Real,GEMPIC_SPACEDIM> const& plo,
                                      amrex::Array4<amrex::Real> jArray,
                                      amrex::IndexType IndT)

{

    splines spl({AMREX_D_DECL(false, false, false)}, {AMREX_D_DECL(false, false, false)});
    amrex::Real lx = (p.pos(0)-plo[0])*dxi[0];
    int i = std::floor(lx);
    amrex::Real xint = lx - i;
    std::vector<amrex::Real> sx_node = spl.spline_eval_vec (spl.shifted_x(xint, GEMPIC_DEG_X, false), GEMPIC_DEG_X);
    std::vector<amrex::Real> sx_cell_1 = spl.spline_eval_vec (spl.shifted_x(xint, GEMPIC_DEG_X, true), GEMPIC_DEG_X);
    std::array<std::vector<amrex::Real>,2> sx_1 = {sx_cell_1, sx_node};
    amrex::Real ly = (p.pos(1)-plo[1])*dxi[1];
    int j = std::floor(ly);
    amrex::Real yint = ly - j;
    std::vector<amrex::Real> sy_node = spl.spline_eval_vec (spl.shifted_x(yint, GEMPIC_DEG_Y, false), GEMPIC_DEG_Y);
    std::vector<amrex::Real> sy_cell_1 = spl.spline_eval_vec (spl.shifted_x(yint, GEMPIC_DEG_Y, true), GEMPIC_DEG_Y);
    std::array<std::vector<amrex::Real>,2> sy_1 = {sy_cell_1, sy_node};
    amrex::Real lz = (p.pos(2)-plo[2])*dxi[2];
    int k = std::floor(lz);
    amrex::Real zint = lz - k;
    std::vector<amrex::Real> sz_node = spl.spline_eval_vec (spl.shifted_x(zint, GEMPIC_DEG_Z, false), GEMPIC_DEG_Z);
    std::vector<amrex::Real> sz_cell_1 = spl.spline_eval_vec (spl.shifted_x(zint, GEMPIC_DEG_Z, true), GEMPIC_DEG_Z);
    std::array<std::vector<amrex::Real>,2> sz_1 = {sz_cell_1, sz_node};
    amrex::Real particle_coeff = dxi[GEMPIC_SPACEDIM]*charge*p.rdata(GEMPIC_VDIM);

    for (int kk = 0; kk <= GEMPIC_DEG_Z; ++kk) {
        for (int jj = 0; jj <= GEMPIC_DEG_Y; ++jj) {
            for (int ii = 0; ii <= GEMPIC_DEG_X; ++ii) {

                // deposit in J and interpolate in E
                double s_coeff = sx_1[IndT[0]==0?0:1][ii]*sy_1[IndT[1]==0?0:1][jj]*sz_1[IndT[2]==0?0:1][kk]
                                                                                                   ;
                std::array<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(spl.compute_cell_index(i+ii, xint, GEMPIC_DEG_X, IndT[0]==0?true:false),
                                                              spl.compute_cell_index(j+jj, yint, GEMPIC_DEG_Y, IndT[1]==0?true:false),
                                                              spl.compute_cell_index(k+kk, zint, GEMPIC_DEG_Z, IndT[2]==0?true:false))};

                amrex::Gpu::Atomic::Add(&(jArray)(cell_index[0]
                                        , cell_index[1]
                        , cell_index[2]
                        , 0), s_coeff*particle_coeff);


            }
        }
    }
}

/*!
     * Push the velocity with a given electric field
     */
template <typename P>
std::array<amrex::Real, GEMPIC_VDIM> push_v_efield(P const& p,
                                                   amrex::Real dt,
                                                   amrex::Real chargemass, //charge/mass
                                                   std::array<amrex::Real,GEMPIC_VDIM> Ep,
                                                   infrastructure infra){
    std::array<amrex::Real,GEMPIC_VDIM> newPos;

    newPos[0] = p.rdata(0) + dt*chargemass*(Ep[0]);
#if (GEMPIC_VDIM > 1)
    newPos[1] = p.rdata(1) + dt*chargemass*(Ep[1]);
#endif
#if (GEMPIC_VDIM > 2)
    newPos[2] = p.rdata(2) + dt*chargemass*(Ep[2]);
#endif
    return(newPos);
}


/*!
     * Push the position and the velocity (only the vxB part) in a nonlinear iteration
     */
template <typename P>
std::array<amrex::Real, GEMPIC_SPACEDIM+GEMPIC_VDIM> push_x_v_bfield(P const& p,
                                                                     amrex::Real dt,
                                                                     amrex::Real chargemass, //charge/mass
                                                                     std::array<amrex::Array4<amrex::Real>,GEMPIC_BDIM> bArray,
                                                                     infrastructure infra){
    std::array<amrex::Real, GEMPIC_SPACEDIM> oldX;
    for (unsigned int d=0;d<GEMPIC_SPACEDIM; ++d)
        oldX[d]= p.pos(d);
    std::array<amrex::Real, GEMPIC_VDIM> oldV;
    for (unsigned int d=0;d<GEMPIC_VDIM;++d)
        oldV[d] = p.rdata(d);
    std::array<amrex::Real, GEMPIC_SPACEDIM> newX = oldX;
    std::array<amrex::Real, GEMPIC_VDIM> newV = oldV;
    std::array<amrex::Real, GEMPIC_SPACEDIM> newXiter;
    std::array<amrex::Real, GEMPIC_VDIM> newViter;
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
            if (d < GEMPIC_VDIM)
                newX[d] = oldX[d] + dt*0.5*( newV[d] + oldV[d] );
        }
        // Compute xMean
        std::array<amrex::Real,GEMPIC_SPACEDIM> xMean;
        for (unsigned int d=0; d<GEMPIC_SPACEDIM; ++d)
        {
            if (d<GEMPIC_VDIM)
                xMean[d] = mod_dom(0.5*(newX[d]+oldX[d]),infra.geom.ProbHi()[d],infra.geom.ProbLo()[d]);
        }
        // evaluate B at xMean
        std::array<amrex::Real,GEMPIC_VDIM> Bp = evaluate_fem_bfield(xMean,
                                                                     infra.dxi,
                                                                     infra.plo,
                                                                     bArray);
#if  (GEMPIC_VDIM == 2)
        newV[0] = oldV[0] + dt*chargemass*( 0.5*(oldV[1]+newViter[1])*Bp[0]);
        newV[1] = oldV[1] + dt*chargemass*( 0.5*(oldV[0]+newViter[0])*Bp[0]);
#elif (GEMPIC_VDIM == 3)
        newV[0] = p.rdata(0) + dt*chargemass*(0.5*(oldV[1]+newViter[1])*Bp[2]-0.5*(oldV[2]+newViter[2])*Bp[1]);
        newV[1] = p.rdata(1) + dt*chargemass*(0.5*(oldV[2]+newViter[2])*Bp[0]-0.5*(oldV[0]+newViter[0])*Bp[2]);
        newV[2] = p.rdata(2) + dt*chargemass*(0.5*(oldV[0]+newViter[0])*Bp[1]-0.5*(oldV[1]+newViter[1])*Bp[0]);
#endif
        diff = 0.0;
        for (unsigned d=0;d<GEMPIC_SPACEDIM;++d)
            diff = std::max(diff, std::abs(newXiter[d] - newX[d]));
        for (unsigned d=0;d<GEMPIC_VDIM;++d)
            diff = std::max(diff, std::abs(newViter[d] - newV[d]));
    }
    if (iter_counter == max_iter)
        amrex::Print() << " Number of itertions: " << iter_counter << " " << diff << std::endl;
    std::array<amrex::Real,GEMPIC_SPACEDIM+GEMPIC_VDIM> newPos;
    for (unsigned int d=0;d<GEMPIC_SPACEDIM;++d)
        newPos[d] = newX[d];
    for (unsigned int d=0;d<GEMPIC_VDIM;++d)
        newPos[GEMPIC_SPACEDIM+d]= newV[d];
    return(newPos);
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
    newPos[0] = mod_dom(p.pos(0) + p.rdata(0)*dt, infra.geom.ProbHi()[0], infra.geom.ProbLo()[0]);
#if (GEMPIC_SPACEDIM > 1)
    newPos[1] = mod_dom(p.pos(1) + p.rdata(1)*dt, infra.geom.ProbHi()[1], infra.geom.ProbLo()[1]);
#endif
#if (GEMPIC_SPACEDIM > 2)
    newPos[2] = mod_dom(p.pos(2) + p.rdata(2)*dt, infra.geom.ProbHi()[2], infra.geom.ProbLo()[2]);
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
    newPos[0] = mod_dom(p.pos(0) + p.rdata(0)*dt, infra.geom.ProbHi()[0], infra.geom.ProbLo()[0]);
    newPos[GEMPIC_SPACEDIM] = p.rdata(0);
#if (GEMPIC_SPACEDIM > 1)
    newPos[1] = mod_dom(p.pos(1) + p.rdata(1)*dt, infra.geom.ProbHi()[1], infra.geom.ProbLo()[1]);
#endif
#if (GEMPIC_SPACEDIM > 2)
    newPos[2] = mod_dom(p.pos(2) + p.rdata(2)*dt, infra.geom.ProbHi()[2], infra.geom.ProbLo()[2]);
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
    newPos[0] = mod_dom(p.pos(0) + newPos[GEMPIC_SPACEDIM]*dt, infra.geom.ProbHi()[0], infra.geom.ProbLo()[0]);
#if (GEMPIC_SPACEDIM > 1)
    newPos[1] = mod_dom(p.pos(1) + newPos[GEMPIC_SPACEDIM+1]*dt, infra.geom.ProbHi()[1], infra.geom.ProbLo()[1]);
#endif
#if (GEMPIC_SPACEDIM > 2)
    newPos[2] = mod_dom(p.pos(2) + newPos[GEMPIC_SPACEDIM+2]*dt, infra.geom.ProbHi()[2], infra.geom.ProbLo()[2]);
#endif


    return(newPos);
}

}
}

#endif
