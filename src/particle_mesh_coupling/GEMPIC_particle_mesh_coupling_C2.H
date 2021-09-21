#ifndef PARTICLE_MESH_COUPLING_C2_H_
#define PARTICLE_MESH_COUPLING_C2_H_

#include <cmath>

#include <AMReX_Array.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Particles.H>
#include <AMReX_Print.H>

#include <GEMPIC_Config.H>

namespace Gempic {
namespace Particles {

template<int deg>
class spline1d_at_particles
{
public :
    std::vector<amrex::Real> primitive;
    int box;

void init_position(amrex::Real position,
            amrex::Real plo,
                   amrex::Real dxi)
    {
        // Compute normalized position xint and box
        amrex::Real xint = (position-plo)*dxi;
        box = std::floor(xint);
        xint = xint - box;

        if ( deg%2 == 0 )
            {
                if (xint< 0.5)
                    {
                        box -= deg/2;
                        xint = 0.5-xint;
                    }
                else
                    {
                        box += -deg/2+1;
                        xint = 1.5-xint;
                    }
            }
        else
            {
                box -= (deg-1)/2;
                xint = 1. - xint;
            }
        primitive = spline_eval_primitive<deg-1> (xint);
    }

private :

    template<int degree>
     	    std::vector<amrex::Real> spline_eval_primitive (amrex::Real x)  {
        std::vector<amrex::Real> s;
        switch (degree) {
        case 0:
            s = {x};
            break;
        case 1:
            s = {pow(x,2)/2., 1.-pow(x-1.,2)/2.};
            break;
        case 2:
            s = {pow(x,3)/6., (((-2.*x+3)*x+3.)*x+1.)/6., pow(x-1.,3)/6.+1.};
            break;
        case 3:
            s = {pow(x,4)/24.,  -pow(x,4)/8.+pow(x,3)/6.+pow(x,2)/4.+x/6.+1./24., pow(x,4)/8.-pow(x,3)/3.+x*2./3.+0.5, -pow((1.-x),4)/24.+1.};
            break;
        case 4:
            s = {  pow(x,5)/120. ,  (((((-4.*x+5.)*x+10.)*x+10.)*x+5.)*x+1.)/120., (((((6.*x-15.)*x-10.)*x+30.)*x+55.)*x+27.)/120.  ,(((((-4.*x+15.)*x-10.)*x-30.)*x+55.)*x+93.)/120.   ,  -pow(1.-x,5)/120.+1. };
            break;
        case 5:
            s = {  pow(x,6)/720. , ((((((-5.*x+6.)*x+15.)*x+20.)*x+15.)*x+6.)*x+1.)/720., ((((((10.*x-24.)*x-30.)*x+40.)*x+150.)*x+156.)*x+58.)/720., ((((-10.*x+36.)*x*x-120.)*x*x+396.)*x+360.)/720., ((((((5.*x-24.)*x+30.)*x+40.)*x-150.)*x+156.)*x+662.)/720., ((((((-x+6.)*x-15.)*x+20.)*x-15.)*x+6.)*x+719.)/720.};
            break;
        case 6:
            s = {pow(x,7)/5040.,
                -pow(x,7)/840. + pow(x,6)/720. + pow(x,5)/240. + pow(x,4)/144. + pow(x,3)/144. + pow(x,2)/240. + x/720. + 1./5040.,
                 pow(x,7)/336. - pow(x,6)/144. - pow(x,5)/80. + pow(x,4)/144. + pow(x,3)/16. + 5.*pow(x,2)/48. + 19.*x/240. + 121./5040.,
                -pow(x,7)/252. + pow(x,6)/72. + pow(x,5)/120. - pow(x,4)/18. - 5.*pow(x,3)/72. + pow(x,2)/6. + 151.*x/360. + 82./315.,
                pow(x,7)/336. - pow(x,6)/72. + pow(x,5)/120. + pow(x,4)/18. - 5.*pow(x,3)/72. - pow(x,2)/6. + 151.*x/360. + 233./315.,
                -pow(x,7)/840. + pow(x,6)/144. - pow(x,5)/80. - pow(x,4)/144. + pow(x,3)/16. - 5.*pow(x,2)/48. + 19.*x/240. + 691./5040.,
                pow(x,7)/5040. - pow(x,6)/720. + pow(x,5)/240. - pow(x,4)/144. + pow(x,3)/144. - pow(x,2)/240. + x/720. + 811./5040.};
            break;
        }
        return(s);
    }


};

template<int degx,int degy,int degz>
class splines_at_particles
{
public :
        amrex::GpuArray<amrex::GpuArray<amrex::Real, std::max(std::max(degx, degy), degz)>,GEMPIC_SPACEDIM> spline_cell;
        amrex::GpuArray<amrex::GpuArray<amrex::Real, std::max(std::max(degx, degy), degz)+1>,GEMPIC_SPACEDIM> spline_node;

        amrex::GpuArray<int, GEMPIC_SPACEDIM> box;

        // Default constructor
        //splines_at_particles();

        template<int dim, int deg>
AMREX_GPU_HOST_DEVICE  AMREX_NO_INLINE           void init_position(amrex::Real position,
                           amrex::Real plo,
                           amrex::Real dxi)
        {
            // Compute normalized position xint and box
            amrex::Real xint = (position-plo)*dxi;
            box[dim] = std::floor(xint);
            xint = xint - box[dim];

            if ( deg%2 == 0 )
                {
                    if (xint< 0.5)
                        {
                            box[dim] -= deg/2;
                            xint = 0.5-xint;
                        }
                    else
                        {
                            box[dim] += -deg/2+1;
                            xint = 1.5-xint;
                        }
                }
            else
                {
                    box[dim] -= (deg-1)/2;
                    xint = 1. - xint;
                }

            spline_cell[dim] = spline_eval<deg-1, std::max(std::max(degx, degy), degz)> (xint);
            spline_node[dim] = spline_eval<deg, std::max(std::max(degx, degy), degz)+1> (xint);

        }

 AMREX_GPU_HOST_DEVICE  AMREX_NO_INLINE                  void init_particles(amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> position,
                            amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> const& plo,
                            amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM+1> const& dxi)
        {
            init_position<0,degx>(position[0],plo[0],dxi[0]);
#if (GEMPIC_SPACEDIM>1)
            init_position<1,degy>(position[1],plo[1],dxi[1]);
#endif
#if (GEMPIC_SPACEDIM>2)
            init_position<2,degz>(position[2],plo[2],dxi[2]);
#endif
        }

private :
    template<int degree,int length>
AMREX_GPU_HOST_DEVICE  AMREX_NO_INLINE       amrex::GpuArray<amrex::Real, length> spline_eval (amrex::Real x)  {
            amrex::GpuArray<amrex::Real, length> s;
            switch (degree) {
            case 0:
                s[0] = 1.;
                break;
            case 1:
                s[0] = x;
                s[1] = 1.-x;
                break;
            case 2:
                s[0] = pow(x,2)/2.;
                s[1] = (1.-x)*x+0.5;
                s[2] = pow(x-1,2)/2.;
                break;
            case 3:
                s[0] = pow(x,3)/6.;
                s[1] = (((-x+1)*x+1.)*3.*x+1.)/6. ;
                s[2] = ((3.*x-6.)*x*x+4.)/6.;
                s[3] = pow((1.-x),3)/6.;
                break;
              case 4:
                s[0] = pow(x,4)/24.;
                s[1] = ((((-x+1.)*4.*x+6.)*x+4.)*x+1.)/24.;
                s[2] = ((((6.*x-12.)*x-6.)*x+12.)*x+11.)/24.;
                s[3] = ((((-4.*x+12.)*x-6.)*x-12.)*x+11.)/24. ;
                s[4] = pow(1.-x,4)/24.;
                break;
            case 5:
                s[0] = pow(x,5)/120.;
                s[1] = (((((-x+1.)*x+2.)*x+2.)*x+1.)*5.*x+1.)/120.;
                s[2] = (((((x-2.)*x-2.)*x+2.)*x+5.)*10.*x+26.)/120.;
                s[3] = (((-x+3.)*x*x-6.)*x*10.*x+66.)/120.;
                s[4] = (((((x-4.)*x+4.)*x+4.)*x-10.)*5.*x+26.)/120.;
                s[5] = (((((-x+5.)*x-10.)*x+10.)*x-5.)*x+1.)/120.;
                break;
            case 6:
                s[0] = pow(x,6)/720.;
                s[1] = -pow(x,6)/120. + pow(x,5)/120. + pow(x,4)/48. + pow(x,3)/36. + pow(x,2)/48. + x/120. + 1./720.;
                s[2] = pow(x,6)/48. - pow(x,5)/24. - pow(x,4)/16. + pow(x,3)/36. + 3.*pow(x,2)/16. + 5.*x/24. + 19./240.;
                s[3] = -pow(x,6)/36. + pow(x,5)/12. + pow(x,4)/24. - 2.*pow(x,3)/9. - 5.*pow(x,2)/24. + 1.*x/3. + 151./360.;
                s[4] = pow(x,6)/48. - pow(x,5)/12. + pow(x,4)/24. + 2.*pow(x,3)/9. - 5.*pow(x,2)/24. - x/3. + 151./360.;
                s[5] = -pow(x,6)/120. + pow(x,5)/24. - pow(x,4)/16. - pow(x,3)/36. + 3.*pow(x,2)/16. - 5.*x/24. + 19./240.;
                s[6] = pow(x,6)/720. - pow(x,5)/120. + pow(x,4)/48. - pow(x,3)/36. + pow(x,2)/48. - x/120. + 1./720.;
                break;
            }
            return(s);
        }


};

template<int vdim, int degx, int degy, int degz>
AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE amrex::GpuArray<amrex::Real, vdim> evaluate_efield(splines_at_particles<degx,degy,degz> &spline,
                                              amrex::GpuArray<amrex::Array4<amrex::Real>,vdim> eArray)
{
    amrex::GpuArray<amrex::Real,vdim> fields;
    for (int comp = 0; comp < vdim; comp++) {
        fields[comp] = 0.;
    }

#if (GEMPIC_SPACEDIM > 2)
    for (int kk = 0; kk <= degz; ++kk)
#endif
      {
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0; jj <= degy; ++jj)
#endif
          {
            for (int ii = 0; ii <= degx; ++ii) {
                amrex::GpuArray<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(spline.box[0]+ii, spline.box[1]+jj, spline.box[2]+kk)};
                if ( ii < degx ) {
                    double s_coeff = spline.spline_cell[0][ii]
        #if(GEMPIC_SPACEDIM > 1)
                            *spline.spline_node[1][jj]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                            *spline.spline_node[2][kk]
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
                        double s_coeff = spline.spline_node[0][ii]
                                *spline.spline_cell[1][jj]
        #if(GEMPIC_SPACEDIM > 2)
                                *spline.spline_node[2][kk]
        #endif
                                ;
                        fields[1] += s_coeff*(eArray[1])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]) ,0);
                    }
                }
#endif
#if(GEMPIC_SPACEDIM > 2)
                if ( kk < degz ) {
                    double s_coeff = spline.spline_node[0][ii]
                            *spline.spline_node[1][jj]
                            *spline.spline_cell[2][kk]
                            ;
                    fields[2] += s_coeff*(eArray[2])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]) ,0);
                }
#endif
            }
        }
    }
    return(fields);
}


#if GEMPIC_SPACEDIM == 3
template<int vdim, int degx, int degy, int degz>
amrex::GpuArray<amrex::Real, vdim> evaluate_bfield(splines_at_particles<degx,degy,degz> &spline,
                                              amrex::GpuArray<amrex::Array4<amrex::Real>,vdim> eArray)
{
    amrex::GpuArray<amrex::Real,vdim> fields;
    for (int comp = 0; comp < vdim; comp++) {
        fields[comp] = 0.;
    }

    for (int kk = 0; kk <= degz; ++kk) {
        for (int jj = 0; jj <= degy; ++jj) {
            for (int ii = 0; ii <= degx; ++ii) {
                amrex::GpuArray<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(spline.box[0]+ii, spline.box[1]+jj, spline.box[2]+kk)};
                if ( ii < degx ) {
                    double s_coeff = spline.spline_node[0][ii]
                            *spline.spline_cell[1][jj]
                            *spline.spline_cell[2][kk]
                            ;
                fields[0] += s_coeff*(eArray[0])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]) ,0);
                }
                if ( jj < degy ) {
                        double s_coeff = spline.spline_node[0][ii]
                                *spline.spline_cell[1][jj]
                                *spline.spline_node[2][kk];
                        fields[1] += s_coeff*(eArray[1])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]) ,0);

                }
                if ( kk < degz ) {
                    double s_coeff = spline.spline_node[0][ii]
                            *spline.spline_node[1][jj]
                            *spline.spline_cell[2][kk]
                            ;
                    fields[2] += s_coeff*(eArray[2])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]) ,0);
                }
              }
        }
    }
    return(fields);
}
#endif

template <int degx, int degy, int degz>
void gempic_deposit_rho_C3 ( splines_at_particles<degx, degy, degz> &spline,
                                amrex::Real weight,
                            amrex::Array4<amrex::Real> const& rho)
{


#if (GEMPIC_SPACEDIM > 2)
    for (int kk = 0; kk <= degz; ++kk)
#endif
      {
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0; jj <= degy; ++jj)
#endif
          {
            for (int ii = 0; ii <= degx; ++ii) {
                amrex::GpuArray<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(spline.box[0]+ii, spline.box[1]+jj, spline.box[2]+kk)};
                double s_coeff = spline.spline_node[0][ii]
        #if(GEMPIC_SPACEDIM > 1)
                        * spline.spline_node[1][jj]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                        * spline.spline_node[2][kk]
        #endif
                        ;
#if (GEMPIC_SPACEDIM == 1)
                amrex::HostDevice::Atomic::Add(rho.ptr(cell_index[0], 0, 0, 0), s_coeff*weight);
#else
        amrex::HostDevice::Atomic::Add(rho.ptr(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0), s_coeff*weight);
#endif
            }
        }
    }
}


template<typename S, int vdim, int degp, int degp1, int degp2, int pdim>
void accumulate_j_update_v_C2(S const& spline,
                              spline1d_at_particles<degp> const& spline_new,
                              spline1d_at_particles<degp> const& spline_old,
                              amrex::Real weight,
                              const amrex::Real* dx,
                              amrex::GpuArray<amrex::Array4<amrex::Real>,int(vdim/2.5)*2+1> bArray,
                              amrex::GpuArray<amrex::Array4<amrex::Real>,vdim> jArray,
                              amrex::GpuArray<amrex::Real, 2> &fields   )
{
    int pdim1 = (pdim+1)%3;
    int pdim2 = (pdim+2)%3;

    fields = {0.,0.};

    int local_size = std::abs(spline_new.box-spline_old.box)+degp;
    int box_start = spline_old.box;
    if (spline_new.box<spline_old.box)
      box_start = spline_new.box;

    std::vector<amrex::Real> primitive(local_size+degp);
    if ( spline_old.box < spline_new.box ) {
            for (unsigned int j=0;j<degp;j++)
              primitive[j] = spline_old.primitive[j]*dx[pdim];
            for (unsigned int j=degp;j<local_size;j++)
              primitive[j] = dx[pdim];

            for (unsigned int j =0;j<degp; j++) {
                primitive[local_size-degp+j] -= spline_new.primitive[j]*dx[pdim];
              }
          }
        else {
            for(unsigned int j=degp;j<local_size;j++) {
                primitive[j] = -dx[pdim];
              }
            for (unsigned int j=0;j<degp;j++) {
                primitive[j] = -spline_new.primitive[j]*dx[pdim];
              }
            for (unsigned int j =0;j<degp; j++) {
                primitive[local_size-degp+j] += spline_old.primitive[j]*dx[pdim];
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
                  cell_index = {AMREX_D_DECL(box_start+ii, spline.box[1]+jj, spline.box[2]+kk)};
                else if (pdim == 1 )
                  cell_index = {AMREX_D_DECL( spline.box[0]+kk, box_start+ii, spline.box[2]+jj)};
                else if (pdim == 2)
                  cell_index = {AMREX_D_DECL( spline.box[0]+jj, spline.box[1]+kk, box_start+ii)};
                double s_coeff = primitive[ii]
                        *spline.spline_node[pdim1][jj]
                        *spline.spline_node[pdim2][kk];
#if (GEMPIC_SPACEDIM == 1)
                amrex::HostDevice::Atomic::Add(jArray[pdim].ptr(cell_index[0], 0, 0 , 0), s_coeff*weight);
#else
                amrex::HostDevice::Atomic::Add(jArray[pdim].ptr(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]) , 0), s_coeff*weight);
#endif

                    if ( jj < degp1 )
                      {
                        double s_coeff = primitive[ii]
                              *spline.spline_cell[pdim1][jj]
                              *spline.spline_node[pdim2][kk];

#if (GEMPIC_SPACEDIM == 1)
                fields[0] += s_coeff*(bArray[pdim2])(cell_index[0], 0, 0, 0);
#else
                fields[0] += s_coeff*(bArray[pdim2])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0);
#endif
                          }
                  if ( kk < degp2 ){
                      double s_coeff = primitive[ii]
                              *spline.spline_node[pdim1][jj]
                              *spline.spline_cell[pdim2][kk];
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

  
} // end namespace particles
} // end namespace GEMPIC

#endif
