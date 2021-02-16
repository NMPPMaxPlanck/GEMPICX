#ifndef PARTICLE_MESH_COUPLING_C2_H_
#define PARTICLE_MESH_COUPLING_C2_H_

#include <cmath>

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

        if ( deg%2 == 1 )
            {
                if (xint< 0.5)
                    {
                        box -= (deg-1)/2;
                        xint = 0.5-xint;
                    }
                else
                    {
                        box += -(deg-1)/2+1;
                        xint = 1.5-xint;
                    }
            }
        else
            {
                box -= deg/2+1;
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
            s = {pow(x,2)/2., 1./2.+x+pow(x,2)/2.};
            break;
        case 2:
            s = {pow(x,3)/6., -pow(x,3)/3.+pow(x,2)/2.+x/2.+1./6., pow(x,3)/6.-pow(x,2)/2.+x*0.5+5./6.};
            break;
        case 3:
            s = {pow(x,4)/24., -pow((1.-x),4)/24.+1.,  -pow(x,4)/8.+pow(x,3)/6.+pow(x,2)/4.+x/6.+1./24., pow(x,4)/8.-pow(x,3)/3.+x*2./3.+0.5};
            break;
        case 4:
            s = {  pow(x,5)/120. ,  (((((-4.*x+5.)*x+10.)*x+10.)*x+5.)*x+1.)/120., (((((6.*x-15.)*x-10.)*x+30.)*x+55.)*x+27.)/120.  ,(((((-4.*x+15.)*x-10.)*x-30.)*x-55.)*x+93.)/120.   ,  -pow(1.-x,5)/120.+1. };
            break;
        }
        return(s);
    }


};

template<int degx,int degy,int degz>
class splines_at_particles
{
public :
        std::array<std::vector<amrex::Real>,GEMPIC_SPACEDIM> spline_cell;
        std::array<std::vector<amrex::Real>,GEMPIC_SPACEDIM> spline_node;

        std::array<int, GEMPIC_SPACEDIM> box;

        // Default constructor
        //splines_at_particles();

        template<int dim, int deg>
        void init_position(amrex::Real position,
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

            spline_cell[dim] = spline_eval<deg-1> (xint);
            spline_node[dim] = spline_eval<deg> (xint);

        }

        void init_particles(std::array<amrex::Real,GEMPIC_SPACEDIM> position,
                            std::array<amrex::Real,GEMPIC_SPACEDIM> const& plo,
                            std::array<amrex::Real,GEMPIC_SPACEDIM+1> const& dxi)
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
    template<int degree>
    std::vector<amrex::Real> spline_eval (amrex::Real x)  {
            std::vector<amrex::Real> s;
            switch (degree) {
            case 0:
                s = {1.};
                break;
            case 1:
                s = {x, 1.-x};
                break;
            case 2:
                s = {pow(x,2)/2., 1./2.+x-pow(x,2), 1./2.-x+pow(x,2)/2.};
                break;
            case 3:
                s = {1./6.*pow(x,3), 1./6.*(1.+3.*x+3.*pow(x,2)-3.*pow(x,3)), 1./6.*(4.-6.*pow(x,2)+3.*pow(x,3)), 1./6.*pow((1.-x),3)};
                break;
              case 4:
                s = {pow(x,4)/24., (((((-x+1.)*4.*x+6.)*x+4.)*x+4.)*x+1.)/24., ((((6.*x-12.)*x-6.)*x+12.)*x+11.)/24.,((((-4.*x+12.)*x-6.)*x-12.)*x+11.)/24. , pow(1.-x,4)/24. };
                break;
            }
            return(s);
        }


};

template<int vdim, int degx, int degy, int degz>
std::array<amrex::Real, vdim> evaluate_efield(splines_at_particles<degx,degy,degz> &spline,
                                              std::array<amrex::Array4<amrex::Real>,vdim> eArray)
{
    std::array<amrex::Real,vdim> fields;
    for (int comp = 0; comp < vdim; comp++) {
        fields[comp] = 0.;
    }

#if (GEMPIC_SPACEDIM > 2)
    for (int kk = 0; kk <= degz; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0; jj <= degy; ++jj) {
#endif
            for (int ii = 0; ii <= degx; ++ii) {
                std::array<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(spline.box[0]+ii, spline.box[1]+jj, spline.box[2]+kk)};
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
#if (GEMPIC_SPACEDIM > 1)
        }
#endif
#if (GEMPIC_SPACEDIM > 2)
    }
#endif
    return(fields);
}


#if GEMPIC_SPACEDIM == 3
template<int vdim, int degx, int degy, int degz>
std::array<amrex::Real, vdim> evaluate_bfield(splines_at_particles<degx,degy,degz> &spline,
                                              std::array<amrex::Array4<amrex::Real>,vdim> eArray)
{
    std::array<amrex::Real,vdim> fields;
    for (int comp = 0; comp < vdim; comp++) {
        fields[comp] = 0.;
    }

    for (int kk = 0; kk <= degz; ++kk) {
        for (int jj = 0; jj <= degy; ++jj) {
            for (int ii = 0; ii <= degx; ++ii) {
                std::array<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(spline.box[0]+ii, spline.box[1]+jj, spline.box[2]+kk)};
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
    for (int kk = 0; kk <= degz; ++kk) {
#endif
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0; jj <= degy; ++jj) {
#endif
            for (int ii = 0; ii <= degx; ++ii) {
                std::array<int,GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(spline.box[0]+ii, spline.box[1]+jj, spline.box[2]+kk)};
                double s_coeff = spline.spline_node[0][ii]
        #if(GEMPIC_SPACEDIM > 1)
                        * spline.spline_node[1][jj]
        #endif
        #if(GEMPIC_SPACEDIM > 2)
                        * spline.spline_node[2][kk]
        #endif
                        ;
#if (GEMPIC_SPACEDIM == 1)
                amrex::Gpu::Atomic::Add(&rho(cell_index[0], 0, 0, 0), s_coeff*weight);
#else
                amrex::Gpu::Atomic::Add(&rho(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0), s_coeff*weight);
#endif
            }
#if (GEMPIC_SPACEDIM > 1)
        }
#endif
#if (GEMPIC_SPACEDIM > 2)
    }
#endif
}


template<typename S, int vdim, int degp, int degp1, int degp2, int pdim>
void accumulate_j_update_v_C2(S const& spline,
                              spline1d_at_particles<degp> const& spline_new,
                              spline1d_at_particles<degp> const& spline_old,
                              amrex::Real weight,
                              const amrex::Real* dx,
                              std::array<amrex::Array4<amrex::Real>,int(vdim/2.5)*2+1> bArray,
                              std::array<amrex::Array4<amrex::Real>,vdim> jArray,
                              std::array<amrex::Real, 2> &fields   )
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
        for (unsigned int j =0;j<degp; j++) {
            primitive[local_size-degp+j] = spline_new.primitive[j]*dx[pdim];
          }
        for(unsigned int j=0;j<local_size-degp;j++) {
            primitive[j] = dx[pdim];
          }
        for (unsigned int j=0;j<degp;j++) {
            primitive[j] -= spline_old.primitive[j]*dx[pdim];
          }
      }
    else {
        for (unsigned int j =0;j<degp; j++) {
            primitive[local_size-degp+j] = -spline_old.primitive[j]*dx[pdim];
          }
        for(unsigned int j=0;j<local_size-degp;j++) {
            primitive[j] = -dx[pdim];
          }
        for (unsigned int j=0;j<degp;j++) {
            primitive[j] += spline_new.primitive[j]*dx[pdim];
          }
      }

    for (int kk = 0; kk <=degp2; ++kk)
      {
        for (int jj = 0; jj <= degp1; ++jj)
          {
            for (int ii = 0; ii <= local_size; ++ii)
              {
                std::array<int,GEMPIC_SPACEDIM> cell_index;
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
                amrex::Gpu::Atomic::Add(&(jArray[pdim])(cell_index[0], 0, 0 , 0), s_coeff*weight);
#else
                amrex::Gpu::Atomic::Add(&(jArray[pdim])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]) , 0), s_coeff*weight);
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
