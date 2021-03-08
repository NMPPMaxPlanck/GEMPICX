#include <AMReX.H>
#include <AMReX_ArrayOfStructs.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_CoordSys.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_Particle.H>
#include <AMReX_Particles.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>
#include <AMReX_StructOfArrays.H>

#include <GEMPIC_infrastructure.H>

#ifndef SPLINES_H_
#define SPLINES_H_

namespace Gempic {
namespace Particles {

class splines
{
public:

    // parameters for splines
    std::array<bool,GEMPIC_SPACEDIM> centered_evaluation;
    std::array<bool,GEMPIC_SPACEDIM> integral_evaluation;

    // The default constructor.
    splines ();

    // Explicit const
    splines (std::array<bool,GEMPIC_SPACEDIM> cent_eval,
             std::array<bool,GEMPIC_SPACEDIM> int_eval){

        centered_evaluation = cent_eval;
        integral_evaluation = int_eval;
    }

    std::vector<amrex::Real> spline_eval_vec (amrex::Real x, int degree)  {
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
        case 5:
            s = {pow(x,5)/120., (((((-x+1.)*x+2.)*x+2.)*x+1.)*5.*x+1.)/120., (((((x-2.)*x-2.)*x+2.)*x+5.)*10.*x+26.)/120., (((-x+3.)*x*x-6.)*x*10.*x+66.)/120., (((((x-4.)*x+4.)*x+4.)*x-10.)*5.*x+26.)/120., (((((-x+5.)*x-10.)*x+10.)*x-5.)*x+1.)/120. };
            break;
        }
        return(s);
    }

    template<int degree>
    std::vector<amrex::Real> spline_eval_prim_vec (amrex::Real x)  {
        std::vector<amrex::Real> s;
        switch (degree) {
        case 0:
            s = {x};
            break;
        case 1:
            s = {1./2.+x-pow(x,2)/2., pow(x,2)/2.};
            break;
        case 2:
            s = {pow(x,3)/6.-pow(x,2)/2.+x*0.5+5./6., -pow(x,3)/3.+pow(x,2)/2.+x/2.+1./6., pow(x,3)/6.};
            break;
        case 3:
            s = { -pow((1.-x),4)/24.+1., pow(x,4)/8.-pow(x,3)/3.+x*2./3.+0.5, -pow(x,4)/8.+pow(x,3)/6.+pow(x,2)/4.+x/6.+1./24., pow(x,4)/24.};
            break;
        case 4:
            s = { -pow(1.-x,5)/120.+1. , (((((-4.*x+15.)*x-10.)*x-30.)*x-55.)*x+93.)/120.   , (((((6.*x-15.)*x-10.)*x+30.)*x+55.)*x+27.)/120.,  (((((-4.*x+5.)*x+10.)*x+10.)*x+5.)*x+1.)/120.  , pow(x,5)/120. };
            break;
        }
        return(s);
    }

    template<int degree>
    std::vector<amrex::Real> spline_eval_vec_back (amrex::Real x)  {
        std::vector<amrex::Real> s;
        switch (degree) {
        case 0:
            s = {1.};
            break;
        case 1:
            s = {1.-x, x};
            break;
        case 2:
            s = { 1./2.-x+pow(x,2)/2., 1./2.+x-pow(x,2), pow(x,2)/2.};
            break;
        case 3:
            s = {1./6.*pow((1.-x),3), 1./6.*(4.-6.*pow(x,2)+3.*pow(x,3)), 1./6.*(1.+3.*x+3.*pow(x,2)-3.*pow(x,3)), 1./6.*pow(x,3)};
            break;
          case 4:
            s = {pow(1.-x,4.)/24.,((((-4.*x+12.)*x-6.)*x-12.)*x+11)/24. ,  ((((6.*x-12.)*x-6.)*x+12.)*x+11.)/24.,(((((-x+1.)*4.*x+6.)*x+4.)*x+4.)*x+1.)/24., pow(x,4)/24. };
            break;
        }
        return(s);
    }

    amrex::Real shifted_x (amrex::Real x, int degree, bool cent_eval) {
        if ((degree%2==0) ^ (cent_eval)) { //even XOR centered
            if (x>1./2.) {
                return(3./2.-x);
            } else {
                return(1./2.-x);
            }
        } else {
            return(1.-x);
        }
    }

    int compute_cell_index (int cell_index, amrex::Real x, int degree, bool cent_eval) {

        if (!cent_eval) {
            if(degree%2 == 0) {
                if (x < 0.5) {
                    return(cell_index-int(std::round(degree/2.)));
                } else {
                    return(cell_index-int(std::round(degree/2.))+1);
                }
            } else { //odd
                return(cell_index-int(std::round((degree-1.)/2.)));
            }
        } else {
            if(degree%2 == 1) {
                if (x < 0.5) {
                    return(cell_index-int(std::round((degree-1)/2.)));
                } else {
                    return(cell_index-int(std::round(degree/2.)+1));
                }
            } else { //even
                return(cell_index-int(std::round(degree/2.)));
            }
        }

    }

};
}
}

#endif

