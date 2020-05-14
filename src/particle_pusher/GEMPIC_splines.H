#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_RealBox.H>
#include <AMReX_REAL.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_Box.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_CoordSys.H>
#include <AMReX_Particle.H>
#include <AMReX_Particles.H>
#include <AMReX_ArrayOfStructs.H>
#include <AMReX_StructOfArrays.H>

#include <GEMPIC_infrastructure.H>

#ifndef SPLINES_H_
#define SPLINES_H_

class splines
{
public:

    // parameters for splines
    amrex::IntVect degree;
    std::array<bool,GEMPIC_SPACEDIM> centered_evaluation;
    std::array<bool,GEMPIC_SPACEDIM> integral_evaluation;

    // The default constructor.
    splines ();

    // Explicit const
    splines (std::array<bool,GEMPIC_SPACEDIM> cent_eval,
             std::array<bool,GEMPIC_SPACEDIM> int_eval){

        degree = amrex::IntVect{AMREX_D_DECL(GEMPIC_DEG_X,GEMPIC_DEG_Y,GEMPIC_DEG_Z)};
        centered_evaluation = cent_eval;
        integral_evaluation = int_eval;
    }

    std::vector<amrex::Real> spline_eval_vec (amrex::Real x, int degr)  {
        std::vector<amrex::Real> s;
        switch (degr) {
        case 0:
            s = {1.};
            break;
        case 1:
            s = {x, 1.-x};
            break;
        case 2:
            s = {pow(x,2)/2., 1./2.+x-pow(x,2), 1./2.-x+pow(x,2)/2};
            break;
        case 3:
            s = {1./6.*pow(x,3), 1./6.*(1.+3.*x+3.*pow(x,2)-3.*pow(x,3)), 1./6.*(4.-6.*pow(x,2)+3.*pow(x,3)), 1./6.*pow((1.-x),3)};
            break;
        }
        return(s);
    }

    amrex::Real shifted_x (amrex::Real x, int dimension, bool cent_eval) {
        if (degree[dimension]%2==0 ^ cent_eval) { //even XOR centered
            if (x>1./2.) {
                return(3./2.-x);
            } else {
                return(1./2.-x);
            }
        } else {
            return(1.-x);
        }
    }

    int compute_cell_index (int cell_index, int dimension, amrex::Real x, bool cent_eval) {
        if (cent_eval) {
            if(degree[dimension]%2 == 0) {
                if (x < 0.5) {
                    return(cell_index-int(degree[dimension]/2.));
                } else {
                    return(cell_index-int(degree[dimension]/2.)+1);
                }
            } else { //odd
                return(cell_index-int((degree[dimension]-1.)/2.));
            }
        } else {
            if(degree[dimension]%2 == 1) {
                if (x < 0.5) {
                    return(cell_index-int((degree[dimension]-1)/2.));
                } else {
                    return(cell_index-int(degree[dimension]/2.)+1);
                }
            } else { //even
                return(cell_index-int(degree[dimension]/2.));
            }
        }

    }

};

#endif

