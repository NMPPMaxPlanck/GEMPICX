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

#include <infrastructure.H>

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
    splines (amrex::IntVect deg,
             std::array<bool,GEMPIC_SPACEDIM> cent_eval,
             std::array<bool,GEMPIC_SPACEDIM> int_eval){

        degree = deg;
        centered_evaluation = cent_eval;
        integral_evaluation = int_eval;
    }

    amrex::Real spline_eval (amrex::Real x,
                             int dimension,
                             int cell) {

        std::vector<amrex::Real> s;
        switch (degree[dimension]) {
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
        return(s[cell]);
    }

    amrex::Real shifted_x (amrex::Real x, int dimension) {
        if (degree[dimension]%2==0 ^ centered_evaluation[dimension]) { //even XOR centered
            if (x>1./2.) {
                return(3./2.-x);
            } else {
                return(1./2.-x);
            }
        } else {
            return(1.-x);
        }
    }

    amrex::Real deposition_coefficient (std::array<amrex::Real, GEMPIC_SPACEDIM> x,
                                        std::array<int, GEMPIC_SPACEDIM> cell) {
        amrex::Real coeff = 1.;
        coeff *= spline_eval(shifted_x(x[0], 0), 0, cell[0]);
#if GEMPIC_SPACEDIM > 1
        coeff *= spline_eval(shifted_x(x[1], 1), 1, cell[1]);
#endif
#if GEMPIC_SPACEDIM > 2
        coeff *= spline_eval(shifted_x(x[2], 2), 2, cell[2]);
#endif
        return(coeff);
    }

    int compute_cell_index (int cell_index, int dimension, amrex::Real x) {
        if (centered_evaluation[dimension]) {
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

