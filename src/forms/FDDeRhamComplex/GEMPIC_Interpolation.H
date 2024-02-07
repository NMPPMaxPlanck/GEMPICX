#ifndef GEMPIC_INTERPOLATION_H_
#define GEMPIC_INTERPOLATION_H_

#include <AMReX_SPACE.H>
#include "GEMPIC_Fields.H"


namespace GEMPIC_Interpolation{
using namespace GEMPIC_Fields;

/**
* @brief Evaluates a 1D Lagrange polynomial
* 
* Description:
* Evaluates a Lagrange polynomial \f$L_i\f$ on the interval \f$[x_k, x_{k+1}]\f$, where \f$L_0\f$ is
* the Lagrange polynomial associated to node \f$x_k\f$, \f$L_{-1}\f$ to \f$x_{k-1}\f$, \f$L_1\f$ to \f$x_{k+1}\f$,
* and so forth.
* The whole grid is assumed to be uniform, evaluation is done using Neville's algorithm.
* 
* @tparam degree : int, degree of basis functions used
* @param polyIndex : int, indicates the polynomial w.r.t. \f$x_k\f$
* @param xk : Real, node \f$x_k\f$ which defines the interval of evaluation
* @param dx : (uniform) grid size
* @param y : Real, point in \f$[x_k, x_{k+1}]\f$, where \f$L_i\f$ is to be evaluated
* 
* @return \f$L_{polyIndex} (y)\f$
*/
template<int degree>
AMREX_GPU_DEVICE amrex::Real evalLagrange (const int polyIndex, const amrex::Real xk, const amrex::Real dx, const amrex::Real y)
{
    
    amrex::GpuArray<amrex::Real, degree> vals;
    amrex::GpuArray<amrex::Real, degree> nodes;
    for (int j = 0; j < degree; ++j)
    {
        if (j == polyIndex + degree/2 - 1){
            vals[j] = 1;
        } else {
            vals[j] = 0;
        }
        nodes[j] = xk + (1 - degree/2 + j) * dx;
    }
    for (int i = 1; i <= degree-1; ++i)
    {
        for (int j = 0; j < degree-i; ++j)
        {
            vals[j] = vals[j+1] + (y - nodes[j+i]) * (vals[j+1] - vals[j]) / (nodes[j+i] - nodes[j]);
        }
    }
    return vals[0];
}

/**
* @brief Evaluates a 1D edge function
* 
* Description:
* Evaluates an edge function \f$e_i\f$ on the interval \f$[x_k, x_{k+1}]\f$, where \f$e_0\f$ is
* the edge function associated to the cell \f$[x_k,  x_{k+1}]\f$, \f$e_{-1}\f$ to \f$[x_{k-1},  x_k]\f$,
* \f$e_1\f$ to \f$[x_{k+1},  x_{k+2}]\f$, and so forth.
* The whole grid is assumed to be uniform, only edge functions of degree 0, 2, 4 can be evaluated.
* 
* @tparam degree : int, degree of basis functions used
* @param polyIndex : int, indicates the edge function w.r.t. \f$[x_k, x_{k+1}]\f$
* @param xk : Real, node \f$x_k\f$ which defines the interval of evaluation
* @param dx : (uniform) grid size
* @param y : Real, point in \f$[x_k, x_{k+1}]\f$, where \f$e_i\f$ is to be evaluated
* 
* @return \f$e_{polyIndex} (y)\f$
*/
template<int degree>
AMREX_GPU_DEVICE amrex::Real evalEdge (const int polyIndex, const amrex::Real xk, const amrex::Real dx, const amrex::Real y)
{
    
    amrex::Real val = 0;
    amrex:: Real diff = y - xk;
    switch (degree)
    {
    case 2:
        val =  1. / dx;
        break;
    case 4:
        switch (polyIndex)
        {
        case -1:
            val = ((diff / (2.*dx) - 1.) * diff / dx + 1./3.) / dx;
            break;
        case 0:
            val = ((-diff / dx + 1.) * diff / dx + 5./6.) / dx;
            break;
        case 1:
            val = (std::pow(diff,2) / (2*std::pow(dx,2)) - 1./6.) / dx;
            break;
        default:
            break;
        }
        break;
    case 6:
        switch (polyIndex)
        {
        case -2:
            val = ((((1./24. * diff / dx -  1./6.) * diff / dx +  1./8.) * diff / dx + 1./12.) * diff / dx - 1./20.) / dx;
            break;
        case -1:
            val = (((((-1./6. * diff) / dx + 1./2.) * diff / dx + 1./4.) * diff / dx - 5./4.) *  diff / dx + 9./20.) / dx;
            break;
        case 0:
            val = (((((1./4. * diff) / dx - 1./2.) * diff / dx - 1.) * diff / dx + 5./4.) * diff / dx + 47./60.) / dx;
            break;
        case 1:
            val = (((((-1./6. * diff) / dx + 1./6.) * diff / dx + 3./4.) * diff / dx - 1./12.) * diff / dx - 13./60.) / dx;
            break;
        case 2:
            val = ((1./24. * std::pow(diff, 2) / std::pow(dx, 2) - 1./8.) * std::pow(diff, 2) / std::pow(dx, 2) + 1./30.) / dx;
            break;
        default:
            break;
        }
        break;
    default:
        break;
    }

    return val;
}

/**
* @brief Computes the maximum norm w.r.t. the values at the cell midpoints
* 
* Description:
* Interpolates the field, evaluates it a the cell midpoint and computes the max error
* 
* @tparam degree : int, degree of basis functions used
* @param geom : Geometry, needed for physical coordinates
* @param func : ParserExecutor, provides the exact function to be evaluated
* @param func_h : MultiFab, contains the DOF values of the numeric approximation
* @param dr : RealVect, contains the spatial grid sizes
* @param form : int, specifies k with func_h in C_k
* @param dual : bual, true if func_h is defined on dual grid
* @param comp : int, specifies which component func_h belongs to
* @param t : Real, time point at which func is to be evaluated
* 
* @return void
*/
template<int degree>
amrex::Real maxErrorMidpoint (amrex::Geometry geom, amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func, amrex::MultiFab &func_h,
                    amrex::RealVect dr, int form, bool dual, int comp = 0, amrex::Real t = 0.)
{
    

    const amrex::BoxArray& ba = func_h.boxArray();
    const amrex::DistributionMapping& dm = func_h.DistributionMap();
    int nghost = func_h.nGrow();
    amrex::MultiFab values(ba, dm, 1, nghost);

    for (amrex::MFIter mfi(func_h); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        amrex::Array4<amrex::Real> const &funcMF = func_h[mfi].array();
        amrex::Array4<amrex::Real> const &valuesMF = values[mfi].array();

        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = geom.ProbLoArray();

        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> drHalf =
        {
            AMREX_D_DECL(dr[xDir] / 2, dr[yDir] / 2, dr[zDir] / 2)
        };
        
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            // Compute the position of the point i, j, k
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> rr = 
            {
                AMREX_D_DECL(r0[xDir] + i*dr[xDir] - (dual ? drHalf[xDir] : 0),
                             r0[yDir] + j*dr[yDir] - (dual ? drHalf[yDir] : 0),
                             r0[zDir] + k*dr[zDir] - (dual ? drHalf[zDir] : 0))
            };

            amrex::Real error = 0.0;

            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r = 
            {
                AMREX_D_DECL(rr[xDir] + drHalf[xDir], rr[yDir] + drHalf[yDir], rr[zDir] + drHalf[zDir])
            };

            amrex::Real fval = 0;
            GEMPIC_D_EXCL(
                int b{0};,
                int c{0};,)
            switch (form)
            {
            case 0:
                GEMPIC_D_LOOP_BEGIN(
                for (int a = 1 - degree/2; a < degree/2 + 1; ++a),
                    for (int b = 1 - degree/2; b < degree/2 + 1; ++b),
                        for (int c = 1 - degree/2; c < degree/2 + 1; ++c))
                            fval += funcMF(i + a, j + b, k + c) * GEMPIC_D_MULT(evalLagrange<degree>(a, rr[xDir] + (dual ? dr[xDir] : 0), dr[xDir], r[xDir] + (dual ? dr[xDir] : 0)),
                                                                                evalLagrange<degree>(b, rr[yDir] + (dual ? dr[yDir] : 0), dr[yDir], r[yDir] + (dual ? dr[yDir] : 0)),
                                                                                evalLagrange<degree>(c, rr[zDir] + (dual ? dr[zDir] : 0), dr[zDir], r[zDir] + (dual ? dr[zDir] : 0)));
                GEMPIC_D_LOOP_END
                break;
            case 1:
                GEMPIC_D_LOOP_BEGIN(
                for (int a = 1 - degree/2; a < degree/2 + (comp == xDir ? 0 : 1); ++a),
                    for (int b = 1 - degree/2; b < degree/2 + (comp == yDir ? 0 : 1); ++b),
                        for (int c = 1 - degree/2; c < degree/2 + (comp == zDir ? 0 : 1); ++c))
                            fval += funcMF(i + a, j + b, k + c) * GEMPIC_D_MULT(comp == xDir ? evalEdge<degree>(a, rr[xDir], dr[xDir], r[xDir]) : evalLagrange<degree>(a, rr[xDir], dr[xDir], r[xDir]),
                                                                                comp == yDir ? evalEdge<degree>(b, rr[yDir], dr[yDir], r[yDir]) : evalLagrange<degree>(b, rr[yDir], dr[yDir], r[yDir]),
                                                                                comp == zDir ? evalEdge<degree>(c, rr[zDir], dr[zDir], r[zDir]) : evalLagrange<degree>(c, rr[zDir], dr[zDir], r[zDir]));
                GEMPIC_D_LOOP_END
                break;
            case 2:
                GEMPIC_D_LOOP_BEGIN(
                for (int a = 1 - degree/2; a < degree/2 + (comp == xDir ? 1 : 0); ++a),
                    for (int b = 1 - degree/2; b < degree/2 + (comp == yDir ? 1 : 0); ++b),
                        for (int c = 1 - degree/2; c < degree/2 + (comp == zDir ? 1 : 0); ++c))
                            fval += funcMF(i + a, j + b, k + c) * GEMPIC_D_MULT(comp == xDir ? evalLagrange<degree>(a, rr[xDir], dr[xDir], r[xDir]) : evalEdge<degree>(a, rr[xDir], dr[xDir], r[xDir]),
                                                                                comp == yDir ? evalLagrange<degree>(b, rr[yDir], dr[yDir], r[yDir]) : evalEdge<degree>(b, rr[yDir], dr[yDir], r[yDir]),
                                                                                comp == zDir ? evalLagrange<degree>(c, rr[zDir], dr[zDir], r[zDir]) : evalEdge<degree>(c, rr[zDir], dr[zDir], r[zDir]));
                GEMPIC_D_LOOP_END
                break;
            case 3:
                GEMPIC_D_LOOP_BEGIN(
                for (int a = 1 - degree/2; a < degree/2; ++a),
                    for (int b = 1 - degree/2; b < degree/2; ++b),
                        for (int c = 1 - degree/2; c < degree/2; ++c))
                            fval += funcMF(i + a, j + b, k + c) * GEMPIC_D_MULT(evalEdge<degree>(a, rr[xDir], dr[xDir], r[xDir]),
                                                                                evalEdge<degree>(b, rr[yDir], dr[yDir], r[yDir]),
                                                                                evalEdge<degree>(c, rr[zDir], dr[zDir], r[zDir]));
                GEMPIC_D_LOOP_END
                break;
            
            default:
                break;
            }
            
            switch (form)
            {
            case 0:
                error += func(AMREX_D_DECL(r[xDir] + (dual ? dr[xDir] : 0),
                                            r[yDir] + (dual ? dr[yDir] : 0),
                                            r[zDir] + (dual ? dr[zDir] : 0)), t) - fval;
                break;
            case 1:
                error += func(AMREX_D_DECL(r[xDir] + (dual && comp != xDir ? dr[xDir] : 0),
                                            r[yDir] + (dual && comp != yDir? dr[yDir] : 0),
                                            r[zDir] + (dual && comp != zDir ? dr[zDir] : 0)), t) - fval;
                break;
            case 2:
                error += func(AMREX_D_DECL(r[xDir] + (dual && comp == xDir ? dr[xDir] : 0),
                                            r[yDir] + (dual && comp == yDir ? dr[yDir] : 0),
                                            r[zDir] + (dual && comp == zDir ? dr[zDir] : 0)), t) - fval;
                break;
            case 3:
                error += func(AMREX_D_DECL(r[xDir], r[yDir], r[zDir]), t) - fval;
                break;
            default:
                break;
            }

            valuesMF(i,j,k) = error;
        });
    }
    return values.norm0();
}

}

#endif