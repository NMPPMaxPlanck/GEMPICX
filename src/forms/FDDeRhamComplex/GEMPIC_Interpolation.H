#ifndef GEMPIC_INTERPOLATION_H_
#define GEMPIC_INTERPOLATION_H_

#include <GEMPIC_Fields.H>
#include <GEMPIC_Params.H>

using namespace GEMPIC_Fields;

namespace GEMPIC_Interpolation{

template<int degree>
AMREX_GPU_DEVICE amrex::Real evalLagrange(const int polyIndex, const amrex::Real xk, const amrex::Real dx, const amrex::Real y)
{
    /**
    * @brief Evaluates a 1D Lagrange polynomial
    * 
    * Description:
    * ----------
    * Evaluates a Lagrange polynomial $L_i$ on the interval $[x_k, x_{k+1}], where $L_0$ is
    * the Lagrange polynomial associated to node $x_k$, $L_{-1}$ to $x_{k-1}$, $L_1$ to $x_{k+1}$,
    * and so forth.
    * The whole grid is assumed to be uniform, evaluation is done using Neville's algorithm.
    * 
    * 
    * Parameters:
    * @tparam degree : int, degree of basis functions used
    * @param polyIndex : int, indicates the polynomial w.r.t. $x_k$
    * @param xk : Real, node $x_k$ which defines the interval of evaluation
    * @param dx : (uniform) grid size
    * @param y : Real, point in $[x_k, x_{k+1}]$, where $L_i$ is to be evaluated
    * 
    * Returns:
    * ----------
    * Returns Real, $L_{polyIndex} (y)$
    */
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

template<int degree>
AMREX_GPU_DEVICE amrex::Real evalEdge(const int polyIndex, const amrex::Real xk, const amrex::Real dx, const amrex::Real y)
{
    /**
    * @brief Evaluates a 1D edge function
    * 
    * Description:
    * ----------
    * Evaluates an edge function $e_i$ on the interval $[x_k, x_{k+1}]$, where $e_0$ is
    * the edge function associated to the cell $[x_k,  x_{k+1}]$, $e_{-1}$ to $[x_{k-1},  x_k]$,
    * $e_1$ to $[x_{k+1},  x_{k+2}]$, and so forth.
    * The whole grid is assumed to be uniform, only edge functions of degree 0, 2, 4 can be evaluated.
    * 
    * 
    * Parameters:
    * @tparam degree : int, degree of basis functions used
    * @param polyIndex : int, indicates the edge function w.r.t. $[x_k, x_{k+1}]$
    * @param xk : Real, node $x_k$ which defines the interval of evaluation
    * @param dx : (uniform) grid size
    * @param y : Real, point in $[x_k, x_{k+1}]$, where $e_i$ is to be evaluated
    * 
    * Returns:
    * ----------
    * Returns Real, $e_{polyIndex} (y)$
    */
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

template<int degree>
amrex::Real maxErrorMidpoint(amrex::Geometry geom, amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func, amrex::MultiFab &func_h,
                    amrex::RealVect dr, int form, bool dual, int comp = 0, amrex::Real t = 0.)
{
    /**
    * @brief Computes the maximum norm w.r.t. the values at the cell midpoints
    * 
    * Description:
    * ----------
    * Interpolates the field, evaluates it a the cell midpoint and computes the max error
    * 
    * 
    * Parameters:
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
    * Returns:
    * ----------
    * Returns void
    */

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
            AMREX_D_DECL(dr[0] / 2, dr[1] / 2, dr[2] / 2)
        };
        
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            // Compute the position of the point i, j, k
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> rr = 
            {
                AMREX_D_DECL(r0[0] + i*dr[0] - (dual ? drHalf[0] : 0),
                             r0[1] + j*dr[1] - (dual ? drHalf[1] : 0),
                             r0[2] + k*dr[2] - (dual ? drHalf[2] : 0))
            };

            amrex::Real error = 0.0;

            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r = 
            {
                AMREX_D_DECL(rr[0] + drHalf[0], rr[1] + drHalf[1], rr[2] + drHalf[2])
            };

            amrex::Real fval = 0;
            int b{0}, c{0};
            switch (form)
            {
            case 0:
                GEMPIC_D_LOOP_BEGIN(
                for (int a = 1 - degree/2; a < degree/2 + 1; ++a),
                    for (int b = 1 - degree/2; b < degree/2 + 1; ++b),
                        for (int c = 1 - degree/2; c < degree/2 + 1; ++c))
                            fval += funcMF(i + a, j + b, k + c) * GEMPIC_D_MULT(evalLagrange<degree>(a, rr[0] + (dual ? dr[0] : 0), dr[0], r[0] + (dual ? dr[0] : 0)),
                                                                                evalLagrange<degree>(b, rr[1] + (dual ? dr[1] : 0), dr[1], r[1] + (dual ? dr[1] : 0)),
                                                                                evalLagrange<degree>(c, rr[2] + (dual ? dr[2] : 0), dr[2], r[2] + (dual ? dr[2] : 0)));
                GEMPIC_D_LOOP_END
                break;
            case 1:
                GEMPIC_D_LOOP_BEGIN(
                for (int a = 1 - degree/2; a < degree/2 + (comp == 0 ? 0 : 1); ++a),
                    for (int b = 1 - degree/2; b < degree/2 + (comp == 1 ? 0 : 1); ++b),
                        for (int c = 1 - degree/2; c < degree/2 + (comp == 2 ? 0 : 1); ++c))
                            fval += funcMF(i + a, j + b, k + c) * GEMPIC_D_MULT(comp == 0 ? evalEdge<degree>(a, rr[0], dr[0], r[0]) : evalLagrange<degree>(a, rr[0], dr[0], r[0]),
                                                                                comp == 1 ? evalEdge<degree>(b, rr[1], dr[1], r[1]) : evalLagrange<degree>(b, rr[1], dr[1], r[1]),
                                                                                comp == 2 ? evalEdge<degree>(c, rr[2], dr[2], r[2]) : evalLagrange<degree>(c, rr[2], dr[2], r[2]));
                GEMPIC_D_LOOP_END
                break;
            case 2:
                GEMPIC_D_LOOP_BEGIN(
                for (int a = 1 - degree/2; a < degree/2 + (comp == 0 ? 1 : 0); ++a),
                    for (int b = 1 - degree/2; b < degree/2 + (comp == 1 ? 1 : 0); ++b),
                        for (int c = 1 - degree/2; c < degree/2 + (comp == 2 ? 1 : 0); ++c))
                            fval += funcMF(i + a, j + b, k + c) * GEMPIC_D_MULT(comp == 0 ? evalLagrange<degree>(a, rr[0], dr[0], r[0]) : evalEdge<degree>(a, rr[0], dr[0], r[0]),
                                                                                comp == 1 ? evalLagrange<degree>(b, rr[1], dr[1], r[1]) : evalEdge<degree>(b, rr[1], dr[1], r[1]),
                                                                                comp == 2 ? evalLagrange<degree>(c, rr[2], dr[2], r[2]) : evalEdge<degree>(c, rr[2], dr[2], r[2]));
                GEMPIC_D_LOOP_END
                break;
            case 3:
                GEMPIC_D_LOOP_BEGIN(
                for (int a = 1 - degree/2; a < degree/2; ++a),
                    for (int b = 1 - degree/2; b < degree/2; ++b),
                        for (int c = 1 - degree/2; c < degree/2; ++c))
                            fval += funcMF(i + a, j + b, k + c) * GEMPIC_D_MULT(evalEdge<degree>(a, rr[0], dr[0], r[0]),
                                                                                evalEdge<degree>(b, rr[1], dr[1], r[1]),
                                                                                evalEdge<degree>(c, rr[2], dr[2], r[2]));
                GEMPIC_D_LOOP_END
                break;
            
            default:
                break;
            }
            
            switch (form)
            {
            case 0:
                error += func(AMREX_D_DECL(r[0] + (dual ? dr[0] : 0),
                                            r[1] + (dual ? dr[1] : 0),
                                            r[2] + (dual ? dr[2] : 0)), t) - fval;
                break;
            case 1:
                error += func(AMREX_D_DECL(r[0] + (dual && comp != 0 ? dr[0] : 0),
                                            r[1] + (dual && comp != 1 ? dr[1] : 0),
                                            r[2] + (dual && comp != 2 ? dr[2] : 0)), t) - fval;
                break;
            case 2:
                error += func(AMREX_D_DECL(r[0] + (dual && comp == 0 ? dr[0] : 0),
                                            r[1] + (dual && comp == 1 ? dr[1] : 0),
                                            r[2] + (dual && comp == 2 ? dr[2] : 0)), t) - fval;
                break;
            case 3:
                error += func(AMREX_D_DECL(r[0], r[1], r[2]), t) - fval;
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