#include <tinyexpr.h>

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_CoordSys.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_Geometry.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLNodeLaplacian.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>

#include <GEMPIC_gempic_norm.H>
#include <GEMPIC_infrastructure.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_mesh_coupling.H>

using namespace Infra;
using namespace Utils;

#ifndef MAXWELL_YEE_H_
#define MAXWELL_YEE_H_

namespace Gempic {
namespace Field_solvers {
/**
 * Class storing all the field information (both analytical (@p E_sol_Array, @p B_sol_Array) and numerical (@p E_Array, @p B_Array, @p J_Array), infrastructure information
 * for the fields (@p E_Index, @p B_Index, @p Nghost, @p Ncomp, @p time) and the necessary components to solve the Poisson problem for the first step (@p phi_init, @p rho,
 * @p phi, @p sigma). In addition, the class can initialize the fields (init_E_B, init_B) and the functions for the Poisson problem (init_rho_phi), advance the complete
 * problem (advance) or parts of it (advance_E, advance_B, advance_time), initialize E by solving the Poisson problem (solve_poisson), fill ghost cells of all relevant MultiFabs
 * (FillBD), compute the analytical solution (exactSolution) and compute the error of the numerical solution (computeError).
 */
class maxwell_yee
{
public:

    int nsteps; // number of simulation steps
    amrex::Real dt;
    std::array<std::unique_ptr<amrex::MultiFab>, GEMPIC_VDIM > E_Array;
    std::array<std::unique_ptr<amrex::MultiFab>, GEMPIC_VDIM > J_Array;
    std::array<std::unique_ptr<amrex::MultiFab>, GEMPIC_BDIM > B_Array;
    std::array<std::unique_ptr<amrex::MultiFab>, GEMPIC_VDIM > E_sol_Array;
    std::array<std::unique_ptr<amrex::MultiFab>, GEMPIC_BDIM > B_sol_Array;
    std::array<std::unique_ptr<amrex::IndexType>, GEMPIC_VDIM> E_Index;
    std::array<std::unique_ptr<amrex::IndexType>, GEMPIC_BDIM> B_Index;
    amrex::MultiFab phi_init; // for Poisson
    amrex::MultiFab rho; // for Poisson
    amrex::MultiFab rho_gauss_law; // for Poisson
    int Nghost;
    amrex::MultiFab phi; // for Poisson
    //amrex::MLMG* mlmg_ptr;

private:

    int Ncomp;
    amrex::Real time;
    amrex::MultiFab sigma; //for Poisson

public:

    // The default constructor.
    /**
     * Default Constructor.
     */
    maxwell_yee ();

    // Explicit const
    /**
     * Explicit constructor: all class attributes can be initialized by passing an object of class initializer, infrastructure and the number of ghost cells.
     */
    maxwell_yee (initializer init, infrastructure ifr, int NG){
        // set index indicating nodal-/cell-centered
        // 0 -> cell-centered | 1 -> node-centered

        E_Index[0].reset(new amrex::IndexType (amrex::IntVect{AMREX_D_DECL(0,1,1)}));
#if (GEMPIC_VDIM > 1)
        E_Index[1].reset(new amrex::IndexType (amrex::IntVect{AMREX_D_DECL(1,0,1)}));
#endif
#if (GEMPIC_VDIM > 2)
        E_Index[2].reset(new amrex::IndexType (amrex::IntVect{AMREX_D_DECL(1,1,0)}));
#endif

        B_Index[0].reset(new amrex::IndexType (amrex::IntVect{AMREX_D_DECL(1,0,0)}));
#if (GEMPIC_BDIM > 1)
        B_Index[1].reset(new amrex::IndexType (amrex::IntVect{AMREX_D_DECL(0,1,0)}));
#endif
#if (GEMPIC_BDIM > 2)
        B_Index[2].reset(new amrex::IndexType (amrex::IntVect{AMREX_D_DECL(0,0,1)}));
#endif

        Nghost = NG;
        Ncomp = 1;

        dt = init.dt;
        time = 0.0;

        // make the multifabs nodal-/cell-centered according to indices
        for (int i = 0; i<GEMPIC_VDIM; i++) {
            E_Array[i].reset(new amrex::MultiFab (convert(ifr.grid, *E_Index[i]),ifr.distriMap,Ncomp,Nghost));
            E_sol_Array[i].reset(new amrex::MultiFab (convert(ifr.grid, *E_Index[i]),ifr.distriMap,Ncomp,Nghost));
            J_Array[i].reset(new amrex::MultiFab (convert(ifr.grid, *E_Index[i]),ifr.distriMap,Ncomp,Nghost));
        }
        for (int i = 0; i<GEMPIC_BDIM; i++) {
            B_Array[i].reset(new amrex::MultiFab (convert(ifr.grid, *B_Index[i]),ifr.distriMap,Ncomp,Nghost));
            B_sol_Array[i].reset(new amrex::MultiFab (convert(ifr.grid, *B_Index[i]),ifr.distriMap,Ncomp,Nghost));
        }

        // for Poisson:
        //node centered BA:
        const amrex::BoxArray& nba = amrex::convert(ifr.grid,amrex::IntVect::TheNodeVector());
        phi     .define(nba        , ifr.distriMap, 1, Nghost); //box, distriMap, nVar, nGrow
        rho     .define(nba        , ifr.distriMap, 1, Nghost);
        rho_gauss_law.define(nba        , ifr.distriMap, 1, Nghost);
        phi_init.define(nba        , ifr.distriMap, 1, Nghost);
        // sigma is cell-centered (default, so no need to pass IndexVector)
        sigma   .define(ifr.grid, ifr.distriMap, 1, 0);
        sigma.setVal(-1.0); // sigma is the identity (lapl = nabla dot ID nabla)

        nsteps = init.n_steps;

    }

    /**
     * Initialize E and B field for time t=0 with a given array of functions.
     */
    void init_E_B (double (*fields[GEMPIC_VDIM+GEMPIC_BDIM]) (std::array<double,GEMPIC_SPACEDIM> x, double t),infrastructure ifr){

        std::array<double,GEMPIC_SPACEDIM> x;
        double half_dt = dt/2.0; // for leap frog initialization

        // E_field
        for (int i = 0; i<GEMPIC_VDIM; i++) {
            for ( amrex::MFIter mfi(*E_Array[i]); mfi.isValid(); ++mfi ){

                const amrex::Box& bx = mfi.validbox();
                amrex::IntVect lo = {bx.smallEnd()};
                amrex::IntVect hi = {bx.bigEnd()};
#if (GEMPIC_SPACEDIM > 2)
                for(int k=lo[2]; k<=hi[2]; k++){
                    x[2] = ifr.geom.ProbLo()[2] + ((double)k+0.5-0.5*(*E_Index[i])[2])*ifr.dx[2];
#endif
#if (GEMPIC_SPACEDIM > 1)
                    for(int j=lo[1]; j<=hi[1]; j++){
                        x[1] = ifr.geom.ProbLo()[1] + ((double)j+0.5-0.5*(*E_Index[i])[1])*ifr.dx[1];
#endif
                        for(int l=lo[0]; l<=hi[0]; l++){
                            x[0] = ifr.geom.ProbLo()[0] + ((double)l+0.5-0.5*(*E_Index[i])[0])*ifr.dx[0];
                            // the box for these values:
                            amrex::Box cc(amrex::IntVect{AMREX_D_DECL(l,j,k)}, amrex::IntVect{AMREX_D_DECL(l,j,k)}, *E_Index[i]);
                            (*E_Array[i])[mfi].setVal((*fields[i])(x,0), cc, 0, 1);
                        }
#if (GEMPIC_SPACEDIM > 1)
                    }
#endif
#if (GEMPIC_SPACEDIM > 2)
                }
#endif
                // next mfi
            }
            (*E_Array[i]).FillBoundary(ifr.geom.periodicity());
        }

        // B_field
        for (int i = 0; i<GEMPIC_BDIM; i++) {
            for ( amrex::MFIter mfi(*B_Array[i]); mfi.isValid(); ++mfi ){

                const amrex::Box& bx = mfi.validbox();
                amrex::IntVect lo = {bx.smallEnd()};
                amrex::IntVect hi = {bx.bigEnd()};

#if (GEMPIC_SPACEDIM > 2)
                for(int k=lo[2]; k<=hi[2]; k++){
                    x[2] = ifr.geom.ProbLo()[2] + ((double)k+0.5-0.5*(*B_Index[i])[2])*ifr.dx[2];
#endif
#if (GEMPIC_SPACEDIM > 1)
                    for(int j=lo[1]; j<=hi[1]; j++){
                        x[1] = ifr.geom.ProbLo()[1] + ((double)j+0.5-0.5*(*B_Index[i])[1])*ifr.dx[1];
#endif
                        for(int l=lo[0]; l<=hi[0]; l++){
                            x[0] = ifr.geom.ProbLo()[0] + ((double)l+0.5-0.5*(*B_Index[i])[0])*ifr.dx[0];
                            // the box for these values:
                            amrex::Box cc(amrex::IntVect{AMREX_D_DECL(l,j,k)}, amrex::IntVect{AMREX_D_DECL(l,j,k)}, *B_Index[i]);
                            (*B_Array[i])[mfi].setVal((*fields[i+GEMPIC_VDIM])(x,half_dt), cc, 0, 1);
                        }
#if (GEMPIC_SPACEDIM > 1)
                    }
#endif
#if (GEMPIC_SPACEDIM > 2)
                }
#endif
                // next mfi
            }
            (*B_Array[i]).FillBoundary(ifr.geom.periodicity());
        }
    }

    /**
     * Initialize E field for time t=0 with a given array of functions.
     */
    void initE (infrastructure infra,
                te_expr *Ex_parse,
                te_expr *Ey_parse,
                te_expr *Ez_parse,
                double *x,
                double *y,
                double *z) {



        for (int i = 0; i<GEMPIC_VDIM; i++) {
            for ( amrex::MFIter mfi(*E_Array[i]); mfi.isValid(); ++mfi ) {
                const amrex::Box& bx = mfi.validbox();
                amrex::IntVect lo = {bx.smallEnd()};
                amrex::IntVect hi = {bx.bigEnd()};
#if (GEMPIC_SPACEDIM > 2)
                for(int k=lo[2]; k<=hi[2]; k++){
                    *z = infra.geom.ProbLo()[2] + ((double)k+0.5-0.5*(*E_Index[i])[2])*infra.dx[2];
#endif
#if (GEMPIC_SPACEDIM > 1)
                    for(int j=lo[1]; j<=hi[1]; j++){
                        *y = infra.geom.ProbLo()[1] + ((double)j+0.5-0.5*(*E_Index[i])[1])*infra.dx[1];
#endif
                        for(int l=lo[0]; l<=hi[0]; l++){
                            *x = infra.geom.ProbLo()[0] + ((double)l+0.5-0.5*(*E_Index[i])[0])*infra.dx[0];
                            // the box for these values:
                            amrex::Box cc(amrex::IntVect{AMREX_D_DECL(l,j,k)}, amrex::IntVect{AMREX_D_DECL(l,j,k)}, *E_Index[i]);
                            const double result = (i==0 ? te_eval(Ex_parse) : (i==1 ? te_eval(Ey_parse) : te_eval(Ez_parse)));
                            (*E_Array[i])[mfi].setVal(result, cc, 0, 1);
                        }
#if (GEMPIC_SPACEDIM > 2)
                    }
#endif
#if (GEMPIC_SPACEDIM > 1)
                }
#endif
                // next mfi
            }
            (*E_Array[i]).FillBoundary(infra.geom.periodicity());
        }
    }

    /**
     * Initialize B field for time t=0 with a given array of functions.
     */
    void initB (infrastructure infra,
                te_expr *Bx_parse,
                te_expr *By_parse,
                te_expr *Bz_parse,
                double *x,
                double *y,
                double *z) {



        for (int i = 0; i<GEMPIC_BDIM; i++) {
            for ( amrex::MFIter mfi(*B_Array[i]); mfi.isValid(); ++mfi ) {
                const amrex::Box& bx = mfi.validbox();
                amrex::IntVect lo = {bx.smallEnd()};
                amrex::IntVect hi = {bx.bigEnd()};
#if (GEMPIC_SPACEDIM > 2)
                for(int k=lo[2]; k<=hi[2]; k++){
                    *z = infra.geom.ProbLo()[2] + ((double)k+0.5-0.5*(*B_Index[i])[2])*infra.dx[2];
#endif
#if (GEMPIC_SPACEDIM > 1)
                    for(int j=lo[1]; j<=hi[1]; j++){
                        *y = infra.geom.ProbLo()[1] + ((double)j+0.5-0.5*(*B_Index[i])[1])*infra.dx[1];
#endif
                        for(int l=lo[0]; l<=hi[0]; l++){
                            *x = infra.geom.ProbLo()[0] + ((double)l+0.5-0.5*(*B_Index[i])[0])*infra.dx[0];
                            // the box for these values:
                            amrex::Box cc(amrex::IntVect{AMREX_D_DECL(l,j,k)}, amrex::IntVect{AMREX_D_DECL(l,j,k)}, *B_Index[i]);
                            const double result = (i==0 ? te_eval(Bx_parse) : (i==1 ? te_eval(By_parse) : te_eval(Bz_parse)));
                            (*B_Array[i])[mfi].setVal(result, cc, 0, 1);
                        }
#if (GEMPIC_SPACEDIM > 2)
                    }
#endif
#if (GEMPIC_SPACEDIM > 1)
                }
#endif
                // next mfi
            }
            (*B_Array[i]).FillBoundary(infra.geom.periodicity());
        }
    }

    /**
     * Fill Boundaries of all relevant Fabs (E, J, B, rho).
     */
    void FillBD (infrastructure ifr){
        for(int cc=0;cc<GEMPIC_VDIM;cc++){
            (*E_Array[cc]).FillBoundary(ifr.geom.periodicity());
            (*E_sol_Array[cc]).FillBoundary(ifr.geom.periodicity()); //TODO: remove
        }
        for(int cc=0;cc<GEMPIC_VDIM;cc++){
            (*J_Array[cc]).FillBoundary(ifr.geom.periodicity());
        }
        for(int cc=0;cc<GEMPIC_BDIM;cc++){
            (*B_Array[cc]).FillBoundary(ifr.geom.periodicity());
        }
        rho.FillBoundary(ifr.geom.periodicity());
    }

    /**
     * Advance time
     */
    void advance_time () {
        time = time+dt;
    }

    /**
     * Advance E by a time step of length @p dt_step with a Yee scheme
     */
    void advance_E (infrastructure ifr, amrex::Real dt_step){

        amrex::Real c1[1], c2[1], c3[1], c4[1], c5[1], cj[1];

        // E-field
        for (int i=0; i<GEMPIC_VDIM; i++) {
            for (amrex::MFIter mfi(*E_Array[i]); mfi.isValid(); ++mfi ) {
                const amrex::Box& bx = mfi.validbox();

                amrex::IntVect lo = {bx.smallEnd()};
                amrex::IntVect hi = {bx.bigEnd()};

#if (GEMPIC_SPACEDIM > 2)
                for(int k=lo[2]; k<=hi[2]; k++){
#endif
#if (GEMPIC_SPACEDIM > 1)
                    for(int j=lo[1]; j<=hi[1]; j++){
#endif
                        for(int l=lo[0]; l<=hi[0]; l++){

                            // the box for these values:
                            amrex::Box cc(amrex::IntVect{AMREX_D_DECL(l,j,k)}, amrex::IntVect{AMREX_D_DECL(l,j,k)}, *E_Index[i]);

                            // old field_out value
                            (*E_Array[i])[mfi].getVal(c1, amrex::IntVect{AMREX_D_DECL(l,j,k)},0,1);
                            // J value
                            (*J_Array[i])[mfi].getVal(cj, amrex::IntVect{AMREX_D_DECL(l,j,k)},0,1);

                            // B-derivatives
#if (GEMPIC_VDIM < 3 & GEMPIC_SPACEDIM == 1)
                            if (i == 0){
                                (*E_Array[0])[mfi].setVal(*c1+dt_step*(-*cj), cc, 0, 1);
                            }
#endif
#if (GEMPIC_VDIM == 2 & GEMPIC_SPACEDIM == 2)
                            //Ex for 2D2V
                            if (i == 0) {
                                (*B_Array[0])[mfi].getVal(c2,amrex::IntVect{AMREX_D_DECL(  l,j,k)},0,1);
                                (*B_Array[0])[mfi].getVal(c3,amrex::IntVect{AMREX_D_DECL(l,j-1,k)},0,1);
                                (*E_Array[0])[mfi].setVal(*c1+dt_step*((*c2-*c3)/ifr.dx[1]-*cj), cc, 0, 1);
                            }
#endif
#if (GEMPIC_VDIM == 2)
                            if (i == 1) {
                                (*B_Array[0])[mfi].getVal(c2,amrex::IntVect{AMREX_D_DECL(  l,j,k)},0,1);
                                (*B_Array[0])[mfi].getVal(c3,amrex::IntVect{AMREX_D_DECL(l-1,j,k)},0,1);
                                (*E_Array[1])[mfi].setVal(*c1+dt_step*(-(*c2-*c3)/ifr.dx[0]-*cj), cc, 0, 1);
                            }
#endif
#if (GEMPIC_VDIM == 3)
                            (*B_Array[(i+2)%GEMPIC_SPACEDIM])[mfi].getVal(c2,amrex::IntVect{AMREX_D_DECL(  l,  j  ,k)},0,1);
                            (*B_Array[(i+2)%GEMPIC_SPACEDIM])[mfi].getVal(c3,amrex::IntVect{AMREX_D_DECL(l-(i%GEMPIC_SPACEDIM==2?1:0),j-(i%GEMPIC_SPACEDIM==0?1:0),k-(i%GEMPIC_SPACEDIM==1?1:0))},0,1);
                            (*B_Array[(i+1)%GEMPIC_SPACEDIM])[mfi].getVal(c4, amrex::IntVect{AMREX_D_DECL(  l,  j,  k)},0,1);
                            (*B_Array[(i+1)%GEMPIC_SPACEDIM])[mfi].getVal(c5,amrex::IntVect{AMREX_D_DECL(l-(i%GEMPIC_SPACEDIM==1?1:0),j-(i%GEMPIC_SPACEDIM==2?1:0),k-(i%GEMPIC_SPACEDIM==0?1:0))},0,1);

                            (*E_Array[i])[mfi].setVal(*c1+dt_step*((*c2-*c3)/ifr.dx[(i+1)%GEMPIC_SPACEDIM]-
                                                      (*c4-*c5)/ifr.dx[(i+2)%GEMPIC_SPACEDIM]-*cj), cc, 0, 1);
#endif
                        }
#if (GEMPIC_SPACEDIM > 1)
                    }
#endif
#if (GEMPIC_SPACEDIM > 2)
                }
#endif
                // next mfi
            }
            (*E_Array[i]).FillBoundary(ifr.geom.periodicity());
        }
    }

    /**
     * Advance E by a time step of length @p dt_step with a Yee scheme by the B-Field contribution
     */
    void advance_E_from_B (infrastructure ifr, amrex::Real dt_step){

        amrex::Real c1[1], c2[1], c3[1], c4[1], c5[1];

        // E-field
        for (int i=0; i<GEMPIC_VDIM; i++) {
            for (amrex::MFIter mfi(*E_Array[i]); mfi.isValid(); ++mfi ) {
                const amrex::Box& bx = mfi.validbox();

                amrex::IntVect lo = {bx.smallEnd()};
                amrex::IntVect hi = {bx.bigEnd()};

#if (GEMPIC_SPACEDIM > 2)
                for(int k=lo[2]; k<=hi[2]; k++){
#endif
#if (GEMPIC_SPACEDIM > 1)
                    for(int j=lo[1]; j<=hi[1]; j++){
#endif
                        for(int l=lo[0]; l<=hi[0]; l++){

                            // the box for these values:
                            amrex::Box cc(amrex::IntVect{AMREX_D_DECL(l,j,k)}, amrex::IntVect{AMREX_D_DECL(l,j,k)}, *E_Index[i]);

                            // old field_out value
                            (*E_Array[i])[mfi].getVal(c1, amrex::IntVect{AMREX_D_DECL(l,j,k)},0,1);

                            // B-derivatives
#if (GEMPIC_VDIM < 3 & GEMPIC_SPACEDIM == 1)
                            if (i == 0){
                                (*E_Array[0])[mfi].setVal(*c1+dt_step, cc, 0, 1);
                            }
#endif
#if (GEMPIC_VDIM == 2 & GEMPIC_SPACEDIM == 2)
                            //Ex for 2D2V
                            if (i == 0) {
                                (*B_Array[0])[mfi].getVal(c2,amrex::IntVect{AMREX_D_DECL(  l,j,k)},0,1);
                                (*B_Array[0])[mfi].getVal(c3,amrex::IntVect{AMREX_D_DECL(l,j-1,k)},0,1);
                                (*E_Array[0])[mfi].setVal(*c1+dt_step*((*c2-*c3)/ifr.dx[1]), cc, 0, 1);
                            }
#endif
#if (GEMPIC_VDIM == 2)
                            if (i == 1) {
                                (*B_Array[0])[mfi].getVal(c2,amrex::IntVect{AMREX_D_DECL(  l,j,k)},0,1);
                                (*B_Array[0])[mfi].getVal(c3,amrex::IntVect{AMREX_D_DECL(l-1,j,k)},0,1);
                                (*E_Array[1])[mfi].setVal(*c1+dt_step*(-(*c2-*c3)/ifr.dx[0]), cc, 0, 1);
                            }
#endif
#if (GEMPIC_VDIM == 3)
                            (*B_Array[(i+2)%GEMPIC_SPACEDIM])[mfi].getVal(c2,amrex::IntVect{AMREX_D_DECL(  l,  j  ,k)},0,1);
                            (*B_Array[(i+2)%GEMPIC_SPACEDIM])[mfi].getVal(c3,amrex::IntVect{AMREX_D_DECL(l-(i%GEMPIC_SPACEDIM==2?1:0),j-(i%GEMPIC_SPACEDIM==0?1:0),k-(i%GEMPIC_SPACEDIM==1?1:0))},0,1);
                            (*B_Array[(i+1)%GEMPIC_SPACEDIM])[mfi].getVal(c4, amrex::IntVect{AMREX_D_DECL(  l,  j,  k)},0,1);
                            (*B_Array[(i+1)%GEMPIC_SPACEDIM])[mfi].getVal(c5,amrex::IntVect{AMREX_D_DECL(l-(i%GEMPIC_SPACEDIM==1?1:0),j-(i%GEMPIC_SPACEDIM==2?1:0),k-(i%GEMPIC_SPACEDIM==0?1:0))},0,1);

                            (*E_Array[i])[mfi].setVal(*c1+dt_step*((*c2-*c3)/ifr.dx[(i+1)%GEMPIC_SPACEDIM]-
                                                      (*c4-*c5)/ifr.dx[(i+2)%GEMPIC_SPACEDIM]), cc, 0, 1);
#endif
                        }
#if (GEMPIC_SPACEDIM > 1)
                    }
#endif
#if (GEMPIC_SPACEDIM > 2)
                }
#endif
                // next mfi
            }
            (*E_Array[i]).FillBoundary(ifr.geom.periodicity());
        }
    }

    /**
     * Advance E by a time step of length @p dt_step with a Yee scheme by the J-Field contribution
     */
    void advance_E_from_J (infrastructure ifr, amrex::Real dt_step){

        amrex::Real c1[1], cj[1];

        // E-field
        for (int i=0; i<GEMPIC_VDIM; i++) {
            for (amrex::MFIter mfi(*E_Array[i]); mfi.isValid(); ++mfi ) {
                const amrex::Box& bx = mfi.validbox();

                amrex::IntVect lo = {bx.smallEnd()};
                amrex::IntVect hi = {bx.bigEnd()};

#if (GEMPIC_SPACEDIM > 2)
                for(int k=lo[2]; k<=hi[2]; k++){
#endif
#if (GEMPIC_SPACEDIM > 1)
                    for(int j=lo[1]; j<=hi[1]; j++){
#endif
                        for(int l=lo[0]; l<=hi[0]; l++){

                            // the box for these values:
                            amrex::Box cc(amrex::IntVect{AMREX_D_DECL(l,j,k)}, amrex::IntVect{AMREX_D_DECL(l,j,k)}, *E_Index[i]);

                            // old field_out value
                            (*E_Array[i])[mfi].getVal(c1, amrex::IntVect{AMREX_D_DECL(l,j,k)},0,1);
                            // J value
                            (*J_Array[i])[mfi].getVal(cj, amrex::IntVect{AMREX_D_DECL(l,j,k)},0,1);
                            (*E_Array[i])[mfi].setVal(*c1+dt_step*(-*cj), cc, 0, 1);

                        }
#if (GEMPIC_SPACEDIM > 1)
                    }
#endif
#if (GEMPIC_SPACEDIM > 2)
                }
#endif
                // next mfi
            }
            (*E_Array[i]).FillBoundary(ifr.geom.periodicity());
        }
    }

    /**
     * Advance B by a time step of length @p dt_step with a Yee scheme
     */
    void advance_B (infrastructure ifr, amrex::Real dt_step){

        amrex::Real c1[1], c2[1], c3[1], c4[1], c5[1];

        // B-field
        for (int i=0; i<GEMPIC_BDIM; i++) {
            for (amrex::MFIter mfi(*B_Array[i]); mfi.isValid(); ++mfi ) {
                const amrex::Box& bx = mfi.validbox();

                amrex::IntVect lo = {bx.smallEnd()};
                amrex::IntVect hi = {bx.bigEnd()};

#if (GEMPIC_SPACEDIM > 2)
                for(int k=lo[2]; k<=hi[2]; k++){
#endif
#if (GEMPIC_SPACEDIM > 1)
                    for(int j=lo[1]; j<=hi[1]; j++){
#endif
                        for(int l=lo[0]; l<=hi[0]; l++){

                            // the box for these values:
                            amrex::Box cc(amrex::IntVect{AMREX_D_DECL(l,j,k)}, amrex::IntVect{AMREX_D_DECL(l,j,k)}, *B_Index[i]);

                            // old field_out value
                            (*B_Array[i])[mfi].getVal(c1, amrex::IntVect{AMREX_D_DECL(l,j,k)},0,1);

                            // E-derivatives
#if (GEMPIC_VDIM == 1)
                            //no B
#elif (GEMPIC_VDIM == 2 & GEMPIC_SPACEDIM == 1)
                            (*E_Array[1])[mfi].getVal(c4, amrex::IntVect{AMREX_D_DECL(l+1,j,k)},0,1);
                            (*E_Array[1])[mfi].getVal(c5, amrex::IntVect{AMREX_D_DECL(l  ,j,k)},0,1);

                            (*B_Array[0])[mfi].setVal(*c1-dt_step*((*c4-*c5)/ifr.dx[0]), cc, 0, 1);
#elif (GEMPIC_VDIM == 2 & GEMPIC_SPACEDIM == 2)
                            (*E_Array[0])[mfi].getVal(c2,amrex::IntVect{AMREX_D_DECL(l,j+1,k)},0,1);
                            (*E_Array[0])[mfi].getVal(c3,amrex::IntVect{AMREX_D_DECL(l,j,k)},0,1);
                            (*E_Array[1])[mfi].getVal(c4,amrex::IntVect{AMREX_D_DECL(l+1,j,k)},0,1);
                            (*E_Array[1])[mfi].getVal(c5, amrex::IntVect{AMREX_D_DECL(l,j,k)},0,1);

                            (*B_Array[0])[mfi].setVal(*c1+dt_step*((*c2-*c3)/ifr.dx[1]-
                                                      (*c4-*c5)/ifr.dx[0]), cc, 0, 1);
#elif (GEMPIC_BDIM == 3)
                            (*E_Array[(i+2)%GEMPIC_SPACEDIM])[mfi].getVal(c2,amrex::IntVect{AMREX_D_DECL(l+(i%GEMPIC_SPACEDIM==2?1:0),j+(i%GEMPIC_SPACEDIM==0?1:0),k+(i%GEMPIC_SPACEDIM==1?1:0))},0,1);
                            (*E_Array[(i+2)%GEMPIC_SPACEDIM])[mfi].getVal(c3,amrex::IntVect{AMREX_D_DECL(  l,  j  ,k)},0,1);
                            (*E_Array[(i+1)%GEMPIC_SPACEDIM])[mfi].getVal(c4,amrex::IntVect{AMREX_D_DECL(l+(i%GEMPIC_SPACEDIM==1?1:0),j+(i%GEMPIC_SPACEDIM==2?1:0),k+(i%GEMPIC_SPACEDIM==0?1:0))},0,1);
                            (*E_Array[(i+1)%GEMPIC_SPACEDIM])[mfi].getVal(c5, amrex::IntVect{AMREX_D_DECL(  l,  j,  k)},0,1);

                            (*B_Array[i])[mfi].setVal(*c1-dt_step*((*c2-*c3)/ifr.dx[(i+1)%GEMPIC_SPACEDIM]-
                                                      (*c4-*c5)/ifr.dx[(i+2)%GEMPIC_SPACEDIM]), cc, 0, 1);
#endif
                        }
#if (GEMPIC_SPACEDIM > 1)
                    }
#endif
#if (GEMPIC_SPACEDIM > 2)
                }
#endif
                // next mfi
            }
            (*B_Array[i]).FillBoundary(ifr.geom.periodicity());
        }
    }

    /**
     * Advance time and both fields by a time step of length @p dt_step with a Yee scheme
     */
    void advance (infrastructure ifr, amrex::Real dt_step){

        time = time+dt_step;

        advance_E(ifr, dt_step);
        advance_B(ifr, dt_step);
    }

    /**
     * Initialize rho and phi with given funtions
     */
    void init_rho_phi(infrastructure ifr,
                      te_expr *phi_parse,
                      te_expr *rho_parse,
                      double *x,
                      double *y,
                      double *z){

        for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi ) {
            const amrex::Box& bx = mfi.validbox();

            amrex::IntVect lo = {bx.smallEnd()};
            amrex::IntVect hi = {bx.bigEnd()};

#if (GEMPIC_SPACEDIM > 2)
            for(int k=lo[2]; k<=hi[2]; k++){
#endif
#if (GEMPIC_SPACEDIM > 1)
                for(int j=lo[1]; j<=hi[1]; j++){
#endif
                    for(int l=lo[0]; l<=hi[0]; l++){

                        // the box for these values:
                        *x = l*ifr.dx[0];
#if (GEMPIC_SPACEDIM > 1)
                        *y = j*ifr.dx[1];
#endif
#if (GEMPIC_SPACEDIM > 2)
                        *z = k*ifr.dx[2];
#endif
                        amrex::Box cc(amrex::IntVect{AMREX_D_DECL(l,j,k)}, amrex::IntVect{AMREX_D_DECL(l,j,k)}, amrex::IntVect::TheNodeVector());

                        const double result_phi = te_eval(phi_parse);
                        const double result_rho = te_eval(rho_parse);
                        phi[mfi].setVal(result_phi, cc, 0, 1);
                        rho[mfi].setVal(result_rho, cc, 0, 1);
                    }
#if (GEMPIC_SPACEDIM > 1)
                }
#endif
#if (GEMPIC_SPACEDIM > 2)
            }
#endif
            // next mfi
        }

        // box with all interior nodes:
        const amrex::Box& interior = amrex::surroundingNodes(amrex::grow(ifr.geom.Domain(), -1));
        //phi.setVal(0.0, interior, 0, 1, 0); // set sol to 0 on interior
        rho.FillBoundary(ifr.geom.periodicity());
        phi.FillBoundary(ifr.geom.periodicity());
    }

    /**
     * Initialize E by solving Poisson problem in two steps: 1) use the MLMG solver of amrex to solve rho -> phi 2) use differences for phi -> E
     */
    void solve_poisson(infrastructure ifr, amrex::Real reltol = 1.e-11){
        //--------------------------------------------------------------------------------
        // rho
#if GEMPIC_SPACEDIM > 1 // for == 1: use analytical solution instead of MLMG solver

        int verbose = 0;
        int bottom_verbose = 0;
        int max_iter = 100;
        int max_fmg_iter = 0; //TODO

        amrex::MLNodeLaplacian linop({ifr.geom}, {ifr.grid}, {ifr.distriMap}); // linear operator class

        linop.setDomainBC({AMREX_D_DECL(amrex::LinOpBCType::Periodic, // for lower ends
                           amrex::LinOpBCType::Periodic,
                           amrex::LinOpBCType::Periodic)},
        {AMREX_D_DECL(amrex::LinOpBCType::Periodic, // for higher ends
         amrex::LinOpBCType::Periodic,
         amrex::LinOpBCType::Periodic)});
        linop.setSigma(0, sigma); // first argument: level-nr
        amrex::MLMG mlmg(linop); // solver class
        mlmg.setMaxIter(max_iter);
        mlmg.setMaxFmgIter(max_fmg_iter);
        mlmg.setVerbose(verbose);
        mlmg.setBottomVerbose(bottom_verbose);
        //mlmg_ptr = &mlmg;

#if (GEMPIC_SPACEDIM > 1)
        amrex::Vector<std::string> varnames = {"phi_init"};
        WriteSingleLevelPlotfile ("SliceData2D/phi_init_plotfile_0", phi, varnames,
                                  ifr.geom, 0*dt, 0);

        varnames = {"rho"};
        WriteSingleLevelPlotfile ("SliceData2D/rho_plotfile_0", rho, varnames,
                                  ifr.geom, 0*dt, 0);
#endif
        amrex::Print() << "rho " << gempic_norm(&rho, ifr, 0) << std::endl;
        //(*mlmg_ptr).solve({&phi}, {&rho}, reltol, 0.0);
        mlmg.solve({&phi}, {&rho}, reltol, 0.0);
        phi.FillBoundary(ifr.geom.periodicity());

#if (GEMPIC_SPACEDIM > 1)
        varnames = {"phi_end"};
        WriteSingleLevelPlotfile ("SliceData2D/phi_end_plotfile_0", phi, varnames,
                                  ifr.geom, 0*dt, 0);
#endif

#endif

        //----------------------------------------------------------------------------------
        // E

        amrex::Real c1[1], c2[1];

        for (int i=0; i<GEMPIC_VDIM; i++) {
            for (amrex::MFIter mfi(*E_Array[i]); mfi.isValid(); ++mfi ) {
                const amrex::Box& bx = mfi.validbox();

                amrex::IntVect lo = {bx.smallEnd()};
                amrex::IntVect hi = {bx.bigEnd()};
#if (GEMPIC_SPACEDIM > 2)
                for(int k=lo[2]; k<=hi[2]; k++){
#endif
#if (GEMPIC_SPACEDIM > 1)
                    for(int j=lo[1]; j<=hi[1]; j++){
#endif
                        for(int l=lo[0]; l<=hi[0]; l++){

                            // the box for these values:
                            amrex::Box cc(amrex::IntVect{AMREX_D_DECL(l,j,k)}, amrex::IntVect{AMREX_D_DECL(l,j,k)}, *E_Index[i]);

                            // old field_out value
                            (*E_Array[i])[mfi].getVal(c1, amrex::IntVect{AMREX_D_DECL(l,j,k)},0,1); //TODO remove

                            // phi-derivative
                            phi[mfi].getVal(c1,amrex::IntVect{AMREX_D_DECL(l+(i==0?1:0),j+(i==1?1:0),k+(i==2?1:0))},0,1);
                            phi[mfi].getVal(c2,amrex::IntVect{AMREX_D_DECL(l,j,k)},0,1);

                            (*E_Array[i])[mfi].setVal(-(*c1-*c2)/ifr.dx[i], cc, 0, 1);
                        }
#if (GEMPIC_SPACEDIM > 2)
                    }
#endif
#if (GEMPIC_SPACEDIM > 1)
                }
#endif
                // next mfi
            }
            (*E_Array[i]).FillBoundary(ifr.geom.periodicity());
        }
        amrex::Vector<std::string> varnamesE = {"Ex_0"};
        WriteSingleLevelPlotfile ("SliceData/Ex_plotfile_0", *E_Array[0], varnamesE, ifr.geom, 0*dt, 0);

#if (GEMPIC_VDIM > 1)
        varnamesE = {"Ey_0"};
        WriteSingleLevelPlotfile ("SliceData/Ey_plotfile_0", *E_Array[1], varnamesE, ifr.geom, 0*dt, 0);
#endif
#if (GEMPIC_VDIM > 2)
        varnamesE = {"Ez_0"};
        WriteSingleLevelPlotfile ("SliceData/Ez_plotfile_0", *E_Array[2], varnamesE, ifr.geom, 0*dt, 0);
#endif

    }

    /**
     * Compute rho for a given rho (for diagnostics: Gauss law)
     */
    void rho_from_E(infrastructure ifr){

        amrex::Real c1[1], c2[1];

        for (amrex::MFIter mfi(rho_gauss_law); mfi.isValid(); ++mfi ) {
            const amrex::Box& bx = mfi.validbox();

            amrex::IntVect lo = {bx.smallEnd()};
            amrex::IntVect hi = {bx.bigEnd()};

#if (GEMPIC_SPACEDIM > 2)
            for(int k=lo[2]; k<=hi[2]; k++){
#endif
#if (GEMPIC_SPACEDIM > 1)
                for(int j=lo[1]; j<=hi[1]; j++){
#endif
                    for(int l=lo[0]; l<=hi[0]; l++){

                        // the box for these values:
                        amrex::Box cc(amrex::IntVect{AMREX_D_DECL(l,j,k)}, amrex::IntVect{AMREX_D_DECL(l,j,k)}, amrex::IntVect::TheNodeVector());

                        amrex::Real rho_val = 0.0;
                        for (int i=0; i<GEMPIC_VDIM; i++) {

                            // E-derivatives
                            (*E_Array[i])[mfi].getVal(c2,amrex::IntVect{AMREX_D_DECL(l-(i==0?1:0),j-(i==1?1:0),k-(i==2?1:0))},0,1);
                            (*E_Array[i])[mfi].getVal(c1,amrex::IntVect{AMREX_D_DECL(l,j,k)},0,1);

                            rho_val += (*c1-*c2)/ifr.dx[i];
                        }

                        (rho_gauss_law)[mfi].setVal(rho_val, cc, 0, 1);
                    }
#if (GEMPIC_SPACEDIM > 2)
                }
#endif
#if (GEMPIC_SPACEDIM > 1)
            }
#endif
            // next mfi
        }
        (rho_gauss_law).FillBoundary(ifr.geom.periodicity());

    }

    /**
     * Compute rho for a given rho (for diagnostics: Gauss law)
     */
    amrex::Real compute_gauss_error(infrastructure ifr, Particles::particle_groups *part_gr){

        rho_from_E(ifr); // fills rho_gauss_law

        // Deposit charges in rho:
        for (int spec=0;spec<(*part_gr).n_species;spec++) {
            (*(*part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in
            for (amrex::ParIter<GEMPIC_VDIM+1,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {
                amrex::Box tilebox;
                amrex::FArrayBox local_rho;

                tilebox = pti.tilebox();
                tilebox.grow(Nghost);
                const amrex::Box tb = amrex::convert(tilebox, amrex::IntVect::TheUnitVector());

                local_rho.resize(tb,1); // second arg: number of comps
                local_rho.setVal(0.0);

                auto& particles = pti.GetArrayOfStructs();
                const long np  = pti.numParticles();

                amrex::Array4<amrex::Real> const& rhoarr = local_rho.array();
                for (int pp=0;pp<np;pp++) {
                    Particles::gempic_deposit_rho(particles[pp], (*part_gr).charge[spec], rhoarr, ifr.plo, ifr.dxi);
                  }
                (rho)[pti].atomicAdd(local_rho,tb,tb,0,0,1);
              }
          }

        std::cout << rho_gauss_law.norm0() << "|" << rho.norm0() << ", dx" << ifr.dx[0] << std::endl;
        rho_gauss_law.minus(rho, 0, 1, 0);
        return(gempic_norm(&(rho_gauss_law), ifr, 2));
    }

    /**
     * Compute norm of errors of E and B field from function with exact solution
     */
    std::array<amrex::Real,GEMPIC_VDIM+GEMPIC_BDIM> computeError(double (*fields[]) (std::array<double,GEMPIC_SPACEDIM> x, double t),
                                                                 bool bool_B,
                                                                 infrastructure ifr){
        std::array<amrex::Real,GEMPIC_VDIM+GEMPIC_BDIM> E_B_error;

        exactSolution(fields, bool_B, ifr);

        for (int i=0; i<GEMPIC_VDIM; i++){
            amrex::Vector<std::string> varnames = {"E"+std::to_string(i)};
            WriteSingleLevelPlotfile ("SliceDataTest/E"+std::to_string(i)+"_plotfile_0", *E_Array[i], varnames,
                                      ifr.geom, dt, 0); //level_step
            varnames = {"E"+std::to_string(i)+"sol"};
            WriteSingleLevelPlotfile ("SliceDataTest/E"+std::to_string(i)+"sol"+"_plotfile_0", *E_sol_Array[i], varnames,
                                      ifr.geom, dt, 0); //level_step
            (*E_sol_Array[i]).minus(*E_Array[i], 0, 1, 0);
            E_B_error[i] = gempic_norm(&(*E_sol_Array[i]), ifr, 2);
        }
        if (bool_B) {
            for (int i=0; i<GEMPIC_BDIM; i++){
                (*B_sol_Array[i]).minus(*B_Array[i], 0, 1, 0);
                E_B_error[i+GEMPIC_VDIM] = gempic_norm(&(*B_sol_Array[i]), ifr, 2);
            }
        }
        return(E_B_error);
    }

private:

    /**
     * Fill the MultiFabs containing the analytical solution with the solution by evaluating the given function.
     */
    void exactSolution(double (*fields[]) (std::array<double,GEMPIC_SPACEDIM> x, double t),
                       bool bool_B,
                       infrastructure ifr){
        std::array<double,GEMPIC_SPACEDIM> x;
        amrex::Real btime = time + dt/2.0;

        // E-field
        for (int i=0; i<GEMPIC_VDIM; i++) {
            for (amrex::MFIter mfi(*E_sol_Array[i]); mfi.isValid(); ++mfi ) {
                const amrex::Box& bx = mfi.validbox();

                amrex::IntVect lo = {bx.smallEnd()};
                amrex::IntVect hi = {bx.bigEnd()};

#if (GEMPIC_SPACEDIM > 2)
                for(int k=lo[2]; k<=hi[2]; k++){
                    x[2] = ifr.geom.ProbLo()[2] + ((double)k+0.5-0.5*(*E_Index[i])[2])*ifr.dx[2];
#endif
#if (GEMPIC_SPACEDIM > 1)
                    for(int j=lo[1]; j<=hi[1]; j++){
                        x[1] = ifr.geom.ProbLo()[1] + ((double)j+0.5-0.5*(*E_Index[i])[1])*ifr.dx[1];
#endif
                        for(int l=lo[0]; l<=hi[0]; l++){
                            x[0] = ifr.geom.ProbLo()[0] + ((double)l+0.5-0.5*(*E_Index[i])[0])*ifr.dx[0];

                            // the box for these values:
                            amrex::Box cc(amrex::IntVect{AMREX_D_DECL(l,j,k)}, amrex::IntVect{AMREX_D_DECL(l,j,k)}, *E_Index[i]);
                            (*E_sol_Array[i])[mfi].setVal((*fields[i])(x,time), cc, 0, 1);
                        }
#if (GEMPIC_SPACEDIM > 1)
                    }
#endif
#if (GEMPIC_SPACEDIM > 2)
                }
#endif
                // next mfi
            }
            (*E_sol_Array[i]).FillBoundary(ifr.geom.periodicity());
        }

        if (bool_B){
            // B-field
            for (int i=0; i<GEMPIC_BDIM; i++) {
                for (amrex::MFIter mfi(*B_sol_Array[i]); mfi.isValid(); ++mfi ) {
                    const amrex::Box& bx = mfi.validbox();

                    amrex::IntVect lo = {bx.smallEnd()};
                    amrex::IntVect hi = {bx.bigEnd()};

#if (GEMPIC_SPACEDIM > 2)
                    for(int k=lo[2]; k<=hi[2]; k++){
                        x[2] = ifr.geom.ProbLo()[2] + ((double)k+0.5-0.5*(*B_Index[i])[2])*ifr.dx[2];
#endif
#if (GEMPIC_SPACEDIM > 1)
                        for(int j=lo[1]; j<=hi[1]; j++){
                            x[1] = ifr.geom.ProbLo()[1] + ((double)j+0.5-0.5*(*B_Index[i])[1])*ifr.dx[1];
#endif
                            for(int l=lo[0]; l<=hi[0]; l++){
                                x[0] = ifr.geom.ProbLo()[0] + ((double)l+0.5-0.5*(*B_Index[i])[0])*ifr.dx[0];

                                // the box for these values:
                                amrex::Box cc(amrex::IntVect{AMREX_D_DECL(l,j,k)}, amrex::IntVect{AMREX_D_DECL(l,j,k)}, *B_Index[i]);
                                (*B_sol_Array[i])[mfi].setVal((*fields[i+GEMPIC_VDIM])(x,btime), cc, 0, 1);
                            }
#if (GEMPIC_SPACEDIM > 1)
                        }
#endif
#if (GEMPIC_SPACEDIM > 2)
                    }
#endif
                    // next mfi
                }
                (*B_sol_Array[i]).FillBoundary(ifr.geom.periodicity());
            }
        }
    }

};
}
}

#endif
