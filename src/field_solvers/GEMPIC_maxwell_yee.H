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
#include <GEMPIC_MLNodeLaplacian_FD.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_mesh_coupling.H>
#include <GEMPIC_particle_mesh_coupling_C2.H>

namespace Gempic {
namespace Vlasov_Maxwell {
template<int vdim, int numspec>
class vlasov_maxwell;
}
}

using namespace Infra;
using namespace Utils;
using namespace Vlasov_Maxwell;
using namespace Particles;

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
template<int vdim>
class maxwell_yee
{
public:

    int nsteps; // number of simulation steps
    amrex::Real dt;
    std::array<std::unique_ptr<amrex::MultiFab>, vdim > E_Array;
    std::array<std::unique_ptr<amrex::MultiFab>, vdim > HE_Array;
    std::array<std::unique_ptr<amrex::MultiFab>, vdim > J_Array;
    std::array<std::unique_ptr<amrex::MultiFab>, int(vdim/2.5)*2+1 > B_Array;
    std::array<std::unique_ptr<amrex::MultiFab>, int(vdim/2.5)*2+1 > HB_Array;
    std::array<std::unique_ptr<amrex::MultiFab>, vdim > E_sol_Array;
    std::array<std::unique_ptr<amrex::MultiFab>, int(vdim/2.5)*2+1 > B_sol_Array;
    std::array<std::unique_ptr<amrex::IndexType>, vdim> E_Index;
    std::array<std::unique_ptr<amrex::IndexType>, int(vdim/2.5)*2+1 > B_Index;
    std::array<std::unique_ptr<amrex::iMultiFab>, vdim> E_Masks;
    std::array<std::unique_ptr<amrex::iMultiFab>, vdim> B_Masks;
    std::unique_ptr<amrex::iMultiFab> nodal_Mask;
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
     * Explicit constructor: all class attributes can be initialized by passing an object of class vlasov maxwell, infrastructure and the number of ghost cells.
     */
    template <int numspec>
    maxwell_yee (vlasov_maxwell<vdim, numspec> VM, infrastructure ifr){
        // set index indicating nodal-/cell-centered
        // 0 -> cell-centered | 1 -> node-centered

        E_Index[0].reset(new amrex::IndexType (amrex::IntVect{AMREX_D_DECL(0,1,1)}));
        if(vdim > 1) {
            E_Index[1].reset(new amrex::IndexType (amrex::IntVect{AMREX_D_DECL(1,0,1)}));
        }
        if (vdim > 2) {
            E_Index[2].reset(new amrex::IndexType (amrex::IntVect{AMREX_D_DECL(1,1,0)}));
        }

        B_Index[0].reset(new amrex::IndexType (amrex::IntVect{AMREX_D_DECL(1,0,0)}));
if (int(vdim/2.5)*2+1 > 1) {
        B_Index[1].reset(new amrex::IndexType (amrex::IntVect{AMREX_D_DECL(0,1,0)}));
}
if (int(vdim/2.5)*2+1 > 2) {
        B_Index[2].reset(new amrex::IndexType (amrex::IntVect{AMREX_D_DECL(0,0,1)}));
}
if (int(vdim/2.5)*2+1 < 3) {
    B_Index[0].reset(new amrex::IndexType (amrex::IntVect{AMREX_D_DECL(0,0,1)}));
}

        Nghost = VM.Nghost;
        Ncomp = 1;

        dt = VM.dt;
        time = 0.0;

        // make the multifabs nodal-/cell-centered according to indices
        for (int i = 0; i<vdim; i++) {
            E_Array[i].reset(new amrex::MultiFab (convert(ifr.grid, *E_Index[i]),ifr.distriMap,Ncomp,Nghost));
            HE_Array[i].reset(new amrex::MultiFab (convert(ifr.grid, *E_Index[i]),ifr.distriMap,Ncomp,Nghost));
            E_sol_Array[i].reset(new amrex::MultiFab (convert(ifr.grid, *E_Index[i]),ifr.distriMap,Ncomp,Nghost));
            J_Array[i].reset(new amrex::MultiFab (convert(ifr.grid, *E_Index[i]),ifr.distriMap,Ncomp,Nghost));
        }
        for (int i = 0; i<int(vdim/2.5)*2+1; i++) {
            B_Array[i].reset(new amrex::MultiFab (convert(ifr.grid, *B_Index[i]),ifr.distriMap,Ncomp,Nghost));
            HB_Array[i].reset(new amrex::MultiFab (convert(ifr.grid, *B_Index[i]),ifr.distriMap,Ncomp,Nghost));
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

        nsteps = VM.n_steps;

        //-----------------------------------------------------------------------------
        // Fill Owner Mask

        for (int comp=0; comp<vdim; comp++) {
            E_Masks[comp].reset(new amrex::iMultiFab (convert(ifr.grid, *E_Index[comp]),ifr.distriMap,1,Nghost));
            (*(E_Masks[comp])).setVal(0.0);
            (*(E_Masks[comp])).FillBoundary(ifr.geom.periodicity());

            for ( amrex::MFIter mfi((*(E_Masks[comp]))); mfi.isValid(); ++mfi ) {
                const amrex::Box& bx = mfi.validbox();
                amrex::IntVect lo = {bx.smallEnd()};
                amrex::IntVect hi = {bx.bigEnd()};
#if (GEMPIC_SPACEDIM > 2)
                for(int k=lo[2]; k<=(comp==2?hi[2]:(hi[2]-1)); k++){
#endif
#if (GEMPIC_SPACEDIM > 1)
                    for(int j=lo[1]; j<=(comp==1?hi[1]:(hi[1]-1)); j++){
#endif
                        for(int l=lo[0]; l<=(comp==0?hi[0]:(hi[0]-1)); l++){
                            // the box for these values:
                            amrex::Box cc(amrex::IntVect{AMREX_D_DECL(l,j,k)}, amrex::IntVect{AMREX_D_DECL(l,j,k)}, (*E_Index[comp]));
                            (*(E_Masks[comp]))[mfi].setVal(1, cc, 0, 1);
                        }

#if (GEMPIC_SPACEDIM > 1)
                    }
#endif
#if (GEMPIC_SPACEDIM > 2)
                }
#endif
            }
        }

        for (int comp=0; comp<int(vdim/2.5)*2+1; comp++) {
            B_Masks[comp].reset(new amrex::iMultiFab (convert(ifr.grid, *B_Index[comp]),ifr.distriMap,1,Nghost));
            (*(B_Masks[comp])).setVal(0.0);
            (*(B_Masks[comp])).FillBoundary(ifr.geom.periodicity());

            for ( amrex::MFIter mfi((*(B_Masks[comp]))); mfi.isValid(); ++mfi ) {
                const amrex::Box& bx = mfi.validbox();
                amrex::IntVect lo = {bx.smallEnd()};
                amrex::IntVect hi = {bx.bigEnd()};
#if (GEMPIC_SPACEDIM > 2)
                for(int k=lo[2]; k<=(comp==2?(hi[2]-1):hi[2]); k++){
#endif
#if (GEMPIC_SPACEDIM > 1)
                    for(int j=lo[1]; j<=(comp==1?(hi[1]-1):hi[1]); j++){
#endif
                        for(int l=lo[0]; l<=(comp==0?(hi[0]-1):hi[0]); l++){
                            // the box for these values:
                            amrex::Box cc(amrex::IntVect{AMREX_D_DECL(l,j,k)}, amrex::IntVect{AMREX_D_DECL(l,j,k)}, (*B_Index[comp]));
                            (*(B_Masks[comp]))[mfi].setVal(1, cc, 0, 1);
                        }

#if (GEMPIC_SPACEDIM > 1)
                    }
#endif
#if (GEMPIC_SPACEDIM > 2)
                }
#endif
            }
            nodal_Mask.reset(new amrex::iMultiFab (convert(ifr.grid, amrex::IntVect::TheNodeVector()),ifr.distriMap,1,Nghost));
            (*nodal_Mask).setVal(0.0);
            (*nodal_Mask).FillBoundary(ifr.geom.periodicity());

            for ( amrex::MFIter mfi(*nodal_Mask); mfi.isValid(); ++mfi ) {
                const amrex::Box& bx = mfi.validbox();
                amrex::IntVect lo = {bx.smallEnd()};
                amrex::IntVect hi = {bx.bigEnd()};
#if (GEMPIC_SPACEDIM > 2)
                for(int k=lo[2]; k<=(hi[2]-1); k++){
#endif
#if (GEMPIC_SPACEDIM > 1)
                    for(int j=lo[1]; j<=(hi[1]-1); j++){
#endif
                        for(int l=lo[0]; l<=(hi[0]-1); l++){
                            // the box for these values:
                            amrex::Box cc(amrex::IntVect{AMREX_D_DECL(l,j,k)}, amrex::IntVect{AMREX_D_DECL(l,j,k)}, amrex::IntVect::TheNodeVector());
                            (*nodal_Mask)[mfi].setVal(1, cc, 0, 1);
                        }

#if (GEMPIC_SPACEDIM > 1)
                    }
#endif
#if (GEMPIC_SPACEDIM > 2)
                }
#endif
            }
        }

    }


    template<int degree>
    void projection (double (field) (std::array<double,GEMPIC_SPACEDIM> x, double t),
                     amrex::Real t,
                     infrastructure ifr,
                     std::array<bool, GEMPIC_SPACEDIM> integrate,
                     amrex::IndexType IndexVect,
                     amrex::MultiFab *MultiF
                     ) {
        std::vector<amrex::Real> stencil;
        switch (degree) {
        case 2:
            stencil = {0};
            break;
        case 4:
            stencil = {1.0/6.0, 4.0/6.0, 1.0/6.0};
            break;
        case 6:
            stencil = {1.0/12.0, 4.0/12.0, 2.0/12.0, 4.0/12.0, 1.0/12.0};
            break;
        default: amrex::Print() << "degree not implemented" << std::endl;

        }
        std::array<amrex::Real, degree-1> x;
        amrex::Real x_midpoint;
#if (GEMPIC_SPACEDIM > 1)
        std::array<amrex::Real, degree-1> y;
        amrex::Real y_midpoint;
#endif
#if (GEMPIC_SPACEDIM > 2)
        std::array<amrex::Real, degree-1> z;
        amrex::Real z_midpoint;
#endif
        double half_dt = dt/2.0; // for leap frog initialization

        for ( amrex::MFIter mfi(*MultiF); mfi.isValid(); ++mfi ){

            const amrex::Box& bx = mfi.validbox();
            amrex::IntVect lo = {bx.smallEnd()};
            amrex::IntVect hi = {bx.bigEnd()};
#if (GEMPIC_SPACEDIM > 2)
            for(int k=lo[2]; k<=hi[2]; k++){
                z_midpoint = ifr.geom.ProbLo()[2] + ((double)k+0.5-0.5*(IndexVect)[2])*ifr.dx[2];
                for (int d = 0; d<(degree-1); d++) z[d] = z_midpoint + (d-(degree/2.0-1))*ifr.dx[0]/pow(2.0,degree/2.0-1);
#endif
#if (GEMPIC_SPACEDIM > 1)
                for(int j=lo[1]; j<=hi[1]; j++){
                    y_midpoint = ifr.geom.ProbLo()[1] + ((double)j+0.5-0.5*(IndexVect)[1])*ifr.dx[1];
                    for (int d = 0; d<(degree-1); d++) y[d] = y_midpoint + (d-(degree/2.0-1))*ifr.dx[0]/pow(2.0,degree/2.0-1);
#endif
                    for(int l=lo[0]; l<=hi[0]; l++){
                        x_midpoint = ifr.geom.ProbLo()[0] + ((double)l+0.5-0.5*(IndexVect)[0])*ifr.dx[0];
                        for (int d = 0; d<(degree-1); d++) x[d] = x_midpoint + (d-(degree/2.0-1))*ifr.dx[0]/pow(2.0,degree/2.0-1);
                        // the box for these values:
                        amrex::Box cc(amrex::IntVect{AMREX_D_DECL(l,j,k)}, amrex::IntVect{AMREX_D_DECL(l,j,k)}, IndexVect);
                        amrex::Real val = 0.0;
                        if (integrate[0]) {
                            for (int d = 0; d<(degree-1); d++) val+= stencil[d]*(field({AMREX_D_DECL(x[d],y[d],z[d])},t));
                        } else {
                            val += field({AMREX_D_DECL(x_midpoint,y_midpoint,z_midpoint)},t);
                        }
                        (*MultiF)[mfi].setVal(val, cc, 0, 1);
                    }
#if (GEMPIC_SPACEDIM > 1)
                }
#endif
#if (GEMPIC_SPACEDIM > 2)
            }
#endif
            // next mfi
        }
        (*MultiF).FillBoundary(ifr.geom.periodicity());
    }

    template<int degree>
    void projection (std::string function_string,
                     std::array<amrex::Real, 3> k,
                     amrex::Real t,
                     infrastructure ifr,
                     std::array<bool, GEMPIC_SPACEDIM> integrate,
                     amrex::IndexType IndexVect,
                     amrex::MultiFab *MultiF
                     ) {
        double x_eval, y_eval, z_eval;
        double t_eval = t;
        double kx = k[0];
        double ky = k[1];
        double kz = k[2];
        int err;
        te_variable read_vars[7] = {{"x", &x_eval}, {"y", &y_eval}, {"z", &z_eval}, {"t", &t_eval}, {"kvarx", &kx}, {"kvary", &ky}, {"kvarz", &kz}};
        int varcount = 7;
        te_expr *function_parse = te_compile(function_string.c_str(), read_vars, varcount, &err);

        std::vector<amrex::Real> stencil;
        switch (degree) {
        case 2:
            stencil = {1.0};
            break;
        case 4:
            stencil = {1.0/6.0, 4.0/6.0, 1.0/6.0};
            break;
        case 6:
            stencil = {1.0/12.0, 4.0/12.0, 2.0/12.0, 4.0/12.0, 1.0/12.0};
            break;
        default: amrex::Print() << "degree not implemented" << std::endl;

        }
        std::array<amrex::Real, degree-1> x;
        amrex::Real x_midpoint;
        std::array<amrex::Real, degree-1> y;
        amrex::Real y_midpoint;
        std::array<amrex::Real, degree-1> z;
        amrex::Real z_midpoint;

        int x_len[GEMPIC_SPACEDIM];
        for (int i=0;i<GEMPIC_SPACEDIM;i++) x_len[i] = (integrate[i]?(degree-1):1);

        for ( amrex::MFIter mfi(*MultiF); mfi.isValid(); ++mfi ){

            const amrex::Box& bx = mfi.validbox();
            amrex::IntVect lo = {bx.smallEnd()};
            amrex::IntVect hi = {bx.bigEnd()};
#if (GEMPIC_SPACEDIM > 2)
            for(int k=lo[2]; k<=hi[2]; k++){
                z_midpoint = ifr.geom.ProbLo()[2] + ((double)k+0.5-0.5*(IndexVect)[2])*ifr.dx[2];
                for (int d = 0; d<x_len[2]; d++) z[d] = z_midpoint + (d-((x_len[2]+1)/2.0-1))*ifr.dx[0]/pow(2.0,(x_len[2]+1)/2.0-1);
#endif
#if (GEMPIC_SPACEDIM > 1)
                for(int j=lo[1]; j<=hi[1]; j++){
                    y_midpoint = ifr.geom.ProbLo()[1] + ((double)j+0.5-0.5*(IndexVect)[1])*ifr.dx[1];
                    for (int d = 0; d<x_len[1]; d++) y[d] = y_midpoint + (d-((x_len[1]+1)/2.0-1))*ifr.dx[0]/pow(2.0,(x_len[1]+1)/2.0-1);
#endif
                    for(int l=lo[0]; l<=hi[0]; l++){
                        x_midpoint = ifr.geom.ProbLo()[0] + ((double)l+0.5-0.5*(IndexVect)[0])*ifr.dx[0];
                        for (int d = 0; d<x_len[0]; d++) x[d] = x_midpoint + (d-((x_len[0]+1)/2.0-1))*ifr.dx[0]/pow(2.0,(x_len[0]+1)/2.0-1);
                        // the box for these values:
                        amrex::Box cc(amrex::IntVect{AMREX_D_DECL(l,j,k)}, amrex::IntVect{AMREX_D_DECL(l,j,k)}, IndexVect);
                        amrex::Real val = 0.0;
#if (GEMPIC_SPACEDIM > 2)
                        for (int zd = 0; zd<x_len[2];zd++)
#endif
                            {
#if (GEMPIC_SPACEDIM > 1)
                            for (int yd =0; yd<x_len[1]; yd++)
#endif
                                {
                                for (int xd =0; xd<x_len[0]; xd++) {
                                    x_eval = x[xd];
                                    y_eval = y[yd];
                                    z_eval = z[zd];
                                    val+= (integrate[0]?stencil[xd]:1)
#if (GEMPIC_SPACEDIM > 1)
                                            *(integrate[1]?stencil[yd]:1)
#endif
#if (GEMPIC_SPACEDIM > 2)
                                            *(integrate[2]?stencil[zd]:1)
#endif
                                            *(te_eval(function_parse));
                                }
                            }
                        }
                        (*MultiF)[mfi].setVal(val, cc, 0, 1);
                    }
#if (GEMPIC_SPACEDIM > 1)
                }
#endif
#if (GEMPIC_SPACEDIM > 2)
            }
#endif
            // next mfi
        }
        (*MultiF).FillBoundary(ifr.geom.periodicity());
    }


    /**
     * Initialize E and B field for time t=0 with a given array of functions.
     */
    template <int degree>
    void init_E_B (std::array<std::string, vdim> fieldsE, std::array<std::string, int(vdim/2.5)*2+1> fieldsB,
                   std::array<amrex::Real, 3> k,infrastructure ifr){
        initE<degree> (fieldsE, k, ifr);
        initB<degree> (fieldsB, k, ifr);
    }

    /**
     * Initialize E field for time t=0 with a given array of functions.
     */
    template <int degree>
    void initE (std::array<std::string, vdim> fields, std::array<amrex::Real, 3> k,infrastructure ifr) {
        // E_field
        for (int i = 0; i<vdim; i++) {
            projection<degree>(fields[i], k, 0.0, ifr, {AMREX_D_DECL(i!=0, i!=1, i!=2)}, *E_Index[i], &(*E_Array[i]));
        }
    }

    /**
     * Initialize B field for time t=0 with a given array of functions.
     */
    template <int degree>
    void initB (std::array<std::string, int(vdim/2.5)*2+1> fields, std::array<amrex::Real, 3> k,infrastructure ifr){

        double half_dt = dt/2.0; // for leap frog initialization

        // B_field
        for (int i = 0; i<int(vdim/2.5)*2+1; i++) {
            projection<degree>(fields[i], k, half_dt, ifr, {AMREX_D_DECL(i!=0, i!=1, i!=2)}, *B_Index[i], &(*B_Array[i]));
        }
    }

    /**
     * Fill Boundaries of all relevant Fabs (E, J, B, rho).
     */
    void FillBD (infrastructure ifr){
        for(int cc=0;cc<vdim;cc++){
            (*E_Array[cc]).FillBoundary(ifr.geom.periodicity());
            (*E_sol_Array[cc]).FillBoundary(ifr.geom.periodicity()); //TODO: remove
        }
        for(int cc=0;cc<vdim;cc++){
            (*J_Array[cc]).FillBoundary(ifr.geom.periodicity());
        }
        for(int cc=0;cc<int(vdim/2.5)*2+1;cc++){
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
    void advance_E (infrastructure ifr, amrex::Real dt_step, bool from_B, bool from_J,
                    std::array<std::unique_ptr<amrex::MultiFab>, int(vdim/2.5)*2+1 > *MultiInput,
                    std::array<std::unique_ptr<amrex::MultiFab>, vdim > *MultiOutput){

        int bdim = int(vdim/2.5)*2+1;

        amrex::Real c1[1], c2[1], c3[1], c4[1], c5[1], cj[1];

        // E-field
        for (int i=0; i<vdim; i++) {
            for (amrex::MFIter mfi(*(*MultiOutput)[i]); mfi.isValid(); ++mfi ) {
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
                            (*(*MultiOutput)[i])[mfi].getVal(c1, amrex::IntVect{AMREX_D_DECL(l,j,k)},0,1);
                            // J value
                            (*J_Array[i])[mfi].getVal(cj, amrex::IntVect{AMREX_D_DECL(l,j,k)},0,1);

                            // B-derivatives
                            if((vdim < 3) & (GEMPIC_SPACEDIM == 1)){
                                if (i == 0){
                                    (*(*MultiOutput)[0])[mfi].setVal(*c1+(from_J?dt_step*(-*cj):0.0), cc, 0, 1);
                                }
                            }
                            if ((vdim == 2) & (GEMPIC_SPACEDIM == 2)){
                                //Ex for 2D2V
                                if (i == 0) {
                                    (*(*MultiInput)[0])[mfi].getVal(c2,amrex::IntVect{AMREX_D_DECL(  l,j,k)},0,1);
                                    (*(*MultiInput)[0])[mfi].getVal(c3,amrex::IntVect{AMREX_D_DECL(l,j-1,k)},0,1);
                                    (*(*MultiOutput)[0])[mfi].setVal(*c1+dt_step*((from_B?(*c2-*c3)/ifr.dx[1]:0.0)-(from_J?*cj:0.0)), cc, 0, 1);
                                }
                            }
                            if (vdim == 2){
                                if (i == 1) {
                                    (*(*MultiInput)[0])[mfi].getVal(c2,amrex::IntVect{AMREX_D_DECL(  l,j,k)},0,1);
                                    (*(*MultiInput)[0])[mfi].getVal(c3,amrex::IntVect{AMREX_D_DECL(l-1,j,k)},0,1);
                                    (*(*MultiOutput)[1])[mfi].setVal(*c1+dt_step*((from_B?-(*c2-*c3)/ifr.dx[0]:0.0)-(from_J?*cj:0.0)), cc, 0, 1);
                                }
                            }
                            if ((vdim == 3) & (GEMPIC_SPACEDIM == 2)) {
                                if (i==2) {
                                    (*(*MultiInput)[(i+2)%vdim])[mfi].getVal(c2,amrex::IntVect{AMREX_D_DECL(  l,  j  ,k)},0,1);
                                    (*(*MultiInput)[(i+2)%vdim])[mfi].getVal(c3,amrex::IntVect{AMREX_D_DECL(l-(i%vdim==2?1:0),j-(i%vdim==0?1:0),k)},0,1);
                                    (*(*MultiInput)[(i+1)%vdim])[mfi].getVal(c4, amrex::IntVect{AMREX_D_DECL(  l,  j,  k)},0,1);
                                    (*(*MultiInput)[(i+1)%vdim])[mfi].getVal(c5,amrex::IntVect{AMREX_D_DECL(l-(i%vdim==1?1:0),j-(i%vdim==2?1:0),k)},0,1);

                                    (*(*MultiOutput)[i])[mfi].setVal(*c1+dt_step*((from_B?(*c2-*c3)/ifr.dx[(i+1)%vdim]-
                                                              (*c4-*c5)/ifr.dx[(i+2)%vdim]:0.0)-(from_J?*cj:0.0)), cc, 0, 1);
                                } else {
                                    (*(*MultiInput)[2])[mfi].getVal(c2, amrex::IntVect{AMREX_D_DECL(  l,  j,  k)},0,1);
                                    (*(*MultiInput)[2])[mfi].getVal(c3, amrex::IntVect{AMREX_D_DECL(l-(i%vdim==1?1:0),j-(i%vdim==0?1:0),k)},0,1);

                                    (*(*MultiOutput)[i])[mfi].setVal(*c1+dt_step*((from_B?(i==0?1:-1)*(*c2-*c3)/ifr.dx[(i+1)%GEMPIC_SPACEDIM]:0.0)-(from_J?*cj:0.0)), cc, 0, 1);
                                }
                            }
                            if ((vdim == 3) & (GEMPIC_SPACEDIM == 3)) {
                                (*(*MultiInput)[(i+2)%bdim])[mfi].getVal(c2,amrex::IntVect{AMREX_D_DECL(  l,  j  ,k)},0,1);
                                (*(*MultiInput)[(i+2)%bdim])[mfi].getVal(c3,amrex::IntVect{AMREX_D_DECL(l-(i%bdim==2?1:0),j-(i%bdim==0?1:0),k-(i%bdim==1?1:0))},0,1);
                                (*(*MultiInput)[(i+1)%bdim])[mfi].getVal(c4, amrex::IntVect{AMREX_D_DECL(  l,  j,  k)},0,1);
                                (*(*MultiInput)[(i+1)%bdim])[mfi].getVal(c5,amrex::IntVect{AMREX_D_DECL(l-(i%bdim==1?1:0),j-(i%bdim==2?1:0),k-(i%bdim==0?1:0))},0,1);

                                (*(*MultiOutput)[i])[mfi].setVal(*c1+dt_step*((from_B?(*c2-*c3)/ifr.dx[(i+1)%bdim]-
                                                          (*c4-*c5)/ifr.dx[(i+2)%bdim]:0.0)-(from_J?*cj:0.0)), cc, 0, 1);
                            }
                        }
#if (GEMPIC_SPACEDIM > 1)
                    }
#endif
#if (GEMPIC_SPACEDIM > 2)
                }
#endif
                // next mfi
            }
            (*(*MultiOutput)[i]).FillBoundary(ifr.geom.periodicity());
        }
    }

    /**
     * Advance B by a time step of length @p dt_step with a Yee scheme
     */
    void advance_B (infrastructure ifr, amrex::Real dt_step,
                    std::array<std::unique_ptr<amrex::MultiFab>, vdim > *MultiInput,
                    std::array<std::unique_ptr<amrex::MultiFab>, int(vdim/2.5)*2+1 > *MultiOutput){

        int bdim = int(vdim/2.5)*2+1;

        amrex::Real c1[1], c2[1], c3[1], c4[1], c5[1];

        // B-field
        for (int i=0; i<bdim; i++) {
            for (amrex::MFIter mfi(*(*MultiOutput)[i]); mfi.isValid(); ++mfi ) {
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
                            (*(*MultiOutput)[i])[mfi].getVal(c1, amrex::IntVect{AMREX_D_DECL(l,j,k)},0,1);

                            // E-derivatives
                            if (vdim == 1){}
                            //no B
                            else if ((vdim == 2) & (GEMPIC_SPACEDIM == 1)){
                                (*(*MultiInput)[1])[mfi].getVal(c4, amrex::IntVect{AMREX_D_DECL(l+1,j,k)},0,1);
                                (*(*MultiInput)[1])[mfi].getVal(c5, amrex::IntVect{AMREX_D_DECL(l  ,j,k)},0,1);

                                (*(*MultiOutput)[0])[mfi].setVal(*c1-dt_step*((*c4-*c5)/ifr.dx[0]), cc, 0, 1);
                            }
                            else if ((vdim == 2) & (GEMPIC_SPACEDIM == 2)){
                                (*(*MultiInput)[0])[mfi].getVal(c2,amrex::IntVect{AMREX_D_DECL(l,j+1,k)},0,1);
                                (*(*MultiInput)[0])[mfi].getVal(c3,amrex::IntVect{AMREX_D_DECL(l,j,k)},0,1);
                                (*(*MultiInput)[1])[mfi].getVal(c4,amrex::IntVect{AMREX_D_DECL(l+1,j,k)},0,1);
                                (*(*MultiInput)[1])[mfi].getVal(c5, amrex::IntVect{AMREX_D_DECL(l,j,k)},0,1);

                                (*(*MultiOutput)[0])[mfi].setVal(*c1+dt_step*((*c2-*c3)/ifr.dx[1]-
                                                          (*c4-*c5)/ifr.dx[0]), cc, 0, 1);
                            }
                            else if ((vdim == 3) & (GEMPIC_SPACEDIM == 2)) {
                                if (i==2) {
                                    (*(*MultiInput)[(i+2)%bdim])[mfi].getVal(c2,amrex::IntVect{AMREX_D_DECL(l+(i%bdim==2?1:0),j+(i%bdim==0?1:0),k)},0,1);
                                    (*(*MultiInput)[(i+2)%bdim])[mfi].getVal(c3,amrex::IntVect{AMREX_D_DECL(  l,  j  ,k)},0,1);
                                    (*(*MultiInput)[(i+1)%bdim])[mfi].getVal(c4,amrex::IntVect{AMREX_D_DECL(l+(i%bdim==1?1:0),j+(i%bdim==2?1:0),k)},0,1);
                                    (*(*MultiInput)[(i+1)%bdim])[mfi].getVal(c5, amrex::IntVect{AMREX_D_DECL(  l,  j,  k)},0,1);

                                    (*(*MultiOutput)[i])[mfi].setVal(*c1-dt_step*((*c2-*c3)/ifr.dx[(i+1)%bdim]-
                                                              (*c4-*c5)/ifr.dx[(i+2)%bdim]), cc, 0, 1);
                                } else {
                                    (*(*MultiInput)[2])[mfi].getVal(c2, amrex::IntVect{AMREX_D_DECL(l+(i%bdim==1?1:0),j+(i%bdim==0?1:0),k)},0,1);
                                    (*(*MultiInput)[2])[mfi].getVal(c3, amrex::IntVect{AMREX_D_DECL(l,j,k)},0,1);

                                    (*(*MultiOutput)[i])[mfi].setVal(*c1+dt_step*((i==0?-1:1)*(*c2-*c3)/ifr.dx[(i+1)%GEMPIC_SPACEDIM]), cc, 0, 1);
                                }
                            }
                            else if (bdim == 3) {
                                (*(*MultiInput)[(i+2)%bdim])[mfi].getVal(c2,amrex::IntVect{AMREX_D_DECL(l+(i%bdim==2?1:0),j+(i%bdim==0?1:0),k+(i%bdim==1?1:0))},0,1);
                                (*(*MultiInput)[(i+2)%bdim])[mfi].getVal(c3,amrex::IntVect{AMREX_D_DECL(  l,  j  ,k)},0,1);
                                (*(*MultiInput)[(i+1)%bdim])[mfi].getVal(c4,amrex::IntVect{AMREX_D_DECL(l+(i%bdim==1?1:0),j+(i%bdim==2?1:0),k+(i%bdim==0?1:0))},0,1);
                                (*(*MultiInput)[(i+1)%bdim])[mfi].getVal(c5, amrex::IntVect{AMREX_D_DECL(  l,  j,  k)},0,1);

                                (*(*MultiOutput)[i])[mfi].setVal(*c1-dt_step*((*c2-*c3)/ifr.dx[(i+1)%GEMPIC_SPACEDIM]-
                                                          (*c4-*c5)/ifr.dx[(i+2)%GEMPIC_SPACEDIM]), cc, 0, 1);
                            }
                        }
#if (GEMPIC_SPACEDIM > 1)
                    }
#endif
#if (GEMPIC_SPACEDIM > 2)
                }
#endif
                // next mfi
            }
            (*(*MultiOutput)[i]).FillBoundary(ifr.geom.periodicity());
        }
    }

#if (GEMPIC_SPACEDIM ==3)
    template<int degree>
    void hodge_full (infrastructure ifr, std::array<std::unique_ptr<amrex::MultiFab>, vdim > *MultiInput,
                     std::array<std::unique_ptr<amrex::MultiFab>, vdim > *MultiOutput, bool is_E) {
        for (int field=0; field<vdim; field++) {
            hodge<degree,0>(ifr, (field==0?0:1), &(*(*MultiInput)[field]), &(*(*MultiOutput)[field]), (is_E?*E_Index[field]:*B_Index[field]));
#if (GEMPIC_SPACEDIM > 1)
            hodge<degree,1>(ifr, (field==1?0:1), &(*(*MultiOutput)[field]), &(*(*MultiInput)[field]), (is_E?*E_Index[field]:*B_Index[field]));
#endif
#if (GEMPIC_SPACEDIM > 2)
            hodge<degree,2>(ifr, (field==2?0:1), &(*(*MultiInput)[field]), &(*(*MultiOutput)[field]), (is_E?*E_Index[field]:*B_Index[field]));
#endif
        }
    }
#endif

    template <int degree, int dimension>
    void hodge (infrastructure ifr, int start_from, amrex::MultiFab *MultiInput, amrex::MultiFab *MultiOutput,
                amrex::IndexType IndexVect){

        amrex::Real c1[1];

        for (amrex::MFIter mfi(*MultiOutput); mfi.isValid(); ++mfi ) {
                const amrex::Box& bx = mfi.validbox();

                amrex::IntVect lo = {bx.smallEnd()};
                amrex::IntVect hi = {bx.bigEnd()};

                std::vector<amrex::Real> stencil;
                switch (start_from) {
                case 0:
                    switch (degree) {
                    case 2:
                        stencil = {1.0};
                        break;
                    case 4:
                         stencil = {1.0/8.0, 6.0/8.0, 1.0/8.0};
                        break;
                    case 6:
                        stencil = {-7.0/384.0, 44.0/384.0, 310.0/384.0, 44.0/384.0, -7.0/384.0};
                        break;
                    default: amrex::Print() << "degree not implemented" << std::endl;
                    }
                    break;
                case 1:
                    switch (degree) {
                    case 2:
                        stencil = {1.0};
                        break;
                    case 4:
                        stencil = {-1.0/24.0, 13/12.0, -1.0/24.0};
                        break;
                    case 6:
                        stencil = {3.0/640.0, -29.0/480.0, 1067.0/960.0, -29.0/480.0, 3.0/640.0};
                        break;
                    default: amrex::Print() << "degree not implemented" << std::endl;
                    }
                    break;
                }



#if (GEMPIC_SPACEDIM > 2)
                for(int k=lo[2]; k<=hi[2]; k++){
#endif
#if (GEMPIC_SPACEDIM > 1)
                    for(int j=lo[1]; j<=hi[1]; j++){
#endif
                        for(int l=lo[0]; l<=hi[0]; l++){

                            // the box for these values:
                            amrex::Box cc(amrex::IntVect{AMREX_D_DECL(l,j,k)}, amrex::IntVect{AMREX_D_DECL(l,j,k)}, IndexVect);

                            amrex::Real val = 0.0;
                            for (int d = 0; d<(degree-1); d++) {
                                int shift = d - degree/2 + 1;
                                (*MultiInput)[mfi].getVal(c1, amrex::IntVect{AMREX_D_DECL(l+(dimension==0?shift:0),j+(dimension==1?shift:0),k+(dimension==2?shift:0))},0,1);
                                val += stencil[d]*(*c1);
                            }

                            (*MultiOutput)[mfi].setVal(val, cc, 0, 1);

                        }
#if (GEMPIC_SPACEDIM > 1)
                    }
#endif
#if (GEMPIC_SPACEDIM > 2)
                }
#endif
                // next mfi
            }
            (*MultiOutput).FillBoundary(ifr.geom.periodicity());
    }


    /**
     * Advance time and both fields by a time step of length @p dt_step with a Yee scheme
     */
    void advance (infrastructure ifr, amrex::Real dt_step){

        time = time+dt_step;

        advance_E(ifr, dt_step, true, true, &HB_Array, &E_Array);
        advance_B(ifr, dt_step, true, true, &HE_Array, &B_Array);
    }

    /**
     * Initialize rho and phi with given funtions
     */
    template <int numspec>
    void init_rho_phi(infrastructure ifr,
                      vlasov_maxwell<vdim, numspec> VM){

        double x, y, z;
        int err;
        te_variable read_vars[3] = {{"x", &x}, {"y", &y}, {"z", &z}};
        int varcount = 3;
        te_expr *rho_parse = te_compile(VM.rho.c_str(), read_vars, varcount, &err);
        te_expr *phi_parse = te_compile(VM.phi.c_str(), read_vars, varcount, &err);

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
                        x = l*ifr.dx[0];
#if (GEMPIC_SPACEDIM > 1)
                        y = j*ifr.dx[1];
#endif
#if (GEMPIC_SPACEDIM > 2)
                        z = k*ifr.dx[2];
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

        rho.FillBoundary(ifr.geom.periodicity());
        phi.FillBoundary(ifr.geom.periodicity());
    }

    template <int degree>
    void init_rho_phi (std::array<std::string, 2> fields, std::array<amrex::Real, 3> k,infrastructure ifr){

        // rho
        projection<degree>(fields[0], k, 0.0, ifr, {AMREX_D_DECL(true, true, true)}, amrex::IntVect::TheNodeVector(), &(rho));
        // phi
        projection<degree>(fields[1], k, 0.0, ifr, {AMREX_D_DECL(true, true, true)}, amrex::IntVect::TheNodeVector(), &(phi));
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

        rho.FillBoundary(ifr.geom.periodicity());
        phi.FillBoundary(ifr.geom.periodicity());
    }

    template <int stencil_length>
    void matrix_mult (infrastructure ifr, std::array<std::array<amrex::Real, stencil_length>, GEMPIC_SPACEDIM> stencil, amrex::MultiFab *MultiInput, amrex::MultiFab *MultiOutput,
                amrex::IndexType IndexVect){

        amrex::Real c1[1];

        for (amrex::MFIter mfi(*MultiOutput); mfi.isValid(); ++mfi ) {
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
                            amrex::Box cc(amrex::IntVect{AMREX_D_DECL(l,j,k)}, amrex::IntVect{AMREX_D_DECL(l,j,k)}, IndexVect);

                            amrex::Real val = 0.0;
                            for (int dimension=0; dimension < GEMPIC_SPACEDIM; dimension++) {
                                for (int d = 0; d<(stencil_length); d++) {
                                    int shift = d - (stencil_length+1)/2 + 1;
                                    (*MultiInput)[mfi].getVal(c1, amrex::IntVect{AMREX_D_DECL(l+(dimension==0?shift:0),j+(dimension==1?shift:0),k+(dimension==2?shift:0))},0,1);
                                    val += stencil[dimension][d]*(*c1);
                                }
                            }

                            (*MultiOutput)[mfi].setVal(val, cc, 0, 1);

                        }
#if (GEMPIC_SPACEDIM > 1)
                    }
#endif
#if (GEMPIC_SPACEDIM > 2)
                }
#endif
                // next mfi
            }
            (*MultiOutput).FillBoundary(ifr.geom.periodicity());
    }

    template <int stencil_length, int dimension>
    void solve_stencil_CG(std::array<std::array<amrex::Real, stencil_length>, GEMPIC_SPACEDIM> stencil,
                          amrex::MultiFab *b, amrex::MultiFab *x, infrastructure ifr, amrex::IndexType IndexVect,
                          int verbose = 0, amrex::Real reltol = 1.e-11, int maxsteps = 100){
        if(verbose > 0) {
            amrex::Print() << "----------------------------------------------------------------" << std::endl;
            amrex::Print() << "Started solve_CG" << std::endl;
        }

        amrex::MultiFab r(convert(ifr.grid, *E_Index[dimension]),ifr.distriMap,Ncomp,Nghost);
        amrex::MultiFab d(convert(ifr.grid, *E_Index[dimension]),ifr.distriMap,Ncomp,Nghost);
        amrex::MultiFab z(convert(ifr.grid, *E_Index[dimension]),ifr.distriMap,Ncomp,Nghost);
        amrex::MultiFab tmp(convert(ifr.grid, *E_Index[dimension]),ifr.distriMap,Ncomp,Nghost);

        amrex::Real alpha, beta, rr;

        // r = A*x
        matrix_mult<stencil_length>(ifr, stencil, x, &r, IndexVect);
        // r = b-A*x
        r.mult(-1.0, 0, 1, Nghost);
        r.plus(*b, 0, 1, Nghost);
        // d = r
        amrex::MultiFab::Copy(d, r, 0, 0, 1, Nghost);

        bool contin = true;
        int step = 0;
        amrex::Real norm_r = 2*reltol;
        while(contin) {

            // z = A*d
            matrix_mult<stencil_length>(ifr, stencil, &d, &z, IndexVect);

            // alpha = <r,r>/<d,z>
            rr = amrex::MultiFab::Dot(*E_Masks[dimension], r, 0, r, 0, 1, 0);
            alpha = rr/amrex::MultiFab::Dot(*E_Masks[dimension], d, 0, z, 0, 1, 0);

            // x = x + alpha*d
            d.mult(alpha, 0, 1, Nghost);
            x->plus(d, 0, 1, Nghost);
            d.mult(1.0/alpha, 0, 1, Nghost);

            // r = r - alpha*z
            z.mult(alpha, 0, 1, Nghost);
            r.minus(z, 0, 1, Nghost);

            // beta = <r,r>/rr (rr is old residuum scalar-prod)
            beta = amrex::MultiFab::Dot(*E_Masks[dimension], r, 0, r, 0, 1, 0)/rr;

            // d = r + beta*d
            d.mult(beta, 0, 1, Nghost);
            d.plus(r, 0, 1, Nghost);

            step++;
            norm_r = r.norm2(0, ifr.geom.periodicity());
            if(verbose > 1) {
                amrex::Print() << "step: " << step << ", residuum: " << norm_r << std::endl;
            }
            if (step >= maxsteps || norm_r < reltol) {
                contin = false;
            }
        }

        if(verbose > 0) {
            amrex::Print() << "Number of iterations: " << step << ", final residuum: " << norm_r << std::endl;
            amrex::Print() << "Finished solve_CG" << std::endl;
            amrex::Print() << "----------------------------------------------------------------" << std::endl;
        }
    }

    template<int degree>
    void solve_hodge_CG(amrex::MultiFab *b, amrex::MultiFab *x, amrex::MultiFab *k, infrastructure ifr, int dimension, int verbose = 0, amrex::Real reltol = 1.e-11,
                        int maxsteps = 100){
        if(verbose > 0) {
            amrex::Print() << "----------------------------------------------------------------" << std::endl;
            amrex::Print() << "Started solve_hodge_CG" << std::endl;
        }

        amrex::MultiFab r(convert(ifr.grid, *E_Index[dimension]),ifr.distriMap,Ncomp,Nghost);
        amrex::MultiFab d(convert(ifr.grid, *E_Index[dimension]),ifr.distriMap,Ncomp,Nghost);
        amrex::MultiFab z(convert(ifr.grid, *E_Index[dimension]),ifr.distriMap,Ncomp,Nghost);
        amrex::MultiFab tmp(convert(ifr.grid, *E_Index[dimension]),ifr.distriMap,Ncomp,Nghost);
        amrex::MultiFab k_scaled(convert(ifr.grid, *E_Index[dimension]),ifr.distriMap,Ncomp,Nghost);

        amrex::Real alpha, beta, rr, kern_factor;

        // r = A*x
        hodge<degree,0>(ifr, (dimension==0?0:1), x, &r, *E_Index[dimension]);
        hodge<degree,1>(ifr, (dimension==1?0:1), &r, &tmp, *E_Index[dimension]);
        hodge<degree,2>(ifr, (dimension==2?0:1), &tmp, &r, *E_Index[dimension]);
        kern_factor = amrex::MultiFab::Dot(*E_Masks[dimension], *k, 0, r, 0, 1, 0)/amrex::MultiFab::Dot(*E_Masks[dimension], *k, 0, *k, 0, 1, 0);
        amrex::MultiFab::Copy(k_scaled,*k, 0, 0, 1, Nghost);
        k_scaled.mult(kern_factor, 0, 1, Nghost);
        r.plus(k_scaled, 0, 1, Nghost);

        // r = b-A*x
        r.mult(-1.0, 0, 1, Nghost);
        r.plus(*b, 0, 1, Nghost);
        // d = r
        amrex::MultiFab::Copy(d, r, 0, 0, 1, Nghost);

        int step = 0;
        for(step = 0; step<maxsteps; step++) {
            // rr
            rr = amrex::MultiFab::Dot(*E_Masks[dimension], r, 0, r, 0, 1, 0);
            if(verbose > 1) {
                amrex::Print() << "step: " << step << ", residuum: " << sqrt(rr)*ifr.dx[0]*ifr.dx[1]*ifr.dx[2] << " , rr: " << rr << std::endl;
            }
            if (sqrt(rr)*ifr.dx[0]*ifr.dx[1]*ifr.dx[2] < reltol) break;


            // z = A*d
            hodge<degree,0>(ifr, (dimension==0?0:1), &d, &z, *E_Index[dimension]);
            hodge<degree,1>(ifr, (dimension==1?0:1), &z, &tmp, *E_Index[dimension]);
            hodge<degree,2>(ifr, (dimension==2?0:1), &tmp, &z, *E_Index[dimension]);

            kern_factor = amrex::MultiFab::Dot(*E_Masks[dimension], *k, 0, z, 0, 1, 0)/amrex::MultiFab::Dot(*E_Masks[dimension], *k, 0, *k, 0, 1, 0);
            amrex::MultiFab::Copy(k_scaled,*k, 0, 0, 1, Nghost);
            k_scaled.mult(kern_factor, 0, 1, Nghost);
            z.plus(k_scaled, 0, 1, Nghost);

            // alpha = <r,r>/<d,z>
            alpha = rr/amrex::MultiFab::Dot(*E_Masks[dimension], d, 0, z, 0, 1, 0);

            // x = x + alpha*d
            amrex::MultiFab::Copy(k_scaled,d, 0, 0, 1, Nghost);
            k_scaled.mult(alpha, 0, 1, Nghost);
            x->plus(k_scaled, 0, 1, Nghost);

            // r = r - alpha*z
            z.mult(alpha, 0, 1, Nghost);
            r.minus(z, 0, 1, Nghost);

            // beta = <r,r>/rr (rr is old residuum scalar-prod)
            beta = amrex::MultiFab::Dot(*E_Masks[dimension], r, 0, r, 0, 1, 0)/rr;

            // d = r + beta*d
            d.mult(beta, 0, 1, Nghost);
            d.plus(r, 0, 1, Nghost);

        }

        if(verbose > 0) {
            amrex::Print() << "Number of iterations: " << step << ", final residuum: " << sqrt(rr)*ifr.dx[0]*ifr.dx[1]*ifr.dx[2] << std::endl;
            amrex::Print() << "Finished solve_hodge_CG" << std::endl;
            amrex::Print() << "----------------------------------------------------------------" << std::endl;
        }

        (*x).FillBoundary(ifr.geom.periodicity());
    }

    template <int degree>
    void poisson_operator (amrex::MultiFab *MultiIn, amrex::MultiFab *MultiOut, amrex::MultiFab *kx,
                           amrex::MultiFab *ky, amrex::MultiFab *kz, infrastructure ifr, int verbose = 0,
                           amrex::Real reltol = 1.e-11, int maxsteps = 100) {

        int dimension = 0;
        amrex::Real kern_factor;

        const int stencil_length = 3;
        std::array<std::array<amrex::Real, stencil_length>, GEMPIC_SPACEDIM> stencil_x;
        stencil_x[0] = {-1.0/ifr.dx[0], 1.0/ifr.dx[0], 0.0};
        stencil_x[1] = {0.0, 0.0, 0.0};
        stencil_x[2] = {0.0, 0.0, 0.0};
        std::array<std::array<amrex::Real, stencil_length>, GEMPIC_SPACEDIM> stencil_y;
        stencil_y[0] = {0.0, 0.0, 0.0};
        stencil_y[1] = {-1.0/ifr.dx[1], 1.0/ifr.dx[1], 0.0};
        stencil_y[2] = {0.0, 0.0, 0.0};
        std::array<std::array<amrex::Real, stencil_length>, GEMPIC_SPACEDIM> stencil_z;
        stencil_z[0] = {0.0, 0.0, 0.0};
        stencil_z[1] = {0.0, 0.0, 0.0};
        stencil_z[2] = {-1.0/ifr.dx[2], 1.0/ifr.dx[2], 0.0};


        amrex::MultiFab Drho_x(convert(ifr.grid, *E_Index[0]),ifr.distriMap,Ncomp,Nghost);
        amrex::MultiFab Drho_y(convert(ifr.grid, *E_Index[1]),ifr.distriMap,Ncomp,Nghost);
        amrex::MultiFab Drho_z(convert(ifr.grid, *E_Index[2]),ifr.distriMap,Ncomp,Nghost);

        amrex::MultiFab HDrho_x(convert(ifr.grid, *E_Index[0]),ifr.distriMap,Ncomp,Nghost);
        HDrho_x.setVal(1.0,0);
        amrex::MultiFab HDrho_y(convert(ifr.grid, *E_Index[1]),ifr.distriMap,Ncomp,Nghost);
        HDrho_y.setVal(1.0,0);
        amrex::MultiFab HDrho_z(convert(ifr.grid, *E_Index[2]),ifr.distriMap,Ncomp,Nghost);
        HDrho_z.setVal(1.0,0);

        // 1) Apply D
        matrix_mult<stencil_length>(ifr, stencil_x, MultiIn, &Drho_x, *E_Index[0]);
        matrix_mult<stencil_length>(ifr, stencil_y, MultiIn, &Drho_y, *E_Index[1]);
        matrix_mult<stencil_length>(ifr, stencil_z, MultiIn, &Drho_z, *E_Index[2]);

        // 2) Apply hodge
            // to x comp
            amrex::MultiFab::Copy(HDrho_x, Drho_x, 0, 0, 1, Nghost);
            solve_hodge_CG<degree>(&Drho_x, &HDrho_x, kx, ifr, dimension, 0, reltol, maxsteps);
            //amrex::MultiFab::Copy(HDrho_x, Drho_x, 0, 0, 1, Nghost);

            // to y comp
            dimension = 1;
            amrex::MultiFab::Copy(HDrho_y, Drho_y, 0, 0, 1, Nghost);
            solve_hodge_CG<degree>(&Drho_y, &HDrho_y, ky, ifr, dimension, 0, reltol, maxsteps);
            //amrex::MultiFab::Copy(HDrho_y, Drho_y, 0, 0, 1, Nghost);

            // to z comp
            dimension = 2;
            amrex::MultiFab::Copy(HDrho_z, Drho_z, 0, 0, 1, Nghost);
            solve_hodge_CG<degree>(&Drho_z, &HDrho_z, kz, ifr, dimension, 0, reltol, maxsteps);
            //amrex::MultiFab::Copy(HDrho_z, Drho_z, 0, 0, 1, Nghost);;

        // 3) Apply D^T

        stencil_x[0] = {0.0, 1.0/ifr.dx[0], -1.0/ifr.dx[0]};
        stencil_x[1] = {0.0, 0.0, 0.0};
        stencil_x[2] = {0.0, 0.0, 0.0};

        stencil_y[0] = {0.0, 0.0, 0.0};
        stencil_y[1] = {0.0, 1.0/ifr.dx[1], -1.0/ifr.dx[1]};
        stencil_y[2] = {0.0, 0.0, 0.0};

        stencil_z[0] = {0.0, 0.0, 0.0};
        stencil_z[1] = {0.0, 0.0, 0.0};
        stencil_z[2] = {0.0, 1.0/ifr.dx[2], -1.0/ifr.dx[2]};

        matrix_mult<3>(ifr, stencil_x, &HDrho_x, &Drho_x, *E_Index[0]);
        matrix_mult<3>(ifr, stencil_y, &HDrho_y, &Drho_y, *E_Index[1]);
        matrix_mult<3>(ifr, stencil_z, &HDrho_z, &Drho_z, *E_Index[2]);


        amrex::MultiFab::Copy(*MultiOut, Drho_x, 0, 0, 1, Nghost);
        MultiOut->plus(Drho_y, 0, 1, Nghost);
        MultiOut->plus(Drho_z, 0, 1, Nghost);

        MultiOut->FillBoundary(ifr.geom.periodicity());
    }

    template<int degree>
    void solve_poisson_CG(amrex::MultiFab *b, amrex::MultiFab *x, amrex::MultiFab *kx, amrex::MultiFab *ky, amrex::MultiFab *kz,
                          infrastructure ifr, int verbose = 0, amrex::Real reltol = 1.e-11,
                        int maxsteps = 100){
        if(verbose > 0) {
            amrex::Print() << "----------------------------------------------------------------" << std::endl;
            amrex::Print() << "Started solve_poisson_CG" << std::endl;
        }

        amrex::MultiFab r(convert(ifr.grid, amrex::IntVect::TheNodeVector()),ifr.distriMap,Ncomp,Nghost);
        amrex::MultiFab d(convert(ifr.grid, amrex::IntVect::TheNodeVector()),ifr.distriMap,Ncomp,Nghost);
        amrex::MultiFab z(convert(ifr.grid, amrex::IntVect::TheNodeVector()),ifr.distriMap,Ncomp,Nghost);
        amrex::MultiFab tmp(convert(ifr.grid, amrex::IntVect::TheNodeVector()),ifr.distriMap,Ncomp,Nghost);

        amrex::MultiFab k(convert(ifr.grid, amrex::IntVect::TheNodeVector()),ifr.distriMap,Ncomp,Nghost);
        amrex::MultiFab k_scaled(convert(ifr.grid, amrex::IntVect::TheNodeVector()),ifr.distriMap,Ncomp,Nghost);
        k.setVal(1.0,0);
        k.FillBoundary(ifr.geom.periodicity());
        amrex::Real kern_factor;

        amrex::Real alpha, beta, rr;

        // subtract kernel from x0:
        kern_factor = amrex::MultiFab::Dot(*nodal_Mask, k, 0, *b, 0, 1, 0)/amrex::MultiFab::Dot(*nodal_Mask, k, 0, k, 0, 1, 0);
        amrex::MultiFab::Copy(k_scaled,k, 0, 0, 1, Nghost);
        k_scaled.mult(kern_factor, 0, 1, Nghost);
        b->minus(k_scaled, 0, 1, Nghost);

        // r = A*x
        poisson_operator<degree> (x, &r, kx, ky, kz, ifr, verbose, reltol, maxsteps);
        kern_factor = amrex::MultiFab::Dot(*nodal_Mask, k, 0, r, 0, 1, 0)/amrex::MultiFab::Dot(*nodal_Mask, k, 0, k, 0, 1, 0);
        amrex::MultiFab::Copy(k_scaled,k, 0, 0, 1, Nghost);
        k_scaled.mult(kern_factor, 0, 1, Nghost);
        r.plus(k_scaled, 0, 1, Nghost);


        // r = b-A*x
        r.mult(-1.0, 0, 1, Nghost);
        r.plus(*b, 0, 1, Nghost);

        // d = r
        amrex::MultiFab::Copy(d, r, 0, 0, 1, Nghost);

        int step = 0;
        for(step = 0; step<maxsteps; step++) {
            // rr
            rr = amrex::MultiFab::Dot(*nodal_Mask, r, 0, r, 0, 1, 0);
            if(verbose > 1) {
                amrex::Print() << "step: " << step << ", residuum: " << sqrt(rr) << " , rr: " << rr*ifr.dx[0]*ifr.dx[1]*ifr.dx[2] << std::endl;
            }
            if (sqrt(rr)*ifr.dx[0]*ifr.dx[1]*ifr.dx[2] < reltol) break;

            // z = A*d
            poisson_operator<degree> (&d, &z, kx, ky, kz, ifr, verbose, reltol, maxsteps);
            kern_factor = amrex::MultiFab::Dot(*nodal_Mask, k, 0, d, 0, 1, 0)/amrex::MultiFab::Dot(*nodal_Mask, k, 0, k, 0, 1, 0);
            amrex::MultiFab::Copy(k_scaled,k, 0, 0, 1, Nghost);
            k_scaled.mult(kern_factor, 0, 1, Nghost);
            d.plus(k_scaled, 0, 1, Nghost);

            // alpha = <r,r>/<d,z>
            alpha = rr/amrex::MultiFab::Dot(*nodal_Mask, d, 0, z, 0, 1, 0);

            // x = x + alpha*d
            amrex::MultiFab::Copy(k_scaled,d, 0, 0, 1, Nghost);
            k_scaled.mult(alpha, 0, 1, Nghost);
            x->plus(k_scaled, 0, 1, Nghost);

            // r = r - alpha*z
            z.mult(alpha, 0, 1, Nghost);
            r.minus(z, 0, 1, Nghost);

            // beta = <r,r>/rr (rr is old residuum scalar-prod)
            beta = amrex::MultiFab::Dot(*nodal_Mask, r, 0, r, 0, 1, 0)/rr;

            // d = r + beta*d
            d.mult(beta, 0, 1, Nghost);
            d.plus(r, 0, 1, Nghost);

        }

        if(verbose > 0) {
            amrex::Print() << "Number of iterations: " << step << ", final residuum: " << sqrt(rr)*ifr.dx[0]*ifr.dx[1]*ifr.dx[2] << std::endl;
            amrex::Print() << "Finished solve_poisson_CG" << std::endl;
            amrex::Print() << "----------------------------------------------------------------" << std::endl;
        }

        (*x).FillBoundary(ifr.geom.periodicity());
    }

    template<int degree>
    void D_from_phi(amrex::MultiFab *phi_mf, std::array<std::unique_ptr<amrex::MultiFab>, vdim > *D,
                    amrex::MultiFab *kx, amrex::MultiFab *ky, amrex::MultiFab *kz,infrastructure ifr, int verbose = 0,
                    amrex::Real reltol = 1.e-11, int maxsteps = 100) {

        std::array<std::unique_ptr<amrex::MultiFab>, vdim > tmp;
        for (int i=0; i<vdim;i++) {
            tmp[i].reset(new amrex::MultiFab (convert(ifr.grid, *E_Index[i]),ifr.distriMap,Ncomp,Nghost));
        }
        // ---------------------------------------------------------------------
        // D-operator

        amrex::Real c1[1], c2[1];

        for (int i=0; i<vdim; i++) {
            for (amrex::MFIter mfi(*tmp[i]); mfi.isValid(); ++mfi ) {
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

                            // phi-derivative
                            (*phi_mf)[mfi].getVal(c1,amrex::IntVect{AMREX_D_DECL(l+(i==0?1:0),j+(i==1?1:0),k+(i==2?1:0))},0,1);
                            (*phi_mf)[mfi].getVal(c2,amrex::IntVect{AMREX_D_DECL(l,j,k)},0,1);

                            (*tmp[i])[mfi].setVal(-(*c1-*c2)/ifr.dx[i], cc, 0, 1);
                        }
#if (GEMPIC_SPACEDIM > 2)
                    }
#endif
#if (GEMPIC_SPACEDIM > 1)
                }
#endif
                // next mfi
            }
            (*tmp[i]).FillBoundary(ifr.geom.periodicity());
        }

        // ---------------------------------------------------------------------
        // H⁻¹-operator
        // to x comp
        int dimension = 0;
        amrex::MultiFab::Copy(*(*D)[dimension], *tmp[dimension], 0, 0, 1, Nghost);
        solve_hodge_CG<degree>(&(*tmp[dimension]), &(*(*D)[dimension]), kx, ifr, dimension, verbose, reltol, maxsteps);
        // to x comp
        dimension = 1;
        amrex::MultiFab::Copy(*(*D)[dimension], *tmp[dimension], 0, 0, 1, Nghost);
        solve_hodge_CG<degree>(&(*tmp[dimension]), &(*(*D)[dimension]), ky, ifr, dimension, verbose, reltol, maxsteps);
        // to x comp
        dimension = 2;
        amrex::MultiFab::Copy(*(*D)[dimension], *tmp[dimension], 0, 0, 1, Nghost);
        solve_hodge_CG<degree>(&(*tmp[dimension]), &(*(*D)[dimension]), kz, ifr, dimension, verbose, reltol, maxsteps);

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

        amrex::MLNodeLaplacian_FD linop({ifr.geom}, {ifr.grid}, {ifr.distriMap}); // linear operator class

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

        mlmg.solve({&phi}, {&rho}, reltol, 0.0);
        phi.FillBoundary(ifr.geom.periodicity());

#endif

        //----------------------------------------------------------------------------------
        // E

        amrex::Real c1[1], c2[1];

        for (int i=0; i<vdim; i++) {
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
                        for (int i=0; i<vdim; i++) {

                            // E-derivatives
                            bool backward = true; // false: central
                            if(backward) {
                                (*E_Array[i])[mfi].getVal(c2,amrex::IntVect{AMREX_D_DECL(l-(i==0?1:0),j-(i==1?1:0),k-(i==2?1:0))},0,1);
                                (*E_Array[i])[mfi].getVal(c1,amrex::IntVect{AMREX_D_DECL(l,j,k)},0,1);

                                rho_val += (*c1-*c2)/ifr.dx[i];
                            } else {
                                (*E_Array[i])[mfi].getVal(c2,amrex::IntVect{AMREX_D_DECL(l-(i==0?1:0),j-(i==1?1:0),k-(i==2?1:0))},0,1);
                                (*E_Array[i])[mfi].getVal(c1,amrex::IntVect{AMREX_D_DECL(l+(i==0?1:0),j+(i==1?1:0),k+(i==2?1:0))},0,1);

                                rho_val += 0.5*(*c1-*c2)/ifr.dx[i];
                            }

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
    template<int numspec, int degx, int degy, int degz>
    amrex::Real compute_gauss_error(infrastructure ifr, Particles::particle_groups<vdim, numspec> *part_gr){

        rho_from_E(ifr); // fills rho_gauss_law

        // Deposit charges in rho:
        rho.setVal(0.0);
        rho.FillBoundary(ifr.geom.periodicity());
        for (int spec=0;spec<numspec;spec++) {
            (*(*part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in
            for (amrex::ParIter<vdim+1,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {
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
                    Particles::gempic_deposit_rho<amrex::Particle<vdim+1>,vdim,degx,degy,degz>(particles[pp], (*part_gr).charge[spec], rhoarr, ifr.plo, ifr.dxi);
                }
                (rho)[pti].atomicAdd(local_rho,tb,tb,0,0,1);
            }
        }

        rho.SumBoundary(ifr.geom.periodicity());
        rho.OverrideSync(*nodal_Mask, ifr.geom.periodicity());
        rho.FillBoundary(ifr.geom.periodicity());

        // we cannot use rho.sum(0) because this would count nodal data at edges twice, whereas norm1 doesn't

        // 1) make all values positive
        rho.plus(-rho.min(0),0,1);
        // 2) compute new mean with L1 norm
        amrex::Real rhomean = rho.norm1(0, ifr.geom.periodicity())/(ifr.n_cell[0]*ifr.n_cell[1]*ifr.n_cell[2]);
        // 3) subtract new mean from rho
        rho.plus(-rhomean,0,1);

        // repeat for rho_gauss_law
        rho_gauss_law.plus(-rho_gauss_law.min(0),0,1);
        amrex::Real rho_gauss_mean = rho_gauss_law.norm1(0, ifr.geom.periodicity())/(ifr.n_cell[0]*ifr.n_cell[1]*ifr.n_cell[2]);
        rho_gauss_law.plus(-rho_gauss_mean,0,1);

        rho_gauss_law.minus(rho, 0, 1, 0);
        return(gempic_norm(&(rho_gauss_law), ifr, 2));
    }

    /**
     * Compute E for rho from particles (for diagnostics: Gauss law)
     */
    template<int numspec>
    amrex::Real compute_gauss_error_E(infrastructure ifr, Particles::particle_groups<vdim, numspec> *part_gr){

        (*E_sol_Array[0]).Copy(*E_sol_Array[0], *E_Array[0],0,0,1,Nghost);
        (*E_sol_Array[1]).Copy(*E_sol_Array[1], *E_Array[1],0,0,1,Nghost);
        (*E_sol_Array[2]).Copy(*E_sol_Array[2], *E_Array[2],0,0,1,Nghost);

        // init phi for poisson solver
        double x, y, z;
        int err;
        std::string phi_str = "4 * 0.5 * cos(0.5 * x)";
        std::string rho_str = "0.0";

        te_variable read_vars[] = {{"x", &x}, {"y", &y}, {"z", &z}};
        int varcount = 3;
        te_expr *rho_parse = te_compile(rho_str.c_str(), read_vars, varcount, &err);
        te_expr *phi_parse = te_compile(phi_str.c_str(), read_vars, varcount, &err);
        init_rho_phi(ifr, phi_parse, rho_parse, &x, &y, &z);

        // Deposit charges in rho:

        rho.setVal(0.0);
        rho.FillBoundary(ifr.geom.periodicity());
        for (int spec=0;numspec;spec++) {
            (*(*part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in
            for (amrex::ParIter<vdim+1,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {
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
                    Particles::gempic_deposit_rho<amrex::Particle<vdim+1>,vdim>(particles[pp], (*part_gr).charge[spec], rhoarr, ifr.plo, ifr.dxi);
                }
                (rho)[pti].atomicAdd(local_rho,tb,tb,0,0,1);
            }
        }

        rho.SumBoundary(ifr.geom.periodicity());
        rho.OverrideSync(*nodal_Mask, ifr.geom.periodicity());
        rho.FillBoundary(ifr.geom.periodicity());

        solve_poisson(ifr);

        (*E_Array[0]).minus(*E_sol_Array[0], 0, 1, 0);
        (*E_Array[1]).minus(*E_sol_Array[1], 0, 1, 0);
        (*E_Array[2]).minus(*E_sol_Array[2], 0, 1, 0);

        amrex::Real res = gempic_norm(&(*E_Array[0]), ifr, 2)+gempic_norm(&(*E_Array[1]), ifr, 2)+gempic_norm(&(*E_Array[2]), ifr, 2);

        (*E_Array[0]).Copy(*E_Array[0], *E_sol_Array[0],0,0,1,Nghost);
        (*E_Array[1]).Copy(*E_Array[1], *E_sol_Array[1],0,0,1,Nghost);
        (*E_Array[2]).Copy(*E_Array[2], *E_sol_Array[2],0,0,1,Nghost);

        return(res);
    }

    template<int numspec, int degx, int degy, int degz, int degree>
    amrex::Real compute_gauss_error_hodge(infrastructure ifr, Particles::particle_groups<vdim, numspec> *part_gr){

        amrex::MultiFab kx(convert(ifr.grid, *E_Index[0]),ifr.distriMap,1,Nghost);
        kx.setVal(1.0, 0);
        kx.FillBoundary(ifr.geom.periodicity());
        amrex::MultiFab ky(convert(ifr.grid, *E_Index[1]),ifr.distriMap,1,Nghost);
        ky.setVal(1.0, 0);
        ky.FillBoundary(ifr.geom.periodicity());
        amrex::MultiFab kz(convert(ifr.grid, *E_Index[2]),ifr.distriMap,1,Nghost);
        kz.setVal(1.0, 0);
        kz.FillBoundary(ifr.geom.periodicity());

        // init phi for poisson solver
        double x, y, z;
        int err;
        std::string phi_str = "4 * 0.5 * cos(0.5 * x)";
        std::string rho_str = "0.0";

        te_variable read_vars[] = {{"x", &x}, {"y", &y}, {"z", &z}};
        int varcount = 3;
        te_expr *rho_parse = te_compile(rho_str.c_str(), read_vars, varcount, &err);
        te_expr *phi_parse = te_compile(phi_str.c_str(), read_vars, varcount, &err);
        init_rho_phi(ifr, phi_parse, rho_parse, &x, &y, &z);

        // Deposit charges in rho:

        rho.setVal(0.0);
        rho.FillBoundary(ifr.geom.periodicity());
        for (int spec=0;numspec;spec++) {
            (*(*part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in
            for (amrex::ParIter<vdim+1,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {
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
                    splines_at_particles<degx, degy, degz> spline;
                    std::array<amrex::Real,GEMPIC_SPACEDIM> position;
                    for (unsigned int d=0;d<GEMPIC_SPACEDIM;++d)
                      position[d] = particles[pp].pos(d);
                    spline.init_particles( position, ifr.plo, ifr.dxi);
                    gempic_deposit_rho_C3<degx, degy, degz> ( spline,
                                                  (*part_gr).charge[spec]*ifr.dxi[GEMPIC_SPACEDIM]*particles[pp].rdata(vdim),
                                                  rhoarr);
                }
                (rho)[pti].atomicAdd(local_rho,tb,tb,0,0,1);
            }
        }

        rho.SumBoundary(ifr.geom.periodicity());
        rho.OverrideSync(*nodal_Mask, ifr.geom.periodicity());
        rho.FillBoundary(ifr.geom.periodicity());

        solve_poisson(ifr, 1e-11);
        solve_poisson_CG<degree>(&rho, &phi, &kx, &ky, &kz, ifr);

         D_from_phi<degree>(&phi, &(E_sol_Array), &kx, &ky, &kz, ifr);

        (*E_sol_Array[0]).minus(*E_Array[0], 0, 1, 0);
        (*E_sol_Array[1]).minus(*E_Array[1], 0, 1, 0);
        (*E_sol_Array[2]).minus(*E_Array[2], 0, 1, 0);

        amrex::Real res = gempic_norm(&(*E_Array[0]), ifr, 2)+gempic_norm(&(*E_Array[1]), ifr, 2)+gempic_norm(&(*E_Array[2]), ifr, 2);

        return(res);
    }

    /**
     * Compute norm of errors of E and B field from function with exact solution
     */
    template<int degree>
        std::array<amrex::Real,vdim+int(vdim/2.5)*2+1> computeError(std::array<std::string, vdim> fields_E,
                                                                    std::array<std::string, int(vdim/2.5)*2+1> fields_B, std::array<amrex::Real, 3> k,
                                                              bool bool_B,
                                                              infrastructure ifr){
            std::array<amrex::Real,vdim+int(vdim/2.5)*2+1> E_B_error;

            exactSolution<degree>(fields_E, fields_B, k, bool_B, ifr);

            for (int i=0; i<vdim; i++){
                (*E_sol_Array[i]).minus(*E_Array[i], 0, 1, 0);
                E_B_error[i] = gempic_norm(&(*E_sol_Array[i]), ifr, 2);
            }
            if (bool_B) {
                for (int i=0; i<int(vdim/2.5)*2+1; i++){
                    (*B_sol_Array[i]).minus(*B_Array[i], 0, 1, 0);
                    E_B_error[i+vdim] = gempic_norm(&(*B_sol_Array[i]), ifr, 2);
                }
            }
            return(E_B_error);
        }

    private:


        template<int degree>
        void exactSolution(std::array<std::string, vdim> fields_E,
                           std::array<std::string, int(vdim/2.5)*2+1> fields_B, std::array<amrex::Real, 3> k,
                           bool bool_B, infrastructure ifr){
            amrex::Real btime = time + dt/2.0;

            // E-field
            for (int i=0; i<vdim; i++) {
                projection<degree>(fields_E[i], k, time, ifr, {AMREX_D_DECL(i!=0, i!=1, i!=2)}, *E_Index[i], &(*E_sol_Array[i]));
            }

            if (bool_B){
                // B-field
                for (int i=0; i<int(vdim/2.5)*2+1; i++) {
                    projection<degree>(fields_B[i], k, btime, ifr, {AMREX_D_DECL(i!=0, i!=1, i!=2)}, *B_Index[i], &(*B_sol_Array[i]));
                }
            }
        }

};
}
}

#endif
