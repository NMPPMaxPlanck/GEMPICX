#ifndef MAXWELL_YEE_H_
#define MAXWELL_YEE_H_

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
#include <GEMPIC_computational_domain.H>
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

using namespace CompDom;
using namespace Utils;
using namespace Vlasov_Maxwell;
using namespace Particles;

using funcToProjType = amrex::Real(*)(amrex::Real, amrex::Real, amrex::Real, amrex::Real, int);

namespace Gempic {
namespace Field_solvers {
/** @brief Class Maxwell_Yee
 *
 * Class storing all the field information (both analytical (@p E_sol_Array, @p B_sol_Array) and numerical (@p E_Array, @p B_Array, @p J_Array), infrastructure information
 * for the fields (@p E_Index, @p B_Index, @p Nghost, @p Ncomp, @p time) and the necessary components to solve the Poisson problem for the first step (@p phi_init, @p rho,
 * @p phi, @p sigma). In addition, the class can initialize the fields (init_E_B, ) and the functions for the Poisson problem (init_rho_phi), advance the complete
 * problem (advance) or parts of it (advance_E, advance_B, advance_time), initialize E by solving the Poisson problem (solve_poisson), fill ghost cells of all relevant MultiFabs
 * (FillBD), compute the analytical solution (exactSolution) and compute the error of the numerical solution (computeError).
 *
 * @authors Irene Garnelo, Katharina Kormann
 *
*/
template<int vdim>
class maxwell_yee
{
public:

    int nsteps; /** number of simulation steps */
    amrex::Real dt; /** stepsize*/
    amrex::GpuArray<std::unique_ptr<amrex::MultiFab>, vdim > E_Array; /** Array storing components of E-Field */
    amrex::GpuArray<std::unique_ptr<amrex::MultiFab>, vdim > Alfven_Tensor; /** Array storing the Alfven tensor */
    amrex::GpuArray<std::unique_ptr<amrex::MultiFab>, vdim > HE_Array;
    amrex::GpuArray<std::unique_ptr<amrex::MultiFab>, vdim > J_Array;
    amrex::GpuArray<std::unique_ptr<amrex::MultiFab>, int(vdim/2.5)*2+1 > B_Array;
    amrex::GpuArray<std::unique_ptr<amrex::MultiFab>, int(vdim/2.5)*2+1 > HB_Array;
    amrex::GpuArray<std::unique_ptr<amrex::MultiFab>, vdim > E_sol_Array;
    amrex::GpuArray<std::unique_ptr<amrex::MultiFab>, int(vdim/2.5)*2+1 > B_sol_Array;
    amrex::GpuArray<std::unique_ptr<amrex::IndexType>, vdim> E_Index;
    amrex::GpuArray<std::unique_ptr<amrex::IndexType>, int(vdim/2.5)*2+1 > B_Index;
    amrex::GpuArray<std::unique_ptr<amrex::iMultiFab>, vdim> E_Masks;
    amrex::GpuArray<std::unique_ptr<amrex::iMultiFab>, vdim> B_Masks;
    std::unique_ptr<amrex::iMultiFab> nodal_Mask;
    amrex::MultiFab phi_init; // for Poisson
    amrex::MultiFab rho; // for Poisson
    amrex::MultiFab rho_gauss_law; // for Poisson
    int Nghost;
    amrex::MultiFab phi; // for Poisson
    //amrex::MLMG* mlmg_ptr;

    amrex::Real epref_ep0 = 1.;
    amrex::Real curlB_scaling = 1.;
    amrex::Real energy_b_scaling = 1.;

private:

    int Ncomp; /** Number of components for each MultiFab. Is 1 because the different field components are differently staggered */
    amrex::Real time;
    amrex::MultiFab sigma; //for Poisson

public:

    // The default constructor.
    /**
     * Default Constructor.
     */
    maxwell_yee ();

    // Explicit const
    maxwell_yee  (computational_domain ifr, amrex::Real dt_in, amrex::Real nsteps_in, int Nghost_in,  amrex::Real epref_ep0_in,
                  amrex::Real curlB_scaling_in ,
                  amrex::Real energy_b_scaling_in){
      initialize(ifr, dt_in, nsteps_in, Nghost_in, epref_ep0_in, curlB_scaling_in, energy_b_scaling_in);
    }

    void initialize (computational_domain ifr, amrex::Real dt_in, amrex::Real nsteps_in, int Nghost_in,  amrex::Real epref_ep0_in,
                  amrex::Real curlB_scaling_in ,
                  amrex::Real energy_b_scaling_in){
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

        Nghost = Nghost_in;
        Ncomp = 1;

        dt = dt_in;
        time = 0.0;

        // make the multifabs nodal-/cell-centered according to indices
        for (int i = 0; i<vdim; i++) {
            E_Array[i].reset(new amrex::MultiFab (convert(ifr.grid, *E_Index[i]),ifr.distriMap,Ncomp,Nghost));
            Alfven_Tensor[i].reset(new amrex::MultiFab (convert(ifr.grid, *E_Index[i]),ifr.distriMap,Ncomp,Nghost));
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

        nsteps = nsteps_in;

        curlB_scaling = curlB_scaling_in;
        epref_ep0 = epref_ep0_in;
        energy_b_scaling = energy_b_scaling_in;

        //-----------------------------------------------------------------------------
        // Fill Owner Mask

        for (int comp=0; comp<vdim; comp++) {
            E_Masks[comp].reset(new amrex::iMultiFab (convert(ifr.grid, *E_Index[comp]),ifr.distriMap,1,Nghost));
            (*(E_Masks[comp])).setVal(0.0);
            (*(E_Masks[comp])).FillBoundary(ifr.geom.periodicity());

            for ( amrex::MFIter mfi((*(E_Masks[comp]))); mfi.isValid(); ++mfi ) {
                const amrex::Box& bx = mfi.validbox();
                amrex::IntVect hi = {bx.bigEnd()};

                amrex::Array4<int> const& mask_arr = (*(E_Masks[comp]))[mfi].array();
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    // if-loop to exclude ownership for the point that is at the upper boundary for nodal dimensions
                    if ((i<=(comp==0?hi[0]:(hi[0]-1))) && (j<=(comp==1?hi[1]:(hi[1]-1))) && (k<=(comp==2?hi[2]:(hi[2]-1))))
                        mask_arr(i,j,k) = 1.0;
                });


            }
        }


        for (int comp=0; comp<int(vdim/2.5)*2+1; comp++) {
            B_Masks[comp].reset(new amrex::iMultiFab (convert(ifr.grid, *B_Index[comp]),ifr.distriMap,1,Nghost));
            (*(B_Masks[comp])).setVal(0.0);
            (*(B_Masks[comp])).FillBoundary(ifr.geom.periodicity());

            for ( amrex::MFIter mfi((*(B_Masks[comp]))); mfi.isValid(); ++mfi ) {
                const amrex::Box& bx = mfi.tilebox();
                amrex::IntVect hi = {bx.bigEnd()};

                amrex::Array4<int> const& mask_arr = (*(B_Masks[comp]))[mfi].array();
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    // if-loop to exclude ownership for the point that is at the upper boundary for nodal dimensions
                    if ((i<=(comp==0?(hi[0]-1):hi[0])) && (j<=(comp==1?(hi[1]-1):hi[1])) && (k<=(comp==2?(hi[2]-1):hi[2])))
                        mask_arr(i,j,k) = 1.0;
                });


            }
        }

        nodal_Mask.reset(new amrex::iMultiFab (convert(ifr.grid, amrex::IntVect::TheNodeVector()),ifr.distriMap,1,Nghost));
        (*nodal_Mask).setVal(0.0);
        (*nodal_Mask).FillBoundary(ifr.geom.periodicity());

        for ( amrex::MFIter mfi(*nodal_Mask); mfi.isValid(); ++mfi ) {
            const amrex::Box& bx = mfi.validbox();
            amrex::IntVect hi = {bx.bigEnd()};

            amrex::Array4<int> const& mask_arr = (*nodal_Mask)[mfi].array();
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                // if-loop to exclude ownership for the point that is at the upper boundary for nodal dimensions
                if ((i<=(hi[0]-1)) && (j<=(hi[1]-1)) && (k<=(hi[2]-1)))
                    mask_arr(i,j,k) = 1.0;
            });

        }

    }

    /**
         * Explicit constructor. Initializes Indextypes, MultiFabs and OwnerMasks.
         *  @param VM - object of type vlasov-maxwell with all relevant information
         *  @param ifr - object of type infrastructure
         */
    template <int numspec>
    maxwell_yee (vlasov_maxwell<vdim, numspec> VM, computational_domain ifr){
        *this = maxwell_yee(ifr, VM.dt, VM.n_steps, VM.Nghost, VM.epref_ep0, VM.curlB_scaling, VM.energy_b_scaling);

    }

    template<int degree>
    void projection (std::string function_string,
                     amrex::GpuArray<amrex::Real, 3> k,
                     amrex::Real t,
                     computational_domain ifr,
                     amrex::GpuArray<bool, GEMPIC_SPACEDIM> integrate,
                     amrex::IndexType IndexVect,
                     amrex::MultiFab *MultiF
                     ) {
        double t_eval = t;
        double kx = k[0];
        double ky = k[1];
        double kz = k[2];
        int varcount = 7;

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


        int x_len[GEMPIC_SPACEDIM];
        for (int i=0;i<GEMPIC_SPACEDIM;i++) x_len[i] = (integrate[i]?(degree-1):1);

        for ( amrex::MFIter mfi(*MultiF); mfi.isValid(); ++mfi ){

            const amrex::Box& bx = mfi.tilebox();

            amrex::Array4<amrex::Real> const& proj_arr = (*MultiF)[mfi].array();
            //ParallelFor(bx, [=] (int i, int j, int k) {
            amrex::IntVect lo = {bx.smallEnd()};
            amrex::IntVect hi = {bx.bigEnd()};
            for (int k=lo[2];k<=hi[2];++k)
                for (int j=lo[1];j<=hi[1];++j)
                    for (int i=lo[0];i<=hi[0];++i)
                {
                double x_eval, y_eval, z_eval;
                int err;
                te_variable read_vars[7] = {{"x", &x_eval}, {"y", &y_eval}, {"z", &z_eval}, {"t", &t_eval}, {"kvarx", &kx}, {"kvary", &ky}, {"kvarz", &kz}};
                te_expr *function_parse = te_compile(function_string.c_str(), read_vars, varcount, &err);

                amrex::GpuArray<amrex::Real, degree-1> x;
                amrex::Real x_midpoint;
                amrex::GpuArray<amrex::Real, degree-1> y;
                amrex::Real y_midpoint;
                amrex::GpuArray<amrex::Real, degree-1> z;
                amrex::Real z_midpoint;

                z_midpoint = ifr.geom.ProbLo()[2] + ((double)k+0.5-0.5*(IndexVect)[2])*ifr.dx[2];
                for (int d = 0; d<x_len[2]; d++) z[d] = z_midpoint + (d-((x_len[2]+1)/2.0-1))*ifr.dx[0]/pow(2.0,(x_len[2]+1)/2.0-1);

                y_midpoint = ifr.geom.ProbLo()[1] + ((double)j+0.5-0.5*(IndexVect)[1])*ifr.dx[1];
                for (int d = 0; d<x_len[1]; d++) y[d] = y_midpoint + (d-((x_len[1]+1)/2.0-1))*ifr.dx[0]/pow(2.0,(x_len[1]+1)/2.0-1);

                x_midpoint = ifr.geom.ProbLo()[0] + ((double)i+0.5-0.5*(IndexVect)[0])*ifr.dx[0];
                for (int d = 0; d<x_len[0]; d++) x[d] = x_midpoint + (d-((x_len[0]+1)/2.0-1))*ifr.dx[0]/pow(2.0,(x_len[0]+1)/2.0-1);

                amrex::Real val = 0.0;
                for (int zd = 0; zd<x_len[2];zd++)
                {
                    for (int yd =0; yd<x_len[1]; yd++)
                    {
                        for (int xd =0; xd<x_len[0]; xd++) {
                            x_eval = x[xd];
                            y_eval = y[yd];
                            z_eval = z[zd];
                            val+= (integrate[0]?stencil[xd]:1)*(integrate[1]?stencil[yd]:1)*(integrate[2]?stencil[zd]:1)*(te_eval(function_parse));
                        }
                    }
                }
                proj_arr(i,j,k) = val;
                te_free(function_parse);
            }//);
        }
        (*MultiF).FillBoundary(ifr.geom.periodicity());
    }


    template<int degree,
             funcToProjType function_to_project>
    void projection (amrex::Real t,
                     computational_domain ifr,
                     amrex::GpuArray<bool, GEMPIC_SPACEDIM> integrate,
                     amrex::IndexType IndexVect,
                     amrex::MultiFab *MultiF,
                     int funcSelect
                     ) {

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

        std::cout <<"Function: " << funcSelect << " ->  " <<  function_to_project(0, 0, 0, 0,funcSelect) << "\n";
        int x_len[GEMPIC_SPACEDIM];
        for (int i=0;i<GEMPIC_SPACEDIM;i++) x_len[i] = (integrate[i]?(degree-1):1);
        for ( amrex::MFIter mfi(*MultiF); mfi.isValid(); ++mfi ){
            const amrex::Box& bx = mfi.tilebox();
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> plo = ifr.plo;
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> dx = ifr.dx;
            amrex::Array4<amrex::Real> const& proj_arr = (*MultiF)[mfi].array();
            //ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            amrex::IntVect lo = {bx.smallEnd()};
            amrex::IntVect hi = {bx.bigEnd()};
            for (int k=lo[2];k<=hi[2];++k)
                for (int j=lo[1];j<=hi[1];++j)
                    for (int i=lo[0];i<=hi[0];++i)
            {
                double x_eval, y_eval, z_eval;

                amrex::GpuArray<amrex::Real, degree-1> x;
                amrex::Real x_midpoint;
                amrex::GpuArray<amrex::Real, degree-1> y;
                amrex::Real y_midpoint;
                amrex::GpuArray<amrex::Real, degree-1> z;
                amrex::Real z_midpoint;

                z_midpoint = plo[2] + ((double)k+0.5-0.5*(IndexVect)[2])*dx[2];
                for (int d = 0; d<x_len[2]; d++) z[d] = z_midpoint + (d-((x_len[2]+1)/2.0-1))*dx[0]/pow(2.0,(x_len[2]+1)/2.0-1);

                y_midpoint = plo[1] + ((double)j+0.5-0.5*(IndexVect)[1])*ifr.dx[1];
                for (int d = 0; d<x_len[1]; d++) y[d] = y_midpoint + (d-((x_len[1]+1)/2.0-1))*dx[0]/pow(2.0,(x_len[1]+1)/2.0-1);

                x_midpoint = plo[0] + ((double)i+0.5-0.5*(IndexVect)[0])*dx[0];
                for (int d = 0; d<x_len[0]; d++) x[d] = x_midpoint + (d-((x_len[0]+1)/2.0-1))*dx[0]/pow(2.0,(x_len[0]+1)/2.0-1);

                amrex::Real val = 0.0;
                for (int zd = 0; zd<x_len[2];zd++)
                {
                    for (int yd =0; yd<x_len[1]; yd++)
                    {
                        for (int xd =0; xd<x_len[0]; xd++) {
                            x_eval = x[xd];
                            y_eval = y[yd];
                            z_eval = z[zd];
                            val+= (integrate[0]?stencil[xd]:1)*
                                  (integrate[1]?stencil[yd]:1)*
                                  (integrate[2]?stencil[zd]:1)*
                                  (function_to_project(x_eval, y_eval, z_eval,  t, funcSelect));
                        }
                    }
                }
                proj_arr(i,j,k) = val;
            }//);
	}
        (*MultiF).FillBoundary(ifr.geom.periodicity());
    }


    /**
     * Initialize E and B field for time t=0 with a given array of functions.
     */
    template <int degree>
    void init_E_B (amrex::GpuArray<std::string, vdim> fieldsE,
                   amrex::GpuArray<std::string, int(vdim/2.5)*2+1> fieldsB,
                   amrex::GpuArray<amrex::Real, 3> k,
                   computational_domain ifr){
        initE<degree> (fieldsE, k, ifr);
        initB<degree> (fieldsB, k, ifr);
    }


    /**
     * Initialize E field for time t=0 with a given array of functions.
     */
    template <int degree,
             funcToProjType function_to_project
             >
    void initE ( computational_domain ifr , int funcSelect[3] ){

        for(int i=0;i<3;i++)
          projection<degree,function_to_project>( 0.0, ifr, {AMREX_D_DECL(false, true, true)}, *E_Index[0], &(*E_Array[0]), funcSelect[i]);
    };

    /**
     * Initialize E field for time t=0 with a given array of functions.
     */
    template <int degree>
    void initE (amrex::GpuArray<std::string, vdim> fields, amrex::GpuArray<amrex::Real, 3> k,computational_domain ifr) {
        // E_field
        for (int i = 0; i<vdim; i++) {
            projection<degree>(fields[i], k, 0.0, ifr, {AMREX_D_DECL(i!=0, i!=1, i!=2)}, *E_Index[i], &(*E_Array[i]));
        }
    };

    /**
     * Initialize B field for time t=0 with a given array of functions.
     */
    template <int degree, 
             funcToProjType function_to_project
             >
    void initB ( computational_domain ifr , int funcSelect[3]){

        double half_dt = dt/2.0; // for leap frog initialization

        // B_field
        for(int i=0;i<3;i++)
          projection<degree,function_to_project>( half_dt, ifr, {AMREX_D_DECL(false, true, true)}, *B_Index[0], &(*B_Array[0]), funcSelect[i]);
    }

    /**
     * Initialize B field for time t=0 with a given array of functions.
     */
    template <int degree>
    void initB (amrex::GpuArray<std::string, int(vdim/2.5)*2+1> fields, amrex::GpuArray<amrex::Real, 3> k,computational_domain ifr){

        double half_dt = dt/2.0; // for leap frog initialization

        // B_field
        for (int i = 0; i<int(vdim/2.5)*2+1; i++) {
            projection<degree>(fields[i], k, half_dt, ifr, {AMREX_D_DECL(i!=0, i!=1, i!=2)}, *B_Index[i], &(*B_Array[i]));
        }
    }

    /**
     * Fill Boundaries of all relevant Fabs (E, J, B, rho).
     */
    void FillBD (computational_domain ifr){
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
    void advance_time (amrex::Real coeff = 1.0) {
        time = time+dt*coeff;
    }


    /**
     * Advance E by a time step of length @p dt_step with a Yee scheme
     */
    void advance_E (computational_domain ifr, amrex::Real dt_step, bool from_B, bool from_J,
                    amrex::GpuArray<std::unique_ptr<amrex::MultiFab>, int(vdim/2.5)*2+1 > *MultiInput,
                    amrex::GpuArray<std::unique_ptr<amrex::MultiFab>, vdim > *MultiOutput){

        int bdim = int(vdim/2.5)*2+1;

        // make copies for GPU
        amrex::Real dt_step_cp = dt_step;
        amrex::Real curlB_scaling_cp = curlB_scaling;
        amrex::Real epref_ep0_cp = epref_ep0;

        for (int d=0; d<vdim; d++) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(*(*MultiOutput)[d]); mfi.isValid(); ++mfi ) {
                const amrex::Box& bx = mfi.tilebox();

                amrex::Array4<amrex::Real> const& vecIn_1 = (*(*MultiInput)[(d+2)%vdim])[mfi].array();
                amrex::Array4<amrex::Real> const& vecIn_2 = (*(*MultiInput)[(d+1)%vdim])[mfi].array();
                amrex::Array4<amrex::Real> const& vecJ = (*J_Array[d])[mfi].array();
                amrex::Array4<amrex::Real> const& vecOut = (*(*MultiOutput)[d])[mfi].array();
                amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> dx = ifr.dx;
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    if((vdim < 3) & (GEMPIC_SPACEDIM == 1)){
                        if (d == 0){
                            vecOut(i,j,k) = vecOut(i,j,k) + dt_step*epref_ep0*(from_J?(-vecJ(i,j,k)):0.0);
                        }
                    }
                    if ((vdim == 2) & (GEMPIC_SPACEDIM == 2)){
                        //Ex for 2D2V
                        if (d == 0) {
                            vecOut(i,j,k) = vecOut(i,j,k) + dt_step*(curlB_scaling*(from_B?(vecIn_1(i,j,k)-vecIn_1(i,j-1,k))/dx[1]:0.0) + epref_ep0*(from_J?(-vecJ(i,j,k)):0.0));
                        }
                    }
                    if (vdim == 2){
                        if (d == 1) {
                            vecOut(i,j,k) = vecOut(i,j,k) + dt_step*(curlB_scaling*(from_B?(-(vecIn_2(i,j,k)-vecIn_2(i-1,j,k)))/dx[0]:0.0) + epref_ep0*(from_J?(-vecJ(i,j,k)):0.0));
                        }
                    }
                    if ((vdim == 3) & (GEMPIC_SPACEDIM == 2)) {
                        if (d==2) {
                            vecOut(i,j,k) = vecOut(i,j,k) + dt_step*(curlB_scaling*(from_B?(vecIn_2(i,j,k)-vecIn_2(i-(d%vdim==2?1:0),j-(d%vdim==0?1:0),k))/dx[(d+1)%vdim]
                                                                                    -(vecIn_2(i,j,k)-vecIn_2(i-(d%vdim==1?1:0),j-(d%vdim==2?1:0),k))/dx[(d+2)%vdim]:0.0)
                                    + epref_ep0*(from_J?(-vecJ(i,j,k)):0.0));
                        } else {
                            vecOut(i,j,k) = vecOut(i,j,k) + dt_step*(curlB_scaling*(from_B?((d==0?1:-1)*(vecIn_1(i,j,k)-vecIn_1(i-(d%vdim==1?1:0),j-(d%vdim==0?1:0),k)))/dx[0]:0.0) + epref_ep0*(from_J?(-vecJ(i,j,k)):0.0));
                        }
                    }
                    if ((vdim == 3) & (GEMPIC_SPACEDIM == 3)) {
                        vecOut(i,j,k) = vecOut(i,j,k) + dt_step_cp*(curlB_scaling_cp*(from_B?((vecIn_1(i,j,k)-vecIn_1(i-(d%bdim==2?1:0),j-(d%bdim==0?1:0),k-(d%bdim==1?1:0)))/dx[(d+1)%vdim]
                                                                                       -(vecIn_2(i,j,k)-vecIn_2(i-(d%bdim==1?1:0),j-(d%bdim==2?1:0),k-(d%bdim==0?1:0)))/dx[(d+2)%vdim]):0.0)
                                + epref_ep0_cp*(from_J?(-vecJ(i,j,k)):0.0));
                    }
                });
            }
            (*(*MultiOutput)[d]).FillBoundary(ifr.geom.periodicity());
        }
    }

    /**
     * Advance E by a time step of length @p dt_step with a Yee scheme
     */
    void advance_E (computational_domain ifr, amrex::Real dt_step, bool from_B, bool from_J,
                    amrex::GpuArray<std::unique_ptr<amrex::MultiFab>, int(vdim/2.5)*2+1 > *MultiInput,
                    amrex::GpuArray<std::unique_ptr<amrex::MultiFab>, vdim > *MultiAlfvenTensor,
                    amrex::GpuArray<std::unique_ptr<amrex::MultiFab>, vdim > *MultiOutput){

        int bdim = int(vdim/2.5)*2+1;

        for (int d=0; d<vdim; d++) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(*(*MultiOutput)[d]); mfi.isValid(); ++mfi ) {
                const amrex::Box& bx = mfi.tilebox();

                amrex::Array4<amrex::Real> const& vecIn_1 = (*(*MultiInput)[(d+2)%vdim])[mfi].array();
                amrex::Array4<amrex::Real> const& vecIn_2 = (*(*MultiInput)[(d+1)%vdim])[mfi].array();
                amrex::Array4<amrex::Real> const& vecJ = (*J_Array[d])[mfi].array();
                amrex::Array4<amrex::Real> const& vecAT = (*(*MultiAlfvenTensor)[d])[mfi].array();
                amrex::Array4<amrex::Real> const& vecOut = (*(*MultiOutput)[d])[mfi].array();
                amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> dx = ifr.dx;
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    if((vdim < 3) & (GEMPIC_SPACEDIM == 1)){
                        if (d == 0){
                            vecOut(i,j,k) = vecOut(i,j,k) + dt_step*(from_J?(-vecJ(i,j,k)):0.0);
                        }
                    }
                    if ((vdim == 2) & (GEMPIC_SPACEDIM == 2)){
                        //Ex for 2D2V
                        if (d == 0) {
                            vecOut(i,j,k) = vecOut(i,j,k) + dt_step*((from_B?(vecIn_1(i,j,k)-vecIn_1(i,j-1,k))/dx[1]:0.0) + (from_J?(-vecJ(i,j,k)):0.0));
                        }
                    }
                    if (vdim == 2){
                        if (d == 1) {
                            vecOut(i,j,k) = vecOut(i,j,k) + dt_step*((from_B?(-(vecIn_2(i,j,k)-vecIn_2(i-1,j,k)))/dx[0]:0.0) + (from_J?(-vecJ(i,j,k)):0.0));
                        }
                    }
                    if ((vdim == 3) & (GEMPIC_SPACEDIM == 2)) {
                        if (d==2) {
                            vecOut(i,j,k) = vecOut(i,j,k) + dt_step*((from_B?(vecIn_2(i,j,k)-vecIn_2(i-(d%vdim==2?1:0),j-(d%vdim==0?1:0),k))/dx[(d+1)%vdim]
                                                                      -(vecIn_2(i,j,k)-vecIn_2(i-(d%vdim==1?1:0),j-(d%vdim==2?1:0),k))/dx[(d+2)%vdim]:0.0)
                                    + (from_J?(-vecJ(i,j,k)):0.0));
                        } else {
                            vecOut(i,j,k) = vecOut(i,j,k) + dt_step*((from_B?((d==0?1:-1)*(vecIn_1(i,j,k)-vecIn_1(i-(d%vdim==1?1:0),j-(d%vdim==0?1:0),k)))/dx[0]:0.0) + (from_J?(-vecJ(i,j,k)):0.0));
                        }
                    }
                    if ((vdim == 3) & (GEMPIC_SPACEDIM == 3)) {
                        vecOut(i,j,k) = vecOut(i,j,k) + dt_step*(vecAT(i,j,k)*(from_B?((vecIn_1(i,j,k)-vecIn_1(i-(d%bdim==2?1:0),j-(d%bdim==0?1:0),k-(d%bdim==1?1:0)))/dx[(d+1)%vdim]
                                                                                      -(vecIn_2(i,j,k)-vecIn_2(i-(d%bdim==1?1:0),j-(d%bdim==2?1:0),k-(d%bdim==0?1:0)))/dx[(d+2)%vdim]):0.0)
                                + (from_J?(-vecJ(i,j,k)):0.0));
                    }
                });
            }
            (*(*MultiOutput)[d]).FillBoundary(ifr.geom.periodicity());
        }
    }


    /**
     * Advance B by a time step of length @p dt_step with a Yee scheme
     */
    void advance_B (computational_domain ifr, amrex::Real dt_step,
                    amrex::GpuArray<std::unique_ptr<amrex::MultiFab>, vdim > *MultiInput,
                    amrex::GpuArray<std::unique_ptr<amrex::MultiFab>, int(vdim/2.5)*2+1 > *MultiOutput){

        int bdim = int(vdim/2.5)*2+1;
        amrex::Real dt_step_cp = dt_step;

        // B-field
        for (int d=0; d<bdim; d++) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(*(*MultiOutput)[d]); mfi.isValid(); ++mfi ) {
                const amrex::Box& bx = mfi.tilebox();

                amrex::Array4<amrex::Real> const& vecIn_1 = (*(*MultiInput)[(d+2)%bdim])[mfi].array();
                amrex::Array4<amrex::Real> const& vecIn_2 = (*(*MultiInput)[(d+1)%bdim])[mfi].array();
                amrex::Array4<amrex::Real> const& vecOut = (*(*MultiOutput)[d])[mfi].array();
                amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> dx = ifr.dx;
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    if (vdim == 1){}
                    //no B
                    else if ((vdim == 2) & (GEMPIC_SPACEDIM == 1)){
                        vecOut(i,j,k) = vecOut(i,j,k) - dt_step*((vecIn_1(i+1,j,k)-vecIn_1(i,j,k))/dx[0]);
                    }
                    else if ((vdim == 2) & (GEMPIC_SPACEDIM == 2)){
                        vecOut(i,j,k) = vecOut(i,j,k) - dt_step*((vecIn_1(i,j+1,k)-vecIn_1(i,j,k))/dx[0] - (vecIn_2(i+1,j,k)-vecIn_2(i,j,k))/dx[0]);
                    }
                    else if ((vdim == 3) & (GEMPIC_SPACEDIM == 2)) {
                        if (i==2) {
                            vecOut(i,j,k) = vecOut(i,j,k) - dt_step*((vecIn_1(i+(d%bdim==2?1:0),j+(d%bdim==0?1:0),k)-vecIn_1(i,j,k))/dx[(d+1)%bdim] -
                                    (vecIn_2(i+(d%bdim==1?1:0),j+(d%bdim==2?1:0),k)-vecIn_2(i,j,k))/dx[(d+2)%bdim]);
                        } else {
                            vecOut(i,j,k) = vecOut(i,j,k) - dt_step*(-(d==0?-1:1)*(vecIn_1(i+(d%bdim==1?1:0),j+(d%bdim==0?1:0),k)-vecIn_1(i,j,k))/dx[(d+1)%GEMPIC_SPACEDIM]);
                        }
                    }
                    else if (bdim == 3) {
                        vecOut(i,j,k) = vecOut(i,j,k) - dt_step_cp*((vecIn_1(i+(d%bdim==2?1:0),j+(d%bdim==0?1:0),k+(d%bdim==1?1:0))-vecIn_1(i,j,k))/dx[(d+1)%bdim] -
                                (vecIn_2(i+(d%bdim==1?1:0),j+(d%bdim==2?1:0),k+(d%bdim==0?1:0))-vecIn_2(i,j,k))/dx[(d+2)%bdim]);
                    }
                });
            }
            (*(*MultiOutput)[d]).FillBoundary(ifr.geom.periodicity());
        }
    }

#if (GEMPIC_SPACEDIM ==3)
    template<int degree>
    void hodge_full (computational_domain ifr, amrex::GpuArray<std::unique_ptr<amrex::MultiFab>, vdim > *MultiInput,
                     amrex::GpuArray<std::unique_ptr<amrex::MultiFab>, vdim > *MultiOutput, bool is_E) {
        for (int field=0; field<vdim; field++) {

            amrex::MultiFab tmp(convert(ifr.grid, (is_E?*E_Index[field]:*B_Index[field])),ifr.distriMap,Ncomp,Nghost);

            hodge<degree,0>(ifr, (field==0?0:1), &(*(*MultiInput)[field]), &(*(*MultiOutput)[field]), (is_E?*E_Index[field]:*B_Index[field]));
#if (GEMPIC_SPACEDIM > 1)
            hodge<degree,1>(ifr, (field==1?0:1), &(*(*MultiOutput)[field]), &tmp, (is_E?*E_Index[field]:*B_Index[field]));
#endif
#if (GEMPIC_SPACEDIM > 2)
            hodge<degree,2>(ifr, (field==2?0:1), &tmp, &(*(*MultiOutput)[field]), (is_E?*E_Index[field]:*B_Index[field]));
#endif
        }
    }
#endif

    template <int degree, int dimension>
    void hodge (computational_domain ifr, int start_from, amrex::MultiFab *MultiInput, amrex::MultiFab *MultiOutput,
                amrex::IndexType IndexVect){

        amrex::GpuArray<amrex::Real, degree-1> stencil;
        switch (start_from) {
        case 0:
            switch (degree) {
            case 2:
                stencil[0] = 1.0;
                break;
            case 4:
                stencil[0] = 1.0/8.0;
                stencil[1] = 6.0/8.0;
                stencil[2] = 1.0/8.0;
                break;
            case 6:
                stencil[0] = -7.0/384.0;
                stencil[1] = 44.0/384.0;
                stencil[2] = 310.0/384.0;
                stencil[3] = 44.0/384.0;
                stencil[4] = -7.0/384.0;
                break;
            default: amrex::Print() << "degree not implemented" << std::endl;
            }
            break;
        case 1:
            switch (degree) {
            case 2:
                stencil[0] = 1.0;
                break;
            case 4:
                stencil[0] = -1.0/24.0;
               stencil[1] = 13/12.0;
              stencil[2] =  -1.0/24.0;
                break;
            case 6:
                stencil[0] = 3.0/640.0;
               stencil[1] = -29.0/480.0;
              stencil[2] =  1067.0/960.0;
             stencil[3] = -29.0/480.0;
            stencil[4] =  3.0/640.0;
                break;
            default: amrex::Print() << "degree not implemented" << std::endl;
            }
            break;
        }

        for (amrex::MFIter mfi(*MultiOutput); mfi.isValid(); ++mfi ) {
            const amrex::Box& bx = mfi.validbox();

            amrex::Array4<amrex::Real> const& vecIn = (*MultiInput)[mfi].array();
            amrex::Array4<amrex::Real> const& vecOut = (*MultiOutput)[mfi].array();
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {

                amrex::Real val = 0.0;
                for (int d = 0; d<(degree-1); d++) {
                    int shift = d - degree/2 + 1;
                    val += stencil[d]*(vecIn(i+(dimension==0?shift:0),j+(dimension==1?shift:0),k+(dimension==2?shift:0)));
                }
                vecOut(i,j,k) = val;

            });

            // next mfi
        }
        (*MultiOutput).FillBoundary(ifr.geom.periodicity());
    }


    /**
     * Advance time and both fields by a time step of length @p dt_step with a Yee scheme
     */
    void advance (computational_domain ifr, amrex::Real dt_step){

        time = time+dt_step;

        advance_E(ifr, dt_step, true, true, &HB_Array, &E_Array);
        advance_B(ifr, dt_step, true, true, &HE_Array, &B_Array);
    }

    template <int degree>
    void init_rho_phi (amrex::GpuArray<std::string, 2> fields, amrex::GpuArray<amrex::Real, 3> k,computational_domain ifr){

        // rho
        projection<degree>(fields[0], k, 0.0, ifr, {AMREX_D_DECL(true, true, true)}, amrex::IndexType(amrex::IntVect::TheNodeVector()), &(rho));
        // phi
        projection<degree>(fields[1], k, 0.0, ifr, {AMREX_D_DECL(true, true, true)}, amrex::IndexType(amrex::IntVect::TheNodeVector()), &(phi));
    }

    template <int degree>
    void init_rho_phi (amrex::Real (*function_to_project_rho)(amrex::Real, amrex::Real, amrex::Real, amrex::Real),
                       amrex::Real (*function_to_project_phi)(amrex::Real, amrex::Real, amrex::Real, amrex::Real),
                       computational_domain ifr){

        // rho
        projection<degree>(function_to_project_rho, 0.0, ifr, {AMREX_D_DECL(true, true, true)}, amrex::IndexType(amrex::IntVect::TheNodeVector()), &(rho));
        // phi
        projection<degree>(function_to_project_phi, 0.0, ifr, {AMREX_D_DECL(true, true, true)}, amrex::IndexType(amrex::IntVect::TheNodeVector()), &(phi));
    }

    template <int stencil_length>
    void matrix_mult (computational_domain ifr, amrex::GpuArray<amrex::GpuArray<amrex::Real, stencil_length>, GEMPIC_SPACEDIM> stencil, amrex::MultiFab *MultiInput, amrex::MultiFab *MultiOutput,
                      amrex::IndexType IndexVect){

        for (amrex::MFIter mfi(*MultiOutput); mfi.isValid(); ++mfi ) {
            const amrex::Box& bx = mfi.validbox();

            amrex::Array4<amrex::Real> const& vecIn = (*MultiInput)[mfi].array();
            amrex::Array4<amrex::Real> const& vecOut = (*MultiOutput)[mfi].array();

            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::Real val = 0.0;
                for (int dimension=0; dimension < GEMPIC_SPACEDIM; dimension++) {
                    for (int d = 0; d<(stencil_length); d++) {
                        int shift = d - (stencil_length+1)/2 + 1;
                        val += stencil[dimension][d]*vecIn(i+(dimension==0?shift:0),j+(dimension==1?shift:0),k+(dimension==2?shift:0));
                    }
                }
                vecOut(i,j,k) = val;
            });
        }
        (*MultiOutput).FillBoundary(ifr.geom.periodicity());
    }

    template <int stencil_length, int dimension>
    void solve_stencil_CG(amrex::GpuArray<amrex::GpuArray<amrex::Real, stencil_length>, GEMPIC_SPACEDIM> stencil,
                          amrex::MultiFab *b, amrex::MultiFab *x, computational_domain ifr, amrex::IndexType IndexVect,
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
    void solve_hodge_CG(amrex::MultiFab *b, amrex::MultiFab *x, amrex::MultiFab *k, computational_domain ifr, int dimension, int verbose = 0, amrex::Real reltol = 1.e-11,
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
                           amrex::MultiFab *ky, amrex::MultiFab *kz, computational_domain ifr, int verbose = 0,
                           amrex::Real reltol = 1.e-11, int maxsteps = 100) {

        int dimension = 0;

        const int stencil_length = 3;
        amrex::GpuArray<amrex::GpuArray<amrex::Real, stencil_length>, GEMPIC_SPACEDIM> stencil_x;
        stencil_x[0] = {-1.0/ifr.dx[0], 1.0/ifr.dx[0], 0.0};
        stencil_x[1] = {0.0, 0.0, 0.0};
        stencil_x[2] = {0.0, 0.0, 0.0};
        amrex::GpuArray<amrex::GpuArray<amrex::Real, stencil_length>, GEMPIC_SPACEDIM> stencil_y;
        stencil_y[0] = {0.0, 0.0, 0.0};
        stencil_y[1] = {-1.0/ifr.dx[1], 1.0/ifr.dx[1], 0.0};
        stencil_y[2] = {0.0, 0.0, 0.0};
        amrex::GpuArray<amrex::GpuArray<amrex::Real, stencil_length>, GEMPIC_SPACEDIM> stencil_z;
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
                          computational_domain ifr, int verbose = 0, amrex::Real reltol = 1.e-11,
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

        (*x).mult(epref_ep0, 0, 1, Nghost);
        (*x).FillBoundary(ifr.geom.periodicity());
    }

    template<int degree>
    void D_from_phi(amrex::MultiFab *phi_mf, amrex::GpuArray<std::unique_ptr<amrex::MultiFab>, vdim > *D,
                    amrex::MultiFab *kx, amrex::MultiFab *ky, amrex::MultiFab *kz,computational_domain ifr, int verbose = 0,
                    amrex::Real reltol = 1.e-11, int maxsteps = 100) {

        amrex::GpuArray<std::unique_ptr<amrex::MultiFab>, vdim > tmp;
        for (int i=0; i<vdim;i++) {
            tmp[i].reset(new amrex::MultiFab (convert(ifr.grid, *E_Index[i]),ifr.distriMap,Ncomp,Nghost));
        }
        // ---------------------------------------------------------------------
        // D-operator

        for (int comp=0; comp<vdim; comp++) {
            for (amrex::MFIter mfi(*tmp[comp]); mfi.isValid(); ++mfi ) {
                const amrex::Box& bx = mfi.validbox();

                amrex::Array4<amrex::Real> const& phi_arr = (*phi_mf)[mfi].array();
                amrex::Array4<amrex::Real> const& tmp_arr = (*tmp[comp])[mfi].array();
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    tmp_arr(i,j,k) = -(phi_arr(i+(comp==0?1:0),j+(comp==1?1:0),k+(comp==2?1:0))-phi_arr(i,j,k))/ifr.dx[comp];
                });

            }
            (*tmp[comp]).FillBoundary(ifr.geom.periodicity());
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
    void solve_poisson(computational_domain ifr, amrex::Real reltol = 1.e-11, int max_coarsening_level = 30){
        //--------------------------------------------------------------------------------
        // rho
#if GEMPIC_SPACEDIM > 1 // for == 1: use analytical solution instead of MLMG solver

        int verbose = 0;
        int bottom_verbose = 0;
        int max_iter = 1000;
        int max_fmg_iter = 0; //TODO
        int mg_bottom_maxiter = 10000;

        amrex::LPInfo lp_info;
        lp_info.setMaxCoarseningLevel(max_coarsening_level);
        amrex::MLNodeLaplacian linop({ifr.geom}, {ifr.grid}, {ifr.distriMap}, lp_info); // linear operator class

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
        mlmg.setBottomMaxIter(mg_bottom_maxiter);
        mlmg.setVerbose(verbose);
        mlmg.setBottomVerbose(bottom_verbose);
        mlmg.setBottomSolver(amrex::BottomSolver::cg);

        phi.setVal(0.0); // initialise phi
        mlmg.solve({&phi}, {&rho}, reltol, 0.0);
        phi.FillBoundary(ifr.geom.periodicity());

#endif

        phi.mult(epref_ep0, 0, 1, Nghost);
        //----------------------------------------------------------------------------------
        // E


        for (int comp=0; comp<vdim; comp++) {
            for (amrex::MFIter mfi(*E_Array[comp]); mfi.isValid(); ++mfi ) {
                const amrex::Box& bx = mfi.validbox();

                amrex::Array4<amrex::Real> const& phi_arr = phi[mfi].array();
                amrex::Array4<amrex::Real> const& E_arr = (*E_Array[comp])[mfi].array();
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    E_arr(i,j,k) = -(phi_arr(i+(comp==0?1:0),j+(comp==1?1:0),k+(comp==2?1:0))-phi_arr(i,j,k))/ifr.dx[comp];
                });

            }
            (*E_Array[comp]).FillBoundary(ifr.geom.periodicity());
        }
    }

    /**
     * Compute rho for a given rho (for diagnostics: Gauss law)
     */
    void rho_from_E(computational_domain ifr){

        for (amrex::MFIter mfi(rho_gauss_law); mfi.isValid(); ++mfi ) {
            const amrex::Box& bx = mfi.validbox();

            amrex::Array4<amrex::Real> const& rho_arr = rho_gauss_law[mfi].array();
            amrex::GpuArray<amrex::Array4<amrex::Real>, 3> const& E_arr = {(*E_Array[0])[mfi].array(), (*E_Array[1])[mfi].array(), (*E_Array[2])[mfi].array()};
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                rho_arr(i,j,k) = (E_arr[0](i,j,k)-E_arr[0](i-1,j,k))/ifr.dx[0];
                rho_arr(i,j,k) += (E_arr[1](i,j,k)-E_arr[1](i,j-1,k))/ifr.dx[1];
                rho_arr(i,j,k) += (E_arr[2](i,j,k)-E_arr[2](i,j,k-1))/ifr.dx[2];

            });

        }
        (rho_gauss_law).FillBoundary(ifr.geom.periodicity());

    }

    /**
     * Compute rho for a given rho (for diagnostics: Gauss law)
     */
    template<int numspec, int degx, int degy, int degz>
    amrex::Real compute_gauss_error(computational_domain ifr, Particles::particle_groups<vdim, numspec> *part_gr){

        rho_from_E(ifr); // fills rho_gauss_law

        // Deposit charges in rho:
        rho.setVal(0.0);
        rho.FillBoundary(ifr.geom.periodicity());
        for (int spec=0;spec<numspec;spec++) {
            (*(*part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in
            for (amrex::ParIter<vdim+1,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {

                auto& particles = pti.GetArrayOfStructs();
                const long np  = pti.numParticles();

                amrex::Array4<amrex::Real> const& rhoarr = rho[pti].array();

                for (int pp=0;pp<np;pp++) {
                    splines_at_particles<degx, degy, degz> spline;
                    amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> position;
                    for (unsigned int d=0;d<GEMPIC_SPACEDIM;++d)
                        position[d] = particles[pp].pos(d);
                    spline.init_particles( position, ifr.plo, ifr.dxi);
                    gempic_deposit_rho_C3<degx, degy, degz> ( spline,
                                                              (*part_gr).charge[spec]*ifr.dxi[GEMPIC_SPACEDIM]*particles[pp].rdata(vdim),
                                                              rhoarr);
                }
            }
        }

        rho.SumBoundary(0, 1, {Nghost, Nghost, Nghost}, {0, 0, 0}, ifr.geom.periodicity());
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
        amrex::Real rho_norm = gempic_norm(&(rho_gauss_law), ifr, 2);
        return(rho_norm*rho_norm);
    }

    /**
     * Compute E for rho from particles (for diagnostics: Gauss law)
     */
    template<int numspec>
    amrex::Real compute_gauss_error_E(computational_domain ifr, Particles::particle_groups<vdim, numspec> *part_gr){

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
        for (int spec=0;spec<numspec;spec++) {
            (*(*part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in
            for (amrex::ParIter<vdim+1,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {

                auto& particles = pti.GetArrayOfStructs();
                const long np  = pti.numParticles();

                amrex::Array4<amrex::Real> const& rhoarr = rho[pti].array();
                for (int pp=0;pp<np;pp++) {
                    Particles::gempic_deposit_rho<amrex::Particle<vdim+1>,vdim>(particles[pp], (*part_gr).charge[spec], rhoarr, ifr.plo, ifr.dxi);
                }
            }
        }

        rho.SumBoundary(0, 1, {Nghost, Nghost, Nghost}, {0, 0, 0}, ifr.geom.periodicity());
        rho.OverrideSync(*nodal_Mask, ifr.geom.periodicity());
        rho.FillBoundary(ifr.geom.periodicity());

        solve_poisson(ifr);

        (*E_Array[0]).minus(*E_sol_Array[0], 0, 1, 0);
        (*E_Array[1]).minus(*E_sol_Array[1], 0, 1, 0);
        (*E_Array[2]).minus(*E_sol_Array[2], 0, 1, 0);

        amrex::Real res1 = gempic_norm(&(*E_Array[0]), ifr, 2);
        amrex::Real res2 = gempic_norm(&(*E_Array[1]), ifr, 2);
        amrex::Real res3 = gempic_norm(&(*E_Array[2]), ifr, 2);
        amrex::Real res = res1*res1 + res2*res2 + res3*res3;

        (*E_Array[0]).Copy(*E_Array[0], *E_sol_Array[0],0,0,1,Nghost);
        (*E_Array[1]).Copy(*E_Array[1], *E_sol_Array[1],0,0,1,Nghost);
        (*E_Array[2]).Copy(*E_Array[2], *E_sol_Array[2],0,0,1,Nghost);

        return(res);
    }

    template<int numspec, int degx, int degy, int degz, int degree>
    amrex::Real compute_gauss_error_hodge(computational_domain ifr, Particles::particle_groups<vdim, numspec> *part_gr){

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
        for (int spec=0;spec<numspec;spec++) {
            (*(*part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in
            for (amrex::ParIter<vdim+1,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {

                auto& particles = pti.GetArrayOfStructs();
                const long np  = pti.numParticles();

                amrex::Array4<amrex::Real> const& rhoarr = rho[pti].array();
                for (int pp=0;pp<np;pp++) {
                    splines_at_particles<degx, degy, degz> spline;
                    amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> position;
                    for (unsigned int d=0;d<GEMPIC_SPACEDIM;++d)
                        position[d] = particles[pp].pos(d);
                    spline.init_particles( position, ifr.plo, ifr.dxi);
                    gempic_deposit_rho_C3<degx, degy, degz> ( spline,
                                                              (*part_gr).charge[spec]*ifr.dxi[GEMPIC_SPACEDIM]*particles[pp].rdata(vdim),
                                                              rhoarr);
                }
            }
        }

        rho.SumBoundary(0, 1, {Nghost, Nghost, Nghost}, {0, 0, 0}, ifr.geom.periodicity());
        rho.OverrideSync(*nodal_Mask, ifr.geom.periodicity());
        rho.FillBoundary(ifr.geom.periodicity());

        solve_poisson(ifr, 1e-11);
        if (degree> 2)
        {
            solve_poisson_CG<degree>(&rho, &phi, &kx, &ky, &kz, ifr);
            D_from_phi<degree>(&phi, &(E_sol_Array), &kx, &ky, &kz, ifr);
        }

        (*E_sol_Array[0]).minus(*E_Array[0], 0, 1, 0);
        (*E_sol_Array[1]).minus(*E_Array[1], 0, 1, 0);
        (*E_sol_Array[2]).minus(*E_Array[2], 0, 1, 0);

        amrex::Real res1 = gempic_norm(&(*E_Array[0]), ifr, 2);
        amrex::Real res2 = gempic_norm(&(*E_Array[1]), ifr, 2);
        amrex::Real res3 = gempic_norm(&(*E_Array[2]), ifr, 2);
        amrex::Real res = res1*res1 + res2*res2 + res3*res3;

        return(res);
    }

    /**
     * Compute norm of errors of E and B field from function with exact solution
     */
    template<int degree>
    amrex::GpuArray<amrex::Real,vdim+int(vdim/2.5)*2+1> computeError(amrex::GpuArray<std::string, vdim> fields_E,
                                                                amrex::GpuArray<std::string, int(vdim/2.5)*2+1> fields_B, amrex::GpuArray<amrex::Real, 3> k,
                                                                bool bool_B,
                                                                computational_domain ifr){
        amrex::GpuArray<amrex::Real,vdim+int(vdim/2.5)*2+1> E_B_error;

        exactSolution<degree>(fields_E, fields_B, k, bool_B, ifr);

        for (int i=0; i<vdim; i++){
            (*E_sol_Array[i]).minus(*E_Array[i], 0, 1, 0);
            amrex::Real norm_tmp = gempic_norm(&(*E_sol_Array[i]), ifr, 2);
            E_B_error[i] = norm_tmp*norm_tmp;
        }
        if (bool_B) {
            for (int i=0; i<int(vdim/2.5)*2+1; i++){
                (*B_sol_Array[i]).minus(*B_Array[i], 0, 1, 0);
                amrex::Real norm_tmp = gempic_norm(&(*B_sol_Array[i]), ifr, 2);
                E_B_error[i+vdim] = norm_tmp*norm_tmp;
            }
        }
        return(E_B_error);
    }

    /**
     * Compute norm of errors of E and B field from function with exact solution (without tinyexpr)
     */
    template<int degree>
    amrex::GpuArray<amrex::Real,vdim+int(vdim/2.5)*2+1> computeError(amrex::Real (*function_to_project_E0)(amrex::Real, amrex::Real, amrex::Real, amrex::Real),
                                                                     amrex::Real (*function_to_project_E1)(amrex::Real, amrex::Real, amrex::Real, amrex::Real),
                                                                     amrex::Real (*function_to_project_E2)(amrex::Real, amrex::Real, amrex::Real, amrex::Real),
                                                                     amrex::Real (*function_to_project_B0)(amrex::Real, amrex::Real, amrex::Real, amrex::Real),
                                                                     amrex::Real (*function_to_project_B1)(amrex::Real, amrex::Real, amrex::Real, amrex::Real),
                                                                     amrex::Real (*function_to_project_B2)(amrex::Real, amrex::Real, amrex::Real, amrex::Real),
                                                                     bool bool_B,
                                                                     computational_domain ifr){
        amrex::GpuArray<amrex::Real,vdim+int(vdim/2.5)*2+1> E_B_error;

        exactSolution<degree>(function_to_project_E0,
                              function_to_project_E1,
                              function_to_project_E2,
                              function_to_project_B0,
                              function_to_project_B1,
                              function_to_project_B2,
                              bool_B,
                              ifr);

        for (int i=0; i<vdim; i++){
            (*E_sol_Array[i]).minus(*E_Array[i], 0, 1, 0);
            amrex::Real norm_tmp = gempic_norm(&(*E_sol_Array[i]), ifr, 2);
            E_B_error[i] = norm_tmp*norm_tmp;
        }
        if (bool_B) {
            for (int i=0; i<int(vdim/2.5)*2+1; i++){
                (*B_sol_Array[i]).minus(*B_Array[i], 0, 1, 0);
                amrex::Real norm_tmp = gempic_norm(&(*B_sol_Array[i]), ifr, 2);
                E_B_error[i+vdim] = norm_tmp*norm_tmp;
            }
        }
        return(E_B_error);
    }

private:


    template<int degree>
    void exactSolution(amrex::GpuArray<std::string, vdim> fields_E,
                       amrex::GpuArray<std::string, int(vdim/2.5)*2+1> fields_B, amrex::GpuArray<amrex::Real, 3> k,
                       bool bool_B, computational_domain ifr){
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

    template<int degree>
    void exactSolution(amrex::Real (*function_to_project_E0)(amrex::Real, amrex::Real, amrex::Real, amrex::Real),
                       amrex::Real (*function_to_project_E1)(amrex::Real, amrex::Real, amrex::Real, amrex::Real),
                       amrex::Real (*function_to_project_E2)(amrex::Real, amrex::Real, amrex::Real, amrex::Real),
                       amrex::Real (*function_to_project_B0)(amrex::Real, amrex::Real, amrex::Real, amrex::Real),
                       amrex::Real (*function_to_project_B1)(amrex::Real, amrex::Real, amrex::Real, amrex::Real),
                       amrex::Real (*function_to_project_B2)(amrex::Real, amrex::Real, amrex::Real, amrex::Real),
                       bool bool_B,
                       computational_domain ifr){
        amrex::Real btime = time + dt/2.0;

        // E-field
        projection<degree>(function_to_project_E0, time, ifr, {false, true, true}, *E_Index[0], &(*E_sol_Array[0]));
        projection<degree>(function_to_project_E1, time, ifr, {true, false, true}, *E_Index[1], &(*E_sol_Array[1]));
        projection<degree>(function_to_project_E2, time, ifr, {true, true, false}, *E_Index[2], &(*E_sol_Array[2]));

        if (bool_B){
            // B-field
            projection<degree>(function_to_project_B0, btime, ifr, {false, true, true}, *B_Index[0], &(*B_sol_Array[0]));
            projection<degree>(function_to_project_B1, btime, ifr, {true, false, true}, *B_Index[1], &(*B_sol_Array[1]));
            projection<degree>(function_to_project_B2, btime, ifr, {true, true, false}, *B_Index[2], &(*B_sol_Array[2]));
        }
    }

};
}
}

#endif
