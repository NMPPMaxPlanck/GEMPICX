#ifndef LOOP_PREPARATION_H_
#define LOOP_PREPARATION_H_

#include <tinyexpr.h>

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_CoordSys.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>

#include <GEMPIC_diagnostics.H>
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_mesh_coupling.H>
#include <GEMPIC_particle_mesh_coupling_C2.H>
#include <GEMPIC_parameters.H>

using namespace Diagnostics_Output;
using namespace Field_solvers;
using namespace CompDom;
using namespace Particles;

namespace Gempic {
namespace Time_Loop {





template<int vdim, int numspec, int degx, int degy, int degz, int degmw, bool output>
void loop_preparation_init (gempic_parameters<vdim, numspec> VM,
                       computational_domain infra,
                       maxwell_yee<vdim> *mw_yee,
                       particle_groups<vdim, numspec> *part_gr) {

    // initial rho by: deposition of particle charge
    //Set rho to 0
    (*mw_yee).rho.setVal(0.0, 0); // value and component
    (*mw_yee).rho.FillBoundary(infra.geom.periodicity());

    // Deposit charges:
    for (int spec=0;spec<numspec;spec++) {
        (*(*part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in
        for (amrex::ParIter<vdim+1,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {

		amrex::Real charge = (*part_gr).charge[spec];
            auto& particles = pti.GetArrayOfStructs();
            const long np  = pti.numParticles();

            amrex::Array4<amrex::Real> const& rhoarr = (*mw_yee).rho[pti].array();
            amrex::ParallelFor(np,[=] AMREX_GPU_DEVICE (long pp )
			    {                                                     
            //for (int pp=0;pp<np;pp++) {
                //gempic_deposit_rho<amrex::Particle<vdim+1>,vdim,degx,degy,degz>(particles[pp], (*part_gr).charge[spec], rhoarr, infra.plo, infra.dxi);
                splines_at_particles<degx, degy, degz> spline;
                amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> position;
                for (unsigned int d=0;d<GEMPIC_SPACEDIM;++d)
                  position[d] = particles[pp].pos(d);
                spline.init_particles( position, infra.plo, infra.dxi);
                gempic_deposit_rho_C3<degx, degy, degz> ( spline,
                                              charge*infra.dxi[GEMPIC_SPACEDIM]*particles[pp].rdata(vdim),
                                              rhoarr);
              });
        }
    }

    //assign particles to tiles they are in TODO: do I need this?
    for(int spec=0;spec<numspec;spec++){
        (*(*part_gr).mypc[spec]).Redistribute();
    }

    //add values from ghost cells and periodic boundary
    (*mw_yee).rho.SumBoundary(0, 1, {(*mw_yee).Nghost, (*mw_yee).Nghost, (*mw_yee).Nghost}, {0, 0, 0}, infra.geom.periodicity());
    (*mw_yee).rho.OverrideSync(*((*mw_yee).nodal_Mask), infra.geom.periodicity());
    (*mw_yee).rho.FillBoundary(infra.geom.periodicity());

    //initial E by rho
    amrex::MultiFab kx(convert(infra.grid, *mw_yee->E_Index[0]),infra.distriMap,1,mw_yee->Nghost);
    kx.setVal(1.0, 0);
    kx.FillBoundary(infra.geom.periodicity());
    amrex::MultiFab ky(convert(infra.grid, *mw_yee->E_Index[1]),infra.distriMap,1,mw_yee->Nghost);
    ky.setVal(1.0, 0);
    ky.FillBoundary(infra.geom.periodicity());
    amrex::MultiFab kz(convert(infra.grid, *mw_yee->E_Index[2]),infra.distriMap,1,mw_yee->Nghost);
    kz.setVal(1.0, 0);
    kz.FillBoundary(infra.geom.periodicity());

     (*mw_yee).solve_poisson(infra, 1e-11, VM.max_coarsening_level);
    if (degmw > 2)
      {
        (*mw_yee).template solve_poisson_CG<degmw>(&(mw_yee->rho), &(mw_yee->phi), &kx, &ky, &kz, infra, 2, 1e-11, 100);
        (*mw_yee).template D_from_phi<degmw>(&(mw_yee->phi), &(mw_yee->E_Array), &kx, &ky, &kz, infra);
      }

}

    template<int vdim, int numspec, int degx, int degy, int degz, int degmw, bool output>
    void loop_staggering (gempic_parameters<vdim, numspec> VM,
                           computational_domain infra,
                           maxwell_yee<vdim> *mw_yee,
                           particle_groups<vdim, numspec> *part_gr,
                           diagnostics<vdim, numspec, degx, degy, degz,degmw> *diagn,
                           bool time_staggered) {

    if (time_staggered) {
        amrex::Print() << "ATTENTION: method is time staggered" << std::endl;
        //------------------------------------------------------------------------------
        //approximate e^1/2
        mw_yee->template hodge_full<degmw>(infra, &(mw_yee->B_Array), &(mw_yee->HB_Array), false);
        (*mw_yee).advance_E(infra, 0.5*(*mw_yee).dt, true, true, &(mw_yee->HB_Array), &(mw_yee->E_Array));
        //------------------------------------------------------------------------------
        //approximate x^1/2
        for (int spec=0;spec<numspec;spec++) {
            for (amrex::ParIter<vdim+1,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {
                amrex::Particle<vdim+1,0>*  AMREX_RESTRICT particles= &(pti.GetArrayOfStructs()[0]); // get particles

                const long np  = pti.numParticles();
                const amrex::Real dt = (*mw_yee).dt;
                amrex::ParallelFor(np,[=] AMREX_GPU_DEVICE (long pp){

                    amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> pos;
                    for (int comp = 0; comp < GEMPIC_SPACEDIM; comp++) {
                        pos[comp] = particles[pp].pos(comp);
                    }
                    amrex::GpuArray<amrex::Real,vdim> vel;
                    for (int comp = 0; comp < vdim; comp++) {
                        vel[comp] = particles[pp].rdata(comp);
                    }

                    amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM+vdim> newPos = push_particle_pos<vdim>(infra.plo, infra.phi, pos, vel, 0.5*dt);
                    for (int xd=0;xd<GEMPIC_SPACEDIM; xd++)
                        particles[pp].pos(xd) = newPos[xd];
                });
            }
            (*(*part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in
        }

    }

    //------------------------------------------------------------------------------
    //Save "step 0" data
    if (output)
        (*diagn).save_step_data(0, infra, mw_yee, part_gr); //false: slices are not saved
}

template<int vdim, int numspec, int degx, int degy, int degz, int degmw, bool output>
void loop_preparation (gempic_parameters<vdim, numspec> VM,
                       computational_domain infra,
                       maxwell_yee<vdim> *mw_yee,
                       particle_groups<vdim, numspec> *part_gr,
                       diagnostics<vdim, numspec, degx, degy, degz,degmw> *diagn,
                       bool time_staggered,
                       amrex::GpuArray<std::string, int(vdim/2.5)*2+1> fields_B) {

    loop_preparation_init<vdim, numspec,degx,degy,degz,degmw,output>(VM, infra, mw_yee, part_gr);

    // initial B_field
    (*mw_yee).template initB<degmw>(fields_B, VM.k, infra);

    loop_staggering<vdim, numspec,degx,degy,degz,degmw,output>(VM, infra, mw_yee, part_gr, diagn, time_staggered);
}



template<int vdim, int numspec, int degx, int degy, int degz, int degmw, bool output>
void loop_preparation (gempic_parameters<vdim, numspec> VM,
                       computational_domain infra,
                       maxwell_yee<vdim> *mw_yee,
                       particle_groups<vdim, numspec> *part_gr,
                       diagnostics<vdim, numspec, degx, degy, degz,degmw> *diagn,
                       bool time_staggered,
                       amrex::Real (*function_to_project_0)(amrex::Real, amrex::Real, amrex::Real, amrex::Real),
                       amrex::Real (*function_to_project_1)(amrex::Real, amrex::Real, amrex::Real, amrex::Real),
                       amrex::Real (*function_to_project_2)(amrex::Real, amrex::Real, amrex::Real, amrex::Real)) {

    loop_preparation_init<vdim, numspec,degx,degy,degz,degmw,output>(VM, infra, mw_yee, part_gr);

    // initial B_field
    (*mw_yee).template initB<degmw>(function_to_project_0,function_to_project_1, function_to_project_2, infra);

    loop_staggering<vdim, numspec,degx,degy,degz,degmw,output>(VM, infra, mw_yee, part_gr, diagn, time_staggered);
}

template<int vdim, int numspec, int degx, int degy, int degz, int degmw, bool output>
void loop_preparation (gempic_parameters<vdim, numspec> VM,
                       computational_domain infra,
                       maxwell_yee<vdim> *mw_yee,
                       particle_groups<vdim, numspec> *part_gr,
                       diagnostics<vdim, numspec, degx, degy, degz,degmw> *diagn,
                       bool time_staggered,
                       amrex::GpuArray<int,int(vdim/2.5)*2+1> funcSelect
                       ) {

    loop_preparation_init<vdim, numspec,degx,degy,degz,degmw,output>(VM, infra, mw_yee, part_gr);

    // initial B_field
    (*mw_yee).template initB<degmw>(infra, funcSelect);

    loop_staggering<vdim, numspec,degx,degy,degz,degmw,output>(VM, infra, mw_yee, part_gr, diagn, time_staggered);
}

}
}

#endif
