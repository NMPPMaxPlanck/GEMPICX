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
#include <GEMPIC_infrastructure.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_mesh_coupling.H>

using namespace Diagnostics_Output;
using namespace Field_solvers;
using namespace Infra;
using namespace Particles;

#ifndef LOOP_PREPARATION_H_
#define LOOP_PREPARATION_H_

namespace Gempic {
namespace Time_Loop {

void loop_preparation (infrastructure infra,
                       maxwell_yee *mw_yee,
                       particle_groups *part_gr,
                       diagnostics *diagn,
                       double (*initB[GEMPIC_BDIM]) (std::array<double,GEMPIC_SPACEDIM> x,double k),
                       amrex::Real initk) {

    // initial rho by: deposition of particle charge
    //Set rho to 0
    (*mw_yee).rho.setVal(0.0, 0); // value and component

    // Deposit charges:
    for (int spec=0;spec<(*part_gr).n_species;spec++) {
        (*(*part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in
        for (amrex::ParIter<GEMPIC_VDIM+1,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {
            amrex::Box tilebox;
            amrex::FArrayBox local_rho;

            tilebox = pti.tilebox();
            tilebox.grow((*mw_yee).Nghost);
            const amrex::Box tb = amrex::convert(tilebox, amrex::IntVect::TheUnitVector());

            local_rho.resize(tb,1); // second arg: number of comps
            local_rho.setVal(0.0);

            auto& particles = pti.GetArrayOfStructs();
            const long np  = pti.numParticles();

            amrex::Array4<amrex::Real> const& rhoarr = local_rho.array();
            for (int pp=0;pp<np;pp++) {
                gempic_deposit_rho(particles[pp], (*part_gr).charge[spec], rhoarr, infra.plo, infra.dxi);
            }
            ((*mw_yee).rho)[pti].atomicAdd(local_rho,tb,tb,0,0,1);
        }
    }

    //assign particles to tiles they are in TODO: do I need this?
    for(int spec=0;spec<(*part_gr).n_species;spec++){
        (*(*part_gr).mypc[spec]).Redistribute();
    }

    //add values from ghost cells and periodic boundary
    (*mw_yee).rho.SumBoundary(infra.geom.periodicity());
    (*mw_yee).rho.FillBoundary(infra.geom.periodicity());

    // output rho
    std::ofstream ofs("rho.output", std::ofstream::out);
    for (amrex::MFIter mfi((*mw_yee).rho); mfi.isValid(); ++mfi ) {
        amrex::Print(ofs) << (*mw_yee).rho[mfi] << std::endl;
    }
    ofs.close();

    //initial E by rho
    (*mw_yee).phi_init.setVal(0.0, 0); // TODO
    (*mw_yee).solve_poisson(infra);

    // initial B_field
    (*mw_yee).initB(infra, initB, initk);

    //------------------------------------------------------------------------------
    //approximate e^1/2
    (*mw_yee).advance_E(infra, 0.5*(*mw_yee).dt);
    //------------------------------------------------------------------------------
    //approximate x^1/2
    for (int spec=0;spec<(*part_gr).n_species;spec++) {
        for (amrex::ParIter<GEMPIC_VDIM+1,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {
            auto& particles = pti.GetArrayOfStructs(); // get particles
            const long np  = pti.numParticles();
            for (int pp=0;pp<np;pp++){
                std::array<amrex::Real,GEMPIC_SPACEDIM+GEMPIC_VDIM> newPos = push_particle_pos(particles[pp], 0.5*(*mw_yee).dt, infra);
                std::copy(begin(newPos),end(newPos),&particles.data()[pp*(GEMPIC_SPACEDIM+GEMPIC_VDIM+2)]);
            }
        }
        (*(*part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in
    }

    //------------------------------------------------------------------------------
    //Save "step 0" data
    (*diagn).save_step_data(0, infra, mw_yee, part_gr); //false: slices are not saved

}
}
}


#endif
