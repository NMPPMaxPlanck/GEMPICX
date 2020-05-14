#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_RealBox.H>
#include <AMReX_REAL.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_Box.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_CoordSys.H>

#include <GEMPIC_particle_groups.H>
#include <GEMPIC_infrastructure.H>
#include <GEMPIC_maxwell_yee.H>

#ifndef PARTICLE_POSITION_H_
#define PARTICLE_POSITION_H_

void save_particle_positions (particle_groups *ppg, std::string ID = "", int species = 0){
    std::ofstream ofss("particle_output/x/P_" + ID + ".output", std::ofstream::out);
    for (amrex::ParIter<GEMPIC_VDIM+1,0,0,0> pti(*(*ppg).mypc[species], 0); pti.isValid(); ++pti) {

        const auto& particles = pti.GetArrayOfStructs();
        const long np = pti.numParticles();
        for (int pp=0;pp<np;pp++) {
            amrex::Print(ofss) << particles[pp].pos(0) << "," <<
                      #if (GEMPIC_SPACEDIM > 1)
                                  particles[pp].pos(1) << "," <<
                      #endif
                      #if (GEMPIC_SPACEDIM > 2)
                                  particles[pp].pos(2) <<
                      #endif
                                  std::endl;
        }
    }
    ofss.close();
}

void save_particle_velocities (particle_groups *ppg, std::string ID = "", int species = 0){
    std::ofstream ofss("particle_output/v/P_" + ID + ".output", std::ofstream::out);
    for (amrex::ParIter<GEMPIC_VDIM+1,0,0,0> pti(*(*ppg).mypc[species], 0); pti.isValid(); ++pti) {

        const auto& particles = pti.GetArrayOfStructs();
        const long np = pti.numParticles();
        for (int pp=0;pp<np;pp++) {
            amrex::Print(ofss) << particles[pp].rdata(0) << "," <<
                      #if (GEMPIC_SPACEDIM > 1)
                                  particles[pp].rdata(1) << "," <<
                      #endif
                      #if (GEMPIC_SPACEDIM > 2)
                                  particles[pp].rdata(2) <<
                      #endif
                                  std::endl;
        }
    }
    ofss.close();
}


#endif
