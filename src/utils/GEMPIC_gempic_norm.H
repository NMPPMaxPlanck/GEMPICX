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

#include <GEMPIC_infrastructure.H>

using namespace Gempic;
using namespace Infra;

#ifndef GEMPIC_NORM_H_
#define GEMPIC_NORM_H_

namespace Gempic {
namespace Utils {

amrex::Real gempic_norm(amrex::MultiFab *MultiF, infrastructure ifr, int p){
    //TODOrefine
    amrex::Real result;
    if (p==0) {
        result = (*MultiF).norm0();
    } else if (p==1) {
        result = (*MultiF).norm1(0, ifr.geom.periodicity());
    } else if (p==2) {
        result = pow((*MultiF).norm2(0, ifr.geom.periodicity()),2.);
    }
    if (p > 0) {
        result *= ifr.dx[0]/ifr.Length[0]
#if(GEMPIC_SPACEDIM > 1)
                *ifr.dx[1]/ifr.Length[1]
#endif
#if(GEMPIC_SPACEDIM > 2)
                *ifr.dx[2]/ifr.Length[2]
#endif
                ;
    }
    return(result);
}
}
}

#endif
