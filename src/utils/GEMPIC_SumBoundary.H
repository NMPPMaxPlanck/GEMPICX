#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_CoordSys.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FPC.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>
#include <AMReX_Utility.H>

#include <GEMPIC_infrastructure.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_particle_groups.H>

using namespace Field_solvers;
using namespace Infra;
using namespace Particles;

#ifndef SUMBOUNDARY_H_
#define SUMBOUNDARY_H_

namespace Gempic {
namespace Utils {

void
Gempic_SumBoundary (maxwell_yee *mw_yee, infrastructure *infra)
{
    for (int i=0; i<GEMPIC_VDIM; i++) {
        (*(*mw_yee).J_Array[i]).SumBoundary(infra->geom.periodicity());
        (*(*mw_yee).J_Array[i]).OverrideSync((*(*mw_yee).E_Masks[i]), infra->geom.periodicity());
    }


}


}

}

#endif
