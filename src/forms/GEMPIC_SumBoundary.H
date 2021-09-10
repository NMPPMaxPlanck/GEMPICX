#include <AMReX.H>
#ifndef SUMBOUNDARY_H_
#define SUMBOUNDARY_H_

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

#include <GEMPIC_computational_domain.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_particle_groups.H>

using namespace Field_solvers;
using namespace CompDom;
using namespace Particles;

namespace Gempic {
namespace Utils {

template<int vdim>
void
Gempic_SumBoundary (maxwell_yee<vdim> *mw_yee, computational_domain *infra)
{
    for (int i=0; i<vdim; i++) {
        (*(*mw_yee).J_Array[i]).SumBoundary(0, 1, {(*mw_yee).Nghost, (*mw_yee).Nghost, (*mw_yee).Nghost}, {0, 0, 0}, infra->geom.periodicity());
        (*(*mw_yee).J_Array[i]).OverrideSync((*(*mw_yee).E_Masks[i]), infra->geom.periodicity());
    }


}


}

}

#endif
