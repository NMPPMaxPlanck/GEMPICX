#ifndef GEMPIC_NORM_H_
#define GEMPIC_NORM_H_

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
#include <assert.h>
#include <GEMPIC_computational_domain.H>

using namespace Gempic;
using namespace CompDom;

namespace Gempic
{
namespace Utils
{
amrex::Real gempic_norm(amrex::MultiFab& MultiF, computational_domain ifr, int p)
{
    // TODOrefine
    assert(p <= 2);
    amrex::Real result = 0.0;
    if (p == 0)
    {
        result = MultiF.norm0();
    }
    else if (p == 1)
    {
        result = MultiF.norm1(0, ifr.geom.periodicity());
    }
    else if (p == 2)
    {
        result = MultiF.norm2(0, ifr.geom.periodicity());
    }

    if (p > 0)
    {
        result *= ((p == 2) ? (1.0 / sqrt(ifr.n_cell[0])) : (1.0 / ifr.n_cell[0]))
#if (GEMPIC_SPACEDIM > 1)
                  * ((p == 2) ? (1.0 / sqrt(ifr.n_cell[1])) : (1.0 / ifr.n_cell[1]))
#endif
#if (GEMPIC_SPACEDIM > 2)
                  * ((p == 2) ? (1.0 / sqrt(ifr.n_cell[2])) : (1.0 / ifr.n_cell[2]))
#endif
            ;
    }
    return (result);
}
}  // namespace Utils
}  // namespace Gempic

#endif
