#ifndef GEMPIC_NORM_H_
#define GEMPIC_NORM_H_

#include <assert.h>

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

#include "GEMPIC_ComputationalDomain.H"

namespace Gempic::Utils
{
inline amrex::Real gempic_norm (const amrex::MultiFab& multiF,
                                const ComputationalDomain& ifr,
                                int p)
{
    BL_PROFILE("Gempic::Utils::gempic_norm()");
    // TODOrefine
    assert(p <= 2);
    amrex::Real result = 0.0;
    if (p == 0)
    {
        result = multiF.norm0();
    }
    else if (p == 1)
    {
        result = multiF.norm1(0, ifr.m_geom.periodicity());
    }
    else if (p == 2)
    {
        result = multiF.norm2(0, ifr.m_geom.periodicity());
    }

    if (p > 0)
    {
        result *= ((p == 2) ? (1.0 / sqrt(ifr.m_nCell[xDir])) : (1.0 / ifr.m_nCell[xDir]))
#if (AMREX_SPACEDIM > 1)
                  * ((p == 2) ? (1.0 / sqrt(ifr.m_nCell[yDir])) : (1.0 / ifr.m_nCell[yDir]))
#endif
#if (AMREX_SPACEDIM > 2)
                  * ((p == 2) ? (1.0 / sqrt(ifr.m_nCell[zDir])) : (1.0 / ifr.m_nCell[zDir]))
#endif
            ;
    }
    return (result);
}
} // namespace Gempic::Utils

#endif
