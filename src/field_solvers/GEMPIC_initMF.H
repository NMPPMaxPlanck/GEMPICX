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
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>

#ifndef INIT_H_
#define INIT_H_

namespace Gempic {
namespace Field_solvers {

void init_const (amrex::MultiFab& MultiF, amrex::Real constant) {

    for ( amrex::MFIter mfi(MultiF); mfi.isValid(); ++mfi ){

            const amrex::Box& bx = mfi.tilebox();

            amrex::Array4<amrex::Real> const& init_arr = (MultiF)[mfi].array();
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                init_arr(i,j,k) = constant;
            });
        }
    }


}
}

#endif

