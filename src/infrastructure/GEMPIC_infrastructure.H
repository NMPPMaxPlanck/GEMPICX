#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_CoordSys.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>

#include <GEMPIC_initializer.H>

using namespace Gempic;

using namespace Init;

#ifndef INFRASTRUCTURE_H_
#define INFRASTRUCTURE_H_

namespace Gempic {
namespace Infra {

/**
 * Class containing all the necessary structures (such as the @p domain and the @p distriMap) and information (such as @p dx and @p n_cell).
 */
class infrastructure
{
public:

    amrex::IntVect n_cell; // number of cells
    int max_grid_size; // maximum number of cells in each direction
    int is_periodic[GEMPIC_SPACEDIM];
    amrex::RealBox real_box; // domain like {min_x,min_y,min_z}{max_x,max_y,max_z}

    amrex::RealVect Length; // Length of domain in x,y,z directions
    amrex::BoxArray grid;
    amrex::Geometry geom;
    amrex::Box domain;
    amrex::DistributionMapping distriMap;
    const amrex::Real* dx;
    std::array<amrex::Real,GEMPIC_SPACEDIM> plo;
    std::array<amrex::Real,GEMPIC_SPACEDIM+1> dxi;

    // The default constructor.
    /**
     * Default Constructor.
     */
    infrastructure ();

    // Explicit const
    /**
     * Explicit constructor: all class attributes can be initialized by passing an object of class initializer.
     */
    infrastructure (initializer init){
        n_cell = init.n_cell;
        max_grid_size = init.max_grid_size;

        for(int cc=0;cc<GEMPIC_SPACEDIM;cc++){
            is_periodic[cc] = init.is_periodic[cc];
        }
        real_box = init.real_box;

        for(int cc=0;cc<GEMPIC_SPACEDIM;cc++){
            Length[cc] = (real_box.hi(cc)-real_box.lo(cc));
        }

        amrex::IntVect dom_lo(AMREX_D_DECL(0, 0, 0));
        amrex::IntVect dom_hi(AMREX_D_DECL(n_cell[0]-1, n_cell[1]-1, n_cell[2]-1));

        domain.setSmall(dom_lo);
        domain.setBig(dom_hi);
        grid.define(domain);
        grid.maxSize(max_grid_size);
        geom.define(domain,&real_box,amrex::CoordSys::cartesian,is_periodic);
        dx = geom.CellSize();

        distriMap.define(grid);

        dxi[GEMPIC_SPACEDIM] = 1.;
        for (int cc=0;cc<GEMPIC_SPACEDIM;cc++){
            plo[cc] = geom.ProbLo()[cc];
            dxi[cc] = 1.0/dx[cc];
            dxi[GEMPIC_SPACEDIM] *= dxi[cc];
        }
    }
};
}
}

#endif
