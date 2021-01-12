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

#include <GEMPIC_Config.H>


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

    amrex::Real tolerance_particle; //

    // The default constructor.
    /**
     * Default Constructor.
     */
    infrastructure (){};

};
}
}

#endif
