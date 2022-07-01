#ifndef COMPUTATIONAL_DOMAIN_H_
#define COMPUTATIONAL_DOMAIN_H_

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_BLProfiler.H>
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

namespace Gempic
{
namespace CompDom
{
/**
 * Class containing all the necessary structures (such as the @p domain and the @p distriMap) and
 * information (such as @p dx and @p n_cell).
 */
class computational_domain
{
public:
    amrex::IntVect n_cell;         // number of cells
    amrex::IntVect max_grid_size;  // maximum number of cells in tile in each direction
    int is_periodic[GEMPIC_SPACEDIM];
    amrex::RealBox real_box;  // domain like {min_x,min_y,min_z}{max_x,max_y,max_z}

    amrex::RealVect Length;  // Length of domain in x,y,z directions
    amrex::BoxArray grid;
    amrex::Geometry geom;
    amrex::Box domain;
    amrex::DistributionMapping distriMap;
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> dx;
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> plo;
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> phi;
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM + 1> dxi;

    amrex::Real tolerance_particle;  //

    // The default constructor.
    /**
     * Default Constructor.
     */
    /*  computational_domain (){};

      computational_domain(amrex::IntVect n_cell_in,
                           amrex::IntVect max_grid_size_in,
                           amrex::IntVect is_periodic_in,
                           amrex::RealBox real_box_in,
                           amrex::Real tolerance_particle_in = 1.e-10){
          initialize(n_cell_in, max_grid_size_in, is_periodic_in, real_box_in,
      tolerance_particle_in);
      };*/

    // private:

    void initialize_computational_domain(amrex::IntVect n_cell_in, amrex::IntVect max_grid_size_in,
                                         amrex::IntVect is_periodic_in, amrex::RealBox real_box_in,
                                         amrex::Real tolerance_particle_in = 1.e-10)
    {
        BL_PROFILE("computational_domain::initialize_computational_domain");

        // Initialize infrastructure:

        n_cell = n_cell_in;
        max_grid_size = max_grid_size_in;

        for (int cc = 0; cc < GEMPIC_SPACEDIM; cc++)
        {
            is_periodic[cc] = is_periodic_in[cc];
        }
        real_box = real_box_in;

        for (int cc = 0; cc < GEMPIC_SPACEDIM; cc++)
        {
            Length[cc] = (real_box.hi(cc) - real_box.lo(cc));
        }

        amrex::IntVect dom_lo(AMREX_D_DECL(0, 0, 0));
        amrex::IntVect dom_hi(AMREX_D_DECL(n_cell[0] - 1, n_cell[1] - 1, n_cell[2] - 1));

        domain.setSmall(dom_lo);
        domain.setBig(dom_hi);
        grid.define(domain);
        grid.maxSize(max_grid_size);
        geom.define(domain, &real_box, amrex::CoordSys::cartesian, is_periodic);
        // ifr->dx = *(ifr->geom.CellSize());

        distriMap.define(grid);

        dxi[GEMPIC_SPACEDIM] = 1.;
        for (int cc = 0; cc < GEMPIC_SPACEDIM; cc++)
        {
            dx[cc] = geom.CellSize()[cc];
            plo[cc] = geom.ProbLo()[cc];
            phi[cc] = geom.ProbHi()[cc];
            dxi[cc] = 1.0 / dx[cc];
            dxi[GEMPIC_SPACEDIM] *= dxi[cc];
        }

        tolerance_particle = tolerance_particle_in;
    }
};
}  // namespace CompDom
}  // namespace Gempic

#endif
