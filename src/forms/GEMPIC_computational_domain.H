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
#include "GEMPIC_parameters.H"

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
    amrex::IntVect n_cell;         // number of cells of the domain. A domain consists of several grids.
    amrex::IntVect max_grid_size;  // maximum number of cells in tile in each direction. AMReX automagically creates grids to cover the domain and not be bigger than a certain, maximum grid size.
    amrex::Array<int, GEMPIC_SPACEDIM> is_periodic;
    amrex::RealBox real_box;  // domain like {min_x,min_y,min_z}{max_x,max_y,max_z}
    amrex::RealVect Length;  // Length of domain in x,y,z directions
    amrex::BoxArray grid;
    amrex::Geometry geom;
    amrex::Box domain;
    amrex::DistributionMapping distriMap;
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> dx;
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> plo;
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> phi;
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM + 1> dxi;  // inverse of dx

    amrex::Real tolerance_particle;  //

    computational_domain (bool) {}; // bool constructor for deferred specification of domain

    computational_domain ()
    {
        BL_PROFILE("computational_domain::computational_domain");
        Parameters params("computational_domain");
        // Initialize infrastructure:
        //params.get("n_cell_vector", n_cell);
        //params.get("max_grid_size_vector", max_grid_size);
        amrex::Vector<int> nCellVector;
        params.get("n_cell_vector", nCellVector);
        amrex::Vector<int> maxGridSizeVector;
        params.get("max_grid_size_vector", maxGridSizeVector);
        for (int i{0}; i < GEMPIC_SPACEDIM; ++i)
        {
            n_cell[i] = nCellVector[i];
            max_grid_size[i] = maxGridSizeVector[i];
        }

        params.get("is_periodic_vector", is_periodic);
        // In periodic directions domain is [0, 2pi / k]. Otherwise, we use domain_lo/_hi directly
        // Probably needs to check that both are not provided for the same direction?
        amrex::Vector<amrex::Real> domain_lo{AMREX_D_DECL(0.0, 0.0, 0.0)};
        params.getOrSet("domain_lo", domain_lo);
        amrex::Vector<amrex::Real> domain_hi{AMREX_D_DECL(1.0, 1.0, 1.0)};
        amrex::Vector<amrex::Real> k;
        // Non-periodic directions exist
        if (!AMREX_D_TERM(is_periodic[xDir],&& is_periodic[yDir],&& is_periodic[zDir]))
        {
            params.getOrSet("domain_hi", domain_hi);
        }
        // Periodic directions exist
        if (AMREX_D_TERM(is_periodic[xDir],|| is_periodic[yDir],|| is_periodic[zDir]))
        {
            // Attempt to use k first
            if (params.exists("k"))
            {
                if (params.isInInputFile("domain_hi"))
                {
                    std::cerr << "Warning: \"domain_hi\" will not be used if \"k\" exists\n";
                }
                params.get("k", k);
                for (int i = 0; i < GEMPIC_SPACEDIM; i++)
                {
                    if (is_periodic[i] == 1)
                    {
                        domain_hi[i] = 2 * M_PI / k[i] + domain_lo[i];
                    }
                }
            }
            // settle for domain_hi
            else
            {
                params.getOrSet("domain_hi", domain_hi);
            }
        }
        real_box = amrex::RealBox({AMREX_D_DECL(domain_lo[xDir], domain_lo[yDir], domain_lo[zDir])},
                                  {AMREX_D_DECL(domain_hi[xDir], domain_hi[yDir], domain_hi[zDir])});

        //define the length of the domain
        for (int cc = 0; cc < GEMPIC_SPACEDIM; cc++)
        {
            Length[cc] = (real_box.hi(cc) - real_box.lo(cc));
        }

        //lower-boundary of domain
        amrex::IntVect dom_lo(AMREX_D_DECL(0, 0, 0));
        domain.setSmall(dom_lo);

        //higher-boundary of domain
        amrex::IntVect dom_hi(AMREX_D_DECL(n_cell[xDir] - 1, n_cell[yDir] - 1, n_cell[zDir] - 1));
        domain.setBig(dom_hi);

        grid.define(domain);
        grid.maxSize(max_grid_size);

        geom.define(domain, real_box, amrex::CoordSys::cartesian, is_periodic);

        //initialize distribution mapping needed to define MultiFabs
        distriMap.define(grid);

        dxi[GEMPIC_SPACEDIM] = 1.;
        for (int cc = 0; cc < GEMPIC_SPACEDIM; cc++)
        {
            dx[cc] = geom.CellSize()[cc];
            plo[cc] = geom.ProbLo()[cc];
            phi[cc] = geom.ProbHi()[cc];
            dxi[cc] = 1.0 / dx[cc];
            dxi[GEMPIC_SPACEDIM] *= dxi[cc];  // inverse volume
        }

        tolerance_particle = 1.e-10;
        params.getOrSet("tolerance_particle", tolerance_particle);
    }
};
}  // namespace CompDom
}  // namespace Gempic

#endif
