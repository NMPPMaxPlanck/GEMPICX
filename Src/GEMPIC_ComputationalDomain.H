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

#include "GEMPIC_Config.H"
#include "GEMPIC_Parameters.H"

namespace Gempic
{
/**
 * Class containing all the necessary structures (such as the @p domain and the @p distriMap) and
 * information (such as @p dx and @p n_cell).
 */
class ComputationalDomain
{
public:
    amrex::IntVect m_nCell;  // number of cells of the domain. A domain consists of several grids.
    amrex::IntVect m_maxGridSize;  // maximum number of cells in tile in each direction. AMReX
                                   // automagically creates grids to cover the domain and not be
                                   // bigger than a certain, maximum grid size.
    amrex::Array<int, GEMPIC_SPACEDIM> m_isPeriodic;
    amrex::RealBox m_realBox;  // domain like {min_x,min_y,min_z}{max_x,max_y,max_z}
    amrex::RealVect m_length;  // Length of domain in x,y,z directions
    amrex::BoxArray m_grid;
    amrex::Geometry m_geom;
    amrex::Box m_domain;
    amrex::DistributionMapping m_distriMap;
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> m_dx;
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> m_plo;
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> m_phi;
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM + 1> m_dxi;  // inverse of dx

    amrex::Real m_toleranceParticle;  //

    ComputationalDomain(bool){};  // bool constructor for deferred specification of domain

    ComputationalDomain()
    {
        BL_PROFILE("ComputationalDomain::ComputationalDomain");
        Io::Parameters params("ComputationalDomain");
        // Initialize infrastructure:
        // params.get("nCellVector", n_cell);
        // params.get("maxGridSizeVector", max_grid_size);
        amrex::Vector<int> nCellVector;
        params.get("nCellVector", nCellVector);
        amrex::Vector<int> maxGridSizeVector;
        params.get("maxGridSizeVector", maxGridSizeVector);
        for (int i{0}; i < GEMPIC_SPACEDIM; ++i)
        {
            m_nCell[i] = nCellVector[i];
            m_maxGridSize[i] = maxGridSizeVector[i];
        }

        params.get("isPeriodicVector", m_isPeriodic);
        // In periodic directions domain is [0, 2pi / k]. Otherwise, we use domain_lo/_hi directly
        // Probably needs to check that both are not provided for the same direction?
        amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
        params.get_or_set("domainLo", domainLo);
        amrex::Vector<amrex::Real> domainHi{AMREX_D_DECL(1.0, 1.0, 1.0)};
        amrex::Vector<amrex::Real> k;
        // Non-periodic directions exist
        if (!AMREX_D_TERM(m_isPeriodic[xDir], &&m_isPeriodic[yDir], &&m_isPeriodic[zDir]))
        {
            params.get_or_set("domainHi", domainHi);
        }
        // Periodic directions exist
        if (AMREX_D_TERM(m_isPeriodic[xDir], || m_isPeriodic[yDir], || m_isPeriodic[zDir]))
        {
            // Attempt to use k first
            if (params.exists("k"))
            {
                if (params.is_in_input_file("domainHi"))
                {
                    std::cerr << "Warning: \"domainHi\" will not be used if \"k\" exists\n";
                }
                params.get("k", k);
                for (int i = 0; i < GEMPIC_SPACEDIM; i++)
                {
                    if (m_isPeriodic[i] == 1)
                    {
                        domainHi[i] = 2 * M_PI / k[i] + domainLo[i];
                    }
                }
            }
            // settle for domain_hi
            else
            {
                params.get_or_set("domainHi", domainHi);
            }
        }
        m_realBox = amrex::RealBox({AMREX_D_DECL(domainLo[xDir], domainLo[yDir], domainLo[zDir])},
                                   {AMREX_D_DECL(domainHi[xDir], domainHi[yDir], domainHi[zDir])});

        // define the length of the domain
        for (int cc = 0; cc < GEMPIC_SPACEDIM; cc++)
        {
            m_length[cc] = (m_realBox.hi(cc) - m_realBox.lo(cc));
        }

        // lower-boundary of domain
        amrex::IntVect domLo(AMREX_D_DECL(0, 0, 0));
        m_domain.setSmall(domLo);

        // higher-boundary of domain
        amrex::IntVect domHi(AMREX_D_DECL(m_nCell[xDir] - 1, m_nCell[yDir] - 1, m_nCell[zDir] - 1));
        m_domain.setBig(domHi);

        m_grid.define(m_domain);
        m_grid.maxSize(m_maxGridSize);

        m_geom.define(m_domain, m_realBox, amrex::CoordSys::cartesian, m_isPeriodic);

        // initialize distribution mapping needed to define MultiFabs
        m_distriMap.define(m_grid);

        m_dxi[GEMPIC_SPACEDIM] = 1.;
        for (int cc = 0; cc < GEMPIC_SPACEDIM; cc++)
        {
            m_dx[cc] = m_geom.CellSize()[cc];
            m_plo[cc] = m_geom.ProbLo()[cc];
            m_phi[cc] = m_geom.ProbHi()[cc];
            m_dxi[cc] = 1.0 / m_dx[cc];
            m_dxi[GEMPIC_SPACEDIM] *= m_dxi[cc];  // inverse volume
        }

        m_toleranceParticle = 1.e-10;
        params.get_or_set("toleranceParticle", m_toleranceParticle);
    }
};
}  // namespace Gempic

#endif
