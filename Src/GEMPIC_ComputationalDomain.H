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
    amrex::IntVect m_nCell; // number of cells of the domain. A domain consists of several grids.
    amrex::BoxArray m_grid;
    amrex::Geometry m_geom;
    amrex::DistributionMapping m_distriMap;

    amrex::Geometry const& geometry () const { return m_geom; }
    // @todo amrex::Geometry is not accessible on GPU. Therefore, we provide
    // the geometry_data() function as an intermediate solution to access
    // geometry data on GPU.
    AMREX_GPU_HOST_DEVICE amrex::GeometryData const& geometry_data () const { return m_geomData; }
    amrex::Box const& box () const { return m_geom.Domain(); }
    AMREX_GPU_HOST_DEVICE amrex::Real cell_volume () const
    {
        double vol{1.0};
        for (size_t dir = 0; dir < AMREX_SPACEDIM; dir++)
        {
            vol *= m_geomData.CellSize(dir);
        }
        return vol;
    }
    // @todo Place holder until we can replace by AMReX Geometry member function
    AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> cell_size_array () const
    {
        return {{AMREX_D_DECL(m_geomData.CellSize(xDir), m_geomData.CellSize(yDir),
                              m_geomData.CellSize(zDir))}};
    }
    // @todo Place holder until we can replace by AMReX Geometry member function
    AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> inv_cell_size_array () const
    {
        return {{AMREX_D_DECL(1.0 / m_geomData.CellSize(xDir), 1.0 / m_geomData.CellSize(yDir),
                              1.0 / m_geomData.CellSize(zDir))}};
    }

    ComputationalDomain(bool) {}; // bool constructor for deferred specification of domain
    ComputationalDomain()
    {
        BL_PROFILE("Gempic::ComputationalDomain::ComputationalDomain()");
        Io::Parameters params("ComputationalDomain");
        // Initialize infrastructure:
        amrex::Vector<int> nCellVector;
        params.get("nCell", nCellVector);
        amrex::Vector<int> maxGridSizeTmp;
        amrex::IntVect maxGridSize;
        params.get("maxGridSize", maxGridSizeTmp);
        for (int i{0}; i < AMREX_SPACEDIM; ++i)
        {
            m_nCell[i] = nCellVector[i];
            maxGridSize[i] = maxGridSizeTmp[i];
        }

        amrex::Array<int, AMREX_SPACEDIM> isPeriodic;
        params.get("isPeriodic", isPeriodic);
        // In periodic directions domain is [0, 2pi / k]. Otherwise, we use domain_lo/_hi directly
        // Probably needs to check that both are not provided for the same direction?
        amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
        params.get_or_set("domainLo", domainLo);
        amrex::Vector<amrex::Real> domainHi{AMREX_D_DECL(1.0, 1.0, 1.0)};
        amrex::Vector<amrex::Real> k;
        // Non-periodic directions exist
        if (!(AMREX_D_TERM(isPeriodic[xDir], &&isPeriodic[yDir], &&isPeriodic[zDir])))
        {
            params.get_or_set("domainHi", domainHi);
        }
        // Periodic directions exist
        if (AMREX_D_TERM(isPeriodic[xDir], || isPeriodic[yDir], || isPeriodic[zDir]))
        {
            // Attempt to use k first
            if (params.exists("k"))
            {
                if (params.is_in_input_file("domainHi"))
                {
                    std::cerr << "Warning: \"domainHi\" will not be used if \"k\" exists\n";
                }
                params.get("k", k);
                for (int i = 0; i < AMREX_SPACEDIM; i++)
                {
                    if (isPeriodic[i] == 1)
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
        amrex::RealBox realBox =
            amrex::RealBox({AMREX_D_DECL(domainLo[xDir], domainLo[yDir], domainLo[zDir])},
                           {AMREX_D_DECL(domainHi[xDir], domainHi[yDir], domainHi[zDir])});

        amrex::Box domain;
        amrex::IntVect domLo(AMREX_D_DECL(0, 0, 0));
        domain.setSmall(domLo);
        amrex::IntVect domHi(AMREX_D_DECL(m_nCell[xDir] - 1, m_nCell[yDir] - 1, m_nCell[zDir] - 1));
        domain.setBig(domHi);

        m_grid.define(domain);
        m_grid.maxSize(maxGridSize);

        m_geom.define(domain, realBox, amrex::CoordSys::cartesian, isPeriodic);
        m_distriMap.define(m_grid);

        m_geomData = m_geom.data();
    }

private:
    amrex::GeometryData m_geomData;
};
} // namespace Gempic

#endif
