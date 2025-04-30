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
 * @class DiscreteGrid
 * @brief Represents a tensor grid axis for equidistant finite difference
 * grids
 *
 * @details A single axis manages data for a 1-D finite difference grid.
 *          The axis is defined by its length, the minimum value, the number of
 *          cells (degrees of freedom) and the type of the axis.
 *
 *          The computational domain of the axis is given by \f$L = min - max\f$
 *
 * Two independent definitions for axes are available depending on where in
 * the cell a degree of freedom is positioned.
 *
 * @ref DiscreteGrid::Position Cell:
 * The degree of freedom is located in the center of a cell. The interval of
 * the cell is a closed interval at the lower boundary and an open interval on
 * the upper boundary.
 *
 * @image html GEMPIC_DiscreteGridAxisCell.png
 *
 * @ref DiscreteGrid::Position Node:
 * The degree of freedom is located on the cell nodes or on the boundareis of
 * the cell. Therefore size-2 cells in an axis have size-1 degrees of freedom.
 *
 * @image html GEMPIC_DiscreteGridAxisNode.png
 *
 */
class DiscreteGrid
{
public:
    enum Position //! Defines the location of the degree of freedom within a cell
    {
        Cell,
        Node
    };
    DiscreteGrid(amrex::Array<amrex::Real, AMREX_SPACEDIM> domainLo,
                 amrex::Array<amrex::Real, AMREX_SPACEDIM> domainHi,
                 amrex::Array<int, AMREX_SPACEDIM> nCells,
                 amrex::Array<DiscreteGrid::Position, AMREX_SPACEDIM> idxPosition,
                 amrex::Array<bool, AMREX_SPACEDIM> periodic = {AMREX_D_DECL(true, true, true)});
    DiscreteGrid(Io::Parameters& params,
                 amrex::Array<DiscreteGrid::Position, AMREX_SPACEDIM> idxPosition);
    /**
     * @brief Calculate the location of the given index in the interval
     *        [@ref DiscreteGrid::min, @ref DiscreteGrid::max] of the physical domain
     *
     * @param dir
     * @param idx idx\f$\in[0,\f$ @ref DiscreteGrid::size - 1\f$]\f$
     */
    AMREX_GPU_HOST_DEVICE
    amrex::Real location (Direction dir, int idx)
    {
        return min(dir) + (m_offset[dir] + static_cast<amrex::Real>(idx)) * dx(dir);
    }
    //! Lower domain boundary
    AMREX_GPU_HOST_DEVICE
    amrex::Real min (Direction dir) { return m_domainLo[dir]; }
    //! Upper domain boundary
    AMREX_GPU_HOST_DEVICE
    amrex::Real max (Direction dir) { return m_domainHi[dir]; }
    //! Length of the domain
    AMREX_GPU_HOST_DEVICE
    amrex::Real length (Direction dir) { return max(dir) - min(dir); }
    //! Width of a single cell
    AMREX_GPU_HOST_DEVICE
    amrex::Real dx (Direction dir) { return m_dx[dir]; }
    //! Position of the degrees of freedom within a cell
    AMREX_GPU_HOST_DEVICE
    DiscreteGrid::Position position (Direction dir) { return m_idxPosition[dir]; }
    //! Number of degrees of freedom in the domain
    AMREX_GPU_HOST_DEVICE
    int size (Direction dir) { return m_degreesOfFreedom[dir]; }
    AMREX_GPU_HOST_DEVICE
    bool is_periodic (Direction dir) { return m_periodicity[dir]; }

private:
    amrex::Array<amrex::Real, AMREX_SPACEDIM> m_domainLo{};
    amrex::Array<amrex::Real, AMREX_SPACEDIM> m_domainHi{};
    amrex::Array<amrex::Real, AMREX_SPACEDIM> m_dx{};
    amrex::Array<DiscreteGrid::Position, AMREX_SPACEDIM> m_idxPosition{};
    amrex::Array<amrex::Real, AMREX_SPACEDIM> m_offset{};
    amrex::Array<int, AMREX_SPACEDIM> m_degreesOfFreedom{};
    amrex::Array<bool, AMREX_SPACEDIM> m_periodicity{};
};
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
    AMREX_GPU_HOST_DEVICE const amrex::GeometryData& geometry_data () const { return m_geomData; }
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

    ComputationalDomain(std::array<amrex::Real, AMREX_SPACEDIM> const& domainLo,
                        std::array<amrex::Real, AMREX_SPACEDIM> const& domainHi,
                        amrex::IntVect const& nCell,
                        amrex::IntVect const& maxGridSize,
                        std::array<int, AMREX_SPACEDIM> const& isPeriodic,
                        amrex::CoordSys::CoordType coordType = amrex::CoordSys::cartesian);

    ComputationalDomain();

private:
    amrex::GeometryData m_geomData;
};
} // namespace Gempic

#endif
