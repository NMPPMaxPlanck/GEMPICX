#ifndef COMPUTATIONAL_DOMAIN_H_
#define COMPUTATIONAL_DOMAIN_H_

#include <array>

#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_REAL.H>

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
    DiscreteGrid() = default;
    DiscreteGrid(std::array<amrex::Real, AMREX_SPACEDIM> domainLo,
                 std::array<amrex::Real, AMREX_SPACEDIM> domainHi,
                 std::array<int, AMREX_SPACEDIM> nCells,
                 std::array<DiscreteGrid::Position, AMREX_SPACEDIM> idxPosition,
                 std::array<bool, AMREX_SPACEDIM> periodic = {AMREX_D_DECL(true, true, true)});
    DiscreteGrid(Io::Parameters& params,
                 std::array<DiscreteGrid::Position, AMREX_SPACEDIM> idxPosition);
    /**
     * @brief Calculate the location of the given index in the interval
     *        [@ref DiscreteGrid::min, @ref DiscreteGrid::max] of the physical domain
     *
     * @param dir
     * @param idx idx\f$\in[0,\f$ @ref DiscreteGrid::size - 1\f$]\f$
     */
    AMREX_GPU_HOST_DEVICE
    amrex::Real location_1D (Direction dir, int idx) const
    {
        return min(dir) + (m_offset[dir] + static_cast<amrex::Real>(idx)) * dx(dir);
    }
    AMREX_GPU_HOST_DEVICE
    std::array<amrex::Real, AMREX_SPACEDIM> location (AMREX_D_DECL(int ix, int iy, int iz))
    {
        std::array<amrex::Real, AMREX_SPACEDIM> loc{};
        loc[0] = location_1D(Direction::xDir, ix);
        #if (AMREX_SPACEDIM > 1)
        loc[1] = location_1D(Direction::yDir, iy);
        #endif
        #if (AMREX_SPACEDIM == 3)
        loc[2] = location_1D(Direction::zDir, iz);
        #endif
        return loc;
    }
    AMREX_GPU_HOST_DEVICE
    std::array<amrex::Real, AMREX_SPACEDIM> location(std::array<int, AMREX_SPACEDIM> const& idx)
    {
        return location(AMREX_D_DECL(idx[0], idx[1], idx[2]));
    }
    //! Lower domain boundary
    AMREX_GPU_HOST_DEVICE
    amrex::Real min (Direction dir) const { return m_domainLo[dir]; }
    std::array<amrex::Real, AMREX_SPACEDIM> min () const { return m_domainLo; }
    //! Upper domain boundary
    AMREX_GPU_HOST_DEVICE
    amrex::Real max (Direction dir) const { return m_domainHi[dir]; }
    std::array<amrex::Real, AMREX_SPACEDIM> max () const { return m_domainHi; }
    //! Length of the domain
    AMREX_GPU_HOST_DEVICE
    amrex::Real length (Direction dir) const { return max(dir) - min(dir); }
    std::array<amrex::Real, AMREX_SPACEDIM> length ()
    {
        std::array<amrex::Real, AMREX_SPACEDIM> l{};
        for (size_t i = 0; i < l.size(); i++)
        {
            l[i] = length(static_cast<Direction>(i));
        }
        return l;
    }
    //! Width of a single cell
    AMREX_GPU_HOST_DEVICE
    amrex::Real dx (Direction dir) const { return m_dx[dir]; }
    std::array<amrex::Real, AMREX_SPACEDIM> dx () const { return m_dx; }
    //! Position of the degrees of freedom within a cell
    AMREX_GPU_HOST_DEVICE
    DiscreteGrid::Position position (Direction dir) const { return m_idxPosition[dir]; }
    std::array<DiscreteGrid::Position, AMREX_SPACEDIM> position () const { return m_idxPosition; }
    //! Number of degrees of freedom in the domain
    AMREX_GPU_HOST_DEVICE
    int size (Direction dir) const { return m_degreesOfFreedom[dir]; }
    std::array<int, AMREX_SPACEDIM> size () const { return m_degreesOfFreedom; }
    AMREX_GPU_HOST_DEVICE
    bool is_periodic (Direction dir) const { return m_periodicity[dir]; }
    std::array<bool, AMREX_SPACEDIM> is_periodic () const { return m_periodicity; }
private:
    std::array<amrex::Real, AMREX_SPACEDIM> m_domainLo{};
    std::array<amrex::Real, AMREX_SPACEDIM> m_domainHi{};
    std::array<amrex::Real, AMREX_SPACEDIM> m_dx{};
    std::array<DiscreteGrid::Position, AMREX_SPACEDIM> m_idxPosition{};
    std::array<amrex::Real, AMREX_SPACEDIM> m_offset{};
    std::array<int, AMREX_SPACEDIM> m_degreesOfFreedom{};
    std::array<bool, AMREX_SPACEDIM> m_periodicity{};
};

namespace Impl
{
amrex::IndexType to_amrex_idx_type(DiscreteGrid const& discreteGrid);
amrex::Box to_amrex_box(DiscreteGrid const& DiscreteGrid);
amrex::RealBox to_amrex_real_box(DiscreteGrid const& discreteGrid);
amrex::Geometry to_amrex_geometry(DiscreteGrid const& DiscreteGrid);
amrex::Periodicity to_amrex_periodicty(DiscreteGrid const& discreteGrid);
} // namespace Impl

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
    AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, 3> cell_size_3darray () const
    {
        return {{GEMPIC_D_PAD_ONE(m_geomData.CellSize(xDir), m_geomData.CellSize(yDir),
                                  m_geomData.CellSize(zDir))}};
    }
    // @todo Place holder until we can replace by AMReX Geometry member function
    AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> inv_cell_size_array () const
    {
        return {{AMREX_D_DECL(1.0 / m_geomData.CellSize(xDir), 1.0 / m_geomData.CellSize(yDir),
                              1.0 / m_geomData.CellSize(zDir))}};
    }
    // @todo Place holder until we can replace by AMReX Geometry member function
    AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, 3> inv_cell_size_3darray () const
    {
        return {{GEMPIC_D_PAD_ONE(1.0 / m_geomData.CellSize(xDir), 1.0 / m_geomData.CellSize(yDir),
                                  1.0 / m_geomData.CellSize(zDir))}};
    }
    // @todo Place holder until we can replace by AMReX Geometry member function
    AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, 3> prob_low_3darray () const
    {
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r0Tmp{
            AMREX_D_DECL(m_geomData.ProbLo(0), m_geomData.ProbLo(1), m_geomData.ProbLo(2))};
        return {{GEMPIC_D_PAD_ONE(r0Tmp[xDir], r0Tmp[yDir], r0Tmp[zDir])}};
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
