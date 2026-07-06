/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef COMPUTATIONAL_DOMAIN_H_
#define COMPUTATIONAL_DOMAIN_H_

#include <array>
#include <string>

#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_REAL.H>

#include "GEMPIC_Config.H"
#include "GEMPIC_Parameters.H"

namespace Gempic
{

class H5GroupHandle;
class DiscreteAxis
{
public:
    enum IndexPosition
    {
        Cell,
        Node
    };

    DiscreteAxis() = default;
    AMREX_GPU_HOST_DEVICE
    DiscreteAxis(amrex::Real domainLo,
                 amrex::Real domainHi,
                 int nCells,
                 IndexPosition idxPosition,
                 bool periodic = true);

    AMREX_GPU_HOST_DEVICE amrex::Real min () const { return m_domainLo; }
    AMREX_GPU_HOST_DEVICE amrex::Real max () const { return m_domainHi; }
    AMREX_GPU_HOST_DEVICE amrex::Real length () const { return m_domainHi - m_domainLo; }
    AMREX_GPU_HOST_DEVICE amrex::Real dx () const { return length() / n_cells(); }
    AMREX_GPU_HOST_DEVICE int size () const { return m_degreesOfFreedom; }
    AMREX_GPU_HOST_DEVICE int n_cells () const
    {
        return position() == IndexPosition::Cell ? size() : size() - 1;
    }
    AMREX_GPU_HOST_DEVICE bool is_periodic () const { return m_periodicity; }
    AMREX_GPU_HOST_DEVICE IndexPosition position () const { return m_idxPosition; }
    AMREX_GPU_HOST_DEVICE amrex::Real location (int idx) const
    {
        return m_domainLo + (m_offset + static_cast<amrex::Real>(idx)) * dx();
    }

    // clang-format off
    /**
     * @brief Describe a location on the axis with an index and the offset between the location and the index
     */
    // clang-format on
    class Location
    {
    public:
        AMREX_GPU_HOST_DEVICE int index () { return m_index; }
        AMREX_GPU_HOST_DEVICE amrex::Real offset () { return m_offset; }

    private:
        friend class DiscreteAxis;
        Location(int index, amrex::Real offset) : m_index{index}, m_offset{offset} {};
        int m_index{};
        amrex::Real m_offset{};
    };
    // clang-format off
    /**
     * @brief Calculate the nearest index to the right of @param x
     *
     * @return @ref IndexLocation Return the nearest index to the right of @param x.
     */
    // clang-format on
    AMREX_GPU_HOST_DEVICE Location index (amrex::Real const& x) const
    {
        amrex::Real const normalizedPos = (x - location(0)) / dx();
        int const index = std::ceil(normalizedPos);
        amrex::Real const offset = location(index) - x;
        return Location{index, offset};
    };

    bool operator==(DiscreteAxis const&) const = default;

private:
    amrex::Real m_domainLo{};
    amrex::Real m_domainHi{};
    int m_degreesOfFreedom{};
    amrex::Real m_offset{};
    IndexPosition m_idxPosition{};
    bool m_periodicity{true};
};

// clang-format off
/**
 * @class DiscreteGrid
 * @brief Represents a tensor grid for equidistant finite differences
 *
 * @details A single axis manages data for a 1-D finite difference grid.
 *          The axis is defined by its length, the minimum value, the number of cells (degrees of freedom) and the type of the axis.
 *
 *          The computational domain of the axis is given by \f$L = min - max\f$
 *
 * Two independent definitions for axes are available depending on where in the cell a degree of freedom is positioned.
 *
 * @ref DiscreteAxis::IndexPosition Cell:
 * The degree of freedom is located in the center of a cell.
 * The interval of the cell is a closed interval at the lower boundary and an open interval on the upper boundary.
 *
 * @image html GEMPIC_DiscreteGridAxisCell.png
 *
 * @ref DiscreteAxis::IndexPosition Node:
 * The degree of freedom is located on the cell nodes or on the boundaries of the cell.
 * Therefore size-2 cells in an axis have size-1 degrees of freedom.
 *
 * @image html GEMPIC_DiscreteGridAxisNode.png
 *
 */
// clang-format on
class DiscreteGrid
{
public:
    //!@cond EXCLUDE_API_DOC
    DiscreteGrid() = default;
    DiscreteGrid(std::array<amrex::Real, AMREX_SPACEDIM> domainLo,
                 std::array<amrex::Real, AMREX_SPACEDIM> domainHi,
                 std::array<int, AMREX_SPACEDIM> nCells,
                 std::array<DiscreteAxis::IndexPosition, AMREX_SPACEDIM> idxPosition,
                 std::array<bool, AMREX_SPACEDIM> periodic = {AMREX_D_DECL(true, true, true)});
    DiscreteGrid(Io::Parameters& params,
                 std::array<DiscreteAxis::IndexPosition, AMREX_SPACEDIM> idxPosition);
    //!@endcond
    /**
     * @brief Calculate the location of the given index in the interval
     *        [@ref DiscreteGrid::min, @ref DiscreteGrid::max] of the physical domain
     *
     * @param dir
     * @param idx idx\f$\in[0,\f$ @ref DiscreteGrid::size - 1\f$]\f$
     */
    AMREX_GPU_HOST_DEVICE
    amrex::Real location_1d (Direction dir, int idx) const { return m_axes[dir].location(idx); }
    AMREX_GPU_HOST_DEVICE
    std::array<amrex::Real, AMREX_SPACEDIM> location (AMREX_D_DECL(int ix, int iy, int iz))
    {
        return std::array<amrex::Real, AMREX_SPACEDIM>{
            AMREX_D_DECL(location_1d(Direction::xDir, ix), location_1d(Direction::yDir, iy),
                         location_1d(Direction::zDir, iz))};
    }
    AMREX_GPU_HOST_DEVICE
    std::array<amrex::Real, AMREX_SPACEDIM> location (std::array<int, AMREX_SPACEDIM> const& idx)
    {
        return location(AMREX_D_DECL(idx[0], idx[1], idx[2]));
    }
    //! Lower domain boundary
    AMREX_GPU_HOST_DEVICE
    amrex::Real min (Direction dir) const { return m_axes[dir].min(); }
    std::array<amrex::Real, AMREX_SPACEDIM> min () const
    {
        std::array<amrex::Real, AMREX_SPACEDIM> result{};
        for (std::size_t d = 0; d < AMREX_SPACEDIM; ++d)
        {
            result[d] = m_axes[d].min();
        }
        return result;
    }
    //! Upper domain boundary
    AMREX_GPU_HOST_DEVICE
    amrex::Real max (Direction dir) const { return m_axes[dir].max(); }
    std::array<amrex::Real, AMREX_SPACEDIM> max () const
    {
        std::array<amrex::Real, AMREX_SPACEDIM> result{};
        for (std::size_t d = 0; d < AMREX_SPACEDIM; ++d)
        {
            result[d] = m_axes[d].max();
        }
        return result;
    }
    //! Length of the domain
    AMREX_GPU_HOST_DEVICE
    amrex::Real length (Direction dir) const { return m_axes[dir].length(); }
    std::array<amrex::Real, AMREX_SPACEDIM> length () const
    {
        std::array<amrex::Real, AMREX_SPACEDIM> result{};
        for (std::size_t d = 0; d < AMREX_SPACEDIM; ++d)
        {
            result[d] = m_axes[d].length();
        }
        return result;
    }
    //! Width of a single cell
    AMREX_GPU_HOST_DEVICE
    amrex::Real dx (Direction dir) const { return m_axes[dir].dx(); }
    std::array<amrex::Real, AMREX_SPACEDIM> dx () const
    {
        std::array<amrex::Real, AMREX_SPACEDIM> result{};
        for (std::size_t d = 0; d < AMREX_SPACEDIM; ++d)
        {
            result[d] = m_axes[d].dx();
        }
        return result;
    }
    //! Position of the degrees of freedom within a cell
    AMREX_GPU_HOST_DEVICE
    DiscreteAxis::IndexPosition position (Direction dir) const { return m_axes[dir].position(); }
    std::array<DiscreteAxis::IndexPosition, AMREX_SPACEDIM> position () const
    {
        std::array<DiscreteAxis::IndexPosition, AMREX_SPACEDIM> result{};
        for (std::size_t d = 0; d < AMREX_SPACEDIM; ++d)
        {
            result[d] = m_axes[d].position();
        }
        return result;
    }
    //! Number of degrees of freedom in the domain
    AMREX_GPU_HOST_DEVICE
    int size (Direction dir) const { return m_axes[dir].size(); }
    std::array<int, AMREX_SPACEDIM> size () const
    {
        std::array<int, AMREX_SPACEDIM> result{};
        for (std::size_t d = 0; d < AMREX_SPACEDIM; ++d)
        {
            result[d] = m_axes[d].size();
        }
        return result;
    }
    //! Number of cells the domain is divided into
    AMREX_GPU_HOST_DEVICE
    int n_cells (Direction dir) const { return m_axes[dir].n_cells(); }
    std::array<int, AMREX_SPACEDIM> n_cells () const
    {
        std::array<int, AMREX_SPACEDIM> result{};
        for (std::size_t d = 0; d < AMREX_SPACEDIM; ++d)
        {
            result[d] = m_axes[d].n_cells();
        }
        return result;
    }
    //! Periodicity of the domain
    AMREX_GPU_HOST_DEVICE
    bool is_periodic (Direction dir) const { return m_axes[dir].is_periodic(); }
    std::array<bool, AMREX_SPACEDIM> is_periodic () const
    {
        std::array<bool, AMREX_SPACEDIM> result{};
        for (std::size_t d = 0; d < AMREX_SPACEDIM; ++d)
        {
            result[d] = m_axes[d].is_periodic();
        }
        return result;
    }

    AMREX_GPU_HOST_DEVICE DiscreteAxis const& axis (Direction dir) const { return m_axes[dir]; }

    bool operator==(DiscreteGrid const&) const = default;

    friend DiscreteGrid convert_dof_position(
        DiscreteGrid const&, std::array<DiscreteAxis::IndexPosition, AMREX_SPACEDIM> const&);

private:
    std::array<DiscreteAxis, AMREX_SPACEDIM> m_axes{};
};

void serialize(std::string const& label, DiscreteGrid const& grid, H5GroupHandle const& group);
void deserialize(std::string const& label, DiscreteGrid& grid, H5GroupHandle const& group);

DiscreteGrid dof_on_node(DiscreteGrid const& grid);
DiscreteGrid dof_on_cell_center(DiscreteGrid const& grid);
std::array<DiscreteGrid, 3> dof_on_edge(DiscreteGrid const& grid);
std::array<DiscreteGrid, 3> dof_on_face(DiscreteGrid const& grid);

namespace Impl
{
amrex::IndexType to_amrex_idx_type(DiscreteGrid const& discreteGrid);
amrex::Box to_amrex_box(DiscreteGrid const& discreteGrid);
amrex::RealBox to_amrex_real_box(DiscreteGrid const& discreteGrid);
amrex::Geometry to_amrex_geometry(DiscreteGrid const& discreteGrid);
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
