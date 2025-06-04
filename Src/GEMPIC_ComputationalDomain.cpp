#include <array>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_Parameters.H"

namespace Gempic
{
DiscreteGrid::DiscreteGrid (std::array<amrex::Real, AMREX_SPACEDIM> domainLo,
                            std::array<amrex::Real, AMREX_SPACEDIM> domainHi,
                            std::array<int, AMREX_SPACEDIM> nCells,
                            std::array<DiscreteGrid::Position, AMREX_SPACEDIM> idxPosition,
                            std::array<bool, AMREX_SPACEDIM> periodicity) :
    m_domainLo{domainLo},
    m_domainHi{domainHi},
    m_idxPosition{idxPosition},
    m_periodicity{periodicity}
{
    for (int dir{0}; dir < AMREX_SPACEDIM; dir++)
    {
        switch (m_idxPosition[dir])
        {
            case Position::Cell:
            {
                m_degreesOfFreedom[dir] = nCells[dir];
                m_offset[dir] = 0.5;
                break;
            }
            case Position::Node:
            {
                m_degreesOfFreedom[dir] = nCells[dir] + 1;
                m_offset[dir] = 0.0;
                break;
            }
        }
        m_dx[dir] = this->length(static_cast<Direction>(dir)) / nCells[dir];
    }
}

DiscreteGrid::DiscreteGrid (Io::Parameters& params,
                            std::array<DiscreteGrid::Position, AMREX_SPACEDIM> idxPosition)
{
    // Initialize infrastructure:
    std::array<int, AMREX_SPACEDIM> nCells{};
    params.get("ComputationalDomain.nCell", nCells);
    std::array<int, AMREX_SPACEDIM> isPeriodic{};
    params.get("ComputationalDomain.isPeriodic", isPeriodic);
    std::array<bool, AMREX_SPACEDIM> periodicity{};
    for (int dir{0}; dir < AMREX_SPACEDIM; dir++)
    {
        if (isPeriodic[dir])
        {
            periodicity[dir] = true;
        }
        else
        {
            periodicity[dir] = false;
        }
    }
    // In periodic directions domain is [0, 2pi / k]. Otherwise, we use domain_lo/_hi directly
    // Probably needs to check that both are not provided for the same direction?
    std::array<amrex::Real, AMREX_SPACEDIM> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
    params.get_or_set("ComputationalDomain.domainLo", domainLo);
    std::array<amrex::Real, AMREX_SPACEDIM> domainHi{AMREX_D_DECL(1.0, 1.0, 1.0)};
    std::array<amrex::Real, AMREX_SPACEDIM> k;
    // Non-periodic directions exist
    if (!(AMREX_D_TERM(periodicity[xDir], &&periodicity[yDir], &&periodicity[zDir])))
    {
        params.get_or_set("ComputationalDomain.domainHi", domainHi);
    }
    // Periodic directions exist
    if (AMREX_D_TERM(periodicity[xDir], || periodicity[yDir], || periodicity[zDir]))
    {
        // Attempt to use k first
        if (params.exists("ComputationalDomain.k"))
        {
            if (params.is_in_input_file("ComputationalDomain.domainHi"))
            {
                std::cerr << "Warning: \"domainHi\" will not be used if \"k\" exists\n";
            }
            params.get("ComputationalDomain.k", k);
            for (int i = 0; i < AMREX_SPACEDIM; i++)
            {
                if (periodicity[i] == 1)
                {
                    domainHi[i] = 2 * M_PI / k[i] + domainLo[i];
                }
            }
        }
        else
        {
            params.get_or_set("ComputationalDomain.domainHi", domainHi);
        }
    }

    *this = DiscreteGrid{domainLo, domainHi, nCells, idxPosition, periodicity};
}

namespace Impl
{
amrex::IndexType to_amrex_idx_type (DiscreteGrid const& discreteGrid)
{
    amrex::IndexType idx{};
    for (int i = 0; i < AMREX_SPACEDIM; i++)
    {
        switch (discreteGrid.position(static_cast<Direction>(i)))
        {
            case DiscreteGrid::Position::Cell:
                idx.setType(i, amrex::CellIndexEnum::CELL);
                break;
            case DiscreteGrid::Position::Node:
                idx.setType(i, amrex::CellIndexEnum::NODE);
                break;
        }
    }
    return idx;
}
amrex::Box to_amrex_box (DiscreteGrid const& discreteGrid)
{
    amrex::IndexType idx{to_amrex_idx_type(discreteGrid)};
    amrex::Box box;
    amrex::IntVect low(AMREX_D_DECL(0, 0, 0));
    std::array<int, AMREX_SPACEDIM> size{discreteGrid.size()};
    amrex::IntVect high(AMREX_D_DECL(size[xDir] - 1, size[yDir] - 1, size[zDir] - 1));
    return amrex::Box{low, high, idx};
}
amrex::RealBox to_amrex_real_box (DiscreteGrid const& discreteGrid)
{
    auto min = discreteGrid.min();
    auto max = discreteGrid.max();
    return amrex::RealBox{min, max};
}
amrex::Geometry to_amrex_geometry (DiscreteGrid const& discreteGrid)
{
    std::array<int, AMREX_SPACEDIM> periodicity{
        AMREX_D_DECL(static_cast<int>(discreteGrid.is_periodic(Direction::xDir)),
                     static_cast<int>(discreteGrid.is_periodic(Direction::yDir)),
                     static_cast<int>(discreteGrid.is_periodic(Direction::zDir)))};
    //  AMReX geometry always requires for some ridiculous reasons a cell centered box.
    //  Well luckily we can simply convert any box to be cell centered.
    //  https://amrex-codes.github.io/amrex/docs_html/Basics.html#realbox-and-geometry
    //  https://amrex-codes.github.io/amrex/doxygen/classamrex_1_1Geometry.html#ab9cc9315f181884f554c5866cd1e68e5
    return amrex::Geometry{
        amrex::convert(to_amrex_box(discreteGrid),
                       amrex::IndexType(amrex::IntVect{AMREX_D_DECL(amrex::CellIndexEnum::CELL,
                                                                    amrex::CellIndexEnum::CELL,
                                                                    amrex::CellIndexEnum::CELL)})),
        to_amrex_real_box(discreteGrid), amrex::CoordSys::cartesian, periodicity};
}
amrex::Periodicity to_amrex_periodicty (DiscreteGrid const& discreteGrid)
{
    // amrex::Periodicity object is not a bool but returns the highest index of a periodic domain
    // To avoid errors we return the periodicity instance that is taken from an amrex geometry
    // and therefore should hopefully be compatible with all calls to amrex functions.
    return to_amrex_geometry(discreteGrid).periodicity();
}
} // namespace Impl


ComputationalDomain::ComputationalDomain(std::array<amrex::Real, AMREX_SPACEDIM> const& domainLo,
                                         std::array<amrex::Real, AMREX_SPACEDIM> const& domainHi,
                                         amrex::IntVect const& nCell,
                                         amrex::IntVect const& maxGridSize,
                                         std::array<int, AMREX_SPACEDIM> const& isPeriodic,
                                         amrex::CoordSys::CoordType coordType) :
    m_nCell{nCell}
{
    BL_PROFILE("Gempic::ComputationalDomain::ComputationalDomain(args)");
    amrex::Box domain;
    amrex::IntVect domLo(AMREX_D_DECL(0, 0, 0));
    domain.setSmall(domLo);
    amrex::IntVect domHi(AMREX_D_DECL(m_nCell[xDir] - 1, m_nCell[yDir] - 1, m_nCell[zDir] - 1));
    domain.setBig(domHi);

    amrex::RealBox realBox = amrex::RealBox(domainLo, domainHi);
    m_geom.define(domain, realBox, coordType, isPeriodic);
    m_geomData = m_geom.data();

    m_grid.define(domain);
    m_grid.maxSize(maxGridSize);
    m_distriMap.define(m_grid);
}

ComputationalDomain::ComputationalDomain()
{
    Io::Parameters params("ComputationalDomain");
    // Initialize infrastructure:
    amrex::IntVect nCell;
    params.get("nCell", nCell);
    amrex::IntVect maxGridSize;
    params.get("maxGridSize", maxGridSize);

    std::array<int, AMREX_SPACEDIM> isPeriodic;
    params.get("isPeriodic", isPeriodic);
    // In periodic directions domain is [0, 2pi / k]. Otherwise, we use domain_lo/_hi directly
    // Probably needs to check that both are not provided for the same direction?
    std::array<amrex::Real, AMREX_SPACEDIM> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
    params.get_or_set("domainLo", domainLo);
    std::array<amrex::Real, AMREX_SPACEDIM> domainHi{AMREX_D_DECL(1.0, 1.0, 1.0)};
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

    auto coordsys = amrex::CoordSys::cartesian;

    *this = ComputationalDomain{domainLo, domainHi, nCell, maxGridSize, isPeriodic, coordsys};
}
} // namespace Gempic
