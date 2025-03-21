#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_Parameters.H"

namespace Gempic
{
DiscreteGrid::DiscreteGrid(amrex::Array<amrex::Real, AMREX_SPACEDIM> domainLo,
                           amrex::Array<amrex::Real, AMREX_SPACEDIM> domainHi,
                           amrex::Array<int, AMREX_SPACEDIM> nCells,
                           amrex::Array<DiscreteGrid::Position, AMREX_SPACEDIM> idxPosition,
                           amrex::Array<bool, AMREX_SPACEDIM> periodicity) :
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

DiscreteGrid::DiscreteGrid(Io::Parameters &params,
                           amrex::Array<DiscreteGrid::Position, AMREX_SPACEDIM> idxPosition)
{
    // Initialize infrastructure:
    amrex::Array<int, AMREX_SPACEDIM> nCells{};
    params.get("ComputationalDomain.nCell", nCells);
    amrex::Array<int, AMREX_SPACEDIM> isPeriodic{};
    params.get("ComputationalDomain.isPeriodic", isPeriodic);
    amrex::Array<bool, AMREX_SPACEDIM> periodicity{};
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
    amrex::Array<amrex::Real, AMREX_SPACEDIM> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
    params.get_or_set("ComputationalDomain.domainLo", domainLo);
    amrex::Array<amrex::Real, AMREX_SPACEDIM> domainHi{AMREX_D_DECL(1.0, 1.0, 1.0)};
    amrex::Array<amrex::Real, AMREX_SPACEDIM> k;
    // Non-periodic directions exist
    if (!(AMREX_D_TERM(periodicity[xDir], &&periodicity[yDir], &&periodicity[zDir])))
    {
        params.get_or_set("ComputationalDomain.domainHi", domainHi);
    }
    // Periodic directions exist
    if (AMREX_D_TERM(periodicity[xDir], || periodicity[yDir], || periodicity[zDir]))
    {
        // Attempt to use k first
        if (params.exists("k"))
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

ComputationalDomain::ComputationalDomain()
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

} // namespace Gempic
