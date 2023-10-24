#include "GEMPIC_Params.H"

Parameters::Parameters() {};

Parameters::Parameters(const amrex::RealBox& realBox, const amrex::IntVect& nCell,
                       const amrex::IntVect& maxGridSize, const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic, const int degree):
            m_realBox(realBox), m_nCell(nCell), m_maxGridSize(maxGridSize),
            m_isPeriodic(isPeriodic), m_degree(degree)
{

    //lower-boundary of domain
    amrex::IntVect domLo(AMREX_D_DECL(0, 0, 0));    
    m_domain.setSmall(domLo); 

    //higher-boundary of domain
    amrex::IntVect domHi(AMREX_D_DECL(nCell[xDir] - 1, nCell[yDir] - 1, nCell[zDir] - 1));
    m_domain.setBig(domHi);

    m_grid.define(m_domain);
    m_grid.maxSize(maxGridSize);

    amrex::Geometry::Setup(&m_realBox, amrex::CoordSys::cartesian, m_isPeriodic.data());
    m_geometry.define(m_domain);

    //initialize distribution mapping needed to define MultiFabs
    m_distriMap.define(m_grid);

    //define the length of the domain
    m_length = amrex::RealVect{AMREX_D_DECL(m_geometry.ProbHi(xDir) - m_geometry.ProbLo(xDir), m_geometry.ProbHi(yDir) - m_geometry.ProbLo(yDir), m_geometry.ProbHi(zDir) - m_geometry.ProbLo(zDir))};

}

const amrex::RealBox& Parameters::realBox() const
{
    return m_realBox;
}

const amrex::IntVect& Parameters::nCell() const
{
    return m_nCell;
}

const amrex::Array<int, GEMPIC_SPACEDIM>& Parameters::isPeriodic() const
{
    return m_isPeriodic;
}

const amrex::BoxArray& Parameters::grid() const
{
    return m_grid;
}

const amrex::Geometry& Parameters::geometry() const
{
    return m_geometry;
}

const amrex::DistributionMapping& Parameters::distriMap() const
{
    return m_distriMap;
}

const amrex::Box& Parameters::domain() const
{
    return m_domain;
}

const amrex::RealVect Parameters::dr() const
{
    return amrex::RealVect{AMREX_D_DECL(m_length[xDir]/m_nCell[xDir], m_length[yDir]/m_nCell[yDir], m_length[zDir]/m_nCell[zDir])};
}

int Parameters::degree() const
{
    return m_degree;
}
