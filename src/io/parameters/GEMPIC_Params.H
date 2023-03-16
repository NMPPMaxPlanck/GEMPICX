#ifndef GEMPIC_PARAMS_H_
#define GEMPIC_PARAMS_H_

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_CoordSys.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_Geometry.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>
#include <AMReX_Periodicity.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>
#include <GEMPIC_Config.H>

class Parameters
{
    public:
        Parameters(const amrex::RealBox& realBox, const amrex::IntVect& nCell,
                   const amrex::IntVect& maxGridSize, const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic, const int degree);

        const amrex::RealBox& realBox() const;
        const amrex::IntVect& nCell() const;
        const amrex::Array<int, GEMPIC_SPACEDIM>& isPeriodic() const;
        const amrex::BoxArray& grid() const;
        const amrex::Geometry& geometry() const;
        const amrex::DistributionMapping& distriMap() const;
        const amrex::Box& domain() const;
        const amrex::RealVect dr() const;
        int degree() const;

    private:
        amrex::RealBox m_realBox;
        amrex::IntVect m_nCell;
        amrex::IntVect m_maxGridSize;
        amrex::Array<int, GEMPIC_SPACEDIM> m_isPeriodic;
        amrex::RealVect m_length;
        amrex::BoxArray m_grid;
        amrex::Box m_domain;
        amrex::Geometry m_geometry;
        amrex::DistributionMapping m_distriMap;
        int m_degree;
};
#endif
