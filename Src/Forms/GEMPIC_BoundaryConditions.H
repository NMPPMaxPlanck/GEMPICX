#ifndef GEMPIC_BOUNDARY_CONDITIONS_H_
#define GEMPIC_BOUNDARY_CONDITIONS_H_

#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_Geometry.H>
#include <AMReX_BCRec.H>
#include <AMReX_BC_TYPES.H>

#include "GEMPIC_Config.H"
#include "GEMPIC_Parameters.H"

namespace Gempic::Forms
{

// enum for mathematical boundary conditions
enum BCType
{
    int_dir = amrex::BCType::int_dir,
    hom_dir = amrex::BCType::ext_dir,
    extrap = amrex::BCType::user_1
};

// map of physical (used in input file) to mathematical boundary conditions
inline std::map<std::string, int> bcTypeMap = {{"Natural", BCType::extrap},
                                               {"PerfectlyConducting", BCType::hom_dir}};

inline std::vector<std::string> read_boundary_condition (int numNonPeriodic,
                                                         std::string const& diagName)
{
    Io::Parameters paramsBC("BoundaryCondition");
    std::vector<std::string> bc;
    paramsBC.get_or_set(diagName, bc);
    size_t userConditions{bc.size()};

    // Default if no boundary conditions are given in input file
    if (userConditions == 0)
    {
        // No default was given in input file
        if (diagName == "Default")
        {
            bc = std::vector<std::string>(numNonPeriodic * 2, "Natural");
        }
        // No user options was given for this field. Read the default
        else
        {
            bc = read_boundary_condition(numNonPeriodic, "Default");
        }
    }
    // 2 boundary conditions needed for each non-periodic dimension
    // Either no or all non-periodic boundary conditions must be given
    else if (userConditions < 2 * numNonPeriodic)
    {
        GEMPIC_ERROR("Not enough boundary conditions given for " + diagName);
    }
    /* Not checked, so that we can use the same input file for all dimensions
    else
    {
        GEMPIC_ERROR("Too many boundary conditions given for " + diagName);
    }
    */
    return bc;
}

inline amrex::Vector<amrex::BCRec> load_boundary_condition (amrex::Geometry geom,
                                                            std::string diagName)
{
    int numNonPeriodic = 0;
    for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
    {
        numNonPeriodic += geom.isPeriodic(dim) ? 0 : 1;
    }

    auto bc{read_boundary_condition(numNonPeriodic, diagName)};

    // Initialize boundary condition record
    amrex::Vector<amrex::BCRec> bcRec(1);
    int bcIdx = 0;
    for (int n = 0; n < 1; ++n)
    {
        for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
        {
            if (geom.isPeriodic(dim))
            {
                bcRec[n].setLo(dim, BCType::int_dir);
                bcRec[n].setHi(dim, BCType::int_dir);
            }
            else
            {
                auto it = bcTypeMap.find(bc[bcIdx++]);
                if (it != bcTypeMap.end())
                {
                    bcRec[n].setLo(dim, it->second);
                }
                else
                {
                    GEMPIC_ERROR("Boundary condition " + bc[bcIdx - 1] + " is not implemented.");
                }
                it = bcTypeMap.find(bc[bcIdx++]);
                if (it != bcTypeMap.end())
                {
                    bcRec[n].setHi(dim, it->second);
                }
                else
                {
                    GEMPIC_ERROR("Boundary condition " + bc[bcIdx - 1] + " is not implemented.");
                }
            }
        }
    }
    return bcRec;
}

enum class DomainEnd
{
    low,
    high
};

struct BoundaryCondition
{
    // Index type cannot be inferred from arguments in operator() and has to be specified in
    // instantiation of functor
    amrex::IndexType m_idxType;
    int m_extrapDegree;
    amrex::GpuArray<amrex::GpuArray<amrex::Real, 6>, 3> m_extrapStencil;
    Grid m_grid;

    BoundaryCondition() = default;
    BoundaryCondition(amrex::IndexType idxType, int extrapDegree, Grid grid) :
        m_idxType(idxType), m_extrapDegree(extrapDegree), m_grid{grid}
    {
        // Select polynomial extrapolation stencil, available for up to 3 ghost cells
        switch (m_extrapDegree)
        {
            case 0:
                m_extrapStencil[0] = {1};
                m_extrapStencil[1] = {1};
                m_extrapStencil[2] = {1};
                break;
            case 1:
                m_extrapStencil[0] = {2, -1};
                m_extrapStencil[1] = {3, -2};
                m_extrapStencil[2] = {4, -3};
                break;
            case 2:
                m_extrapStencil[0] = {3, -3, 1};
                m_extrapStencil[1] = {6, -8, 3};
                m_extrapStencil[2] = {10, -15, 6};
                break;
            case 3:
                m_extrapStencil[0] = {4, -6, 4, -1};
                m_extrapStencil[1] = {10, -20, 15, -4};
                m_extrapStencil[2] = {20, -45, 36, -10};
                break;
            case 4:
                m_extrapStencil[0] = {5, -10, 10, -5, 1};
                m_extrapStencil[1] = {15, -40, 45, -24, 5};
                m_extrapStencil[2] = {35, -105, 126, -70, 15};
                break;
            case 5:
                m_extrapStencil[0] = {6, -15, 20, -15, 6, -1};
                m_extrapStencil[1] = {21, -70, 105, -84, 35, -6};
                m_extrapStencil[2] = {56, -210, 336, -280, 120, -21};
                break;
            default:
                GEMPIC_ERROR("Extrapolation of degree " + std::to_string(m_extrapDegree) +
                             " not implemented.");
        }
    }

    template <Direction direction, DomainEnd domainEnd>
    AMREX_GPU_HOST_DEVICE amrex::Real apply_stencil (amrex::Array4<amrex::Real> const& arr,
                                                     int const distance,
                                                     int const i,
                                                     int const j,
                                                     int const k) const
    {
        amrex::Real val = 0.0;
        for (int shift = 0; shift <= m_extrapDegree; shift++)
        {
            if constexpr (domainEnd == DomainEnd::high)
            {
                val += m_extrapStencil[distance - 1][shift] *
                       arr(GEMPIC_D_PAD(i - (direction == xDir ? shift : 0),
                                        j - (direction == yDir ? shift : 0),
                                        k - (direction == zDir ? shift : 0)));
            }
            else
            {
                val += m_extrapStencil[distance - 1][shift] *
                       arr(GEMPIC_D_PAD(i + (direction == xDir ? shift : 0),
                                        j + (direction == yDir ? shift : 0),
                                        k + (direction == zDir ? shift : 0)));
            }
        }
        return val;
    }

    AMREX_GPU_DEVICE
    void operator()(amrex::IntVect const& iv,
                    amrex::Array4<amrex::Real> const& dest,
                    int const dcomp,
                    int const numcomp,
                    amrex::GeometryData const& geom,
                    amrex::Real const time,
                    amrex::BCRec const* bcr,
                    int const bcomp,
                    int const origComp) const
    {
        amrex::Box const& domainBox = amrex::convert(geom.Domain(), m_idxType);
        auto const &domainLo = domainBox.loVect();
        auto const &domainHi = domainBox.hiVect();

        int const i = iv[xDir];
        int const ilo = domainLo[xDir];
        int const ihi = domainHi[xDir];
#if AMREX_SPACEDIM >= 2
        int const j = iv[yDir];
        int const jlo = domainLo[yDir];
        int const jhi = domainHi[yDir];
#endif
#if AMREX_SPACEDIM == 3
        int const k = iv[zDir];
        int const klo = domainLo[zDir];
        int const khi = domainHi[zDir];
#endif

        // Needs to be iterated on if multiple components are added
        const amrex::BCRec& bc = bcr[0];
        // Sides in x-direction
        if (i < ilo)
        {
            // Physical boundary conditions only on primal grid / de Rham sequence
            // Boundary cell is never accessed directly (not possible in case of purely node
            // centered data with amrex::GpuBndryFuncFab::nddoit) but the value is set implicitly
            // from every outward ghost cell. This ensures correctness of extrapolation independent
            // of index order.
            if (m_grid == Grid::primal)
            {
                if (m_idxType.nodeCentered(xDir) && bc.lo(xDir) == BCType::hom_dir)
                {
                    dest(GEMPIC_D_PAD(ilo, j, k)) = 0.;
                }
            }
            switch (bc.lo(xDir))
            {
                case BCType::hom_dir:
                case BCType::extrap:
                    dest(GEMPIC_D_PAD(i, j, k)) =
                        apply_stencil<xDir, DomainEnd::low>(dest, ilo - i, GEMPIC_D_PAD(ilo, j, k));
                    break;
            }
        }
        else if (i > ihi)
        {
            if (m_grid == Grid::primal)
            {
                if (m_idxType.nodeCentered(xDir) && bc.hi(xDir) == BCType::hom_dir)
                {
                    dest(GEMPIC_D_PAD(ihi, j, k)) = 0.;
                }
            }
            switch (bc.hi(xDir))
            {
                case BCType::hom_dir:
                case BCType::extrap:
                    dest(GEMPIC_D_PAD(i, j, k)) = apply_stencil<xDir, DomainEnd::high>(
                        dest, i - ihi, GEMPIC_D_PAD(ihi, j, k));
                    break;
            }
        }

#if AMREX_SPACEDIM >= 2
        // Sides in y-direction
        if (j < jlo)
        {
            if (m_grid == Grid::primal)
            {
                if (m_idxType.nodeCentered(yDir) && bc.lo(yDir) == BCType::hom_dir)
                {
                    dest(GEMPIC_D_PAD(i, jlo, k)) = 0.;
                }
            }
            switch (bc.lo(yDir))
            {
                case BCType::hom_dir:
                case BCType::extrap:
                    dest(GEMPIC_D_PAD(i, j, k)) =
                        apply_stencil<yDir, DomainEnd::low>(dest, jlo - j, GEMPIC_D_PAD(i, jlo, k));
                    break;
            }
        }
        else if (j > jhi)
        {
            if (m_grid == Grid::primal)
            {
                if (m_idxType.nodeCentered(yDir) && bc.hi(yDir) == BCType::hom_dir)
                {
                    dest(GEMPIC_D_PAD(i, jhi, k)) = 0.;
                }
            }
            switch (bc.hi(yDir))
            {
                case BCType::hom_dir:
                case BCType::extrap:
                    dest(GEMPIC_D_PAD(i, j, k)) = apply_stencil<yDir, DomainEnd::high>(
                        dest, j - jhi, GEMPIC_D_PAD(i, jhi, k));
                    break;
            }
        }
#endif
#if AMREX_SPACEDIM == 3
        // Sides in z-direction
        if (k < klo)
        {
            if (m_grid == Grid::primal)
            {
                if (m_idxType.nodeCentered(zDir) && bc.lo(zDir) == BCType::hom_dir)
                {
                    dest(i, j, klo) = 0.;
                }
            }
            switch (bc.lo(zDir))
            {
                case BCType::hom_dir:
                case BCType::extrap:
                    dest(GEMPIC_D_PAD(i, j, k)) =
                        apply_stencil<zDir, DomainEnd::low>(dest, klo - k, GEMPIC_D_PAD(i, j, klo));
                    break;
            }
        }
        else if (k > khi)
        {
            if (m_grid == Grid::primal)
            {
                if (m_idxType.nodeCentered(zDir) && bc.hi(zDir) == BCType::hom_dir)
                {
                    dest(i, j, khi) = 0.;
                }
            }
            switch (bc.hi(zDir))
            {
                case BCType::hom_dir:
                case BCType::extrap:
                    dest(GEMPIC_D_PAD(i, j, k)) = apply_stencil<zDir, DomainEnd::high>(
                        dest, k - khi, GEMPIC_D_PAD(i, j, khi));
                    break;
            }
        }
#endif
        // Edges and corners do not have to be treated separately. The values are computed by
        // extrapolation from extrapolated values in a previous direction. AMReX ensures correct
        // order of indices (faces -> edges -> corners).
    }
};
} // namespace Gempic::Forms

#endif
