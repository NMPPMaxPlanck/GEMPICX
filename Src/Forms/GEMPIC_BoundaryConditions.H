#ifndef GEMPIC_BOUNDARY_CONDITIONS_H_
#define GEMPIC_BOUNDARY_CONDITIONS_H_

#include <AMReX.H>
#include <AMReX_Print.H>

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
                                                         const std::string& diagName)
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
        std::string msg = "Not enough boundary conditions given for " + diagName;
        amrex::Assert(msg.c_str(), __FILE__, __LINE__);
    }
    /* Not checked, so that we can use the same input file for all dimensions
    else
    {
        std::string msg = "Too many boundary conditions given for " + diagName;
        amrex::Assert(msg.c_str(), __FILE__, __LINE__);
    }
    */
    return bc;
}

inline amrex::Vector<amrex::BCRec> load_boundary_condition (amrex::Geometry geom,
                                                            std::string diagName)
{
    int numNonPeriodic = 0;
    for (int dim = 0; dim < GEMPIC_SPACEDIM; ++dim)
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
                    amrex::Abort("Boundary condition " + bc[bcIdx - 1] + " is not implemented.");
                }
                it = bcTypeMap.find(bc[bcIdx++]);
                if (it != bcTypeMap.end())
                {
                    bcRec[n].setHi(dim, it->second);
                }
                else
                {
                    amrex::Abort("Boundary condition " + bc[bcIdx - 1] + " is not implemented.");
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

template <Grid grid, Space space>
struct BoundaryConditions
{
    // Index type cannot be inferred from arguments in operator() and has to be specified in
    // instantiation of functor
    amrex::IndexType m_idxType;
    int m_extrapDegree;
    amrex::GpuArray<amrex::GpuArray<amrex::Real, 6>, 3> m_extrapStencil;

    BoundaryConditions(amrex::IndexType idxType, int extrapDegree) :
        m_idxType(idxType), m_extrapDegree(extrapDegree)
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
                amrex::Abort("Extrapolation of degree " + std::to_string(m_extrapDegree) +
                             " not implemented.");
        }
    }

    template <Direction direction, DomainEnd domainEnd>
    AMREX_GPU_HOST_DEVICE amrex::Real apply_stencil (amrex::Array4<amrex::Real> const& arr,
                                                     const int distance,
                                                     const int i,
                                                     const int j,
                                                     const int k) const
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
    void operator()(const amrex::IntVect& iv,
                    amrex::Array4<amrex::Real> const& dest,
                    const int dcomp,
                    const int numcomp,
                    amrex::GeometryData const& geom,
                    const amrex::Real time,
                    const amrex::BCRec* bcr,
                    const int bcomp,
                    const int origComp) const
    {
        amrex::Box const& domainBox = amrex::convert(geom.Domain(), m_idxType);
        const auto& domainLo = domainBox.loVect();
        const auto& domainHi = domainBox.hiVect();
        const amrex::BCRec& bc = bcr[0];

        const int i = iv[xDir];
        const int ilo = domainLo[xDir];
        const int ihi = domainHi[xDir];
#if GEMPIC_SPACEDIM >= 2
        const int j = iv[yDir];
        const int jlo = domainLo[yDir];
        const int jhi = domainHi[yDir];
#if GEMPIC_SPACEDIM == 3
        const int k = iv[zDir];
        const int klo = domainLo[zDir];
        const int khi = domainHi[zDir];
#endif
#endif

        // Sides in x-direction
        if (i < ilo)
        {
            // Physical boundary conditions only on primal grid / de Rham sequence
            // Boundary cell is never accessed directly (not possible in case of purely node
            // centered data with amrex::GpuBndryFuncFab::nddoit) but the value is set implicitly
            // from every outward ghost cell. This ensures correctness of extrapolation independent
            // of index order.
            if constexpr (grid == Grid::primal)
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
            if constexpr (grid == Grid::primal)
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

#if GEMPIC_SPACEDIM >= 2
        // Sides in y-direction
        if (j < jlo)
        {
            if constexpr (grid == Grid::primal)
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
            if constexpr (grid == Grid::primal)
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

#if GEMPIC_SPACEDIM == 3
        // Sides in z-direction
        if (k < klo)
        {
            if constexpr (grid == Grid::primal)
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
            if constexpr (grid == Grid::primal)
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
#endif
        // Edges and corners do not have to be treated separately. The values are computed by
        // extrapolation from extrapolated values in a previous direction. AMReX ensures correct
        // order of indices (faces -> edges -> corners).
    }
};
}  // namespace Gempic::Forms

#endif