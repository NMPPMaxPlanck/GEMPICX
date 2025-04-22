#include <gtest/gtest.h>

#include "GEMPIC_TestUtils.H"

namespace Gempic::Test::Utils
{
using condLambda = bool (*)(AMREX_D_DECL(int, int, int));
void check_field (const char file[],
                  int line,
                  amrex::Array4<amrex::Real> const& fieldArr,
                  amrex::Box const& bx,
                  std::vector<condLambda>&& condVec,
                  std::vector<amrex::Real>&& checks,
                  std::optional<amrex::Real> defCheck,
                  amrex::Real tol)
{
    const auto lo = amrex::lbound(bx);
    const auto hi = amrex::ubound(bx);
    int ncomp{fieldArr.nComp()};
    for (int comp{0}; comp < ncomp; comp++)
    {
        for (int i{lo.x}; i <= hi.x; i++)
        {
            for (int j{lo.y}; j <= hi.y; j++)
            {
                for (int k{lo.z}; k <= hi.z; k++)
                {
                    int condNum{0};
                    const amrex::IntVect idx{AMREX_D_DECL(i, j, k)};
                    for (auto cond : condVec)
                    {
                        if (cond(AMREX_D_DECL(i, j, k)))
                        {
                            EXPECT_NEAR(checks[condNum], *fieldArr.ptr(idx, comp), tol)
                                << file << ":" << line << ": Failed condition " << condNum
                                << ".\nIndices: " << string_array(idx, AMREX_SPACEDIM)
                                << "\tComponent: " << comp;
                            break;
                        }
                        condNum++;
                    }
                    if (condNum == condVec.size() && defCheck)
                    {
                        EXPECT_NEAR(defCheck.value(), *fieldArr.ptr(idx, comp), tol)
                            << file << ":" << line
                            << ": Failed default value check: " << defCheck.value()
                            << ".\nIndices: " << string_array(idx, AMREX_SPACEDIM)
                            << "\tComponent: " << comp;
                    }
                }
            }
        }
    }
}
void compare_fields (const char file[],
                     int line,
                     amrex::Array4<const amrex::Real> const& fieldArr,
                     amrex::Array4<const amrex::Real> const& fieldArr2,
                     amrex::Box const& bx,
                     amrex::Real tol,
                     int ncomp)
{
    const auto lo = amrex::lbound(bx);
    const auto hi = amrex::ubound(bx);
    for (int comp{0}; comp < ncomp; comp++)
    {
        for (int i{lo.x}; i <= hi.x; i++)
        {
            for (int j{lo.y}; j <= hi.y; j++)
            {
                for (int k{lo.z}; k <= hi.z; k++)
                {
                    const amrex::IntVect idx{AMREX_D_DECL(i, j, k)};
                    EXPECT_NEAR(*fieldArr.ptr(idx, comp), *fieldArr2.ptr(idx, comp), tol)
                        << file << ":" << line
                        << ": Unequal arrays.\nIndices: " << string_array(idx, AMREX_SPACEDIM)
                        << "\tComponent: " << comp;
                }
            }
        }
    }
}
double l_inf_error (amrex::Array4<const amrex::Real> const& a,
                    amrex::Array4<const amrex::Real> const& b,
                    amrex::Box const& bx,
                    int ncomp)
{
    const auto lo = amrex::lbound(bx);
    const auto hi = amrex::ubound(bx);
    double lerr{};
    for (int comp{0}; comp < ncomp; comp++)
    {
        for (int i{lo.x}; i <= hi.x; i++)
        {
            for (int j{lo.y}; j <= hi.y; j++)
            {
                for (int k{lo.z}; k <= hi.z; k++)
                {
                    const amrex::IntVect idx{AMREX_D_DECL(i, j, k)};
                    lerr = std::max(std::abs(*a.ptr(idx, comp) - *b.ptr(idx, comp)), lerr);
                }
            }
        }
    }
    return lerr;
}

/*
ComputationalDomain get_default_compdom ()
{
    const amrex::Array<amrex::Real, AMREX_SPACEDIM> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
    const amrex::Array<amrex::Real, AMREX_SPACEDIM> domainHi{
        AMREX_D_DECL(2 * M_PI, 2 * M_PI, 2 * M_PI)};
    const amrex::IntVect nCell{AMREX_D_DECL(16, 16, 16)};
    const amrex::IntVect maxGridSize{AMREX_D_DECL(8, 16, 16)};
    const amrex::Array<int, AMREX_SPACEDIM> isPeriodic{AMREX_D_DECL(1, 1, 1)};

    return ComputationalDomain(domainLo, domainHi, nCell, maxGridSize, isPeriodic,
                               amrex::CoordSys::cartesian);
}
*/
ComputationalDomain get_default_compdom ()
{
    const std::array<amrex::Real, AMREX_SPACEDIM> domainLo{
        AMREX_D_DECL(-M_PI + 0.3, -M_PI + 0.6, -M_PI + 0.4)};
    const std::array<amrex::Real, AMREX_SPACEDIM> domainHi{
        AMREX_D_DECL(M_PI + 0.3, M_PI + 0.6, M_PI + 0.4)};
    const amrex::IntVect nCell{AMREX_D_DECL(9, 11, 7)};
    const amrex::IntVect maxGridSize{AMREX_D_DECL(3, 4, 5)};
    const std::array<int, AMREX_SPACEDIM> isPeriodic{AMREX_D_DECL(1, 1, 1)};

    return ComputationalDomain(domainLo, domainHi, nCell, maxGridSize, isPeriodic);
}

ComputationalDomain get_compdom (int gSize)
{
    const std::array<amrex::Real, AMREX_SPACEDIM> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
    amrex::Real gLength{static_cast<amrex::Real>(gSize)};
    const std::array<amrex::Real, AMREX_SPACEDIM> domainHi{AMREX_D_DECL(gLength, gLength, gLength)};
    const amrex::IntVect nCell{AMREX_D_DECL(gSize, gSize, gSize)};
    const amrex::IntVect maxGridSize{AMREX_D_DECL(gSize, gSize, gSize)};
    const std::array<int, AMREX_SPACEDIM> isPeriodic{AMREX_D_DECL(1, 1, 1)};

    return ComputationalDomain(domainLo, domainHi, nCell, maxGridSize, isPeriodic);
}

ComputationalDomain get_compdom (const amrex::IntVect& nCell)
{
    const amrex::Array<amrex::Real, AMREX_SPACEDIM> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
    const amrex::Array<amrex::Real, AMREX_SPACEDIM> domainHi{
        AMREX_D_DECL(2 * M_PI, 2 * M_PI, 2 * M_PI)};
    const amrex::IntVect maxGridSize{AMREX_D_DECL(8, 8, 8)};
    const amrex::Array<int, AMREX_SPACEDIM> isPeriodic{AMREX_D_DECL(1, 1, 1)};

    return ComputationalDomain(domainLo, domainHi, nCell, maxGridSize, isPeriodic,
                               amrex::CoordSys::cartesian);
}
} // namespace Gempic::Test::Utils