#include <gtest/gtest.h>

#include "GEMPIC_TestUtils.H"

namespace Gempic::Test::Utils
{
using condLambda = bool (*)(AMREX_D_DECL(int, int, int));
void check_field (char const file[],
                  int line,
                  amrex::Array4<amrex::Real> const& fieldArr,
                  amrex::Box const& bx,
                  std::vector<condLambda>&& condVec,
                  std::vector<amrex::Real>&& checks,
                  std::optional<amrex::Real> defCheck,
                  amrex::Real tol)
{
    auto const lo = amrex::lbound(bx);
    auto const hi = amrex::ubound(bx);
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
                    amrex::IntVect const idx{AMREX_D_DECL(i, j, k)};
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
void compare_fields (char const file[],
                     int line,
                     amrex::Array4<amrex::Real const> const& fieldArr,
                     amrex::Array4<amrex::Real const> const& fieldArr2,
                     amrex::Box const& bx,
                     amrex::Real tol,
                     int ncomp)
{
    auto const lo = amrex::lbound(bx);
    auto const hi = amrex::ubound(bx);
    for (int comp{0}; comp < ncomp; comp++)
    {
        for (int i{lo.x}; i <= hi.x; i++)
        {
            for (int j{lo.y}; j <= hi.y; j++)
            {
                for (int k{lo.z}; k <= hi.z; k++)
                {
                    amrex::IntVect const idx{AMREX_D_DECL(i, j, k)};
                    EXPECT_NEAR(*fieldArr.ptr(idx, comp), *fieldArr2.ptr(idx, comp), tol)
                        << file << ":" << line
                        << ": Unequal arrays.\nIndices: " << string_array(idx, AMREX_SPACEDIM)
                        << "\tComponent: " << comp;
                }
            }
        }
    }
}
double l_inf_error (amrex::Array4<amrex::Real const> const& a,
                    amrex::Array4<amrex::Real const> const& b,
                    amrex::Box const& bx,
                    int ncomp)
{
    auto const lo = amrex::lbound(bx);
    auto const hi = amrex::ubound(bx);
    double lerr{};
    for (int comp{0}; comp < ncomp; comp++)
    {
        for (int i{lo.x}; i <= hi.x; i++)
        {
            for (int j{lo.y}; j <= hi.y; j++)
            {
                for (int k{lo.z}; k <= hi.z; k++)
                {
                    amrex::IntVect const idx{AMREX_D_DECL(i, j, k)};
                    lerr = std::max(std::abs(*a.ptr(idx, comp) - *b.ptr(idx, comp)), lerr);
                }
            }
        }
    }
    return lerr;
}

ComputationalDomain get_default_compdom ()
{
    std::array<amrex::Real, AMREX_SPACEDIM> const domainLo{
        AMREX_D_DECL(-M_PI + 0.3, -M_PI + 0.6, -M_PI + 0.4)};
    std::array<amrex::Real, AMREX_SPACEDIM> const domainHi{
        AMREX_D_DECL(M_PI + 0.3, M_PI + 0.6, M_PI + 0.4)};
    amrex::IntVect const nCell{AMREX_D_DECL(9, 11, 7)};
    amrex::IntVect const maxGridSize{AMREX_D_DECL(3, 4, 5)};
    std::array<int, AMREX_SPACEDIM> const isPeriodic{AMREX_D_DECL(1, 1, 1)};

    return ComputationalDomain(domainLo, domainHi, nCell, maxGridSize, isPeriodic);
}

ComputationalDomain get_compdom (int gSize)
{
    std::array<amrex::Real, AMREX_SPACEDIM> const domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
    amrex::Real gLength{static_cast<amrex::Real>(gSize)};
    std::array<amrex::Real, AMREX_SPACEDIM> const domainHi{AMREX_D_DECL(gLength, gLength, gLength)};
    amrex::IntVect const nCell{AMREX_D_DECL(gSize, gSize, gSize)};
    amrex::IntVect const maxGridSize{AMREX_D_DECL(gSize, gSize, gSize)};
    std::array<int, AMREX_SPACEDIM> const isPeriodic{AMREX_D_DECL(1, 1, 1)};

    return ComputationalDomain(domainLo, domainHi, nCell, maxGridSize, isPeriodic);
}

ComputationalDomain get_compdom (amrex::IntVect const& nCell)
{
    amrex::Array<amrex::Real, AMREX_SPACEDIM> const domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
    amrex::Array<amrex::Real, AMREX_SPACEDIM> const domainHi{
        AMREX_D_DECL(2 * M_PI, 2 * M_PI, 2 * M_PI)};
    amrex::IntVect const maxGridSize{AMREX_D_DECL(8, 8, 8)};
    amrex::Array<int, AMREX_SPACEDIM> const isPeriodic{AMREX_D_DECL(1, 1, 1)};

    return ComputationalDomain(domainLo, domainHi, nCell, maxGridSize, isPeriodic,
                               amrex::CoordSys::cartesian);
}
} // namespace Gempic::Test::Utils