#include <gtest/gtest.h>

#include "GEMPIC_TestUtils.H"

namespace Gempic::Test::Utils
{
/* Helper function to check entries of a field given a series of conditions and a default
 * value. Check order is prioritized, so a set of indices only fulfill the first successful
 * condition.
 *
 * Parameters:
 * ----------
 * @param line: int, the line from which the function was called
 * @param fieldArr: amrex::Array4, array containing field values in an easily reached accessor
 * @param top: Dim3, top boundaries of box for fieldArray
 * @param condVec: vector<condLambda>, Vector of lambdas that check if the {SPACEDIM} indices
 * fulfill a given condition.
 * @param checks: vector<amrex::Real>, Vector of values to compare to if indices fulfill the
 * corresponding condVec condition.
 * @param defCheck: amrex::Real, Default value for all indices not fulfilling any of the given
 * conditions.
 */
using condLambda = bool (*)(AMREX_D_DECL(int, int, int));
void check_field (const char file[],
                  int line,
                  amrex::Array4<amrex::Real> const& fieldArr,
                  amrex::Dim3 const&& top,
                  std::vector<condLambda>&& condVec,
                  std::vector<amrex::Real>&& checks,
                  std::optional<amrex::Real> defCheck,
                  amrex::Real tol)
{
    for (int i{0}; i <= top.x; i++)
    {
        for (int j{0}; j <= top.y; j++)
        {
            for (int k{0}; k <= top.z; k++)
            {
                int condNum{0};
                const amrex::IntVect idx{AMREX_D_DECL(i, j, k)};
                for (auto cond : condVec)
                {
                    if (cond(AMREX_D_DECL(i, j, k)))
                    {
                        EXPECT_NEAR(checks[condNum], *fieldArr.ptr(idx, 0), tol)
                            << file << ":" << line << ": Failed condition " << condNum
                            << ".\nIndices: " << string_array(idx, GEMPIC_SPACEDIM);
                        break;
                    }
                    condNum++;
                }
                if (condNum == condVec.size() && defCheck)
                {
                    EXPECT_NEAR(defCheck.value(), *fieldArr.ptr(idx, 0), tol)
                        << file << ":" << line
                        << ": Failed default value check: " << defCheck.value()
                        << ".\nIndices: " << string_array(idx, GEMPIC_SPACEDIM);
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
                        << ": Unequal arrays.\nIndices: " << string_array(idx, GEMPIC_SPACEDIM)
                        << "\tComponent: " << comp;
                    break;
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

}  // namespace Gempic::Test::Utils