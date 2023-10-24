#include "gtest/gtest.h"
#include "GEMPIC_test_utils.H"

namespace GEMPIC_TestUtils {
/* Helper function to check entries of a field given a series of conditions and a default
    * value. Check order is prioritized, so a set of indices only fulfill the first succesful
    * condition.
    * 
    * Parameters:
    * ----------
    * @param line: int, the line from which the function was called
    * @param fieldArr: amrex::Array4, array containing field values in an easily reached accessor
    * @param top: Dim3, top boundaries of box for fieldArray
    * @param condVec: vector<condLambda>, Vector of lambdas that check if the {SPACEDIM} indices fulfill a given condition.
    * @param checks: vector<amrex::Real>, Vector of values to compare to if indices fulfill the corresponding condVec condition.
    * @param defCheck: amrex::Real, Default value for all indices not fulfilling any of the given conditions.
    */
using condLambda = bool(*)(AMREX_D_DECL(int, int, int));
void checkField(const char file[], int line,
                amrex::Array4<amrex::Real> const& fieldArr,
                amrex::Dim3 const&& top,
                std::vector<condLambda>&& condVec,
                std::vector<amrex::Real>&& checks,
                amrex::Real defCheck) {
    for (int i{0}; i <= top.x; i++) { 
        for (int j{0}; j <= top.y; j++) {
            for (int k{0}; k <= top.z; k++) {
                int condNum{0};
                const amrex::IntVect idx{AMREX_D_DECL(i, j, k)};
                for (auto cond : condVec) {
                    if (cond(AMREX_D_DECL(i, j, k))) {
                        EXPECT_NEAR(checks[condNum], *fieldArr.ptr(idx, 0), 1e-14) <<
                            file << ":" << line << ": Failed condition " << condNum <<
                            ".\nIndices: " << stringArray(idx, GEMPIC_SPACEDIM);
                            break;
                    }
                    condNum++;
                }
                if (condNum == condVec.size()) {
                    EXPECT_NEAR(defCheck, *fieldArr.ptr(idx, 0), 1e-14) <<
                        file << ":" << line << ": Failed default value check:" << defCheck <<
                        ".\nIndices: " << stringArray(idx, GEMPIC_SPACEDIM);
                }
            }
        }
    }
}
} // namespace GEMPIC_TestUtils