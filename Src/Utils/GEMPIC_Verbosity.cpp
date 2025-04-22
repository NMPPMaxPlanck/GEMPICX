#include <iostream>

#include <AMReX_BLassert.H>
#include <AMReX_Print.H>

#include "GEMPIC_Config.H"
#include "GEMPIC_Verbosity.H"

namespace Gempic::Utils
{
void Verbosity::set_level (int level)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        s_settable, "Verbosity level cannot be set twice or after it's been used the first time!");
    if (level < 0)
    {
        level = 0;
        amrex::Warning("Negative Verbosity level corrected to 0");
    }
    s_verbosityLevel = level;
    s_settable = false;
    amrex::Print() << "new level: " << s_verbosityLevel << '\n';
}

Verbosity::Verbosity(int level) { set_level(level); }

int Verbosity::level ()
{
    s_settable = false;
    return s_verbosityLevel;
}
} // namespace Gempic::Utils