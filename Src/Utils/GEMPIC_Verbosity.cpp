#include <iostream>

#include "GEMPIC_Config.H"
#include "GEMPIC_Verbosity.H"

namespace Gempic::Utils
{
void Verbosity::set_level (int level)
{
    if (s_settable)
    {
        if (level < 0)
        {
            level = 0;
            std::cerr << "Warning: Negative Verbosity level corrected to 0\n";
        }
        s_verbosityLevel = level;
        s_settable = false;
    }
    else
    {
        std::cerr
            << "Verbosity level cannot be set twice or after it's been used the first time!\n";
        std::exit(Error::VerbosityAlreadySet);
    }
}

Verbosity::Verbosity(int level) { set_level(level); }

int Verbosity::level ()
{
    s_settable = false;
    return s_verbosityLevel;
}
}  // namespace Gempic::Utils