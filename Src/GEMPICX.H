#ifndef _GEMPICX_HEADER_
#define _GEMPICX_HEADER_

#include <AMReX.H>

#include "GEMPIC_AmrexInit.H"
#include "GEMPIC_Config.H"

// Making sure all namespaces are defined so they can be included in the gempicx namespace
namespace Gempic
{
namespace FieldSolvers
{
}
namespace Filter
{
}
namespace Forms
{
}
namespace Io
{
}
namespace Particle
{
}
namespace ParticleMeshCoupling
{
}
namespace TimeLoop
{
}
namespace Utils
{
}
} //namespace Gempic

namespace Gempicx
{
using namespace Gempic;
using namespace FieldSolvers;
using namespace Filter;
using namespace Forms;
using namespace Io;
using namespace Particle;
using namespace ParticleMeshCoupling;
using namespace TimeLoop;
using namespace Utils;
} //namespace Gempicx
#endif
