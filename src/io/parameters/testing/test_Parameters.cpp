#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>
#include <GEMPIC_Params.H>

int main(int argc, char *argv[])
{
    amrex::Initialize(argc, argv);

    const amrex::RealBox realBox({AMREX_D_DECL(-1.0,-1.0,-1.0)},{AMREX_D_DECL( 1.0, 1.0, 1.0)});
    const amrex::IntVect nCell = {AMREX_D_DECL(10, 10, 10)}; 
    const amrex::IntVect maxGridSize = {AMREX_D_DECL(4, 4, 4)};
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic = {1, 1, 1};
    const int hodgeDegree = 2;

    Parameters params(realBox, nCell, maxGridSize, isPeriodic, hodgeDegree);

    amrex::Print() << params.grid() << std::endl;
    amrex::Finalize();
}
