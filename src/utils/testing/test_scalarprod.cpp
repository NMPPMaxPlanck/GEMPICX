#include <tinyexpr.h>

#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Particles.H>

#include <GEMPIC_Config.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_sampler.H>
#include <GEMPIC_vlasov_maxwell.H>
#include <GEMPIC_particle_groups.H>

using namespace std;
using namespace amrex;
using namespace Gempic;

using namespace Field_solvers;
using namespace Particles;
using namespace Sampling;
using namespace Vlasov_Maxwell;

template<int vdim, int numspec, int degx, int degy, int degz>
void main_main ()
{
    const int degmw = 2;
    bool ctest = true;
    vlasov_maxwell<vdim, numspec> VlMa;
    VlMa.init_Nghost(degx, degy, degz);
    VlMa.set_params();
    if (int(vdim/2.5)*2+1 < 3) {
        VlMa.Bx = VlMa.Bz;
        VlMa.Bz = "0.0";
    }
    std::array<std::string, int(vdim/2.5)*2+1> fields_B;
    fields_B[0] = VlMa.Bx;
    if (int(vdim/2.5)*2+1 > 1) {
        fields_B[1] = VlMa.By;
    }
    if (int(vdim/2.5)*2+1 > 1) {
        fields_B[2] = VlMa.Bz;
    }
    if (GEMPIC_SPACEDIM==1 & vdim==1) {
        // For 1D1V change parameters to make a Landau damping
        VlMa.sim_name = "Landau";
        VlMa.n_part_per_cell = {10000};
        VlMa.k = {0.5,0.5,0.5};
        VlMa.WF = "1.0 + 0.5 * cos(kvarx * x)";
        VlMa.Bz = "0.0";
    }
    VlMa.n_steps = 10;
    VlMa.set_computed_params();

    // ------------------------------------------------------------------------------
    // ------------INITIALIZE GEMPIC-STRUCTURES--------------------------------------

    // infrastructure
    infrastructure infra;
    VlMa.initialize_infrastructure(&infra);

    // maxwell_yee
    maxwell_yee<vdim> mw_yee(VlMa, infra);
    (mw_yee).template initB<degmw>(fields_B, VlMa.k, infra);

    amrex::Real ScalarProd = amrex::MultiFab::Dot(*mw_yee.B_Masks[2], *mw_yee.B_Array[2], 0, *mw_yee.B_Array[2], 0, 1, VlMa.Nghost);
    amrex::Real NormSquared = pow((*mw_yee.B_Array[2]).norm2(0, infra.geom.periodicity()),2.);
    amrex::Real Norm = (*mw_yee.B_Array[2]).norm2(0, infra.geom.periodicity());
    std::cout << "<x,x> = " << ScalarProd << std::endl;
    std::cout << "||x||2 = " << NormSquared << std::endl;
    std::cout << "||x|| = " << Norm << std::endl;
    std::cout << "<x,x>-||x||2 = " << ScalarProd-NormSquared << std::endl;

}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main<3, 1, 1, 1, 1>();

    amrex::Finalize();
}



