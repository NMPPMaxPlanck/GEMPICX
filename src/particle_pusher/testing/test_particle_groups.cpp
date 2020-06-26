#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include <GEMPIC_Config.H>
#include <GEMPIC_initializer.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_mesh_coupling.H>

using namespace std;
using namespace amrex;
using namespace Gempic;

using namespace Field_solvers;
using namespace Init;
using namespace Particles;

double B_x(std::array<double,GEMPIC_SPACEDIM> x,double k){return(0);}
#if (GEMPIC_BDIM > 1)
double B_y(std::array<double,GEMPIC_SPACEDIM> x,double k){return(0);}
#endif
#if (GEMPIC_BDIM > 2)
double B_z(std::array<double,GEMPIC_SPACEDIM> x,double k){return(0);}
#endif

void main_main ()
{
    //------------------------------------------------------------------------------
    //build objects:

    double (*initB[GEMPIC_BDIM]) (std::array<double,GEMPIC_SPACEDIM> x,double k);
    initB[0] = B_x;
#if (GEMPIC_BDIM > 1)
    initB[1] = B_y;
#endif
#if (GEMPIC_BDIM > 2)
    initB[2] = B_z;
#endif

    //initializer
    initializer init;
    amrex::IntVect is_periodic(AMREX_D_DECL(1,1,1));
    amrex::IntVect n_cell(AMREX_D_DECL(8,8,8));

    std::array<std::vector<amrex::Real>, GEMPIC_VDIM> VM{};
    std::array<std::vector<amrex::Real>, GEMPIC_VDIM> VD{};
    std::array<std::vector<amrex::Real>, GEMPIC_VDIM> VW{};

    VM[0].push_back(0.0);
    VD[0].push_back(1.0);
    VW[0].push_back(1.0);
#if (GEMPIC_VDIM > 1)
    VM[1].push_back(0.0);
    VD[1].push_back(1.0);
    VW[1].push_back(1.0);
#endif
#if (GEMPIC_VDIM > 2)
    VM[2].push_back(0.0);
    VD[2].push_back(1.0);
    VW[2].push_back(1.0);
#endif

    init.initialize_from_parameters(n_cell,4,is_periodic,1,0.01,0,{1.0},{1.0},1000,0.5,VM,VD,VW);

    // infrastructure
    infrastructure infra(init);

    // maxwell_yee
    //maxwell_yee mw_yee(init, infra, init.Nghost);

    // particles
    particle_groups part_gr(init, infra);

    //------------------------------------------------------------------------------
    // initialize particles:
    std::array<amrex::Real,GEMPIC_VDIM> velocity = {0.,0.,0.};
    amrex::Real weight = 1.;
    int species = 0;
    std::array<amrex::Real,GEMPIC_SPACEDIM> position;

    std::ofstream ofs("test_particle_groups.output", std::ofstream::out);
    amrex::Print(ofs) << endl;

    for (amrex::MFIter mfi= (*(part_gr).mypc[species]).MakeMFIter(0); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.validbox();
        amrex::IntVect lo = {bx.smallEnd()};
        amrex::IntVect hi = {bx.bigEnd()};

        using ParticleType = amrex::Particle<GEMPIC_VDIM+1, 0>; // Particle template
        auto& particles = (*(part_gr).mypc[species]).GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

#if (GEMPIC_SPACEDIM > 2)
        for(int k=lo[2]; k<=hi[2]; k++){
            position[2] = infra.geom.ProbLo()[2] + ((double)k+0.5)*infra.dx[2];
#endif
#if (GEMPIC_SPACEDIM > 1)
            for(int j=lo[1]; j<=hi[1]; j++){
                position[1] = infra.geom.ProbLo()[1] + ((double)j+0.5)*infra.dx[1];
#endif
                for(int l=lo[0]; l<=hi[0]; l++){
                    position[0] = infra.geom.ProbLo()[0] + ((double)l+0.5)*infra.dx[0];
                    amrex::Print(ofs) << position[0] << "|" << position[1] << "|" << position[2] << std::endl;

                    part_gr.add_particle(position, velocity, weight, particles);
                }
#if (GEMPIC_SPACEDIM > 1)
            }
#endif
#if (GEMPIC_SPACEDIM > 2)
        }
#endif
    }

    //check positions
    for (amrex::ParIter<GEMPIC_VDIM+1,0,0,0> pti(*part_gr.mypc[species], 0); pti.isValid(); ++pti) {

        const auto& particles = pti.GetArrayOfStructs();
        const long np = pti.numParticles();
        for (int pp=0;pp<np;pp++) {
            amrex::Print(ofs) << particles[pp].pos(0) << "," <<
             #if (GEMPIC_SPACEDIM > 1)
                         particles[pp].pos(1) << "," <<
             #endif
             #if (GEMPIC_SPACEDIM > 2)
                         particles[pp].pos(2) <<
             #endif
                         std::endl;
        }
    }
    ofs.close();

}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main();

    amrex::Finalize();
}

