
#include <AMReX.H>
#include <AMReX_Print.H>
#include <particle_groups.H>
#include <maxwell_yee.H>
#include <particle_mesh_coupling.H>
#include <sampler.H>
#include <initializer.H>
#include <time_loop.H>
#include <time_loop_avg.H>
#include <gempic_Config.H>

#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Particles.H>

using namespace std;
using namespace amrex;

double WF (std::array<double,GEMPIC_SPACEDIM> x, std::array<double,GEMPIC_VDIM> v,double k) {
    double alpha = 0.5;
    return((1.0 + alpha*cos(k*x[0])));
};

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
    init.initialize_from_parameters(n_cell,4,is_periodic,1,0.01,0,{1.0},{1.0},1000,0.5,
    {0.0},{1.0},{1.0},WF);
    
    // infrastructure
    infrastructure infra(init);

    // maxwell_yee
    maxwell_yee mw_yee(init, infra, init.Nghost);

    // particles
    particle_groups part_gr(init, infra);

    //------------------------------------------------------------------------------
    // initialize particles:
    //part_gr.add_particle({AMREX_D_DECL(0.5,0.0,0.0)}, {AMREX_V_DECL(0.0,0.0,0.0)}, 0.25);
    //part_gr.add_particle({AMREX_D_DECL(2.0,0.5,0.0)}, {AMREX_V_DECL(0.0,0.0,0.0)}, 0.25);
    //part_gr.add_particle({AMREX_D_DECL(7.0,0.5,0.5)}, {AMREX_V_DECL(0.0,0.0,0.0)}, 0.25);
    // //part_gr.add_particle({AMREX_D_DECL(10.996,10.996,10.996)}, {AMREX_D_DECL(0.0,0.0,0.0)}, 0.25);
    //part_gr.add_particle({AMREX_D_DECL(12.56637,12.56637,12.56637)}, {AMREX_V_DECL(0.0,0.0,0.0)}, 0.25);
    // //part_gr.add_particle({AMREX_D_DECL(0.5*12.56637,0.5*12.56637,0.5*12.56637)}, {AMREX_D_DECL(0.0,0.0,0.0)}, 0.25);

    std::array<amrex::Real,GEMPIC_VDIM> velocity = {0.,0.,0.};
    amrex::Real weight = 1.;
    int species = 1;
    std::array<amrex::Real,GEMPIC_SPACEDIM> position;

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
                    std::cout << "a" << std::endl;
                    std::cout << position[0] << "|" << position[1] << "|" << position[2] << std::endl;

                    part_gr.add_particle(position, velocity, weight, particles);
                    std::cout << "b" << std::endl;
                }
#if (GEMPIC_SPACEDIM > 1)
            }
#endif
#if (GEMPIC_SPACEDIM > 2)
        }
#endif
    }

    //check positions
    for (amrex::ParIter<GEMPIC_VDIM+1,0,0,0> pti(*part_gr.mypc[0], 0); pti.isValid(); ++pti) {

        const auto& particles = pti.GetArrayOfStructs();
        const long np = pti.numParticles();
        for (int pp=0;pp<np;pp++) {
            std::cout << particles[pp].pos(0) << "," <<
             #if (GEMPIC_SPACEDIM > 1)
                         particles[pp].pos(1) << "," <<
             #endif
             #if (GEMPIC_SPACEDIM > 2)
                         particles[pp].pos(2) <<
             #endif
                         std::endl;
        }
    }

    //------------------------------------------------------------------------------
    // solve:
    //time_loop_avg(infra, &mw_yee, &part_gr, initB, init.k);
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main();

    amrex::Finalize();
}



