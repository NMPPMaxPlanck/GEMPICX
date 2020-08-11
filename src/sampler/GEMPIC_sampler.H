#include <tinyexpr.h>

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_CoordSys.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_Particles.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>

#include <GEMPIC_infrastructure.H>
#include <GEMPIC_particle_groups.H>

using namespace Infra;
using namespace Particles;

#ifndef SAMPLER_H_
#define SAMPLER_H_

namespace Gempic {
namespace Sampling {

void init_particles_cellwise(infrastructure infra,
                             particle_groups &ppg,
                             initializer init,
                             int species,
                             te_expr *WF_parse,
                             double *x,
                             double *y,
                             double *z,
                             int rSeed = 925){

    amrex::Real x_c,y_c,z_c; // cell-corner coordinates
    std::array<amrex::Real,GEMPIC_SPACEDIM> position;
    std::array<amrex::Real,GEMPIC_VDIM> velocity;
    amrex::Real weight;

    int num_gauss = init.Vmean[0].size();

    srand(rSeed);
    // cumulative sum of vel_weights
    std::array<std::vector<amrex::Real>,GEMPIC_VDIM> vel_cum;
    for(int i=0;i<GEMPIC_VDIM;i++) {
        vel_cum[i].resize(num_gauss);
        std::partial_sum(init.Vweight[i].begin(), init.Vweight[i].end(), vel_cum[i].begin());
    }

    // (0,1)-normally distributed random number generator:
    std::mt19937 gen(rSeed);
    std::normal_distribution<> norm01(0,1);
    std::uniform_real_distribution<> unif(0.0, 1.0);

    for (int pp=0;pp<init.Np_cell;pp++) {
        //position in model cell [0,dx]x[0,dy]x[0,dz]:
        position[0] = unif(gen)*infra.dx[0]; // uniform in [0, dx]
#if (GEMPIC_SPACEDIM > 1)
        position[1] = unif(gen)*infra.dx[1];
#endif
#if (GEMPIC_SPACEDIM > 2)
        position[2] = unif(gen)*infra.dx[2];
#endif

        amrex::Real gauss_choice[GEMPIC_VDIM];
        for(int i=0;i<GEMPIC_VDIM;i++){
            gauss_choice[i] = ((amrex::Real) std::rand() / (RAND_MAX)); // uniform in [0, 1]
        }
        std::array<int,GEMPIC_VDIM> gauss_idx = {AMREX_V_DECL(0,0,0)};
        for(int i=0;i<GEMPIC_VDIM;i++){
            for (int gg=1;gg<num_gauss;gg++){
                if (vel_cum[i][gg-1]-gauss_choice[i] < 0 & vel_cum[i][gg]-gauss_choice[i] >= 0){
                    gauss_idx[i] = gg;
                }
            }
        }

        for (int cc = 0; cc < GEMPIC_VDIM; cc++) {
            velocity[cc] = init.Vdev[cc][gauss_idx[cc]]*norm01(gen)+init.Vmean[cc][gauss_idx[cc]];
        }

        //Adding particle from the model cell to all cells
        for(amrex::MFIter mfi=(*(ppg).mypc[species]).MakeMFIter(0); mfi.isValid(); ++mfi) {

                using ParticleType = amrex::Particle<GEMPIC_VDIM+1, 0>; // Particle template
                amrex::ParticleTile<4, 0, 0, 0>& particles = (*(ppg).mypc[species]).GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
                const amrex::Box& bx = mfi.validbox();
                amrex::IntVect lo = {bx.smallEnd()};
                amrex::IntVect hi = {bx.bigEnd()};

    #if (GEMPIC_SPACEDIM > 2)
                for(int k=lo[2]; k<=hi[2]; k++){
                    z_c = infra.geom.ProbLo()[2] + (double)k*infra.dx[2];
                    //shifted_position[2] = position[2] + z;
                    *z = position[2] + z_c;
    #endif
    #if (GEMPIC_SPACEDIM > 1)
                    for(int j=lo[1]; j<=hi[1]; j++){
                        y_c = infra.geom.ProbLo()[1] + (double)j*infra.dx[1];
                        //shifted_position[1] = position[1] + y;
                        *y = position[1] + y_c;
    #endif
                        for(int l=lo[0]; l<=hi[0]; l++){
                            x_c = infra.geom.ProbLo()[0] + (double)l*infra.dx[0];
                            //shifted_position[0] = position[0] + x;
                            *x = position[0] + x_c;

                            //TODOrefine
                            const double result = te_eval(WF_parse);
                            weight = result/init.Np_cell*infra.dx[0]
            #if (GEMPIC_SPACEDIM > 1)
                                    *infra.dx[1]
            #endif
            #if (GEMPIC_SPACEDIM > 2)
                                    *infra.dx[2]
            #endif
                                    ;


                            (ppg).add_particle({*x, *y, *z}, velocity, weight, particles);
                        }
    #if (GEMPIC_SPACEDIM > 1)
                    }
    #endif
    #if (GEMPIC_SPACEDIM > 2)
                }
    #endif
        }
    }
}

void init_particles_full_domain(infrastructure infra,
                                particle_groups &ppg,
                                initializer init,
                                int species,
                                te_expr *WF_parse,
                                double *x,
                                double *y,
                                double *z,
                                int rSeed = 925){
    amrex::Real x_c,y_c,z_c; // cell-corner coordinates
    std::array<amrex::Real,GEMPIC_SPACEDIM> position;
    std::array<amrex::Real,GEMPIC_VDIM> velocity;
    amrex::Real weight;

    int num_gauss = init.Vmean[0].size();

    // cumulative sum of vel_weights
    std::array<std::vector<amrex::Real>,GEMPIC_VDIM> vel_cum;
    for(int i=0;i<GEMPIC_VDIM;i++) {
        vel_cum[i].resize(num_gauss);
        std::partial_sum(init.Vweight[i].begin(), init.Vweight[i].end(), vel_cum[i].begin());
    }

    // (0,1)-normally distributed random number generator:




    //MFI that adds particle from the model cell to all cells
    for (amrex::MFIter mfi= (*(ppg).mypc[species]).MakeMFIter(0); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.validbox();
        amrex::IntVect lo = {bx.smallEnd()};
        amrex::IntVect hi = {bx.bigEnd()};

        srand(rSeed+lo[0]+lo[1]*10+lo[2]*100);
        std::mt19937 gen(rSeed);
        std::normal_distribution<> norm01(0,1);



        using ParticleType = amrex::Particle<GEMPIC_VDIM+1, 0>; // Particle template
        auto& particles = (*(ppg).mypc[species]).GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

#if (GEMPIC_SPACEDIM > 2)
        for(int k=lo[2]; k<=hi[2]; k++){

#endif
#if (GEMPIC_SPACEDIM > 1)
            for(int j=lo[1]; j<=hi[1]; j++){

#endif
                for(int l=lo[0]; l<=hi[0]; l++){

                    for (int pp=0;pp<init.Np_cell;pp++) {
                        for (int i = 0; i < GEMPIC_SPACEDIM; i++) {
                            position[i] = ((amrex::Real) std::rand() / (RAND_MAX))*infra.dx[i];
                        }

                        amrex::Real gauss_choice[GEMPIC_VDIM];
                        for(int i=0;i<GEMPIC_VDIM;i++){
                            gauss_choice[i] = ((amrex::Real) std::rand() / (RAND_MAX));
                        }
                        std::array<int,GEMPIC_VDIM> gauss_idx = {AMREX_V_DECL(0,0,0)};
                        for(int i=0;i<GEMPIC_VDIM;i++){
                            for (int gg=1;gg<num_gauss;gg++){
                                if (vel_cum[i][gg-1]-gauss_choice[i] < 0 & vel_cum[i][gg]-gauss_choice[i] >= 0){
                                    gauss_idx[i] = gg;
                                }
                            }
                        }

                        for (int cc = 0; cc < GEMPIC_VDIM; cc++) {
                            velocity[cc] = init.Vdev[cc][gauss_idx[cc]]*norm01(gen)+init.Vmean[cc][gauss_idx[cc]];
                        }

                        z_c = infra.geom.ProbLo()[2] + (double)k*infra.dx[2];
                        *z = position[2] + z_c;

                        y_c = infra.geom.ProbLo()[1] + (double)j*infra.dx[1];
                        *y = position[1] + y_c;

                        x_c = infra.geom.ProbLo()[0] + (double)l*infra.dx[0];
                        *x = position[0] + x_c;

                        const double result = te_eval(WF_parse);
                        weight = result/init.Np_cell*infra.dx[0]
        #if (GEMPIC_SPACEDIM > 1)
                                *infra.dx[1]
        #endif
        #if (GEMPIC_SPACEDIM > 2)
                                *infra.dx[2]
        #endif
                                ;
                        (ppg).add_particle({*x, *y, *z}, velocity, weight, particles);
                    }
#if (GEMPIC_SPACEDIM > 1)
                }
#endif
#if (GEMPIC_SPACEDIM > 2)
            }
#endif
        }
    }
}

void init_one_particle_cellwise(infrastructure infra,
                                particle_groups *ppg,
                                initializer init,
                                int species){
    amrex::Real x,y,z;
    std::array<amrex::Real,GEMPIC_SPACEDIM> position;
    std::array<amrex::Real,GEMPIC_SPACEDIM> shifted_position;
    std::array<amrex::Real,GEMPIC_VDIM> velocity;
    amrex::Real weight;

    //position in model cell [0,dx]x[0,dy]x[0,dz]:
    position[0] = 0.5*infra.dx[0];
#if (GEMPIC_SPACEDIM > 1)
    position[1] = 0.5*infra.dx[1];
#endif
#if (GEMPIC_SPACEDIM > 2)
    position[2] = 0.5*infra.dx[2];
#endif

    velocity[0] = 0.;
#if (GEMPIC_VDIM > 1)
    velocity[1] = 0.;
#endif
#if (GEMPIC_VDIM > 2)
    velocity[2] = 0.;
#endif

    //MFI that adds particle from the model cell to all cells
    for (amrex::MFIter mfi = (*(*ppg).mypc[species]).MakeMFIter(0); mfi.isValid(); ++mfi ){ // 0 is level
        const amrex::Box& bx = mfi.validbox();
        amrex::IntVect lo = {bx.smallEnd()};
        amrex::IntVect hi = {bx.bigEnd()};

        using ParticleType = amrex::Particle<GEMPIC_VDIM+1, 0>; // Particle template
        auto& particles = (*(*ppg).mypc[species]).GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

#if (GEMPIC_SPACEDIM > 2)
        for(int k=lo[2]; k<=hi[2]; k++){
            z = infra.geom.ProbLo()[2] + (double)k*infra.dx[2];
            shifted_position[2] = position[2] + z;
#endif
#if (GEMPIC_SPACEDIM > 1)
            for(int j=lo[1]; j<=hi[1]; j++){
                y = infra.geom.ProbLo()[1] + (double)j*infra.dx[1];
                shifted_position[1] = position[1] + y;
#endif
                for(int l=lo[0]; l<=hi[0]; l++){
                    x = infra.geom.ProbLo()[0] + (double)l*infra.dx[0];
                    shifted_position[0] = position[0] + x;

                    weight = 1.; //DIM
                    (*ppg).add_particle(shifted_position, velocity, weight, particles);
                }
#if (GEMPIC_SPACEDIM > 1)
            }
#endif
#if (GEMPIC_SPACEDIM > 2)
        }

#endif
    }

}
}
}

#endif
