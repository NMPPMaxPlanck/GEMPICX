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

template<int vdim, int numspec>
void init_particles_cellwise(infrastructure infra,
                             particle_groups<vdim, numspec> &ppg,
                             initializer<vdim, numspec> init,
                             int species,
                             te_expr *WF_parse,
                             double *x,
                             double *y,
                             double *z,
                             int rSeed = 925){

    amrex::Real x_c
        #if GEMPIC_SPACEDIM > 1
            ,y_c
        #endif
        #if GEMPIC_SPACEDIM > 2
            ,z_c
        #endif
            ; // cell-corner coordinates
    std::array<amrex::Real,GEMPIC_SPACEDIM> position;
    std::array<amrex::Real,vdim> velocity;
    amrex::Real weight;

    int num_gauss = init.Vmean[0].size();

    srand(rSeed);
    // cumulative sum of vel_weights
    std::array<std::vector<amrex::Real>,vdim> vel_cum;
    for(int i=0;i<vdim;i++) {
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

        amrex::Real gauss_choice[vdim];
        std::array<amrex::Real,vdim> gauss_idx;
        for(int i=0;i<vdim;i++){
            gauss_choice[i] = ((amrex::Real) std::rand() / (RAND_MAX)); // uniform in [0, 1]
        }

        for(int i=0;i<vdim;i++){
            for (int gg=1;gg<num_gauss;gg++){
                if ((vel_cum[i][gg-1]-gauss_choice[i] < 0) & (vel_cum[i][gg]-gauss_choice[i] >= 0)){
                    gauss_idx[i] = gg;
                }
            }
        }

        for (int cc = 0; cc < vdim; cc++) {
            velocity[cc] = init.Vdev[cc][gauss_idx[cc]]*norm01(gen)+init.Vmean[cc][gauss_idx[cc]];
        }

        //Adding particle from the model cell to all cells
        for(amrex::MFIter mfi=(*(ppg).mypc[species]).MakeMFIter(0); mfi.isValid(); ++mfi) {

            amrex::ParticleTile<vdim+1, 0, 0, 0>& particles = (*(ppg).mypc[species]).GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
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


                        (ppg).add_particle({AMREX_D_DECL(*x, *y, *z)}, velocity, weight, particles);
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

template<int vdim, int numspec>
void init_particles_full_domain(infrastructure infra,
                                particle_groups<vdim, numspec> &ppg,
                                initializer<vdim, numspec> init,
                                int species,
                                te_expr *WF_parse,
                                double *x,
                                double *y,
                                double *z,
                                int rSeed = 925){
    amrex::Real x_c,y_c,z_c; // cell-corner coordinates
    std::array<amrex::Real,GEMPIC_SPACEDIM> position;
    std::array<amrex::Real,vdim> velocity;
    amrex::Real weight;

    int num_gauss = init.Vmean[0].size();

    // cumulative sum of vel_weights
    std::array<std::vector<amrex::Real>,vdim> vel_cum;
    for(int i=0;i<vdim;i++) {
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

                        amrex::Real gauss_choice[vdim];
                        for(int i=0;i<vdim;i++){
                            gauss_choice[i] = ((amrex::Real) std::rand() / (RAND_MAX));
                        }
                        std::array<amrex::Real,vdim> gauss_idx;
                        switch (vdim) {
                        case 1:
                            gauss_idx = {0};
                            break;
                        case 2:
                            gauss_idx = {0,0};
                            break;
                        case 3:
                            gauss_idx = {0,0,0};
                            break;
                        }
                        for(int i=0;i<vdim;i++){
                            for (int gg=1;gg<num_gauss;gg++){
                                if ((vel_cum[i][gg-1]-gauss_choice[i] < 0) & (vel_cum[i][gg]-gauss_choice[i] >= 0)){
                                    gauss_idx[i] = gg;
                                }
                            }
                        }

                        for (int cc = 0; cc < vdim; cc++) {
                            velocity[cc] = init.Vdev[cc][gauss_idx[cc]]*norm01(gen)+init.Vmean[cc][gauss_idx[cc]];
                        }

#if (GEMPIC_SPACEDIM > 2)
                        z_c = infra.geom.ProbLo()[2] + (double)k*infra.dx[2];
                        *z = position[2] + z_c;
#endif
#if (GEMPIC_SPACEDIM > 1)
                        y_c = infra.geom.ProbLo()[1] + (double)j*infra.dx[1];
                        *y = position[1] + y_c;
#endif
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
                        (ppg).add_particle({AMREX_D_DECL(*x, *y, *z)}, velocity, weight, particles);
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

template <int vdim>
void init_one_particle_cellwise(std::array<amrex::Real,GEMPIC_SPACEDIM> const& dx,
                                std::array<amrex::Real,GEMPIC_SPACEDIM> const& plo,
                                amrex::ParticleContainer<3+1, 0, 0, 0> *mypc,
                                std::array<amrex::Real,GEMPIC_SPACEDIM> position){
    amrex::Real x,y,z;
    std::array<amrex::Real,GEMPIC_SPACEDIM> shifted_position;
    std::array<amrex::Real,vdim> velocity;

    velocity[0] = 1.;
    if (vdim > 1) {
        velocity[1] = 1.;
    }
    if (vdim > 2) {
        velocity[2] = 1.;
    }

    //MFI that adds particle from the model cell to all cells
    for (amrex::MFIter mfi = (*mypc).MakeMFIter(0); mfi.isValid(); ++mfi ){ // 0 is level
        const amrex::Box& bx = mfi.validbox();
        amrex::IntVect lo = {bx.smallEnd()};
        amrex::IntVect hi = {bx.bigEnd()};

        using ParticleType = amrex::Particle<vdim+1, 0>; // Particle template
        auto& particles = (*mypc).GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

#if (GEMPIC_SPACEDIM > 2)
        for(int k=lo[2]; k<=hi[2]; k++){
            z = plo[2] + (double)k*dx[2];
            shifted_position[2] = position[2] + z;
#endif
#if (GEMPIC_SPACEDIM > 1)
            for(int j=lo[1]; j<=hi[1]; j++){
                y = plo[1] + (double)j*dx[1];
                shifted_position[1] = position[1] + y;
#endif
                for(int l=lo[0]; l<=hi[0]; l++){

                    x = plo[0] + (double)l*dx[0];
                    shifted_position[0] = position[0] + x;

                    ParticleType p;
                    p.id() = ParticleType::NextID();
                    p.cpu() = amrex::ParallelDescriptor::MyProc();

                    p.pos(0) = shifted_position[0];
#if (GEMPIC_SPACEDIM > 1)
                    p.pos(1) = shifted_position[1];
#endif
#if (GEMPIC_SPACEDIM > 2)
                    p.pos(2) = shifted_position[2];
#endif

                    p.rdata(0) = velocity[0];
                    if (vdim > 1) {
                        p.rdata(1) = velocity[1];
                    }
                    if (vdim > 2) {
                        p.rdata(2) = velocity[2];
                    }

                    p.rdata(vdim) = 1.0/64.0; //weight

                    (particles).push_back(p);

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
