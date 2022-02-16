#ifndef SAMPLER_H_
#define SAMPLER_H_

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_CoordSys.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_Particles.H>
#include <AMReX_Parser.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>

#include <GEMPIC_computational_domain.H>
#include <GEMPIC_particle_groups.H>

using namespace CompDom;
using namespace Particles;

namespace Gempic {
namespace Sampling {

template<int vdim, int numspec>
void init_particles_cellwise(computational_domain infra,
                             particle_groups<vdim, numspec> &ppg,
                             std::array<int, numspec> n_part_per_cell,
                             std::vector<std::vector<amrex::Real>> Vmean,
                             std::vector<std::vector<amrex::Real>> Vdev,
                             std::vector<std::vector<amrex::Real>> Vweight,
                             int species,
                             amrex::Real (*wave_function)(amrex::Real, amrex::Real, amrex::Real),
                             int rSeed = 925){

    amrex::Real x_c
        #if GEMPIC_SPACEDIM > 1
            ,y_c
        #endif
        #if GEMPIC_SPACEDIM > 2
            ,z_c
        #endif
            ; // cell-corner coordinates
    amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> position;
    amrex::GpuArray<amrex::Real,vdim> velocity;
    amrex::Real weight;
    constexpr amrex::Real epsilon = std::numeric_limits<amrex::Real>::epsilon();
    amrex::Real u1, u2, z0;

    //int num_gauss = Vmean[0].size();
    int num_gauss = Vmean.size();

    // cumulative sum of vel_weights
    // amrex::GpuArray<std::vector<amrex::Real>,vdim> vel_cum;
    // for(int i=0;i<vdim;i++) {
    //     vel_cum[i].resize(num_gauss);
    //     std::partial_sum(Vweight[i].begin(), Vweight[i].end(), vel_cum[i].begin());
    // }
    std::vector<amrex::Real> zeroVec(vdim,0); 
    std::vector<std::vector<amrex::Real>> vel_cum(num_gauss,zeroVec);
    for (int i = 0; i < vdim; i++)
    {
        for (int j = 0; j < num_gauss; j++)
        {
            // vel_cum[i].resize(num_gauss);
            // std::partial_sum(Vweight[i].begin(), Vweight[i].end(), vel_cum[i].begin());
            vel_cum[j][i] = Vweight[j][i];
            for (int k = 0; k < j; k++)
            {
                vel_cum[j][i] += Vweight[k][i];
            }
        }
    }
    

    // (0,1)-normally distributed random number generator:
    //srand(rSeed);
    std::mt19937 gen(rSeed);
    std::mt19937 genv(rSeed+17);
    //std::normal_distribution<> norm01(0,1);
    std::uniform_real_distribution<> unif(0.0, 1.0);
    std::uniform_real_distribution<> unifv(0.0, 1.0);


    for (int pp=0;pp<n_part_per_cell[species];pp++) {
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
            gauss_choice[i] = unif(gen); // uniform in [0, 1]
            gauss_idx[i] = 0;
        }

        for(int i=0;i<vdim;i++){
            for (int gg=1;gg<num_gauss;gg++){
                if ((vel_cum[gg-1][i]-gauss_choice[i] < 0) & (vel_cum[gg][i]-gauss_choice[i] >= 0)){
                    gauss_idx[i] = gg;
                }
            }
        }

        for (int cc = 0; cc < vdim; cc++) {
            do {
                u1 = unifv(genv);
                u2 = unifv(genv);
            }
            while (u1 < epsilon); // make sure epsilon large enough to take log
            // first value from Box-Muller algorithm
            z0  = sqrt(-2.0 * log(u1)) * cos(2* M_PI * u2);
            velocity[cc] = Vdev[gauss_idx[cc]][cc] * z0 + Vmean[gauss_idx[cc]][cc];                          
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
                amrex::Real z = position[2] + z_c;
#endif
#if (GEMPIC_SPACEDIM > 1)
                for(int j=lo[1]; j<=hi[1]; j++){
                    y_c = infra.geom.ProbLo()[1] + (double)j*infra.dx[1];
                    //shifted_position[1] = position[1] + y;
                    amrex::Real y = position[1] + y_c;
#endif
                    for(int l=lo[0]; l<=hi[0]; l++){
                        x_c = infra.geom.ProbLo()[0] + (double)l*infra.dx[0];
                        //shifted_position[0] = position[0] + x;
                        amrex::Real x = position[0] + x_c;

                        //TODOrefine
                        //const double result = WF_eval(WF_parse);
                        amrex::Real result = wave_function(x,y,z);
                        weight = result/n_part_per_cell[species]*infra.dx[0]
        #if (GEMPIC_SPACEDIM > 1)
                                *infra.dx[1]
        #endif
        #if (GEMPIC_SPACEDIM > 2)
                                *infra.dx[2]
        #endif
                                ;


                        (ppg).add_particle({AMREX_D_DECL(x, y, z)}, velocity, weight, particles);
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
void init_particles_full_domain(computational_domain infra,
                                particle_groups<vdim, numspec> &ppg,
                                std::array<int, numspec> n_part_per_cell,
                                amrex::Vector<amrex::Real> k_vec,
                                std::string wave_function,
                                std::vector<std::vector<amrex::Real>> Vmean,
                                std::vector<std::vector<amrex::Real>> Vdev,
                                std::vector<std::vector<amrex::Real>> Vweight,
                                int species,
                                int rSeed = 925){

    double x, y, z;
    double kx = k_vec[0];
    double ky = k_vec[1];
    double kz = k_vec[2];
    
    amrex::Parser parser(wave_function);
    parser.setConstant("kvarx", kx);
    parser.setConstant("kvary", ky);
    parser.setConstant("kvarz", kz);
    parser.registerVariables({"x","y","z"});
    auto WF_eval = parser.compile<3>();  // 3 because there are three variables.
  
    amrex::Real x_c,y_c,z_c; // cell-corner coordinates
    amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> position;
    amrex::GpuArray<amrex::Real,vdim> velocity;
    amrex::Real weight;
    constexpr amrex::Real epsilon = std::numeric_limits<amrex::Real>::epsilon();
    amrex::Real u1, u2, z0;
    bool check_rnd = false;
    int npart = 0;
    int num_gauss = Vmean.size(); // same size for all v-dims. Superposition of Gauss distributions 

    // cumulative sum of vel_weights
    std::vector<amrex::Real> zeroVec(vdim,0); 
    std::vector<std::vector<amrex::Real>> vel_cum(num_gauss,zeroVec);
    //amrex::Print() << "vel_cum " << vel_cum << "\n";
    for (int i = 0; i < vdim; i++)
    {
        for (int j = 0; j < num_gauss; j++)
        {
            // vel_cum[i].resize(num_gauss);
            // std::partial_sum(Vweight[i].begin(), Vweight[i].end(), vel_cum[i].begin());
            vel_cum[j][i] = Vweight[j][i];
            for (int k = 0; k < j; k++)
            {
                vel_cum[j][i] += Vweight[k][i];
            }
        }
    }

    //MFI that adds particle from the model cell to all cells
    for (amrex::MFIter mfi= (ppg.mypc[species])->MakeMFIter(0); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.validbox();
        amrex::IntVect lo = {bx.smallEnd()};
        amrex::IntVect hi = {bx.bigEnd()};

        srand(rSeed+lo[0]+lo[1]*10+lo[2]*100+species*1000);
        std::mt19937 gen(rSeed+lo[0]+lo[1]*10+lo[2]*100+species*1000);
        std::mt19937 genv(rSeed+lo[0]+lo[1]*10+lo[2]*100+species*1000 + 17); // different from above
        //std::normal_distribution<> norm01(0,1);
        std::uniform_real_distribution<> unif(0.0,1.0);
        std::uniform_real_distribution<> unifv(0.0, 1.0);

        auto& particles = (ppg.mypc[species])->GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())]; // only level 0

#if (GEMPIC_SPACEDIM > 2)
        for(int k=lo[2]; k<=hi[2]; k++){

#endif
#if (GEMPIC_SPACEDIM > 1)
            for(int j=lo[1]; j<=hi[1]; j++){

#endif
                for(int l=lo[0]; l<=hi[0]; l++){

                    for (int pp=0;pp<n_part_per_cell[species];pp++) {
                        for (int i = 0; i < GEMPIC_SPACEDIM; i++) {
                           //position[i] = ((amrex::Real) std::rand() / (RAND_MAX))*infra.dx[i];
                           position[i] = unif(gen)*infra.dx[i];
                        }
                        amrex::Real gauss_choice[vdim];
                        for(int i=0;i<vdim;i++){
                            //gauss_choice[i] = ((amrex::Real) std::rand() / (RAND_MAX));
                            gauss_choice[i] = unif(gen);
                        }
                        amrex::GpuArray<int,vdim> gauss_idx;
                        for (int comp = 0; comp < vdim; comp++) {
                            gauss_idx[comp] = 0;
                        }
                        for(int i=0;i<vdim;i++){
                            for (int gg=1;gg<num_gauss;gg++) {
                                if ((vel_cum[gg-1][i]-gauss_choice[i] < 0) & (vel_cum[gg][i]-gauss_choice[i] >= 0)){
                                    gauss_idx[i] = gg;
                                }
                            }
                        }
                        for (int cc = 0; cc < vdim; cc++) {
                            do {
                                u1 = unifv(genv);
                                u2 = unifv(genv);
                            }
                            while (u1 < epsilon); // make sure epsilon large enough to take log
                            // first value from Box-Muller algorithm
                            z0  = sqrt(-2.0 * log(u1)) * cos(2* M_PI * u2);
                            velocity[cc] = Vdev[gauss_idx[cc]][cc] * z0 + Vmean[gauss_idx[cc]][cc];
                            //velocity[cc] = Vdev[cc][gauss_idx[cc]]*norm01(gen)+Vmean[cc][gauss_idx[cc]];
                        }
#if (GEMPIC_SPACEDIM > 2)
                        z_c = infra.geom.ProbLo()[2] + (double)k*infra.dx[2];
                        z = position[2] + z_c;
#endif
#if (GEMPIC_SPACEDIM > 1)
                        y_c = infra.geom.ProbLo()[1] + (double)j*infra.dx[1];
                        y = position[1] + y_c;
#endif
                        x_c = infra.geom.ProbLo()[0] + (double)l*infra.dx[0];
                        x = position[0] + x_c;

                        // parser // const double result = WF_eval(WF_parse);
                        const double factor = WF_eval(x,y,z);
                        weight = (factor/n_part_per_cell[species]) * infra.dx[0]
        #if (GEMPIC_SPACEDIM > 1)
                                *infra.dx[1]
        #endif
        #if (GEMPIC_SPACEDIM > 2)
                                *infra.dx[2]
        #endif
                                ;
                        ppg.add_particle({AMREX_D_DECL(x, y, z)}, velocity, weight, particles);
                        // Check random number
                        if (check_rnd) {
                            npart++;
                            if (npart <10 ) amrex::Print() << npart << " " << x << " "<< y << " " << z << " " << velocity[0] << " " << velocity[1] 
                            << " " << velocity[2] << " " << weight << "\n";
                        }
                    }
#if (GEMPIC_SPACEDIM > 1)
                }
#endif
#if (GEMPIC_SPACEDIM > 2)
            }
#endif
        }
    }
    //parser te_free(WF_parse);
}

template<int vdim, int numspec>
void init_particles_full_domain(computational_domain infra,
                                particle_groups<vdim, numspec> &ppg,
                                std::array<int, numspec> n_part_per_cell,
                                std::vector<std::vector<amrex::Real>> Vmean,
                                std::vector<std::vector<amrex::Real>> Vdev,
                                std::vector<std::vector<amrex::Real>> Vweight,
                                int species,
                                amrex::Real (*wave_function)(amrex::Real, amrex::Real, amrex::Real),
                                int rSeed = 925){

    double x, y, z;

    amrex::Real x_c,y_c,z_c; // cell-corner coordinates
    amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> position;
    amrex::GpuArray<amrex::Real,vdim> velocity;
    amrex::Real weight;
    constexpr amrex::Real epsilon = std::numeric_limits<amrex::Real>::epsilon();
    amrex::Real u1, u2, z0;
    //int num_gauss = Vmean[0].size();
int num_gauss = Vmean.size();

    // cumulative sum of vel_weights
    // amrex::GpuArray<std::vector<amrex::Real>,vdim> vel_cum;
    // for(int i=0;i<vdim;i++) {
    //     vel_cum[i].resize(num_gauss);
    //     std::partial_sum(Vweight[i].begin(), Vweight[i].end(), vel_cum[i].begin());
    // }
    std::vector<amrex::Real> zeroVec(vdim,0); 
    std::vector<std::vector<amrex::Real>> vel_cum(num_gauss,zeroVec);
    for (int i = 0; i < vdim; i++)
    {
        for (int j = 0; j < num_gauss; j++)
        {
            // vel_cum[i].resize(num_gauss);
            // std::partial_sum(Vweight[i].begin(), Vweight[i].end(), vel_cum[i].begin());
            vel_cum[j][i] = Vweight[j][i];
            for (int k = 0; k < j; k++)
            {
                vel_cum[j][i] += Vweight[k][i];
            }
        }
    }

   //MFI that adds particle from the model cell to all cells
    for (amrex::MFIter mfi= (*(ppg).mypc[species]).MakeMFIter(0); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.validbox();
        amrex::IntVect lo = {bx.smallEnd()};
        amrex::IntVect hi = {bx.bigEnd()};

        srand(rSeed+lo[0]+lo[1]*10+lo[2]*100+species*1000);
        std::mt19937 gen(rSeed+lo[0]+lo[1]*10+lo[2]*100+species*1000);
        std::mt19937 genv(rSeed+lo[0]+lo[1]*10+lo[2]*100+species*1000 + 17);
        std::uniform_real_distribution<> unif(0.0, 1.0);
        std::uniform_real_distribution<> unifv(0.0,1.0);

        auto& particles = (*ppg.mypc[species]).GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

#if (GEMPIC_SPACEDIM > 2)
        for(int k=lo[2]; k<=hi[2]; k++){

#endif
#if (GEMPIC_SPACEDIM > 1)
            for(int j=lo[1]; j<=hi[1]; j++){

#endif
                for(int l=lo[0]; l<=hi[0]; l++){

                    for (int pp=0;pp<n_part_per_cell[species];pp++) {
                        for (int i = 0; i < GEMPIC_SPACEDIM; i++) {
                            //position[i] = ((amrex::Real) std::rand() / (RAND_MAX))*infra.dx[i];
                            position[i] = unif(gen)*infra.dx[i];
                        }
                        amrex::Real gauss_choice[vdim];
                        for(int i=0;i<vdim;i++){
                            //gauss_choice[i] = ((amrex::Real) std::rand() / (RAND_MAX));
                            gauss_choice[i] = unif(gen);
                        }
                        amrex::GpuArray<int,vdim> gauss_idx;
                        for (int comp = 0; comp < vdim; comp++) {
                            gauss_idx[comp] = 0;
                        }
                        for(int i=0;i<vdim;i++){
                            for (int gg=1;gg<num_gauss;gg++){
                                if ((vel_cum[gg-1][i]-gauss_choice[i] < 0) & (vel_cum[gg][i]-gauss_choice[i] >= 0)){
                                    gauss_idx[i] = gg;
                                }
                            }
                        }
                        for (int cc = 0; cc < vdim; cc++) {
                            do {
                                u1 = unifv(genv);
                                u2 = unifv(genv);
                            }
                            while (u1 < epsilon); // make sure epsilon large enough to take log
                            // first value from Box-Muller algorithm
                            z0  = sqrt(-2.0 * log(u1)) * cos(2* M_PI * u2);
                            velocity[cc] = Vdev[gauss_idx[cc]][cc] * z0 + Vmean[gauss_idx[cc]][cc];   
                            // velocity[cc] = Vdev[cc][gauss_idx[cc]]*norm01(genv)+Vmean[cc][gauss_idx[cc]]; 
                        }
#if (GEMPIC_SPACEDIM > 2)
                        z_c = infra.geom.ProbLo()[2] + (double)k*infra.dx[2];
                        z = position[2] + z_c;
#endif
#if (GEMPIC_SPACEDIM > 1)
                        y_c = infra.geom.ProbLo()[1] + (double)j*infra.dx[1];
                        y = position[1] + y_c;
#endif
                        x_c = infra.geom.ProbLo()[0] + (double)l*infra.dx[0];
                        x = position[0] + x_c;

                        const double result = wave_function(x,y,z);
                        weight = result/n_part_per_cell[species]*infra.dx[0]
        #if (GEMPIC_SPACEDIM > 1)
                                *infra.dx[1]
        #endif
        #if (GEMPIC_SPACEDIM > 2)
                                *infra.dx[2]
        #endif
                                ;
                        ppg.add_particle({AMREX_D_DECL(x, y, z)}, velocity, weight, particles);
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
void init_one_particle_cellwise(amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> const& dx,
                                amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> const& plo,
                                amrex::ParticleContainer<3+1, 0, 0, 0> *mypc,
                                amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> position){
    amrex::Real x,y,z;
    amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> shifted_position;
    amrex::GpuArray<amrex::Real,vdim> velocity;

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
