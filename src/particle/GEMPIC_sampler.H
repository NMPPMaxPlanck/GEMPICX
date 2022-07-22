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
#include <AMReX_Parser.H>
#include <AMReX_Particles.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_Random.H>
#include <AMReX_RealBox.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_particle_groups.H>

using namespace CompDom;
using namespace Particles;

namespace Gempic 
{
namespace Sampling 
{
template <int vdim, int numspec>
void init_particles_cellwise(
    computational_domain infra, particle_groups<vdim, numspec> &ppg,
    amrex::GpuArray<int, numspec> n_part_per_cell, amrex::Vector<amrex::Vector<amrex::Real>> Vmean,
    amrex::Vector<amrex::Vector<amrex::Real>> Vdev, amrex::Vector<amrex::Real> Vweight, int species,
    amrex::Real (*wave_function)(amrex::Real, amrex::Real, amrex::Real, amrex::Real),
    int rSeed = 925)
{
    amrex::Real x_c
#if GEMPIC_SPACEDIM > 1
        ,
        y_c
#endif
#if GEMPIC_SPACEDIM > 2
        ,
        z_c
#endif
        ;  // cell-corner coordinates
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
    amrex::GpuArray<amrex::Real, vdim> velocity;
    amrex::Real weight;
    constexpr amrex::Real epsilon = std::numeric_limits<amrex::Real>::epsilon();
    amrex::Real u1, u2, z0;

    int num_gauss = Vmean.size();

    // cumulative sum of vel_weights
    // amrex::GpuArray<std::vector<amrex::Real>,vdim> vel_cum;
    // for(int i=0;i<vdim;i++) {
    //     vel_cum[i].resize(num_gauss);
    //     std::partial_sum(Vweight[i].begin(), Vweight[i].end(), vel_cum[i].begin());
    // }
    // std::vector<amrex::Real> zeroVec(vdim,0);
    // std::vector<std::vector<amrex::Real>> vel_cum(num_gauss,zeroVec);
    amrex::Vector<amrex::Real> vel_cum(num_gauss);
    std::partial_sum(Vweight.begin(), Vweight.end(), vel_cum.begin());
    // for (int i = 0; i < vdim; i++)
    // {
    //     for (int j = 0; j < num_gauss; j++)
    //     {
    //         // vel_cum[i].resize(num_gauss);
    //         // std::partial_sum(Vweight[i].begin(), Vweight[i].end(), vel_cum[i].begin());
    //         vel_cum[j][i] = Vweight[j][i];
    //         for (int k = 0; k < j; k++)
    //         {
    //             vel_cum[j][i] += Vweight[k][i];
    //         }
    //     }
    // }

    // (0,1)-normally distributed random number generator:
    // srand(rSeed);
    std::mt19937 gen(rSeed);
    std::mt19937 genv(rSeed + 17);
    // std::normal_distribution<> norm01(0,1);
    std::uniform_real_distribution<> unif(0.0, 1.0);
    std::uniform_real_distribution<> unifv(0.0, 1.0);

    for (int pp = 0; pp < n_part_per_cell[species]; pp++)
    {
        // position in model cell [0,dx]x[0,dy]x[0,dz]:
        position[0] = unif(gen) * infra.dx[0];  // uniform in [0, dx]
#if (GEMPIC_SPACEDIM > 1)
        position[1] = unif(gen) * infra.dx[1];
#endif
#if (GEMPIC_SPACEDIM > 2)
        position[2] = unif(gen) * infra.dx[2];
#endif

        amrex::Real gauss_choice[vdim];
        amrex::Array<amrex::Real, vdim> gauss_idx;
        for (int i = 0; i < vdim; i++)
        {
            gauss_choice[i] = unif(gen);  // uniform in [0, 1]
            gauss_idx[i] = 0;
        }

        for (int i = 0; i < vdim; i++)
        {
            for (int gg = 1; gg < num_gauss; gg++)
            {
                if ((vel_cum[gg - 1] - gauss_choice[i] < 0) & (vel_cum[gg] - gauss_choice[i] >= 0))
                {
                    gauss_idx[i] = gg;
                }
            }
        }

        for (int cc = 0; cc < vdim; cc++)
        {
            do
            {
                u1 = unifv(genv);
                u2 = unifv(genv);
            } while (u1 < epsilon);  // make sure epsilon large enough to take log
            // first value from Box-Muller algorithm
            z0 = sqrt(-2.0 * log(u1)) * cos(2 * M_PI * u2);
            velocity[cc] = Vdev[gauss_idx[cc]][cc] * z0 + Vmean[gauss_idx[cc]][cc];
        }

        //Adding particle from the model cell to all cells
        for(amrex::MFIter mfi=(*(ppg[species]).mypc).MakeMFIter(0); mfi.isValid(); ++mfi) {

            amrex::ParticleTile<0, 0, vdim+1, 0>& particles = 
                (*(ppg[species]).mypc).GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
            const amrex::Box& bx = mfi.validbox();
            amrex::IntVect lo = {bx.smallEnd()};
            amrex::IntVect hi = {bx.bigEnd()};

            amrex::Real z = 0;
#if (GEMPIC_SPACEDIM > 2)
            for (int k = lo[2]; k <= hi[2]; k++)
            {
                z_c = infra.geom.ProbLo()[2] + (double)k * infra.dx[2];
                // shifted_position[2] = position[2] + z;
                amrex::Real z = position[2] + z_c;
#endif
#if (GEMPIC_SPACEDIM > 1)
                for (int j = lo[1]; j <= hi[1]; j++)
                {
                    y_c = infra.geom.ProbLo()[1] + (double)j * infra.dx[1];
                    // shifted_position[1] = position[1] + y;
                    amrex::Real y = position[1] + y_c;
#endif
                    for (int l = lo[0]; l <= hi[0]; l++)
                    {
                        x_c = infra.geom.ProbLo()[0] + (double)l * infra.dx[0];
                        // shifted_position[0] = position[0] + x;
                        amrex::Real x = position[0] + x_c;

                        // TODOrefine
                        // const double result = densityEval(density_parse);
                        amrex::Real result = wave_function(x, y, z, 0.);
                        weight = result / n_part_per_cell[species] * infra.dx[0]
#if (GEMPIC_SPACEDIM > 1)
                                 * infra.dx[1]
#endif
#if (GEMPIC_SPACEDIM > 2)
                                 * infra.dx[2]
#endif
                            ;

                        (ppg[species]).add_particle({AMREX_D_DECL(x, y, z)}, velocity, weight, particles);
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

template <int vdim, int numspec, typename functionType>
void init_particles_full_domain_gpu(computational_domain infra, particle_groups<vdim, numspec> &ppg,
                                amrex::GpuArray<int, numspec> n_part_per_cell,
                                amrex::Vector<amrex::Vector<amrex::Real>> Vmean,
                                amrex::Vector<amrex::Vector<amrex::Real>> Vdev,
                                amrex::Vector<amrex::Real> Vweight, int species,
                                functionType densityEval,
                                // amrex::ParserExecutor<4> densityEval,
                                int rSeed = 925)
{
    BL_PROFILE("init_particles_full_domain");

    constexpr amrex::Real epsilon = std::numeric_limits<amrex::Real>::epsilon();
    bool check_rnd = false;
    int num_gauss = Vmean.size();  // same size for all v-dims. Superposition of Gauss distributions

    // cumulative sum of vel_weights
    amrex::Vector<amrex::Real> vel_cum(num_gauss);
    std::partial_sum(Vweight.begin(), Vweight.end(), vel_cum.begin());

    //MFI that adds particle from the model cell to all cells
    for (amrex::MFIter mfi= (ppg[species].mypc)->MakeMFIter(0); mfi.isValid(); ++mfi )
    {
        const amrex::Box& bx = mfi.validbox();
        amrex::IntVect lo = {bx.smallEnd()};
        amrex::IntVect hi = {bx.bigEnd()};
        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> ProbLo = {
            infra.geom.ProbLo()[0], infra.geom.ProbLo()[1], infra.geom.ProbLo()[2]};
        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> dx;
        for (int i = 0; i < GEMPIC_SPACEDIM; i++)
        {
            dx[i] = infra.dx[i];
        }
        amrex::Array2D<amrex::Real, 0, MAXGAUSS, 0, vdim> VdevGpu, VmeanGpu;
        // actual number of Gaussians in velocity is smaller than maximum number set in
        // GEMPIC_Config.H
        assert(num_gauss < MAXGAUSS);
        for (int j = 0; j < vdim; j++)
        {
            for (int i = 0; i < num_gauss; i++)
            {
                VdevGpu(i, j) = Vdev[i][j];
                VmeanGpu(i, j) = Vmean[i][j];
            }
        }

        srand(rSeed + lo[0] + lo[1] * 10 + lo[2] * 100 + species * 1000);
        // std::mt19937 gen(rSeed + lo[0] + lo[1] * 10 + lo[2] * 100 + species * 1000);
        // std::mt19937 genv(rSeed + lo[0] + lo[1] * 10 + lo[2] * 100 + species * 1000 + 17);
        // std::uniform_real_distribution<> unif(0.0, 1.0);
        // std::uniform_real_distribution<> unifv(0.0, 1.0);
        amrex::RandomEngine gen;
        amrex::RandomEngine genv;
        amrex::InitRandom(rSeed + lo[0] + lo[1] * 10 + lo[2] * 100 + species * 1000);
        amrex::Random(gen);
        amrex::InitRandom(rSeed + lo[0] + lo[1] * 10 + lo[2] * 100 + species * 1000 + 17);
        amrex::Random(genv);

        auto &particles =
            (ppg.mypc[species])->GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
        // #if (GEMPIC_SPACEDIM > 2)
        //         for (int k = lo[2]; k <= hi[2]; k++)
        //         {
        // #endif
        // #if (GEMPIC_SPACEDIM > 1)
        //             for (int j = lo[1]; j <= hi[1]; j++)
        //             {
        // #endif
        //                 for (int l = lo[0]; l <= hi[0]; l++)
        ParallelFor(bx, [=](int l, int j, int k) {
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
            amrex::GpuArray<amrex::Real, vdim> velocity;
            amrex::Real x_c, y_c, z_c;  // cell-corner coordinates
            amrex::Real x, y, z;
            amrex::Real t = 0.0;  // initial time
            amrex::Real weight;
            int npart = 0;
            for (int pp = 0; pp < n_part_per_cell[species]; pp++)
            {
                for (int i = 0; i < GEMPIC_SPACEDIM; i++)
                {
                    // position[i] = ((amrex::Real) std::rand() / (RAND_MAX))*infra.dx[i];
                    position[i] = amrex::Random(gen) * dx[i];
                }
                amrex::Real gauss_choice[vdim];
                for (int i = 0; i < vdim; i++)
                {
                    // gauss_choice[i] = ((amrex::Real) std::rand() / (RAND_MAX));
                    gauss_choice[i] = amrex::Random(gen);
                }
                amrex::GpuArray<int, vdim> gauss_idx;
                for (int comp = 0; comp < vdim; comp++)
                {
                    gauss_idx[comp] = 0;
                }
                for (int i = 0; i < vdim; i++)
                {
                    for (int gg = 1; gg < num_gauss; gg++)
                    {
                        if ((vel_cum[gg - 1] - gauss_choice[i] < 0) &
                            (vel_cum[gg] - gauss_choice[i] >= 0))
                        {
                            gauss_idx[i] = gg;
                        }
                    }
                }
                amrex::Real u1, u2, z0;
                for (int cc = 0; cc < vdim; cc++)
                {
                    do
                    {
                        u1 = amrex::Random(genv);
                        u2 = amrex::Random(genv);
                    } while (u1 < epsilon);  // make sure epsilon large enough to take log
                    // first value from Box-Muller algorithm
                    z0 = sqrt(-2.0 * log(u1)) * cos(2 * M_PI * u2);
                    // velocity[cc] = Vdev[gauss_idx[cc]][cc] * z0 + Vmean[gauss_idx[cc]][cc];
                    velocity[cc] = VdevGpu(gauss_idx[cc], cc) * z0 + VmeanGpu(gauss_idx[cc], cc);
                }
#if (GEMPIC_SPACEDIM > 2)
                z_c = ProbLo[2] + (double)k * dx[2];
                z = position[2] + z_c;
#endif
#if (GEMPIC_SPACEDIM > 1)
                y_c = ProbLo[1] + (double)j * dx[1];
                y = position[1] + y_c;
#endif
                x_c = ProbLo[0] + (double)l * dx[0];
                x = position[0] + x_c;

                // parser // const double result = densityEval(density_parse);
                const double factor = densityEval(x, y, z, t);
                weight = (factor / n_part_per_cell[species]) * dx[0]
#if (GEMPIC_SPACEDIM > 1)
                         * dx[1]
#endif
#if (GEMPIC_SPACEDIM > 2)
                         * dx[2]
#endif
                    ;
                    // Doesn't work on GPU as a pushback is called in add_particle.
                    // The size of the new particle array should be determined before
                    // calling the parallel for and the index of the new particle in this
                    // array should be computable separately for each thread
                    // See Electromagnetic PIC tutorial in AMReX tutorials (InitParticles)
                ppg.add_particle({AMREX_D_DECL(x, y, z)}, velocity, weight, particles);
                // Check random number
                if (check_rnd)
                {
                    npart++;
                    if (npart < 10)
                        amrex::Print()
                            << npart << " " << x << " " << y << " " << z << " " << velocity[0]
                            << " " << velocity[1] << " " << velocity[2] << " " << weight << "\n";
                }
            }
            // #if (GEMPIC_SPACEDIM > 1)
            //                 }
            // #endif
            // #if (GEMPIC_SPACEDIM > 2)
            //             }
            // #endif
        });
    }
}

template <int vdim, int numspec, typename functionType>
void init_particles_full_domain(computational_domain infra, particle_groups<vdim, numspec> &ppg,
                                amrex::GpuArray<int, numspec> n_part_per_cell,
                                amrex::Vector<amrex::Vector<amrex::Real>> Vmean,
                                amrex::Vector<amrex::Vector<amrex::Real>> Vdev,
                                amrex::Vector<amrex::Real> Vweight, int species,
                                functionType densityEval, int rSeed = 925)
{
    BL_PROFILE("init_particles_full_domain (..., amrex::Real (*wave_function)");
    double x, y, z, t=0.0;

    amrex::Real x_c, y_c, z_c;  // cell-corner coordinates
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
    amrex::GpuArray<amrex::Real, vdim> velocity;
    amrex::Real weight;
    constexpr amrex::Real epsilon = std::numeric_limits<amrex::Real>::epsilon();
    amrex::Real u1, u2, z0;
    bool check_rnd = false;
    int npart = 0;
    int num_gauss = Vmean.size();

    // cumulative sum of vel_weights
    amrex::Vector<amrex::Real> vel_cum(num_gauss);
    std::partial_sum(Vweight.begin(), Vweight.end(), vel_cum.begin());

    // MFI that adds particle from the model cell to all cells
    for (amrex::MFIter mfi = (*(ppg).mypc[species]).MakeMFIter(0); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        amrex::IntVect lo = {bx.smallEnd()};
        amrex::IntVect hi = {bx.bigEnd()};

        srand(rSeed + lo[0] + lo[1] * 10 + lo[2] * 100 + species * 1000);
        std::mt19937 gen(rSeed + lo[0] + lo[1] * 10 + lo[2] * 100 + species * 1000);
        std::mt19937 genv(rSeed + lo[0] + lo[1] * 10 + lo[2] * 100 + species * 1000 + 17);
        std::uniform_real_distribution<> unif(0.0, 1.0);
        std::uniform_real_distribution<> unifv(0.0, 1.0);

        auto &particles =
            (ppg.mypc[species])->GetParticles(0)[std::make_pair(mfi.index(),
            mfi.LocalTileIndex())];
#if (GEMPIC_SPACEDIM > 2)
        for (int k = lo[2]; k <= hi[2]; k++)
        {
#endif
#if (GEMPIC_SPACEDIM > 1)
            for (int j = lo[1]; j <= hi[1]; j++)
            {
#endif
                for (int l = lo[0]; l <= hi[0]; l++)
                {
                    for (int pp = 0; pp < n_part_per_cell[species]; pp++)
                    {
                        for (int i = 0; i < GEMPIC_SPACEDIM; i++)
                        {
                            position[i] = unif(gen) * infra.dx[i];
                        }
                        amrex::Real gauss_choice[vdim];
                        for (int i = 0; i < vdim; i++)
                        {
                            gauss_choice[i] = unif(gen);
                        }
                        amrex::GpuArray<int, vdim> gauss_idx;
                        for (int comp = 0; comp < vdim; comp++)
                        {
                            gauss_idx[comp] = 0;
                        }
                        for (int i = 0; i < vdim; i++)
                        {
                            for (int gg = 1; gg < num_gauss; gg++)
                            {
                                if ((vel_cum[gg - 1] - gauss_choice[i] < 0) &
                                    (vel_cum[gg] - gauss_choice[i] >= 0))
                                {
                                    gauss_idx[i] = gg;
                                }
                            }
                        }
                        for (int cc = 0; cc < vdim; cc++)
                        {
                            do
                            {
                                u1 = unifv(genv);
                                u2 = unifv(genv);
                            } while (u1 < epsilon);  // make sure epsilon large enough to take log
                            // first value from Box-Muller algorithm
                            z0 = sqrt(-2.0 * log(u1)) * cos(2 * M_PI * u2);
                            velocity[cc] = Vdev[gauss_idx[cc]][cc] * z0 + Vmean[gauss_idx[cc]][cc];
                        }
#if (GEMPIC_SPACEDIM > 2)
                        z_c = infra.geom.ProbLo()[2] + (double)k * infra.dx[2];
                        z = position[2] + z_c;
#endif
#if (GEMPIC_SPACEDIM > 1)
                        y_c = infra.geom.ProbLo()[1] + (double)j * infra.dx[1];
                        y = position[1] + y_c;
#endif
                        x_c = infra.geom.ProbLo()[0] + (double)l * infra.dx[0];
                        x = position[0] + x_c;

                        const double factor = densityEval(x, y, z, t);
                        weight = factor / n_part_per_cell[species] * infra.dx[0]
#if (GEMPIC_SPACEDIM > 1)
                                 * infra.dx[1]
#endif
#if (GEMPIC_SPACEDIM > 2)
                                 * infra.dx[2]
#endif
                            ;
                        ppg.add_particle({AMREX_D_DECL(x, y, z)}, velocity, weight, particles);
                        // Check random number
                        if (check_rnd)
                        {
                            npart++;
                            if (npart < 10)
                                amrex::Print() << npart << " " << x << " " << y << " " << z << " "
                                               << velocity[0] << " " << velocity[1] << " "
                                               << velocity[2] << " " << weight << "\n";
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
}

template <int vdim>
void init_one_particle_cellwise(amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> const& dx,
                                amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> const& plo,
                                amrex::ParticleContainer<0, 0, vdim+1, 0> *mypc,
                                amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> position)
{
    amrex::Real x,y,z;
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> shifted_position;
    amrex::GpuArray<amrex::Real, vdim> velocity;

    velocity[0] = 1.;
    if (vdim > 1)
    {
        velocity[1] = 1.;
    }
    if (vdim > 2)
    {
        velocity[2] = 1.;
    }

    // MFI that adds particle from the model cell to all cells
    for (amrex::MFIter mfi = (*mypc).MakeMFIter(0); mfi.isValid(); ++mfi)
    {  // 0 is level
        const amrex::Box &bx = mfi.validbox();
        amrex::IntVect lo = {bx.smallEnd()};
        amrex::IntVect hi = {bx.bigEnd()};

        //using ParticleType = amrex::Particle<vdim+1, 0>; // Particle template
        using ParticleType = amrex::Particle<0, 0>; // Particle template
        auto& particles = (*mypc).GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

#if (GEMPIC_SPACEDIM > 2)
        for (int k = lo[2]; k <= hi[2]; k++)
        {
            z = plo[2] + (double)k * dx[2];
            shifted_position[2] = position[2] + z;
#endif
#if (GEMPIC_SPACEDIM > 1)
            for (int j = lo[1]; j <= hi[1]; j++)
            {
                y = plo[1] + (double)j * dx[1];
                shifted_position[1] = position[1] + y;
#endif
                for (int l = lo[0]; l <= hi[0]; l++)
                {
                    x = plo[0] + (double)l * dx[0];
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

                    //p.rdata(0) = velocity[0];
                    particles.push_back_real(0,velocity[0]);
                    if (vdim > 1) {
                        //p.rdata(1) = velocity[1];
                        particles.push_back_real(1,velocity[1]);
                    }
                    if (vdim > 2) {
                        //p.rdata(2) = velocity[2];
                        particles.push_back_real(2,velocity[2]);
                    }

                    //p.rdata(vdim) = 1.0/64.0; //weight
                    particles.push_back_real(vdim,1.0/64.0);//weight

                    particles.push_back(p);
                }
#if (GEMPIC_SPACEDIM > 1)
            }
#endif
#if (GEMPIC_SPACEDIM > 2)
        }

#endif
    }
}
}  // namespace Sampling
}  // namespace Gempic

#endif
