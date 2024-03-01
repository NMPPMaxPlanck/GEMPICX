#ifndef SAMPLER_H_
#define SAMPLER_H_

#include <cassert>

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

#include "GEMPIC_Config.H"
#include "GEMPIC_FunctionParse.H"
#include "GEMPIC_computational_domain.H"
#include "GEMPIC_particle_groups.H"

namespace Gempic
{
namespace Sampling
{
using namespace CompDom;
using namespace Particles;

using ParticleType = amrex::Particle<0, 0>;  // For SoA attributes
template <unsigned int vdim, unsigned int numspec, unsigned int ndata>
void init_particles_cellwise (
    ComputationalDomain infra,
    amrex::GpuArray<std::shared_ptr<ParticleGroups<vdim, ndata>>, numspec>& ppg,
    int species,
    int rSeed = 925)
{
    BL_PROFILE("init_particles_cellwise_new_params");
    amrex::Real AMREX_D_DECL(x_c, y_c, z_c);  // cell-corner coordinates
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
    amrex::GpuArray<amrex::Real, vdim> velocity;
    amrex::Real weight;
    constexpr amrex::Real epsilon = std::numeric_limits<amrex::Real>::epsilon();
    amrex::Real u1, u2, z0;

    std::string specString = "particle.species" + std::to_string(species);
    Parameters params(specString);

    int nPartPerCell;
    params.get("n_part_per_cell", nPartPerCell);
    auto [densityParse, densityEval] = Utils::parse_function("density", params);

    int numGauss{1};
    params.get_or_set("num_gaussians", numGauss);
    amrex::Vector<amrex::Vector<amrex::Real>> vmean, vdev;
    amrex::Vector<amrex::Real> vweight;
    for (int ng = 0; ng < numGauss; ng++)
    {
        amrex::Vector<amrex::Real> vm, vt;
        std::string vmString = "vMean_g" + std::to_string(ng);
        params.get(vmString, vm);
        vmean.push_back(vm);
        std::string vtString = "vThermal_g" + std::to_string(ng);
        params.get(vtString, vt);
        vdev.push_back(vt);
        std::string vwString = "vWeight_g" + std::to_string(ng);
        amrex::Real vw;
        params.get(vwString, vw);
        vweight.push_back(vw);
    }

    // cumulative sum of vel_weights
    // amrex::GpuArray<std::vector<amrex::Real>,vdim> vel_cum;
    // for(int i=0;i<vdim;i++) {
    //     vel_cum[i].resize(num_gauss);
    //     std::partial_sum(Vweight[i].begin(), Vweight[i].end(), vel_cum[i].begin());
    // }
    // std::vector<amrex::Real> zeroVec(vdim,0);
    // std::vector<std::vector<amrex::Real>> vel_cum(num_gauss,zeroVec);
    amrex::Vector<amrex::Real> velCum(numGauss);
    std::partial_sum(vweight.begin(), vweight.end(), velCum.begin());
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

    for (int pp = 0; pp < nPartPerCell; pp++)
    {
        // position in model cell [0,dx]x[0,dy]x[0,dz]:
        AMREX_D_TERM(position[xDir] = unif(gen) * infra.m_dx[xDir];
                     ,  // uniform in [0, dx]
                     position[yDir] = unif(gen) * infra.m_dx[yDir];
                     , position[zDir] = unif(gen) * infra.m_dx[zDir];)

        amrex::Real gaussChoice[vdim];
        amrex::Array<amrex::Real, vdim> gaussIdx;
        for (int i = 0; i < vdim; i++)
        {
            gaussChoice[i] = unif(gen);  // uniform in [0, 1]
            gaussIdx[i] = 0;
        }

        for (int i = 0; i < vdim; i++)
        {
            for (int gg = 1; gg < numGauss; gg++)
            {
                if ((velCum[gg - 1] - gaussChoice[i] < 0) && (velCum[gg] - gaussChoice[i] >= 0))
                {
                    gaussIdx[i] = gg;
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
            velocity[cc] = vdev[gaussIdx[cc]][cc] * z0 + vmean[gaussIdx[cc]][cc];
        }

        // Adding particle from the model cell to all cells
        for (amrex::MFIter mfi = ppg[species]->MakeMFIter(0); mfi.isValid(); ++mfi)
        {
            amrex::ParticleTile<ParticleType, vdim + 1, 0>& particles =
                ppg[species]->GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
            const amrex::Box& bx = mfi.validbox();
            amrex::IntVect lo = {bx.smallEnd()};
            amrex::IntVect hi = {bx.bigEnd()};

#if (GEMPIC_SPACEDIM > 2)
            for (int k = lo[zDir]; k <= hi[zDir]; k++)
            {
                z_c = infra.m_geom.ProbLo()[zDir] + (double)k * infra.m_dx[zDir];
                // shifted_position[zDir] = position[zDir] + z;
                amrex::Real z = position[zDir] + z_c;
#endif
#if (GEMPIC_SPACEDIM > 1)
                for (int j = lo[yDir]; j <= hi[yDir]; j++)
                {
                    y_c = infra.m_geom.ProbLo()[yDir] + (double)j * infra.m_dx[yDir];
                    // shifted_position[yDir] = position[yDir] + y;
                    amrex::Real y = position[yDir] + y_c;
#endif
                    for (int l = lo[xDir]; l <= hi[xDir]; l++)
                    {
                        x_c = infra.m_geom.ProbLo()[xDir] + (double)l * infra.m_dx[xDir];
                        // shifted_position[xDir] = position[xDir] + x;
                        amrex::Real x = position[xDir] + x_c;

                        // TODOrefine
                        // const double result = densityEval(density_parse);

                        amrex::Real result = densityEval(AMREX_D_DECL(x, y, z), 0.);
                        weight =
                            result / nPartPerCell *
                            GEMPIC_D_MULT(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir]);

                        ppg[species]->add_particle({AMREX_D_DECL(x, y, z)}, velocity, weight,
                                                   particles);
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

template <unsigned int vdim, unsigned int numspec, unsigned int ndata = 1>
void init_particles_full_domain_gpu (
    ComputationalDomain infra,
    amrex::GpuArray<std::shared_ptr<ParticleGroups<vdim, ndata>>, numspec>& ppg,
    int species,
    int rSeed = 925)
{
    BL_PROFILE("init_particles_full_domain_gpu_parameters");

    constexpr amrex::Real epsilon = std::numeric_limits<amrex::Real>::epsilon();

    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> probLo{AMREX_D_DECL(
        infra.m_geom.ProbLo()[xDir], infra.m_geom.ProbLo()[yDir], infra.m_geom.ProbLo()[zDir])};
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> dx{
        AMREX_D_DECL(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir])};

    std::string specString = "particle.species" + std::to_string(species);
    Parameters params(specString);

    int nPartPerCell;
    params.get("n_part_per_cell", nPartPerCell);
    // we capture by std::tie instead of by structured binding because of idiosyncracies with lambda
    // expressions. May or may not be resolved in C++20, but not in e.g. Clang.
    amrex::Parser densityParse;
    amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> densityEval;
    std::tie(densityParse, densityEval) = Utils::parse_function("density", params);

    int numGauss{1};
    params.get_or_set("num_gaussians", numGauss);
    amrex::Vector<amrex::Vector<amrex::Real>> vMean, vDev;
    amrex::Vector<amrex::Real> vWeight;
    for (int ng = 0; ng < numGauss; ng++)
    {
        amrex::Vector<amrex::Real> vm, vt;
        std::string vmString = "vMean_g" + std::to_string(ng);
        params.get(vmString, vm);
        vMean.push_back(vm);
        std::string vtString = "vThermal_g" + std::to_string(ng);
        params.get(vtString, vt);
        vDev.push_back(vt);
        std::string vwString = "vWeight_g" + std::to_string(ng);
        amrex::Real vw;
        params.get(vwString, vw);
        vWeight.push_back(vw);
    }

    amrex::Array2D<amrex::Real, 0, MAXGAUSS - 1, 0, static_cast<int>(vdim) - 1> vDevGpu, vMeanGpu;
    // actual number of Gaussians in velocity is smaller than maximum number set in
    // GEMPIC_Config.H
    assert(numGauss <= MAXGAUSS);
    for (int j = 0; j < vdim; j++)
    {
        for (int i = 0; i < numGauss; i++)
        {
            vDevGpu(i, j) = vDev[i][j];
            vMeanGpu(i, j) = vMean[i][j];
        }
    }

    // cumulative sum of vel_weights
    amrex::GpuArray<amrex::Real, MAXGAUSS> velCum;
    std::partial_sum(vWeight.begin(), vWeight.end(), velCum.begin());

#if (GEMPIC_SPACEDIM == 3)
    // MFI that adds particle from the model cell to all cells
    for (amrex::MFIter mfi = ppg[species]->MakeMFIter(0); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        amrex::IntVect lo = {bx.smallEnd()};
        amrex::IntVect hi = {bx.bigEnd()};
        int nCellsTile =
            (hi[xDir] - lo[xDir] + 1) * (hi[yDir] - lo[yDir] + 1) * (hi[zDir] - lo[zDir] + 1);
        // AMREX_D_TERM(hi[xDir] - lo[xDir] + 1, *(hi[yDir] - lo[yDir] + 1), *(hi[zDir] - lo[zDir] +
        // 1));
        int npart = nPartPerCell * nCellsTile;
        auto& particles =
            ppg[species]->GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

        // resize particles structure before launching on GPU
        auto oldSize = particles.GetArrayOfStructs().size();
        auto newSize = oldSize + npart;
        particles.resize(newSize);

        ParticleType* AMREX_RESTRICT pstruct = particles.GetArrayOfStructs()().data();
        amrex::GpuArray<amrex::ParticleReal*, vdim> velArray;
        for (int i = 0; i < vdim; i++)
        {
            velArray[i] = particles.GetStructOfArrays().realarray()[i];
        }
        amrex::ParticleReal* weightArray = particles.GetStructOfArrays().realarray()[vdim];
        int procID = amrex::ParallelDescriptor::MyProc();

        amrex::InitRandom(rSeed + lo[xDir] + lo[yDir] * 10 + lo[zDir] * 100 + species * 1000);

        ParallelForRNG(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::RandomEngine const& gen) noexcept
            {
                int ix = i - lo[xDir];
                int nx = hi[xDir] - lo[xDir] + 1;
#if (GEMPIC_SPACEDIM > 1)
                int iy = j - lo[yDir];
                int ny = hi[yDir] - lo[yDir] + 1;
#endif
#if (GEMPIC_SPACEDIM > 2)
                int iz = k - lo[zDir];
#endif
                // int cellid = (ix * ny + iy) * nz + iz;
                int cellid = (iz * ny + iy) * nx + ix;

                int pidx = cellid * nPartPerCell;
                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
                amrex::GpuArray<amrex::Real, vdim> velocity;
                amrex::Real xC, yC, zC;  // cell-corner coordinates
                amrex::Real x, y, z;
                amrex::Real t = 0.0;  // initial time

                for (int pp = 0; pp < nPartPerCell; pp++)
                {
                    for (int comp = 0; comp < GEMPIC_SPACEDIM; comp++)
                    {
                        position[comp] = amrex::Random(gen) * dx[comp];
                    }
                    amrex::GpuArray<amrex::Real, vdim> gaussChoice;
                    for (int comp = 0; comp < vdim; comp++)
                    {
                        gaussChoice[comp] = amrex::Random(gen);
                    }
                    amrex::GpuArray<int, vdim> gaussIdx;
                    for (int comp = 0; comp < vdim; comp++)
                    {
                        gaussIdx[comp] = 0;
                    }
                    for (int comp = 0; comp < vdim; comp++)
                    {
                        for (int gg = 1; gg < numGauss; gg++)
                        {
                            if ((velCum[gg - 1] - gaussChoice[comp] < 0) &&
                                (velCum[gg] - gaussChoice[comp] >= 0))
                            {
                                gaussIdx[comp] = gg;
                            }
                        }
                    }
                    amrex::Real u1, u2, z0;
                    for (int cc = 0; cc < vdim; cc++)
                    {
                        do
                        {
                            u1 = amrex::Random(gen);
                            u2 = amrex::Random(gen);
                        } while (u1 < epsilon);  // make sure epsilon large enough to take log
                        // first value from Box-Muller algorithm
                        z0 = sqrt(-2.0 * log(u1)) * cos(2 * M_PI * u2);
                        velocity[cc] = vDevGpu(gaussIdx[cc], cc) * z0 + vMeanGpu(gaussIdx[cc], cc);
                    }
#if (GEMPIC_SPACEDIM > 2)
                    zC = probLo[zDir] + (double)k * dx[zDir];
                    z = position[zDir] + zC;
#endif
#if (GEMPIC_SPACEDIM > 1)
                    yC = probLo[yDir] + (double)j * dx[yDir];
                    y = position[yDir] + yC;
#endif
                    xC = probLo[xDir] + (double)i * dx[xDir];
                    x = position[xDir] + xC;

                    const double factor = densityEval(x, y, z, t);
                    amrex::Real weight =
                        factor / nPartPerCell * GEMPIC_D_MULT(dx[xDir], dx[yDir], dx[zDir]);
                    ParticleType& p = pstruct[pidx];
                    p.id() = pidx;
                    p.cpu() = procID;
                    p.pos(xDir) = x;
                    p.pos(yDir) = y;
                    p.pos(zDir) = z;

                    velArray[xDir][pidx] = velocity[xDir];
                    velArray[yDir][pidx] = velocity[yDir];
                    velArray[zDir][pidx] = velocity[zDir];
                    weightArray[pidx] = weight;

                    ++pidx;
                }
            });
    }
#endif
    amrex::Print() << "init_full_domain_gpu OK\n";
}

/**
 * @brief Random number generator. Particle positions are generated using a uniform random
 * distribution giving position on each cell and then remapped to full domain.
 * Velocity distribution is generated fomr uniform distribution using the Box-Muller algorithm
 * as random_normal is machine dependent. This enables for a given seed to always get the same
 * random sequence
 *
 * @tparam vdim Number of velocity components associated with the particles
 * @tparam numspec Number of particle species
 * @tparam ndata Number of further amount of data components to be stored, e.g. particle weight
 * @param infra Computational Domain
 * @param ppg Particle group
 * @param species
 * @param rSeed
 */
template <unsigned int vdim, unsigned int numspec, unsigned int ndata>
void init_particles_full_domain (
    ComputationalDomain infra,
    amrex::GpuArray<std::shared_ptr<ParticleGroups<vdim, ndata>>, numspec>& ppg,
    int species,
    int rSeed = 925)
{
    BL_PROFILE("init_particles_full_domain_new_params");
    double x, y, z, t = 0.0;

    amrex::Real AMREX_D_DECL(x_c, y_c, z_c);  // cell-corner coordinates
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
    amrex::GpuArray<amrex::Real, vdim> velocity;
    amrex::Real weight;
    constexpr amrex::Real epsilon = std::numeric_limits<amrex::Real>::epsilon();
    amrex::Real u1, u2, z0;

    std::string specString = "particle.species" + std::to_string(species);
    Parameters params(specString);

    int nPartPerCell;
    params.get("n_part_per_cell", nPartPerCell);
    double charge;
    params.get("charge", charge);
    double mass;
    params.get("mass", mass);
    auto [densityParse, densityEval] = Utils::parse_function("density", params);

    int numGauss{1};
    params.get_or_set("num_gaussians", numGauss);
    amrex::Vector<amrex::Vector<amrex::Real>> vmean, vdev;
    amrex::Vector<amrex::Real> vweight;
    for (int ng = 0; ng < numGauss; ng++)
    {
        amrex::Vector<amrex::Real> vm, vt;
        std::string vmString = "vMean_g" + std::to_string(ng);
        params.get(vmString, vm);
        vmean.push_back(vm);
        std::string vtString = "vThermal_g" + std::to_string(ng);
        params.get(vtString, vt);
        vdev.push_back(vt);
        std::string vwString = "vWeight_g" + std::to_string(ng);
        amrex::Real vw;
        params.get(vwString, vw);
        vweight.push_back(vw);
    }

    // cumulative sum of vel_weights
    amrex::Vector<amrex::Real> velCum(numGauss);
    std::partial_sum(vweight.begin(), vweight.end(), velCum.begin());

    // MFI that adds particle from the model cell to all cells
    //#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
    for (amrex::MFIter mfi = ppg[species]->MakeMFIter(0); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        amrex::IntVect lo = {bx.smallEnd()};
        amrex::IntVect hi = {bx.bigEnd()};

        std::mt19937 gen(GEMPIC_D_ADD(rSeed + lo[xDir], lo[yDir] * 10, lo[zDir] * 100) +
                         species * 1000);
        std::mt19937 genv(GEMPIC_D_ADD(rSeed + lo[xDir], lo[yDir] * 10, lo[zDir] * 100) +
                          species * 1000 + 17);
        std::uniform_real_distribution<> unif(0.0, 1.0);
        std::uniform_real_distribution<> unifv(0.0, 1.0);

        using myParticleTile = amrex::ParticleTile<ParticleType, vdim + ndata, 0>;
        myParticleTile& particles =
            ppg[species]->GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

#if (GEMPIC_SPACEDIM > 2)
        for (int k = lo[zDir]; k <= hi[zDir]; k++)
        {
#endif
#if (GEMPIC_SPACEDIM > 1)
            for (int j = lo[yDir]; j <= hi[yDir]; j++)
            {
#endif
                for (int l = lo[xDir]; l <= hi[xDir]; l++)
                {
                    for (int pp = 0; pp < nPartPerCell; pp++)
                    {
                        for (int i = 0; i < GEMPIC_SPACEDIM; i++)
                        {
                            position[i] = unif(gen) * infra.m_dx[i];
                        }
                        amrex::Real gaussChoice[vdim];
                        for (int i = 0; i < vdim; i++)
                        {
                            gaussChoice[i] = unif(gen);
                        }
                        amrex::GpuArray<int, vdim> gaussIdx;
                        for (int comp = 0; comp < vdim; comp++)
                        {
                            gaussIdx[comp] = 0;
                        }
                        for (int i = 0; i < vdim; i++)
                        {
                            for (int gg = 1; gg < numGauss; gg++)
                            {
                                if ((velCum[gg - 1] - gaussChoice[i] < 0) &&
                                    (velCum[gg] - gaussChoice[i] >= 0))
                                {
                                    gaussIdx[i] = gg;
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
                            velocity[cc] = vdev[gaussIdx[cc]][cc] * z0 + vmean[gaussIdx[cc]][cc];
                        }
#if (GEMPIC_SPACEDIM > 2)
                        z_c = infra.m_geom.ProbLo()[zDir] + (double)k * infra.m_dx[zDir];
                        z = position[zDir] + z_c;
#endif
#if (GEMPIC_SPACEDIM > 1)
                        y_c = infra.m_geom.ProbLo()[yDir] + (double)j * infra.m_dx[yDir];
                        y = position[yDir] + y_c;
#endif
                        x_c = infra.m_geom.ProbLo()[xDir] + (double)l * infra.m_dx[xDir];
                        x = position[xDir] + x_c;

                        const double factor = densityEval(AMREX_D_DECL(x, y, z), t);
                        weight =
                            factor / nPartPerCell *
                            GEMPIC_D_MULT(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir]);
                        ppg[species]->add_particle({AMREX_D_DECL(x, y, z)}, velocity, weight,
                                                   particles);
                        // ppg[species]->add_particle(position, velocity,
                        //                                    weight, particles);
                        // Check random number
                        // if (check_rnd)
                        // {
                        //     npart++;
                        //     if (npart < 10)
                        //         amrex::Print() << npart << " " << x << " " << y << " " << z << "
                        //         "
                        //                        << velocity[xDir] << " " << velocity[yDir] << " "
                        //                        << velocity[zDir] << " " << weight << "\n";
                        // }
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

template <unsigned int vdim>
void init_one_particle_cellwise (amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const& dx,
                                 amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const& plo,
                                 ParticleGroups<vdim>& pg,
                                 amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position)
{
    amrex::Real x, y, z;
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> shiftedPosition;
    amrex::GpuArray<amrex::Real, vdim> velocity;

    velocity[xDir] = 1.;
    if constexpr (vdim > 1)
    {
        velocity[yDir] = 1.;
    }
    if constexpr (vdim > 2)
    {
        velocity[zDir] = 1.;
    }

    // MFI that adds particle from the model cell to all cells
    for (amrex::MFIter mfi = pg.MakeMFIter(0); mfi.isValid(); ++mfi)
    {  // 0 is level
        const amrex::Box& bx = mfi.validbox();
        amrex::IntVect lo = {bx.smallEnd()};
        amrex::IntVect hi = {bx.bigEnd()};

        auto& particles = pg.GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

#if (GEMPIC_SPACEDIM > 2)
        for (int k = lo[zDir]; k <= hi[zDir]; k++)
        {
            z = plo[zDir] + (double)k * dx[zDir];
            shiftedPosition[zDir] = position[zDir] + z;
#endif
#if (GEMPIC_SPACEDIM > 1)
            for (int j = lo[yDir]; j <= hi[yDir]; j++)
            {
                y = plo[yDir] + (double)j * dx[yDir];
                shiftedPosition[yDir] = position[yDir] + y;
#endif
                for (int l = lo[xDir]; l <= hi[xDir]; l++)
                {
                    x = plo[xDir] + (double)l * dx[xDir];
                    shiftedPosition[xDir] = position[xDir] + x;

                    ParticleType p;
                    p.id() = ParticleType::NextID();
                    p.cpu() = amrex::ParallelDescriptor::MyProc();

                    AMREX_D_TERM(p.pos(xDir) = shiftedPosition[xDir];
                                 , p.pos(yDir) = shiftedPosition[yDir];
                                 , p.pos(zDir) = shiftedPosition[zDir];)

                    // p.rdata(0) = velocity[xDir];
                    particles.push_back_real(0, velocity[xDir]);
                    if (vdim > 1)
                    {
                        // p.rdata(1) = velocity[yDir];
                        particles.push_back_real(1, velocity[yDir]);
                    }
                    if (vdim > 2)
                    {
                        // p.rdata(2) = velocity[zDir];
                        particles.push_back_real(2, velocity[zDir]);
                    }

                    // p.rdata(vdim) = 1.0/64.0; //weight
                    particles.push_back_real(vdim, 1.0 / 64.0);  // weight

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
