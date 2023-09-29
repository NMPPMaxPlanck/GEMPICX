#ifndef TEST_UTILS_H_
#define TEST_UTILS_H_

#include "AMReX.H"
#include <algorithm> // std::all_of, for add_particles function
#include <GEMPIC_particle_groups.H>

using namespace Particles;
namespace GEMPIC_TestUtils {
    // Nghost helper function
    inline const int initNGhost(int degX, int degY, int degZ)
    {
        amrex::Array<int, GEMPIC_SPACEDIM> degs = {AMREX_D_DECL(degX, degY, degZ)};
        const int maxDeg{*(std::max_element(degs.begin(), degs.end()))};
        return maxDeg;
    }

    // Stream arr helper function
    template<typename T>
    std::string stringArray(const T& valArray, int length) {
        std::stringstream stream;
        stream << "(" << valArray[0];
        for (int i = 1; i < length; i++) {
            stream << ", " << valArray[i];
        }
        stream << ")";
        return stream.str();
    }

    // Add-particle helper function. User must supply coordinates (but not necessarily velocities)
    // of all particles to be added.
    template<int vdim, int numSpec, int numParticles>
    void addSingleParticles(
        amrex::GpuArray<std::unique_ptr<Gempic::Particles::particle_groups<vdim>>, numSpec>& particleGroup,
        computational_domain& infra,
        amrex::Array<amrex::Real, numParticles>& weights,
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles>& positions,
        int spec = 0,
        amrex::Array<amrex::GpuArray<amrex::Real, vdim>, numParticles>&& velocities = {})
    {
        amrex::Array<bool, numParticles> checklist{false};
        
        for (amrex::MFIter mfi{particleGroup[0]->MakeMFIter(0)}; mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx{mfi.validbox()};
            amrex::IntVect lo{bx.smallEnd()};
            amrex::IntVect hi{bx.bigEnd()};

            amrex::ParticleTile<ParticleType, vdim + 1, 0>& particles{
                particleGroup[spec]->GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())]};
            for (int i{0}; i < numParticles; i++) {
                // make sure particle is within box (and not previously added? Why?)
                if (!checklist[i] AMREX_D_TERM(
                    && (lo[0]*infra.dx[0] <= positions[i][0]) && 
                                            (positions[i][0] < (hi[0] + 1)*infra.dx[0]),
                    && (lo[1]*infra.dx[1] <= positions[i][1]) &&
                                            (positions[i][1] < (hi[1] + 1)*infra.dx[1]),
                    && (lo[2]*infra.dx[2] <= positions[i][2]) &&
                                            (positions[i][2] < (hi[2] + 1)*infra.dx[2])))
                {
                    particleGroup[spec]->add_particle(positions[i], velocities[i], weights[i],
                                                      particles);
                    checklist[i] = true;
                }
            }
        }

        // Check that all particles have been added
        int idx{-1};
        auto result1 = std::find_if(std::begin(checklist), std::end(checklist), [&idx] (bool i) {idx++; return !i;});
        if (result1 != std::end(checklist)) {
            std::cerr << "Invalid position given to add_particle function:\n" << 
                        stringArray(positions[idx], GEMPIC_SPACEDIM) << 
                        " is out of simulation bounds.\nLower: " << 
                        stringArray(infra.geom.ProbLo(), GEMPIC_SPACEDIM) << 
                        "\nUpper: " << stringArray(infra.geom.ProbHi(), GEMPIC_SPACEDIM);
        }
    }
}
#endif