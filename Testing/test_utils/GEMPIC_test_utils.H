#ifndef TEST_UTILS_H_
#define TEST_UTILS_H_

#include <AMReX.H>
#include <algorithm> // std::all_of, for add_particles function
#include "GEMPIC_particle_groups.H"
#include "GEMPIC_computational_domain.H"

namespace GEMPIC_TestUtils {
    // Nghost helper function
    inline int initNGhost (int degX, int degY, int degZ)
    {
        amrex::Array<int, GEMPIC_SPACEDIM> degs = {AMREX_D_DECL(degX, degY, degZ)};
        const int maxDeg{*(std::max_element(degs.begin(), degs.end()))};
        return maxDeg;
    }

    // Stream arr helper function
    template<typename T>
    std::string stringArray (const T& valArray, int length) {
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
    template<unsigned int vdim, unsigned int numSpec, size_t numParticles>
    void addSingleParticles (
        amrex::GpuArray<std::unique_ptr<Gempic::Particles::particle_groups<vdim>>, numSpec>& particleGroup,
        Gempic::CompDom::computational_domain& infra,
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

            amrex::ParticleTile<Gempic::Particles::ParticleType, vdim + 1, 0>& particles{
                particleGroup[spec]->GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())]};
            for (int i{0}; i < numParticles; i++) {
                // make sure particle is within box (and not previously added? Why?)
                if (!checklist[i] AMREX_D_TERM(
                    && (lo[xDir]*infra.dx[xDir] <= positions[i][xDir]) && 
                                            (positions[i][xDir] < (hi[xDir] + 1)*infra.dx[xDir]),
                    && (lo[yDir]*infra.dx[yDir] <= positions[i][yDir]) &&
                                            (positions[i][yDir] < (hi[yDir] + 1)*infra.dx[yDir]),
                    && (lo[zDir]*infra.dx[zDir] <= positions[i][zDir]) &&
                                            (positions[i][zDir] < (hi[zDir] + 1)*infra.dx[zDir])))
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

    /* Helper function to check entries of a field given a series of conditions and a default
     * value. Check order is prioritized, so a set of indices only fulfill the first succesful
     * condition.
     * 
     * Parameters:
     * ----------
     * @param line: int, the line from which the function was called
     * @param fieldArr: amrex::Array4, array containing field values in an easily reached accessor
     * @param top: Dim3, top boundaries of box for fieldArray
     * @param condVec: vector<condLambda>, Vector of lambdas that check if the {SPACEDIM} indices fulfill a given condition.
     * @param checks: vector<amrex::Real>, Vector of values to compare to if indices fulfill the corresponding condVec condition.
     * @param defCheck: amrex::Real, Default value for all indices not fulfilling any of the given conditions.
     */
    using condLambda = bool(*)(AMREX_D_DECL(int, int, int));
    void checkField(const char file[], int line,
                    amrex::Array4<amrex::Real> const& fieldArr,
                    amrex::Dim3 const&& top,
                    std::vector<condLambda>&& condVec,
                    std::vector<amrex::Real>&& checks,
                    amrex::Real defCheck);

    /* Helper function to check that the entries of two fields are equal.
     * 
     * Parameters:
     * ----------
     * @param line: int, the line from which the function was called
     * @param fieldArr: amrex::Array4, array containing field values in an easily reached accessor
     * @param fieldArr2: amrex::Array4, array containing field values in an easily reached accessor
     * @param top: Dim3, top boundaries of box for fieldArray
     */
    void compareFields(const char file[], int line,
                       amrex::Array4<amrex::Real> const& fieldArr,
                       amrex::Array4<amrex::Real> const& fieldArr2,
                       amrex::Box const& bx,
                       int ncomp = 1);

}
#endif