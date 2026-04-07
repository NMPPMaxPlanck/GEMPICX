/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef TEST_UTILS_H_
#define TEST_UTILS_H_

#include <algorithm> // std::all_of, for add_particles function
#include <optional>

#include <AMReX.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Particle.H"

namespace Gempic::Test::Utils
{
// Nghost helper function
inline int init_n_ghost (int degX, int degY, int degZ)
{
    amrex::Array<int, AMREX_SPACEDIM> degs = {AMREX_D_DECL(degX, degY, degZ)};
    int const maxDeg{*(std::max_element(degs.begin(), degs.end()))};
    return maxDeg;
}

// Stream arr helper function
template <typename T>
std::string string_array (T const& valArray, int length)
{
    std::stringstream stream;
    stream << "(" << valArray[0];
    for (int i = 1; i < length; i++)
    {
        stream << ", " << valArray[i];
    }
    stream << ")";
    return stream.str();
}

// Add-particle helper function. User must supply coordinates (but not necessarily velocities)
// of all particles to be added.
template <unsigned int vDim, unsigned nData, size_t numParticles>
void add_single_particles (
    Gempic::Particle::ParticleSpecies<vDim, nData>* particleSpecies,
    Gempic::ComputationalDomain const& compDom,
    amrex::Array<amrex::Real, numParticles> const& weights,
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> const& positions,
    amrex::Array<amrex::GpuArray<amrex::Real, vDim>, numParticles> const& velocities = {})
{
    std::array<int, numParticles> checklist;
    checklist.fill(false);
    amrex::Geometry const& geom{compDom.geometry()};

    for (amrex::MFIter mfi{particleSpecies->MakeMFIter(0)}; mfi.isValid(); ++mfi)
    {
        std::array<amrex::Real, AMREX_SPACEDIM> lower;
        std::array<amrex::Real, AMREX_SPACEDIM> upper;

        amrex::Box const& bx{mfi.validbox()};
        for (int d{0}; d < AMREX_SPACEDIM; d++)
        {
            lower[d] = geom.ProbLo(d) + bx.smallEnd(d) * geom.CellSize(d);
            upper[d] = geom.ProbLo(d) + (bx.bigEnd(d) + 1) * geom.CellSize(d);
        }
        using PTType = typename amrex::ParticleContainerPureSoA<AMREX_SPACEDIM + vDim + nData,
                                                                0>::ParticleTileType;

        PTType& particles{
            particleSpecies->GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())]};
        for (size_t i{0}; i < numParticles; i++)
        {
            // make sure particle is within box (and not previously added? Why?)
            if (!checklist[i] AMREX_D_TERM(
                    &&(lower[xDir] <= positions[i][xDir]) && (positions[i][xDir] < upper[xDir]),
                    &&(lower[yDir] <= positions[i][yDir]) && (positions[i][yDir] < upper[yDir]),
                    &&(lower[zDir] <= positions[i][zDir]) && (positions[i][zDir] < upper[zDir])))
            {
                particleSpecies->add_particle(positions[i], velocities[i], weights[i], particles);
                checklist[i] = true;
            }
        }
    }

    // Check that all particles have been added
    amrex::ParallelDescriptor::ReduceIntSum(checklist.data(), numParticles);
    for (size_t i{0}; i < numParticles; i++)
    {
        if (!checklist[i])
        {
            std::cerr << "Invalid position given to add_particle function:\n"
                      << string_array(positions[i], AMREX_SPACEDIM)
                      << " is out of simulation bounds.\nLower: "
                      << string_array(geom.ProbLo(), AMREX_SPACEDIM)
                      << "\nUpper: " << string_array(geom.ProbHi(), AMREX_SPACEDIM);
        }
    }
    particleSpecies->Redistribute(); // assign particles to the tile they are in
}

/** Helper function to check entries of a field given a series of conditions and a default
 * value. Check order is prioritized, so a set of indices only fulfill the first succesful
 * condition.
 *
 * Parameters:
 * ----------
 * @param file string, the file from which the function was called
 * @param line int, the line from which the function was called
 * @param fieldArr amrex::Array4, array containing field values in an easily reached accessor
 * @param top Dim3, top boundaries of box for fieldArray
 * @param condVec vector<condLambda>, Vector of lambdas that check if the {SPACEDIM} indices
 * fulfill a given condition.
 * @param checks vector<amrex::Real>, Vector of values to compare to if indices fulfill the
 * corresponding condVec condition.
 * @param defCheck optional amrex::Real, Default value for all indices not fulfilling any of the
 * given conditions. Defaults to {} (empty optional), meaning no check is performed.
 * @param tol amrex::Real, relative tolerance
 */
using condLambda = bool (*)(AMREX_D_DECL(int, int, int));
void check_field(char const file[],
                 int line,
                 amrex::Array4<amrex::Real> const& fieldArr,
                 amrex::Box const& bx,
                 std::vector<condLambda>&& condVec,
                 std::vector<amrex::Real>&& checks,
                 std::optional<amrex::Real> defCheck = {},
                 amrex::Real tol = 1e-14);

/** Helper function to check that the entries of two fields are equal.
 *
 * Parameters:
 * ----------
 * @param file string, the file from which the function was called
 * @param line int, the line from which the function was called
 * @param fieldArr amrex::Array4, array containing field values in an easily reached accessor
 * @param fieldArr2 amrex::Array4, array containing field values in an easily reached accessor
 * @param bx amrex::Box used to find the boundaries of fieldArr and fieldArr2
 * @param tol amrex::Real, relative tolerance
 * @param ncomp number of components of the two multifabs
 */
void compare_fields(char const file[],
                    int line,
                    amrex::Array4<amrex::Real const> const& fieldArr,
                    amrex::Array4<amrex::Real const> const& fieldArr2,
                    amrex::Box const& bx,
                    amrex::Real tol = 1e-14,
                    int ncomp = 1);

double l_inf_error(amrex::Array4<amrex::Real const> const& a,
                   amrex::Array4<amrex::Real const> const& b,
                   amrex::Box const& bx,
                   int ncomp = 1);

ComputationalDomain get_default_compdom();
ComputationalDomain get_compdom(int gSize);
ComputationalDomain get_compdom(amrex::IntVect const& nCell);
void init_multilevel_domain(amrex::Vector<amrex::Geometry>& geom,
                            amrex::Vector<amrex::BoxArray>& ba,
                            amrex::Vector<amrex::DistributionMapping>& dm);
} // namespace Gempic::Test::Utils

/** @brief Helper macro for the @ref Gempic::Test::Utils::check_field function */
#define CHECK_FIELD(...) Gempic::Test::Utils::check_field(__FILE__, __LINE__, __VA_ARGS__)
/** @brief Helper macro for the @ref Gempic::Test::Utils::compare_fields function */
#define COMPARE_FIELDS(...) Gempic::Test::Utils::compare_fields(__FILE__, __LINE__, __VA_ARGS__)
#endif
