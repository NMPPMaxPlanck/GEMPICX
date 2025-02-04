#ifndef TEST_UTILS_H_
#define TEST_UTILS_H_

#include <algorithm>  // std::all_of, for add_particles function
#include <optional>

#include <AMReX.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_ParticleGroups.H"

namespace Gempic::Test::Utils
{
// Nghost helper function
inline int init_n_ghost (int degX, int degY, int degZ)
{
    amrex::Array<int, GEMPIC_SPACEDIM> degs = {AMREX_D_DECL(degX, degY, degZ)};
    const int maxDeg{*(std::max_element(degs.begin(), degs.end()))};
    return maxDeg;
}

// Stream arr helper function
template <typename T>
std::string string_array (const T& valArray, int length)
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
template <unsigned int vdim, size_t numParticles>
void add_single_particles (
    Gempic::Particle::ParticleGroups<vdim>* particleGroup,
    const Gempic::ComputationalDomain& infra,
    const amrex::Array<amrex::Real, numParticles>& weights,
    const amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles>& positions,
    const amrex::Array<amrex::GpuArray<amrex::Real, vdim>, numParticles>& velocities = {})
{
    amrex::Array<bool, numParticles> checklist{false};

    for (amrex::MFIter mfi{particleGroup->MakeMFIter(0)}; mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx{mfi.validbox()};
        amrex::IntVect lo{bx.smallEnd()};
        amrex::IntVect hi{bx.bigEnd()};

        amrex::ParticleTile<Gempic::Particle::ParticleType, vdim + 1, 0>& particles{
            particleGroup->GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())]};
        for (int i{0}; i < numParticles; i++)
        {
            // make sure particle is within box (and not previously added? Why?)
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> dx = infra.geometry().CellSizeArray();
            if (!checklist[i] AMREX_D_TERM(&&(lo[xDir] * dx[xDir] <= positions[i][xDir]) &&
                                               (positions[i][xDir] < (hi[xDir] + 1) * dx[xDir]),
                                           &&(lo[yDir] * dx[yDir] <= positions[i][yDir]) &&
                                               (positions[i][yDir] < (hi[yDir] + 1) * dx[yDir]),
                                           &&(lo[zDir] * dx[zDir] <= positions[i][zDir]) &&
                                               (positions[i][zDir] < (hi[zDir] + 1) * dx[zDir])))
            {
                particleGroup->add_particle(positions[i], velocities[i], weights[i], particles);
                checklist[i] = true;
            }
        }
    }

    // Check that all particles have been added
    int idx{-1};
    auto result1 = std::find_if(std::begin(checklist), std::end(checklist),
                                [&idx] (bool i)
                                {
                                    idx++;
                                    return !i;
                                });
    if (result1 != std::end(checklist))
    {
        std::cerr << "Invalid position given to add_particle function:\n"
                  << string_array(positions[idx], GEMPIC_SPACEDIM)
                  << " is out of simulation bounds.\nLower: "
                  << string_array(infra.m_geom.ProbLo(), GEMPIC_SPACEDIM)
                  << "\nUpper: " << string_array(infra.m_geom.ProbHi(), GEMPIC_SPACEDIM);
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
 * @param condVec: vector<condLambda>, Vector of lambdas that check if the {SPACEDIM} indices
 * fulfill a given condition.
 * @param checks: vector<amrex::Real>, Vector of values to compare to if indices fulfill the
 * corresponding condVec condition.
 * @param defCheck: optional amrex::Real, Default value for all indices not fulfilling any of the
 * given conditions. Defaults to {} (empty optional), meaning no check is performed.
 * @param tol: amrex::Real, relative tolerance
 */
using condLambda = bool (*)(AMREX_D_DECL(int, int, int));
void check_field(const char file[],
                 int line,
                 amrex::Array4<amrex::Real> const& fieldArr,
                 amrex::Dim3 const&& top,
                 std::vector<condLambda>&& condVec,
                 std::vector<amrex::Real>&& checks,
                 std::optional<amrex::Real> defCheck = {},
                 amrex::Real tol = 1e-14);

/* Helper function to check that the entries of two fields are equal.
 *
 * Parameters:
 * ----------
 * @param line: int, the line from which the function was called
 * @param fieldArr: amrex::Array4, array containing field values in an easily reached accessor
 * @param fieldArr2: amrex::Array4, array containing field values in an easily reached accessor
 * @param bx: amrex::Box used to find the boundaries of fieldArr and fieldArr2
 * @param tol: amrex::Real, relative tolerance
 * @param ncomp: number of components of the two multifabs
 */
void compare_fields(const char file[],
                    int line,
                    amrex::Array4<const amrex::Real> const& fieldArr,
                    amrex::Array4<const amrex::Real> const& fieldArr2,
                    amrex::Box const& bx,
                    amrex::Real tol = 1e-14,
                    int ncomp = 1);

double l_inf_error(amrex::Array4<const amrex::Real> const& a,
                   amrex::Array4<const amrex::Real> const& b,
                   amrex::Box const& bx,
                   int ncomp = 1);
}  // namespace Gempic::Test::Utils
#endif