#ifndef GEMPICX_FLUSHFORMAT_H_
#define GEMPICX_FLUSHFORMAT_H_

#include <string>

#include <AMReX_Geometry.H>
#include <AMReX_Vector.H>

#include "GEMPIC_particle_groups.H"

template <int vdim, int numspec, int ndata>
class FlushFormat
{
public:
    FlushFormat() {}

    /** Flush fields and particles to file */
    virtual void write_to_file(
        const amrex::Vector<std::string> varnames,
        const amrex::MultiFab& mf,
        amrex::Geometry& geom,
        const int iteration,
        const double time,
        amrex::GpuArray<std::shared_ptr<Gempic::Particles::ParticleGroups<vdim, ndata>>, numspec>
            particles,
        int nlev,
        const std::string prefix,
        const int fileMinDigits,
        bool plotParticles,
        int bufferID = 1,
        int numBuffers = 1) const = 0;

    virtual ~FlushFormat() {}
};

#endif  // GEMPICX_FLUSHFORMAT_H_
