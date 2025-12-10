#ifndef GEMPICX_FLUSHFORMAT_H_
#define GEMPICX_FLUSHFORMAT_H_

#include <string>

#include <AMReX_Geometry.H>
#include <AMReX_Vector.H>

#include "GEMPIC_Particle.H"

template <int vdim, int ndata>
class FlushFormat
{
public:
    FlushFormat() {}

    /** Flush fields and particles to file */
    virtual void write_to_file(
        amrex::Vector<std::string> const& varnames,
        amrex::MultiFab const& mf,
        amrex::Geometry& geom,
        int const iteration,
        double const time,
        std::vector<std::shared_ptr<Gempic::Particle::ParticleSpecies<vdim, ndata>>> const&
            particles,
        int nlev,
        std::string const& prefix,
        int fileMinDigits,
        bool plotParticles,
        int bufferID = 1,
        int numBuffers = 1) const = 0;

    virtual ~FlushFormat() {}
};

#endif // GEMPICX_FLUSHFORMAT_H_
