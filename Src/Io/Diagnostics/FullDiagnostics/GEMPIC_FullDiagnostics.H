#ifndef GEMPICX_FULLDIAGNOSTICS_H_
#define GEMPICX_FULLDIAGNOSTICS_H_

#include <memory>
#include <string>

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_BLassert.H>
#include <AMReX_Config.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Particles.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include "GEMPIC_CellCenterFunctor.H"
#include "GEMPIC_FlushFormat.H"
#include "GEMPIC_FlushFormatPlotfile.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_ParticleGroups.H"
#include "GEMPIC_RawFunctor.H"

namespace Gempic::Io
{
template <int vdim, int numspec, int ndata>
class FullDiagnostics
{
public:
    FullDiagnostics(int diagIndex, std::string name, amrex::Real dt, int saveInterval = 100) :
        m_diagName{std::move(name)}, m_diagIndex{diagIndex}, m_dt{dt}, m_saveInterval{saveInterval}
    {
        read_parameters();
    }

    /** Whether the last timestep is always dumped */
    bool do_dump_last_timestep () const { return m_dumpLastTimestep; }

    /** Pack (stack) all fields in the cell-centered output MultiFab m_mf_outputs.
     *
     * Fields are computed (e.g., cell-centered or back-transformed)
       on-the-fly using a functor.
     */
    void compute_and_pack ()
    {
        // compute the necessary fields and store result in m_mf_outputs.
        if (!m_plotParticles)
        {
            for (int i = 0; i < m_nComps; i++)
            {
                if (m_mfOutputs.size() > 1)
                {
                    m_allFieldFunctors[i]->operator()(*(m_mfOutputs[i]), i);
                }
                else
                {
                    m_allFieldFunctors[i]->operator()(*(m_mfOutputs[0]), i);
                }
            }
        }
    }

    /** Initialize pointers to main fields and allocate output multifab m_mf_outputs. */
    void init_data (
        ComputationalDomain& compDom,
        std::map<std::string, amrex::MultiFab*>& fieldsDiagnostics,
        std::map<std::string, amrex::Real>& fieldsScaling,
        amrex::GpuArray<std::shared_ptr<Particle::ParticleGroups<vdim, ndata>>, numspec> ions,
        amrex::IntVect& nGhost)
    {
        // initialize member variables and arrays in base class::Diagnostics
        init_base_data();
        initialize_field_functors_and_buffer_data(compDom, fieldsDiagnostics, fieldsScaling, ions,
                                                  nGhost);
    }

    /** Perform necessary operations with user-defined diagnostic parameters
     *  to filter (coarsen, slice), compute (cell-center, back-transform),
     *  and flush the output data stored in buffers, m_mfOutputs.
     * \param[in] step current timestep
     * \param[in] forceFlush used to force-fully write data stored in buffers.
     */
    void filter_compute_pack_flush (int step, bool forceFlush = false)
    {
        // GEMPIC_PROFILE("Diagnostics::FilterComputePackFlush()");

        if (do_compute_and_pack(step, forceFlush))
        {
            compute_and_pack();
            flush(step);
        }
    }
    /// getter for number of components of full diagnostics
    int get_ncomp () { return m_nComps; }

    /** Vector of (pointers to) functors to compute output fields, per level,
     * per component. This allows for simple operations (averaging to
     * cell-center for standard EB fields) as well as more involved operations */
    amrex::Vector<std::unique_ptr<ComputeDiagFunctor>> m_allFieldFunctors;

private:
    /** Read user-requested parameters for full diagnostics */
    void read_parameters ()  // computational_domain infra)
    {
        // Read list of full diagnostics fields requested by the user.
        Parameters ppDiagName("FullDiagnostics." + m_diagName);

        // prefix
        m_filePrefix = "FullDiagnostics/plt_" + m_diagName;  // default for paraview
        ppDiagName.get_or_set("filePrefix", m_filePrefix);

        ppDiagName.get_or_set("fileMinDigits", m_fileMinDigits);
        ppDiagName.get_or_set("dumpLastTimestep", m_dumpLastTimestep);

        // reading variable names for diagnostic
        ppDiagName.get_or_set("varNames", m_diagVarnames);
        m_nComps = static_cast<int>(m_diagVarnames.size());
        m_allFieldFunctors.resize(m_nComps);

        // reading format of file to be written
        ppDiagName.get_or_set("format", m_format);

        // reading functor
        ppDiagName.get_or_set("functor", m_diagFunctor);

        ppDiagName.get_or_set("saveInterval", m_saveInterval);

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_format == "plotfile"  //|| m_format == "openpmd" ||
            // m_format == "checkpoint" || m_format == "ascent" ||
            // m_format == "sensei"
            ,
            "<diag>.format must be plotfile or openpmd or checkpoint or ascent or sensei");
    }

    void init_base_data ()
    /** Initialize formats and data required to construct the buffers
     */
    {
        // Construct Flush class.
        if (m_format == "plotfile")
        {
            m_flushFormat = std::make_unique<FlushFormatPlotfile<vdim, numspec, ndata>>();
            //} else if (m_format == "checkpoint"){
            // creating checkpoint format
            // m_flush_format = std::make_unique<FlushFormatCheckpoint>() ;
        }
        else if (m_format == "openpmd")
        {
#ifdef WARPX_USE_OPENPMD
            m_flushFormat = std::make_unique<FlushFormatOpenPMD>(m_diagName);
#else
            amrex::Abort("To use openpmd output format, need to compile with USE_OPENPMD=TRUE");
#endif
        }
        else
        {
            amrex::Abort("unknown output format");
        }
    }

    /** Whether to plot raw (i.e., NOT cell-centered) fields */
    bool m_plotParticles = false;
    /** Whether to plot guard cells of raw fields */
    bool m_plotRawFieldsGuards = false;
    /** Name of diagnostics: runtime parameter given in the input file. */
    std::string m_diagName;
    /** Variable names in diagnostics: runtime parameter given in the input file. */
    amrex::Vector<std::string> m_diagVarnames;
    /** Type of functor: runtime parameter given in the input file. */
    std::string m_diagFunctor = "raw";  // raw functor as default
    /** Prefix for output directories */
    std::string m_filePrefix;
    /** Minimum number of digits to iteration number in file name */
    int m_fileMinDigits = 6;
    /** Index of diagnostics in MultiDiagnostics::alldiags */
    int m_diagIndex;

    int m_nComps{0};

    /** format for output files, "plotfile" or "openpmd" or "sensei" or "ascent"
     * The checkpoint format is applicable for FullDiagnostics only.
     */
    std::string m_format = "plotfile";  // AMReX format
    /** Whether this iteration has already been dumped, to avoid writing data twice */
    int m_alreadyDone = false;
    /** This class is responsible for flushing the data to file */
    std::unique_ptr<FlushFormat<vdim, numspec, ndata>> m_flushFormat;
    /** output multifab, where all fields are computed (cell-centered or back-transformed)
     *  and stacked.
     *  The first vector is for total number of snapshots. (=1 for FullDiagnostics)
     *  The second vector is loops over the total number of levels.
     */
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> m_mfOutputs;

    /** If true, a dump is performed at the last timestep regardless of the required dump timesteps
     */
    bool m_dumpLastTimestep = true;

    /** Geometry that defines the domain attributes corresponding to output multifab.
     *  Specifically, the user-defined physical co-ordinates for the diagnostics
     *  is used to construct the geometry information for each MultiFab at
     *  the respective levels. This geometry will be used to write out
     *  plotfile data using the WriteToFile() function
     */
    amrex::Geometry m_geomOutput;

    amrex::GpuArray<std::shared_ptr<Particle::ParticleGroups<vdim, ndata>>, numspec> m_particles;

    /** Coarsening ratio such that, fields are averaged to the coarsened grid.
     * The ratio should render the grid to be coarsenable (as defined by AMReX). */
    amrex::IntVect m_crseRatio = amrex::IntVect(1);

    /* simulation dt */
    amrex::Real m_dt;

    /** Determines timesteps at which full diagnostics are written to file */
    int m_saveInterval = 100;

    /** Flush m_mf_outputs and particles to file for the i^th buffer */
    inline void flush (int step)
    {
        if (m_mfOutputs.size() > 1)
        {
            for (int i{0}; i < m_mfOutputs.size(); ++i)
            {
                std::string filePrefix = m_filePrefix + "_" + m_diagVarnames[i];
                amrex::Vector<std::string> singleEntryVarnameVector{m_diagVarnames[i]};
                m_flushFormat->write_to_file(singleEntryVarnameVector, *(m_mfOutputs[i]),
                                             m_geomOutput, step, step * m_dt, m_particles, 1,
                                             filePrefix, m_fileMinDigits, m_plotParticles);
            }
        }
        else
        {
            m_flushFormat->write_to_file(m_diagVarnames, *(m_mfOutputs[0]), m_geomOutput, step,
                                         step * m_dt, m_particles, 1, m_filePrefix, m_fileMinDigits,
                                         m_plotParticles);
        }
    }

    inline bool do_compute_and_pack (int step, bool forceFlush = false)
    /** whether to flush at this time step
     * \param[in] step current time step
     * \param[in] forceFlush if true, return true for any step
     * \return bool, whether to flush
     */
    {
        // Data must be computed and packed for full diagnostics
        // whenever the data needs to be flushed.
        return (forceFlush || (step % m_saveInterval == 0));
    }

    /**
     * Figure out if particles or fields are being plotted. Breaks if particle groups and fields
     * have the identical names.
     */
    std::string get_diag_type (
        std::map<std::string, amrex::MultiFab*>& fieldsDiagnostics,
        amrex::GpuArray<std::shared_ptr<Particle::ParticleGroups<vdim, ndata>>, numspec>& particles)
    {
        bool particleType{false};
        bool fieldType{false};
        for (const auto& diagName : m_diagVarnames)
        {
            if (fieldsDiagnostics.find(diagName) != fieldsDiagnostics.end())
            {
                fieldType = true;
                continue;
            }
            else
            {
                bool diagNameMatched{false};
                for (int species{0}; species < numspec; ++species)
                {
                    if (diagName == particles[species]->get_name())
                    {
                        particleType = true;
                        diagNameMatched = true;
                        break;
                    }
                }
                if (diagNameMatched)
                {
                    continue;
                }
                else
                {
                    amrex::Assert("Variable name neither particle nor field", "FullDiagnostics.H",
                                  __LINE__, diagName.c_str());
                }
            }
        }
        if (fieldType && particleType)
        {
            amrex::Assert("Particles and fields in the same diagnostics group", "FullDiagnostics.H",
                          __LINE__);
            return "";
        }
        else if (fieldType)
        {
            return "field";
        }
        else if (particleType)
        {
            return "Particle";
        }
        else
        {
            amrex::Assert("Empty diagnostics group ", "FullDiagnostics.H", __LINE__,
                          m_diagName.c_str());
            return "";
        }
    }

    /** Initialize functors that store pointers to the fields requested by the user.
     */

    void initialize_field_functors_and_buffer_data (
        ComputationalDomain& compDom,
        std::map<std::string, amrex::MultiFab*>& fieldsDiagnostics,
        std::map<std::string, amrex::Real>& fieldsScaling,
        amrex::GpuArray<std::shared_ptr<Particle::ParticleGroups<vdim, ndata>>, numspec>& particles,
        amrex::IntVect& nghost)
    {
        std::string diagType{get_diag_type(fieldsDiagnostics, particles)};
        if (diagType == "Particle")
        {
            if (!m_plotParticles)  // needs only to be performed once for all particle groups
            {
                m_plotParticles = true;
                m_particles = particles;
                // Multifab output needs to be defined but not filled when using particles
                amrex::IntVect ngrow{AMREX_D_DECL(0, 0, 0)};  // no ghost cells needed
                m_mfOutputs.push_back(std::make_unique<amrex::MultiFab>(
                    amrex::MultiFab(compDom.m_grid, compDom.m_distriMap, m_nComps, ngrow)));
                m_geomOutput = compDom.m_geom;
            }
        }
        else if (diagType == "field")
        {
            if (m_diagFunctor == "CellCenter")
            {
                m_mfOutputs.push_back(std::make_unique<amrex::MultiFab>(
                    amrex::MultiFab(compDom.m_grid, compDom.m_distriMap, m_nComps,
                                    nghost)));  // ngrow required for interpolation
                m_geomOutput = compDom.m_geom;

                for (int i = 0; i < m_nComps; i++)
                {
                    if (fieldsDiagnostics.find(m_diagVarnames[i]) != fieldsDiagnostics.end())
                    {
                        m_allFieldFunctors[i] = std::make_unique<CellCenterFunctor>(
                            fieldsDiagnostics[m_diagVarnames[i]], fieldsScaling[m_diagVarnames[i]],
                            this->m_crseRatio);
                    }
                    else
                    {
                        amrex::Assert("Field for diagnostics not found", "FullDiagnostics.H",
                                      __LINE__, m_diagVarnames[i].c_str());
                    }
                }
            }
            else if (m_diagFunctor == "raw")
            {
                for (int i{0}; i < m_nComps; ++i)
                {
                    if (fieldsDiagnostics.find(m_diagVarnames[i]) != fieldsDiagnostics.end())
                    {
                        m_allFieldFunctors[i] = std::make_unique<RawFunctor>(
                            fieldsDiagnostics[m_diagVarnames[i]], this->m_crseRatio);
                        // map.second->n_grow -> if we want to write ghost cells too, also to be
                        // changed in operator()
                        m_mfOutputs.push_back(std::make_unique<amrex::MultiFab>(amrex::MultiFab(
                            convert(compDom.m_grid, fieldsDiagnostics[m_diagVarnames[i]]->ixType()),
                            compDom.m_distriMap, 1, 0)));  // ncomp 1, ngrow 0 as just copying
                    }
                    else
                    {
                        amrex::Assert("Field for diagnostics not found", "FullDiagnostics.H",
                                      __LINE__, m_diagVarnames[i].c_str());
                    }
                }
                m_geomOutput = compDom.m_geom;
            }
        }
    }
};

}  // namespace Gempic::Io
#endif  // GEMPICX_FULLDIAGNOSTICS_H_
