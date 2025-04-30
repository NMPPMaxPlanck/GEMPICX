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

#include "GEMPIC_CustomDiagnosticStrategies.H"
#include "GEMPIC_DiagnosticStrategies.H"
#include "GEMPIC_FlushFormat.H"
#include "GEMPIC_FlushFormatPlotfile.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_ParticleGroups.H"

namespace Gempic::Io
{
template <int vdim, int ndata>
class FullDiagnostics
{
public:
    FullDiagnostics(int diagIndex,
                    std::string name,
                    int saveInterval = 100,
                    std::string saveFolder = "FullDiagnostics") :
        m_diagName{std::move(name)},
        m_diagIndex{diagIndex},
        m_saveInterval{saveInterval},
        m_saveFolder{saveFolder}
    {
        BL_PROFILE("Gempic::Io::FullDiagnostics::FullDiagnostics()");
        read_parameters();
    }

    /** Whether the last timestep is always dumped */
    bool do_dump_last_timestep () const { return m_dumpLastTimestep; }

    /** Pack (stack) all fields in the cell-centered output MultiFab m_mf_outputs.
     *
     * Fields are computed (e.g., cell-centered or back-transformed)
       on-the-fly using a outputProcessor.
     */
    void compute_and_pack ()
    {
        BL_PROFILE("Gempic::Io::FullDiagnostics::compute_and_pack()");
        // compute the necessary fields and store result in m_mf_outputs.
        if (!m_plotParticles)
        {
            for (int i = 0; i < m_nGroupMembers; i++)
            {
                if (m_mfOutputs.size() > 1)
                {
                    m_allFieldOutputProcessors[i]->operator()(*(m_mfOutputs[i]), 0);
                }
                else
                {
                    m_allFieldOutputProcessors[i]->operator()(*(m_mfOutputs[0]), i);
                }
            }
        }
    }

    /** Initialize pointers to main fields and allocate output multifab m_mf_outputs. */
    void init_data (ComputationalDomain const& compDom,
                    std::map<std::string, amrex::MultiFab const> const& fieldMap,
                    std::map<std::string, amrex::Real const> const& fieldsScaling,
                    std::vector<std::shared_ptr<Particle::ParticleGroups<vdim, ndata>>> const& ions)
    {
        BL_PROFILE("Gempic::Io::FullDiagnostics::init_data()");
        // initialize member variables and arrays in base class::Diagnostics
        init_base_data();
        initialize_field_operators_and_buffer_data(compDom, fieldMap, fieldsScaling, ions);
    }

    /** Perform necessary operations with user-defined diagnostic parameters
     *  to filter (coarsen, slice), compute (cell-center, back-transform),
     *  and flush the output data stored in buffers, m_mfOutputs.
     * \param[in] step current timestep
     * \param[in] simTime current time
     * \param[in] forceFlush used to force-fully write data stored in buffers.
     */
    void filter_compute_pack_flush (int step, amrex::Real simTime, bool forceFlush = false)
    {
        BL_PROFILE("Gempic::Io::FullDiagnostics::filter_compute_pack_flush()");
        // GEMPIC_PROFILE("Diagnostics::FilterComputePackFlush()");

        if (do_compute_and_pack(step, forceFlush))
        {
            compute_and_pack();
            flush(step, simTime);
        }
    }
    /// getter for number of components of full diagnostics
    int get_num_group_members () { return m_nGroupMembers; }

    /** Vector of (pointers to) outputProcessors to compute output fields, per level,
     * per component. This allows for simple operations (averaging to
     * cell-center for standard EB fields) as well as more involved operations */
    amrex::Vector<std::unique_ptr<ComputeDiagOutputProcessor>> m_allFieldOutputProcessors;

private:
    /** Read user-requested parameters for full diagnostics */
    void read_parameters () // computational_domain infra)
    {
        BL_PROFILE("Gempic::Io::FullDiagnostics::read_parameters()");
        // Read list of full diagnostics fields requested by the user.
        Parameters ppDiagName("FullDiagnostics." + m_diagName);

        // prefix
        ppDiagName.get_or_set("saveFolder", m_saveFolder);
        m_filePrefix = "plt_" + m_diagName; // default for paraview
        ppDiagName.get_or_set("filePrefix", m_filePrefix);
        m_filePrefix = m_saveFolder + "/" + m_filePrefix;

        ppDiagName.get_or_set("fileMinDigits", m_fileMinDigits);
        ppDiagName.get_or_set("dumpLastTimestep", m_dumpLastTimestep);

        // reading variable names for diagnostic
        ppDiagName.get_or_set("varNames", m_diagVarnames);
        // Check if user provided names for components. Otherwise, call them "var1", "var2", ...
        for (auto varName : m_diagVarnames)
        {
            Io::Parameters compParameters("componentNames");
            amrex::Vector<std::string> compNames;
            compParameters.get_or_set(m_diagName, compNames);
            if (!compNames.empty()) m_fieldComponentNames.emplace(m_diagName, compNames);
        }
        m_nGroupMembers = static_cast<int>(m_diagVarnames.size());
        m_allFieldOutputProcessors.resize(m_nGroupMembers);

        // reading format of file to be written
        ppDiagName.get_or_set("format", m_format);

        // reading outputProcessor
        ppDiagName.get_or_set("outputProcessor", m_diagOutputProcessor);
        if (m_diagOutputProcessor == "Custom")
        {
            ppDiagName.get("customID", m_customId);
        }

        ppDiagName.get_or_set("saveInterval", m_saveInterval);

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_format == "plotfile" //|| m_format == "openpmd" ||
            // m_format == "checkpoint" || m_format == "ascent" ||
            // m_format == "sensei"
            ,
            "<diag>.format must be plotfile"); //or openpmd or checkpoint or ascent or sensei");
    }

    void init_base_data ()
    /** Initialize formats and data required to construct the buffers
     */
    {
        BL_PROFILE("Gempic::Io::FullDiagnostics::init_base_data()");
        // Construct Flush class.
        if (m_format == "plotfile")
        {
            m_flushFormat = std::make_unique<FlushFormatPlotfile<vdim, ndata>>();
            //} else if (m_format == "checkpoint") {
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
    /** Variable names that may differ if one or more fields have multiple components */
    amrex::Vector<amrex::Vector<std::string>> m_actualVarnames;
    // map of component names for multicomponent field multifabs
    std::map<std::string, amrex::Vector<std::string>> m_fieldComponentNames;
    /** Type of outputProcessor: runtime parameter given in the input file. */
    std::string m_diagOutputProcessor = "Raw"; // raw outputProcessor as default
    /** custom ID when m_diagOutputProcessor is custom: parameter given in input file. */
    std::string m_customId;
    /** Prefix for output directories */
    std::string m_filePrefix;
    /** Minimum number of digits to iteration number in file name */
    int m_fileMinDigits = 6;
    /** Index of diagnostics in MultiDiagnostics::alldiags */
    int m_diagIndex;

    int m_nGroupMembers{0};

    /** format for output files, "plotfile" or "openpmd" or "sensei" or "ascent"
     * The checkpoint format is applicable for FullDiagnostics only.
     */
    std::string m_format = "plotfile"; // AMReX format
    /** Whether this iteration has already been dumped, to avoid writing data twice */
    int m_alreadyDone = false;
    /** This class is responsible for flushing the data to file */
    std::unique_ptr<FlushFormat<vdim, ndata>> m_flushFormat;
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
     *  plotfile data using the write_to_file() function
     */
    amrex::Geometry m_geomOutput;

    std::vector<std::shared_ptr<Particle::ParticleGroups<vdim, ndata>>> m_particles;

    /** Coarsening ratio such that fields are averaged to the coarsened grid.
     * The ratio should render the grid to be coarsenable (as defined by AMReX). */
    amrex::IntVect m_crseRatio = amrex::IntVect(1);

    /** Determines timesteps at which full diagnostics are written to file */
    int m_saveInterval;
    std::string m_saveFolder;

    /** Flush m_mf_outputs and particles to file for the i^th buffer */
    inline void flush (int step, amrex::Real simTime)
    {
        BL_PROFILE("Gempic::Io::FullDiagnostics::flush()");
        int nlev{1}; // number of levels. 1 for now.
        for (int i{0}; i < m_mfOutputs.size(); ++i)
        {
            std::string filePrefix = m_filePrefix;
            if (m_mfOutputs.size() > 1)
            {
                filePrefix += "_" + m_diagVarnames[i];
            }

            m_flushFormat->write_to_file(m_actualVarnames[i], *(m_mfOutputs[i]), m_geomOutput, step,
                                         simTime, m_particles, nlev, filePrefix, m_fileMinDigits,
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
        BL_PROFILE("Gempic::Io::FullDiagnostics::do_compute_and_pack()");
        // Data must be computed and packed for full diagnostics
        // whenever the data needs to be flushed.
        return (forceFlush || (step % m_saveInterval == 0));
    }

    enum class DiagnosticType
    {
        particle,
        field,
        noType
    };
    /**
     * Figure out if particles or fields are being plotted. Breaks if particle groups and fields
     * have the identical names.
     */
    DiagnosticType get_diag_type (
        std::map<std::string, amrex::MultiFab const> const& fieldMap,
        std::vector<std::shared_ptr<Particle::ParticleGroups<vdim, ndata>>> const& particles)
    {
        BL_PROFILE("Gempic::Io::FullDiagnostics::get_diag_type()");
        bool particleType{false};
        bool fieldType{false};
        for (auto const& diagName : m_diagVarnames)
        {
            if (fieldMap.find(diagName) != fieldMap.end())
            {
                fieldType = true;
                continue;
            }
            else
            {
                bool diagNameMatched{false};
                for (auto& particleSpecies : particles)
                {
                    if (diagName == particleSpecies->get_name())
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
            return DiagnosticType::noType;
        }
        else if (fieldType)
        {
            return DiagnosticType::field;
        }
        else if (particleType)
        {
            return DiagnosticType::particle;
        }
        else
        {
            amrex::Assert("Empty diagnostics group ", "FullDiagnostics.H", __LINE__,
                          m_diagName.c_str());
            return DiagnosticType::noType;
        }
    }

    /** Initialize outputProcessors that store pointers to the fields requested by the user.
     */

    void initialize_field_operators_and_buffer_data (
        ComputationalDomain const& compDom,
        std::map<std::string, amrex::MultiFab const> const& fieldMap,
        std::map<std::string, amrex::Real const> const& fieldsScaling,
        std::vector<std::shared_ptr<Particle::ParticleGroups<vdim, ndata>>> const& particles)
    {
        BL_PROFILE("Gempic::Io::FullDiagnostics::initialize_field_operators_and_buffer_data()");
        using namespace Impl;
        DiagnosticType diagType{get_diag_type(fieldMap, particles)};

        m_geomOutput = compDom.m_geom;

        if (diagType == DiagnosticType::particle)
        {
            if (m_diagOutputProcessor != "Raw")
            {
                amrex::Warning("outputProcessor argument has no effect for particle diagnostics\n");
            }
            if (!m_plotParticles) // needs only to be performed once for all particle groups
            {
                m_plotParticles = true;
                m_particles = particles;
                // Multifab output needs to be defined but not filled when using particles
                auto ngrow{amrex::IntVect::TheZeroVector()}; // no ghost cells needed
                m_mfOutputs.push_back(std::make_unique<amrex::MultiFab>(
                    compDom.m_grid, compDom.m_distriMap, m_nGroupMembers, ngrow));

                m_actualVarnames.push_back(m_diagVarnames);
            }
        }
        else if (diagType == DiagnosticType::field)
        {
            if (m_diagOutputProcessor == "CellCenter")
            {
                int nGhost = 0;
                int nCompTotal = 0;
                amrex::Vector<std::string> tmpVarnames;

                for (int i{0}; i < m_nGroupMembers; ++i)
                {
                    m_allFieldOutputProcessors[i] = std::make_unique<CellCenterOutputProcessor>(
                        fieldMap.at(m_diagVarnames[i]), fieldsScaling.at(m_diagVarnames[i]),
                        this->m_crseRatio);

                    int nComp = fieldMap.at(m_diagVarnames[i]).nComp();
                    nCompTotal += nComp;

                    auto compNames{get_comp_names(nComp, i)};
                    tmpVarnames.insert(tmpVarnames.end(), compNames.begin(), compNames.end());
                }
                m_actualVarnames.push_back(tmpVarnames);

                m_mfOutputs.push_back(std::make_unique<amrex::MultiFab>(
                    compDom.m_grid, compDom.m_distriMap, nCompTotal, nGhost));
            }
            else if (m_diagOutputProcessor == "Raw")
            {
                for (int i{0}; i < m_nGroupMembers; ++i)
                {
                    m_allFieldOutputProcessors[i] = std::make_unique<RawOutputProcessor>(
                        fieldMap.at(m_diagVarnames[i]), this->m_crseRatio);
                    // nGhost 0 as just copying.
                    int nGhost = 0;
                    int nComp = fieldMap.at(m_diagVarnames[i]).nComp();
                    m_mfOutputs.push_back(std::make_unique<amrex::MultiFab>(
                        fieldMap.at(m_diagVarnames[i]).boxArray(),
                        fieldMap.at(m_diagVarnames[i]).DistributionMap(), nComp, nGhost));

                    m_actualVarnames.push_back(get_comp_names(nComp, i));
                }
            }
            else if (m_diagOutputProcessor == "Custom")
            {
                for (int i{0}; i < m_nGroupMembers; ++i)
                {
                    m_allFieldOutputProcessors[i] = match_custom_id(m_customId)(
                        fieldMap.at(m_diagVarnames[i]), fieldsScaling.at(m_diagVarnames[i]),
                        this->m_crseRatio);
                    // nGhost 0 as just copying.
                    int nGhost = 0;
                    int nComp = fieldMap.at(m_diagVarnames[i]).nComp();
                    m_mfOutputs.push_back(std::make_unique<amrex::MultiFab>(
                        fieldMap.at(m_diagVarnames[i]).boxArray(),
                        fieldMap.at(m_diagVarnames[i]).DistributionMap(), nComp, nGhost));

                    m_actualVarnames.push_back(get_comp_names(nComp, i));
                }
            }
            else
            {
                std::string msg{"Unknown outputProcessor: '" + m_diagOutputProcessor + "'."};
                amrex::Assert(msg.c_str(), "FullDiagnostics.H", __LINE__);
            }
        }
    }

    amrex::Vector<std::string> get_comp_names (int nComp, int i)
    {
        amrex::Vector<std::string> compNames(nComp);
        if (nComp == 1)
        {
            compNames[0] = m_diagVarnames[i];
        }
        else if (m_fieldComponentNames.find(m_diagVarnames[i]) != m_fieldComponentNames.end())
        {
            compNames = m_fieldComponentNames[m_diagVarnames[i]];
        }
        else
        {
            for (int idx{0}; idx < nComp; ++idx)
            {
                compNames[idx] = "var" + std::to_string(idx + 1);
            }
        }

        return compNames;
    }
};

} // namespace Gempic::Io
#endif // GEMPICX_FULLDIAGNOSTICS_H_
