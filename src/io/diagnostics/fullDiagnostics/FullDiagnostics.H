#ifndef GEMPICX_FULLDIAGNOSTICS_H_
#define GEMPICX_FULLDIAGNOSTICS_H_

#include "CellCenterFunctor.H"
#include "RawFunctor.H"
#include "FlushFormat.H"
#include "FlushFormatPlotfile.H"

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_BLassert.H>
#include <AMReX_Config.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_Particles.H>

#include <GEMPIC_particle_groups.H>
#include "GEMPIC_parameters.H"

#include <memory>
#include <string>

template <int vdim, int numspec, int ndata>
class FullDiagnostics
{
public:
    FullDiagnostics (int diag_index, std::string name, amrex::Real dt, int save_interval = 100) : m_diag_name{std::move(name)}, m_diag_index{diag_index}, m_dt{dt}, m_save_interval{save_interval}
    {
        ReadParameters();
    }

    /** Whether the last timestep is always dumped */
    bool DoDumpLastTimestep () const {return  m_dump_last_timestep;}

    /** Pack (stack) all fields in the cell-centered output MultiFab m_mf_outputs.
     *
     * Fields are computed (e.g., cell-centered or back-transformed)
       on-the-fly using a functor. 
     */
    void ComputeAndPack ()
    {
        // compute the necessary fields and store result in m_mf_outputs.
        if(!m_plot_particles) 
        {
            for (int i = 0; i < nComps; i++)
            { 
                if (m_mf_outputs.size() > 1)
                {
                    m_all_field_functors[i]->operator()(*(m_mf_outputs[i]), i);
                }
                else
                {
                    m_all_field_functors[i]->operator()(*(m_mf_outputs[0]), i);
                }
            }
        }
    }

    /** Initialize pointers to main fields and allocate output multifab m_mf_outputs. */
    void InitData (Gempic::CompDom::computational_domain& compDom, std::map<std::string, amrex::MultiFab*>& fieldsDiagnostics,
                    std::map<std::string, amrex::Real>& fieldsScaling, 
                    amrex::GpuArray<std::shared_ptr<Gempic::Particles::particle_groups<vdim, ndata>>, numspec> ions, amrex::IntVect& nGhost) 
    {
        // initialize member variables and arrays in base class::Diagnostics
        InitBaseData();
        InitializeFieldFunctorsAndBufferData(compDom, fieldsDiagnostics, fieldsScaling, ions, nGhost);
    }
    
    /** Perform necessary operations with user-defined diagnostic parameters
     *  to filter (coarsen, slice), compute (cell-center, back-transform),
     *  and flush the output data stored in buffers, m_mf_outputs.
     * \param[in] step current timestep
     * \param[in] force_flush used to force-fully write data stored in buffers.
     */
    void FilterComputePackFlush (int step, bool force_flush=false)
    {
        //GEMPIC_PROFILE("Diagnostics::FilterComputePackFlush()");

        if ( DoComputeAndPack (step, force_flush) ) {
            ComputeAndPack();
            Flush(step);
        }
    }
    /// getter for number of components of full diagnostics
    int getNcomp()
    {
        return nComps;
    }

    /** Vector of (pointers to) functors to compute output fields, per level,
     * per component. This allows for simple operations (averaging to
     * cell-center for standard EB fields) as well as more involved operations */
    amrex::Vector<std::unique_ptr<ComputeDiagFunctor>> m_all_field_functors;

private:
    /** Read user-requested parameters for full diagnostics */
    void ReadParameters ()//computational_domain infra)
    {
        // Read list of full diagnostics fields requested by the user.
        Parameters pp_diag_name("fullDiagnostics." + m_diag_name);

        //prefix
        m_file_prefix = "fullDiagnostics/plt_" + m_diag_name; //default for paraview
        pp_diag_name.getOrSet("file_prefix", m_file_prefix);
        
        pp_diag_name.getOrSet("file_min_digits", m_file_min_digits);
        pp_diag_name.getOrSet("dump_last_timestep", m_dump_last_timestep);
        
        //reading variable names for diagnostic
        pp_diag_name.getOrSet("var_names", m_diag_varnames);
        nComps = static_cast<int>(m_diag_varnames.size());
        m_all_field_functors.resize(nComps);

        // reading format of file to be written
        pp_diag_name.getOrSet("format", m_format);

        //reading functor
        pp_diag_name.getOrSet("functor", m_diag_functor);
        
        pp_diag_name.getOrSet("save_interval", m_save_interval);

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_format == "plotfile" //|| m_format == "openpmd" ||
            //m_format == "checkpoint" || m_format == "ascent" ||
            //m_format == "sensei"
            ,"<diag>.format must be plotfile or openpmd or checkpoint or ascent or sensei");
    }

    void InitBaseData ()
    /** Initialize formats and data required to construct the buffers
     */
    {
        // Construct Flush class.
        if (m_format == "plotfile"){
            m_flush_format = std::make_unique<FlushFormatPlotfile<vdim, numspec, ndata>>() ;
        //} else if (m_format == "checkpoint"){
            // creating checkpoint format
           // m_flush_format = std::make_unique<FlushFormatCheckpoint>() ;
        } else if (m_format == "openpmd"){
    #ifdef WARPX_USE_OPENPMD
            m_flush_format = std::make_unique<FlushFormatOpenPMD>(m_diag_name);
    #else
            amrex::Abort("To use openpmd output format, need to compile with USE_OPENPMD=TRUE");
    #endif
        } else {
            amrex::Abort("unknown output format");
        }
    }

    /** Whether to plot raw (i.e., NOT cell-centered) fields */
    bool m_plot_particles = false;
    /** Whether to plot guard cells of raw fields */
    bool m_plot_raw_fields_guards = false;
    /** Name of diagnostics: runtime parameter given in the input file. */
    std::string m_diag_name;
    /** Variable names in diagnostics: runtime parameter given in the input file. */
    amrex::Vector<std::string> m_diag_varnames;
    /** Type of functor: runtime parameter given in the input file. */
    std::string m_diag_functor = "raw"; // raw functor as default
    /** Prefix for output directories */
    std::string m_file_prefix;
    /** Minimum number of digits to iteration number in file name */
    int m_file_min_digits = 6;
    /** Index of diagnostics in MultiDiagnostics::alldiags */
    int m_diag_index;

    int nComps{0};
    
    /** format for output files, "plotfile" or "openpmd" or "sensei" or "ascent"
     * The checkpoint format is applicable for FullDiagnostics only.
     */
    std::string m_format = "plotfile"; // AMReX format
    /** Whether this iteration has already been dumped, to avoid writing data twice */
    int m_already_done = false;
    /** This class is responsible for flushing the data to file */
    std::unique_ptr<FlushFormat<vdim, numspec, ndata>> m_flush_format;
    /** output multifab, where all fields are computed (cell-centered or back-transformed)
     *  and stacked.
     *  The first vector is for total number of snapshots. (=1 for FullDiagnostics)
     *  The second vector is loops over the total number of levels.
     */
    amrex::Vector<std::unique_ptr<amrex::MultiFab>>  m_mf_outputs;

    /** If true, a dump is performed at the last timestep regardless of the required dump timesteps */
    bool m_dump_last_timestep = true;
    
    /** Geometry that defines the domain attributes corresponding to output multifab.
     *  Specifically, the user-defined physical co-ordinates for the diagnostics
     *  is used to construct the geometry information for each MultiFab at
     *  the respective levels. This geometry will be used to write out
     *  plotfile data using the WriteToFile() function
     */
    amrex::Geometry m_geom_output;

    amrex::GpuArray<std::shared_ptr<Gempic::Particles::particle_groups<vdim, ndata>>, numspec> m_particles;

    /** Coarsening ratio such that, fields are averaged to the coarsened grid.
      * The ratio should render the grid to be coarsenable (as defined by AMReX). */
    amrex::IntVect m_crse_ratio = amrex::IntVect(1);

    /* simulation dt */
    amrex::Real m_dt;

    /** Determines timesteps at which full diagnostics are written to file */
    int m_save_interval = 100;

    /** Flush m_mf_outputs and particles to file for the i^th buffer */
    inline void Flush (int step)
    {
        if (m_mf_outputs.size() > 1)
        {
            for (int i{0}; i < m_mf_outputs.size(); ++i)
            {
                std::string file_prefix = m_file_prefix + "_" + m_diag_varnames[i]; 
                amrex::Vector<std::string> singleEntryVarnameVector{m_diag_varnames[i]};
                m_flush_format->WriteToFile(
                    singleEntryVarnameVector, *(m_mf_outputs[i]), m_geom_output, step,
                    step*m_dt, m_particles, 1, file_prefix,
                    m_file_min_digits, m_plot_particles);
            }
        }
        else
        {
            m_flush_format->WriteToFile(
                m_diag_varnames, *(m_mf_outputs[0]), m_geom_output, step,
                step*m_dt, m_particles, 1, m_file_prefix,
                m_file_min_digits, m_plot_particles);
        }
    }

    inline bool DoComputeAndPack (int step, bool force_flush=false)
    /** whether to flush at this time step
     * \param[in] step current time step
     * \param[in] force_flush if true, return true for any step
     * \return bool, whether to flush
     */
    {
        // Data must be computed and packed for full diagnostics
        // whenever the data needs to be flushed.
        return (force_flush || (step % m_save_interval == 0));
    }

    /**
     * Figure out if particles or fields are being plotted. Breaks if particle groups and fields have the identical names.
    */
    std::string get_diag_type (std::map<std::string, amrex::MultiFab*>& fieldsDiagnostics, 
                        amrex::GpuArray<std::shared_ptr<Gempic::Particles::particle_groups<vdim, ndata>>, numspec>& particles)
    {
        bool particleType{false};
        bool fieldType{false};
        for (const auto &diagName : m_diag_varnames)
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
                    if (diagName == particles[species]->getName())
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
                    amrex::Assert("Variable name neither particle nor field", "FullDiagnostics.H" , __LINE__, diagName.c_str());
                }
            }
        }
        if (fieldType && particleType)
        {
            amrex::Assert("Particles and fields in the same diagnostics group", "FullDiagnostics.H", __LINE__);
            return "";
        }
        else if (fieldType)
        {
            return "field";
        }
        else if (particleType)
        {
            return "particle";
        }
        else
        {
            amrex::Assert("Empty diagnostics group ", "FullDiagnostics.H", __LINE__, m_diag_name.c_str());
            return "";
        }
    }

    /** Initialize functors that store pointers to the fields requested by the user.
      */
    
    void InitializeFieldFunctorsAndBufferData (Gempic::CompDom::computational_domain& compDom, std::map<std::string, amrex::MultiFab*>& fieldsDiagnostics, 
                                               std::map<std::string, amrex::Real>& fieldsScaling, 
                                            amrex::GpuArray<std::shared_ptr<Gempic::Particles::particle_groups<vdim, ndata>>, numspec>& particles, amrex::IntVect& nghost)
    {
        std::string diag_type{get_diag_type(fieldsDiagnostics, particles)};
        if (diag_type == "particle")
        {
            if(!m_plot_particles) // needs only to be performed once for all particle groups 
            { 
                m_plot_particles = true;
                m_particles = particles;
                // Multifab output needs to be defined but not filled when using particles
                amrex::IntVect ngrow{AMREX_D_DECL(0,0,0)}; // no ghost cells needed
                m_mf_outputs.push_back(std::make_unique<amrex::MultiFab> (amrex::MultiFab(compDom.grid, compDom.distriMap, nComps, ngrow)));
                m_geom_output = compDom.geom;
            }
        }
        else if (diag_type == "field")
        {
            if (m_diag_functor == "CellCenter")
            {
                m_mf_outputs.push_back(std::make_unique<amrex::MultiFab> (amrex::MultiFab(compDom.grid, compDom.distriMap, nComps, nghost))); //ngrow required for interpolation 
                m_geom_output = compDom.geom;

                for (int i = 0; i < nComps; i++)
                {
                    if (fieldsDiagnostics.find(m_diag_varnames[i]) != fieldsDiagnostics.end())
                    {
                        m_all_field_functors[i] = std::make_unique<CellCenterFunctor>(fieldsDiagnostics[m_diag_varnames[i]], fieldsScaling[m_diag_varnames[i]], this->m_crse_ratio);
                    }
                    else
                        amrex::Assert("Field for diagnostics not found",  "FullDiagnostics.H" , __LINE__, m_diag_varnames[i].c_str());
                }
            } 
            else if (m_diag_functor == "raw")
            {
                for (int i{0}; i < nComps; ++i)
                {
                    if (fieldsDiagnostics.find(m_diag_varnames[i]) != fieldsDiagnostics.end())
                    {
                        m_all_field_functors[i] = std::make_unique<RawFunctor>(fieldsDiagnostics[m_diag_varnames[i]], this->m_crse_ratio);
                        //map.second->n_grow -> if we want to write ghost cells too, also to be changed in operator()
                        m_mf_outputs.push_back(std::make_unique<amrex::MultiFab> (amrex::MultiFab(convert(compDom.grid, fieldsDiagnostics[m_diag_varnames[i]]->ixType()), compDom.distriMap, 1, 0))); //ncomp 1, ngrow 0 as just copying
                    }
                    else
                        amrex::Assert("Field for diagnostics not found",  "FullDiagnostics.H" , __LINE__, m_diag_varnames[i].c_str());
                }
                m_geom_output = compDom.geom;
            }
        }
    }
};

#endif // GEMPICX_FULLDIAGNOSTICS_H_
