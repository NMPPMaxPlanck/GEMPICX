/* Copyright 2019-2020 Maxence Thevenet, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
/*
* Modified for GEMPICX
*/

#ifndef REDUCEDDIAGNOSTICS_REDUCEDDIAGNOSTICS_H_
#define REDUCEDDIAGNOSTICS_REDUCEDDIAGNOSTICS_H_

//#include "Utils/IntervalsParser.H"
#include <GEMPIC_Fields.H>

#include <AMReX_REAL.H>
#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>

#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <filesystem>

/**
 *  Base class for reduced diagnostics. Each type of reduced diagnostics is
 *  implemented in a derived class, and must override the (pure virtual)
 *  function ComputeDiags.
 */
template <int vdim, int numspec, int degx, int degy, int degz, int degree, int ndata>
class ReducedDiagnostics
{
public:

    /// output path (default)
    std::string m_dirName = "reducedDiagnostics";
    
    std::string m_path = m_dirName + "/";

    /// output extension (default)
    std::string m_extension = "txt";

    /// diags name
    std::string m_rd_name;

    /// check if it is a restart run
    int m_IsNotRestart = 1;

    /// separator in the output file
    std::string m_sep = " ";

    /// output data
    std::vector<amrex::Real> m_data;

    /// create fields needed for computations if they are missing
    bool m_computeMissingFields;

    /// deRham complex
    std::shared_ptr<GEMPIC_Fields::DeRhamComplex> m_deRham;

    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    ReducedDiagnostics (std::string rd_name)
    {
        m_rd_name = rd_name;
        //BackwardCompatibility();

        amrex::ParmParse pp_rd_name(m_rd_name);

        // read path
        if (!(std::filesystem::exists(m_dirName)))
        {
        std::filesystem::create_directories(m_dirName);
        }
        pp_rd_name.query("path", m_path);

        // read extension
        pp_rd_name.query("extension", m_extension);

        if (amrex::ParallelDescriptor::IOProcessor())
        {
            // create folder
            //if (!UtilCreateDirectory(m_path, 0755))
            //{ CreateDirectoryFailed(m_path); }

            // replace / create output file
            if ( m_IsNotRestart ) // not a restart
            {
                std::ofstream ofs{m_path+m_rd_name+"."+m_extension, std::ios::trunc};
                ofs.close();
            }
        }
/*
        // read separator
        pp_rd_name.query("separator", m_sep);
*/    }
    // end constructor

    /**
     * Virtual destructor for polymorphism
     */
    virtual ~ReducedDiagnostics () = default;

    /**
     * function to initialize data after amr
     * levels are initialized.
     */
    virtual void InitData (std::shared_ptr<GEMPIC_Fields::DeRhamComplex> drc, bool cmf)
    {
        m_deRham = drc;
        m_computeMissingFields = cmf;
    }

    /** Redistribute parallel data structures during load balance
     */
    virtual void LoadBalance (){}

    /**
     * function to compute diags
     */
    virtual void ComputeDiags (Gempic::CompDom::computational_domain& infra, std::map<std::string, amrex::MultiFab*>& fieldsDiagnostics, 
                        amrex::GpuArray<std::shared_ptr<Gempic::Particles::particle_groups<vdim, ndata>>, numspec> &part_gr) = 0;

    /**
     * write to file function
     */
    virtual void WriteToFile (amrex::Real dt, int step) const
    {
        // open file
        std::ofstream ofs{m_path + m_rd_name + "." + m_extension,
        std::ofstream::out | std::ofstream::app};

        // write step
        ofs << step;

        ofs << m_sep;

        // set precision
        ofs << std::fixed << std::setprecision(14) << std::scientific;

        // write time
        ofs << dt * (step);

        // loop over data size and write
        for (const auto& item : m_data) ofs << m_sep << item;

        // end loop over data size

        // end line
        ofs << std::endl;

        // close file
        ofs.close();
    }
    // end ReducedDiagnostics::WriteToFile
};

#endif
