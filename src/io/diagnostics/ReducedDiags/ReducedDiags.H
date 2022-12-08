/* Copyright 2019-2020 Maxence Thevenet, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef REDUCEDDIAGS_REDUCEDDIAGS_H_
#define REDUCEDDIAGS_REDUCEDDIAGS_H_

//#include "Utils/IntervalsParser.H"

#include <AMReX_REAL.H>
#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Utility.H>

#include <string>
#include <vector>
#include <fstream>
#include <iomanip>

/**
 *  Base class for reduced diagnostics. Each type of reduced diagnostics is
 *  implemented in a derived class, and must override the (pure virtual)
 *  function ComputeDiags.
 */
template <int vdim, int numspec, int degx, int degy, int degz, int degree, int ndata>
class ReducedDiags
{
public:

    /// output path (default)
    std::string m_path = "./";

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

    // Domain Volume
    amrex::Real vol = 1.0;

    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    ReducedDiags (std::string rd_name)
    {
        m_rd_name = rd_name;
        //BackwardCompatibility();

        amrex::ParmParse pp_rd_name(m_rd_name);

        // read path
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
    virtual ~ReducedDiags () = default;

    /**
     * function to initialize data after amr
     * levels are initialized.
     */
    virtual void InitData (){}

    /** Redistribute parallel data structures during load balance
     */
    virtual void LoadBalance (){}

    /**
     * function to compute diags
     *
     * @param[in] step current time step
     */
    virtual void ComputeDiags (int step, computational_domain infra, maxwell_yee<vdim> *mw_yee,
                        amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec> &part_gr) = 0;

    /**
     * write to file function
     *
     * @param[in] step current time step
     */
    virtual void WriteToFile (amrex::Real dt, int step) const
    {
        // open file
        std::ofstream ofs{m_path + m_rd_name + "." + m_extension,
        std::ofstream::out | std::ofstream::app};

        // write step
        ofs << step+1;

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
    // end ReducedDiags::WriteToFile
};

#endif
