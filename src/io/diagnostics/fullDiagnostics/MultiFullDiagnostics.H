#ifndef GEMPICX_MULTIDIAGNOSTICS_H_
#define GEMPICX_MULTIDIAGNOSTICS_H_

#include "FullDiagnostics.H"

#include <AMReX_Vector.H>
#include <AMReX.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

/**
 * \brief This class contains a vector of all full diagnostics in the simulation.
 * Full diagnostics save all the field and/or particle data at chose time steps
 */
template <int vdim, int numspec, int ndata>
class MultiDiagnostics
{
public:
    MultiDiagnostics (amrex::Real dt)
    {
        ReadParameters();
        /** Resize alldiags and initialize each element to a pointer to a
         * diagnostics. Calls the corresponding diagnostics constructor.
         */

        //diags_names = {"part", "field"}; //default for testing
        ndiags = static_cast<int>(diags_names.size());
        alldiags.resize( ndiags );
        for (int i=0; i<ndiags; i++){
            alldiags[i] = std::make_unique<FullDiagnostics<vdim, numspec, ndata>>(i, diags_names[i], dt, m_default_save_interval);
        }
    }   
    /** \brief Read Input parameters. Called in constructor. */
    void ReadParameters ()
    {
        Parameters params("fullDiagnostics");

        bool enable_diags{false};
        params.getOrSet("enable", enable_diags);
        if (enable_diags) 
        {
            params.get("group_names", diags_names);
            params.getOrSet("save_interval", m_default_save_interval);
        }
    }
    /** \brief Loop over diags in alldiags and call their InitDiags */
    void InitData (Gempic::CompDom::computational_domain& compDom, std::map<std::string, amrex::MultiFab*>& fieldsDiagnostics, 
                    std::map<std::string, amrex::Real>& fieldsScaling, 
                    amrex::GpuArray<std::shared_ptr<Gempic::Particles::particle_groups<vdim, ndata>>, numspec> particles, amrex::IntVect& nghost)
    {
        for( auto& diag : alldiags ){
            diag->InitData(compDom, fieldsDiagnostics, fieldsScaling, particles, nghost);
        }
    }
    /** \brief Called at each iteration. Compute diags and flush. */
    void FilterComputePackFlush (int step, bool force_flush=false)
    {
        for (auto& diag : alldiags){
            diag->FilterComputePackFlush (step, force_flush);
        }
    }
    /** \brief Called only at the last iteration. Loop over each diag and if m_dump_last_timestep
     *         is true, compute diags and flush with force_flush=true. */
    void FilterComputePackFlushLastTimestep (int step)
    {
        for (auto& diag : alldiags){
            if (diag->DoDumpLastTimestep()){
                constexpr bool force_flush = true;
                diag->FilterComputePackFlush (step, force_flush);
            }
        }
    }

    FullDiagnostics<vdim, numspec, ndata>& GetDiag(int idiag) {return *alldiags[idiag]; }
    int getTotalDiags() {return ndiags;}
    int getNcomp(int idiag) {return alldiags[idiag]->getNcomp();}
private:
    /** Vector of pointers to all diagnostics */
    amrex::Vector<std::unique_ptr<FullDiagnostics<vdim, numspec, ndata>> > alldiags;
    int ndiags = 0; /**< number of different diagnostics */
    std::vector<std::string> diags_names;
    int m_default_save_interval{100}; // default save interval for all full diagnostics
};

#endif // GEMPICX_MULTIDIAGNOSTICS_H_
