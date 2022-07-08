#ifndef GEMPIC_AMREX_INIT_H_
#define GEMPIC_AMREX_INIT_H_

#include <AMReX.H>
#include <AMReX_ParmParse.H>

namespace Gempic
{
    /** Overwrite defaults in AMReX Inputs
         *
         * This overwrites defaults in amrex::ParmParse for inputs.
     */
    void
    overwrite_amrex_parser_defaults ()
    {
        amrex::ParmParse pp_amrex("amrex");

        // Abort the simulation when we run out of GPU memory
        // By default, AMReX starts swapping to CPU, which makes runs very slow.
        // It's more useful to abort and tell the user to request more GPUs or scale
        // down the simulation.
        // https://amrex-codes.github.io/amrex/docs_html/GPU.html#inputs-parameters
        bool abort_on_out_of_gpu_memory = true;  // AMReX' default: false
        pp_amrex.queryAdd("abort_on_out_of_gpu_memory", abort_on_out_of_gpu_memory);
    }
} // namespace Gempic

#endif // GEMPIC_AMREX_INIT_H_