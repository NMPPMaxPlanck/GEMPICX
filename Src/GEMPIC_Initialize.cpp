/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <Kokkos_Core.hpp>

#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_ParmParse.H>

#include "GEMPIC_Initialize.H"

namespace Gempic
{

namespace Impl
{
void overwrite_amrex_parser_defaults ()
{
    amrex::ParmParse ppAmrex("amrex");

    // Abort the simulation when we run out of GPU memory
    // By default, AMReX starts swapping to CPU, which makes runs very slow.
    // It's more useful to abort and tell the user to request more GPUs or scale
    // down the simulation.
    // https://amrex-codes.github.io/amrex/docs_html/GPU.html#inputs-parameters
    bool abortOnOutOfGpuMemory = true; // AMReX's default: false
    ppAmrex.queryAdd("abort_on_out_of_gpu_memory", abortOnOutOfGpuMemory);

    // GEMPICX launches all AMReX kernels on the Kokkos default stream
    // (see Impl::set_amrex_external_stream), so AMReX needs only one stream of its own.
    // A user supplied value would be ignored for kernel launches, hence we reject it.
    if (ppAmrex.contains("max_gpu_streams"))
    {
        amrex::Abort(
            "GEMPICX: amrex.max_gpu_streams must not be set. GEMPICX launches all AMReX "
            "kernels on the Kokkos default stream and sets amrex.max_gpu_streams = 1.");
    }
    ppAmrex.add("max_gpu_streams", 1); // AMReX' default: 4
}

void set_amrex_external_stream ()
{
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        Kokkos::device_id() == amrex::Gpu::Device::deviceId(),
        "GEMPICX: Kokkos and AMReX selected different GPUs. Set the same device for both "
        "(e.g. --kokkos-device-id or amrex.device_id).");
#endif
#if defined(KOKKOS_ENABLE_CUDA)
    amrex::Gpu::Device::setExternalStream(Kokkos::Cuda().cuda_stream());
#elif defined(KOKKOS_ENABLE_HIP)
    amrex::Gpu::Device::setExternalStream(Kokkos::HIP().hip_stream());
#endif
}

void reset_amrex_external_stream ()
{
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    amrex::Gpu::Device::resetExternalStream();
#endif
}

} // namespace Impl

void initialize (int argc, char* argv[])
{
    // Pass all arguments explicitly: with amrex::Initialize(argc, argv, func) the function
    // pointer converts to `bool build_parm_parse` and func is never called.
    amrex::Initialize(argc, argv, true, MPI_COMM_WORLD,
                      Gempic::Impl::overwrite_amrex_parser_defaults);
    Kokkos::initialize(argc, argv);
    Impl::set_amrex_external_stream();
}

void finalize ()
{
    Impl::reset_amrex_external_stream();
    Kokkos::finalize();
    amrex::Finalize();
}

} // namespace Gempic
