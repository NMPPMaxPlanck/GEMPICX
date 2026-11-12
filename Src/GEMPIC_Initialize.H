/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef GEMPIC_INITIALIZE_H_
#define GEMPIC_INITIALIZE_H_

namespace Gempic
{

namespace Impl
{
void overwrite_amrex_parser_defaults();
/** Let AMReX launch all kernels on the Kokkos default stream */
void set_amrex_external_stream();

/** Undo set_amrex_external_stream. Call before Kokkos::finalize. No-op in CPU builds. */
void reset_amrex_external_stream();
} // namespace Impl

void initialize(int argc, char* argv[]);
void finalize();

} // namespace Gempic

#endif // GEMPIC_INITIALIZE_H_
