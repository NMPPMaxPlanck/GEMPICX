/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_REAL.H>

#include "GEMPIC_DiscreteTime.H"
#include "GEMPIC_HDF5Interface.H"

namespace Gempic
{

DiscreteTime::DiscreteTime(amrex::Real dt, size_t finalStep, size_t initialStep) :
    m_dt{dt}, m_initialStep{initialStep}, m_currentStep{initialStep}, m_finalStep{finalStep} {};

amrex::Real const& DiscreteTime::dt() const { return m_dt; }
amrex::Real DiscreteTime::t() const { return m_currentStep * m_dt; }
amrex::Real DiscreteTime::initial() const { return m_initialStep * m_dt; };
amrex::Real DiscreteTime::final() const { return m_finalStep * m_dt; };
size_t const& DiscreteTime::current_step() const { return m_currentStep; }
size_t const& DiscreteTime::initial_step() const { return m_initialStep; }
size_t const& DiscreteTime::final_step() const { return m_finalStep; }

bool DiscreteTime::continue_simulation() const { return m_currentStep <= m_finalStep; };
void DiscreteTime::step() { m_currentStep++; };

void serialize (std::string const& label, DiscreteTime const& time, H5GroupHandle const& group)
{
#if GEMPIC_USE_HDF5
    H5GroupHandle timeGroup{group.h5id(), label, Gempic::H5GroupHandle::Mode::CreateExclusive};

    // 2) All attributes are scalar → reuse a single scalar dataspace
    H5DataspaceHandle scalarSpace{H5DataspaceHandle::Scalar{}};

    H5AttributeHandle dtAttr{timeGroup.h5id(), "dt", Gempic::H5AttributeHandle::Mode::Create,
                             H5T_NATIVE_DOUBLE, scalarSpace.h5id()};
    auto dt = static_cast<double>(time.dt());
    check_hdf5(H5Awrite(dtAttr.h5id(), H5T_NATIVE_DOUBLE, &dt));

    H5AttributeHandle curAttr{timeGroup.h5id(), "current_step",
                              Gempic::H5AttributeHandle::Mode::Create, H5T_NATIVE_UINT64,
                              scalarSpace.h5id()};
    auto n = static_cast<size_t>(time.current_step());
    check_hdf5(H5Awrite(curAttr.h5id(), H5T_NATIVE_UINT64, &n));

    // final_step attribute (m_N)
    H5AttributeHandle finAttr{timeGroup.h5id(), "final_step",
                              Gempic::H5AttributeHandle::Mode::Create, H5T_NATIVE_UINT64,
                              scalarSpace.h5id()};
    auto N = static_cast<size_t>(time.final_step());
    check_hdf5(H5Awrite(finAttr.h5id(), H5T_NATIVE_UINT64, &N));
#else
    // Remove unused warnings by casting datatype
    UNUSED(label);
    UNUSED(time);
    UNUSED(group);
    throw_hdf5_unavailable();
#endif
}

void deserialize (std::string const& label, DiscreteTime& time, H5GroupHandle const& group)
{
#if GEMPIC_USE_HDF5
    H5GroupHandle timeGroup{group.h5id(), label, Gempic::H5GroupHandle::Mode::ReadWrite};
    H5DataspaceHandle scalarSpace{H5DataspaceHandle::Scalar{}};

    H5AttributeHandle dtAttr{timeGroup.h5id(), "dt", Gempic::H5AttributeHandle::Mode::ReadWrite,
                             H5T_NATIVE_DOUBLE, scalarSpace.h5id()};
    double dt{};
    check_hdf5(H5Aread(dtAttr.h5id(), H5T_NATIVE_DOUBLE, &dt));

    H5AttributeHandle curAttr{timeGroup.h5id(), "current_step",
                              Gempic::H5AttributeHandle::Mode::ReadWrite, H5T_NATIVE_UINT64,
                              scalarSpace.h5id()};
    size_t n{};
    check_hdf5(H5Aread(curAttr.h5id(), H5T_NATIVE_UINT64, &n));

    H5AttributeHandle finAttr{timeGroup.h5id(), "final_step",
                              Gempic::H5AttributeHandle::Mode::ReadWrite, H5T_NATIVE_UINT64,
                              scalarSpace.h5id()};
    size_t N{};
    check_hdf5(H5Aread(finAttr.h5id(), H5T_NATIVE_UINT64, &N));

    time = DiscreteTime{dt, N, n};
#else
    // Remove unused warnings by casting datatype
    UNUSED(label);
    UNUSED(time);
    UNUSED(group);
    throw_hdf5_unavailable();
#endif
};
} // namespace Gempic
