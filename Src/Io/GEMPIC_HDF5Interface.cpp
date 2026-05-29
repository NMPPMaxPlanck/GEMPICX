/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
/**
 * This file is part of the BSL6D project
 * https://gitlab.mpcdf.mpg.de/bsl6d/bsl6d/
 *
 * ISC License
 *
 * Copyright 2023 by Nils Schild (nils.schild@ipp.mpg.de)
 *
 * Permission to use, copy, modify, and/or distribute this software for any purpose with
 * or without fee is hereby granted, provided that the above copyright notice and this
 * permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO
 * THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO
 * EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL
 * DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER
 * IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
 * CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#if GEMPIC_USE_HDF5
#include <algorithm>
#include <iostream>
#else
#include <stdexcept>
#endif

#if GEMPIC_USE_HDF5
#include "GEMPIC_ComputationalDomain.H"
#endif
#include "GEMPIC_HDF5Interface.H"

namespace Gempic
{

#if GEMPIC_USE_HDF5

H5FileHandle::H5FileHandle(std::string filename, Mode mode, MPI_Comm comm) :
    m_filename{filename + s_extension}, m_mode{mode}
{
    open(comm);
}

H5FileHandle::~H5FileHandle() { close(); }

H5FileHandle::H5FileHandle(H5FileHandle&& other) noexcept :
    m_filename(std::move(other.m_filename)), m_file(other.m_file)
{
    other.m_file = -1;
}

H5FileHandle& H5FileHandle::operator=(H5FileHandle&& other) noexcept
{
    if (this != &other)
    {
        close();
        m_filename = std::move(other.m_filename);
        m_file = other.m_file;
        other.m_file = -1;
    }
    return *this;
}

void H5FileHandle::open (MPI_Comm comm)
{
    hid_t commProp = check_hdf5(H5Pcreate(H5P_FILE_ACCESS));
    check_hdf5(H5Pset_fapl_mpio(commProp, comm, MPI_INFO_NULL));
    // Suggestions taken from
    // https://github.com/HDFGroup/hdf5/blob/develop/HDF5Examples/C/H5PAR/ph5_hyperslab_by_row.c#L58
    check_hdf5(H5Pset_all_coll_metadata_ops(commProp, true));
    check_hdf5(H5Pset_coll_metadata_write(commProp, true));

    switch (m_mode)
    {
        case Mode::ReadOnly:
            m_file = check_hdf5(H5Fopen(m_filename.c_str(), H5F_ACC_RDONLY, commProp));
            break;
        case Mode::ReadWrite:
            m_file = check_hdf5(H5Fopen(m_filename.c_str(), H5F_ACC_RDWR, commProp));
            break;
        case Mode::Create:
            m_file =
                check_hdf5(H5Fcreate(m_filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, commProp));
            break;
        case Mode::CreateExclusive:
            m_file = check_hdf5(H5Fcreate(m_filename.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, commProp));
            break;
    }

    check_hdf5(H5Pclose(commProp));
}

void H5FileHandle::close () noexcept
{
    if (m_file >= 0)
    {
        H5Fclose(m_file);
        m_file = -1;
    }
}

H5GroupHandle::H5GroupHandle(hid_t location, std::string groupName, Mode mode) :
    m_name(std::move(groupName))
{
    open(location, mode);
}

H5GroupHandle::~H5GroupHandle() { close(); }

H5GroupHandle::H5GroupHandle(H5GroupHandle&& other) noexcept :
    m_name(std::move(other.m_name)), m_group(other.m_group)
{
    other.m_group = -1;
}

H5GroupHandle& H5GroupHandle::operator=(H5GroupHandle&& other) noexcept
{
    if (this != &other)
    {
        close();
        m_name = std::move(other.m_name);
        m_group = other.m_group;
        other.m_group = -1;
    }
    return *this;
}

bool H5GroupHandle::exists (hid_t location, std::string const& groupName)
{
    htri_t exists = H5Lexists(location, groupName.c_str(), H5P_DEFAULT);
    return exists > 0;
}

void H5GroupHandle::open (hid_t location, Mode mode)
{
    switch (mode)
    {
        case Mode::ReadWrite:
            m_group = check_hdf5(H5Gopen2(location, m_name.c_str(), H5P_DEFAULT));
            break;
        case Mode::Create:
        {
            if (exists(location, m_name))
            {
                m_group = check_hdf5(H5Gopen2(location, m_name.c_str(), H5P_DEFAULT));
            }
            else
            {
                m_group = check_hdf5(
                    H5Gcreate2(location, m_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
            }
            break;
        }
        case Mode::CreateExclusive:
            m_group = check_hdf5(
                H5Gcreate2(location, m_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
            break;
    }
}

void H5GroupHandle::close () noexcept
{
    if (m_group >= 0)
    {
        H5Gclose(m_group);
        m_group = -1;
    }
}

H5DataspaceHandle::H5DataspaceHandle(H5DataspaceHandle::Scalar)
{
    open(H5DataspaceHandle::Scalar{});
}

H5DataspaceHandle::H5DataspaceHandle(std::vector<hsize_t> const& dims) { open(dims); }

H5DataspaceHandle::H5DataspaceHandle(DiscreteGrid const& grid)
{
    std::vector<hsize_t> dims{GEMPIC_D_PAD_ONE(static_cast<hsize_t>(grid.size(Direction::xDir)),
                                               static_cast<hsize_t>(grid.size(Direction::yDir)),
                                               static_cast<hsize_t>(grid.size(Direction::zDir)))};
    // HDF5 expects the dimensions in C order, i.e. z, y, x for a 3D grid
    // Therefore, data is transposed when written to the file.
    // This is achieved by reversing the order of the dimensions when creating the dataspace.
    std::reverse(dims.begin(), dims.end());
    *this = H5DataspaceHandle{dims};
}

H5DataspaceHandle::~H5DataspaceHandle() { close(); }

H5DataspaceHandle::H5DataspaceHandle(H5DataspaceHandle&& other) noexcept :
    m_dataspace(other.m_dataspace)
{
    other.m_dataspace = -1;
}

H5DataspaceHandle& H5DataspaceHandle::operator=(H5DataspaceHandle&& other) noexcept
{
    if (this != &other)
    {
        close();
        m_dataspace = other.m_dataspace;
        other.m_dataspace = -1;
    }
    return *this;
}

void H5DataspaceHandle::open (H5DataspaceHandle::Scalar)
{
    m_dataspace = check_hdf5(H5Screate(H5S_SCALAR));
}

void H5DataspaceHandle::open (std::vector<hsize_t> const& dims)
{
    m_dataspace = check_hdf5(H5Screate_simple(dims.size(), dims.data(), nullptr));
}

void H5DataspaceHandle::close () noexcept
{
    if (m_dataspace >= 0)
    {
        check_hdf5(H5Sclose(m_dataspace));
        m_dataspace = -1;
    }
}

H5DatasetHandle::H5DatasetHandle(
    hid_t location, std::string name, Mode mode, hid_t datatype, hid_t dataspace)

{
    open(location, name, mode, datatype, dataspace);
}

H5DatasetHandle::~H5DatasetHandle() { close(); }

H5DatasetHandle::H5DatasetHandle(H5DatasetHandle&& other) noexcept : m_dataset(other.m_dataset)
{
    other.m_dataset = -1;
}

H5DatasetHandle& H5DatasetHandle::operator=(H5DatasetHandle&& other) noexcept
{
    if (this != &other)
    {
        close();
        m_dataset = other.m_dataset;
        other.m_dataset = -1;
    }
    return *this;
}

bool H5DatasetHandle::exists (hid_t location, std::string const& name)
{
    htri_t exists = H5Lexists(location, name.c_str(), H5P_DEFAULT);
    return exists > 0;
}

void H5DatasetHandle::open (
    hid_t location, std::string const& name, Mode mode, hid_t datatype, hid_t dataspace)
{
    switch (mode)
    {
        case Mode::ReadWrite:
            m_dataset = check_hdf5(H5Dopen2(location, name.c_str(), H5P_DEFAULT));
            break;
        case Mode::Create:
        {
            if (exists(location, name))
            {
                m_dataset = check_hdf5(H5Dopen2(location, name.c_str(), H5P_DEFAULT));
            }
            else
            {
                m_dataset = check_hdf5(H5Dcreate2(location, name.c_str(), datatype, dataspace,
                                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
            }
            break;
        }
        case Mode::CreateExclusive:
            m_dataset = check_hdf5(H5Dcreate2(location, name.c_str(), datatype, dataspace,
                                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
            break;
    }
}

void H5DatasetHandle::close () noexcept
{
    if (m_dataset >= 0)
    {
        H5Dclose(m_dataset);
        m_dataset = -1;
    }
}

H5AttributeHandle::H5AttributeHandle(
    hid_t location, std::string name, Mode mode, hid_t datatype, hid_t dataspace) :
    m_name(std::move(name))
{
    open(location, m_name, mode, datatype, dataspace);
}

H5AttributeHandle::~H5AttributeHandle() { close(); }

H5AttributeHandle::H5AttributeHandle(H5AttributeHandle&& other) noexcept :
    m_name(std::move(other.m_name)), m_attribute(other.m_attribute)
{
    other.m_attribute = -1;
}

H5AttributeHandle& H5AttributeHandle::operator=(H5AttributeHandle&& other) noexcept
{
    if (this != &other)
    {
        close();
        m_name = std::move(other.m_name);
        m_attribute = other.m_attribute;
        other.m_attribute = -1;
    }
    return *this;
}

void H5AttributeHandle::open (
    hid_t location, std::string const& name, Mode mode, hid_t datatype, hid_t dataspace)
{
    switch (mode)
    {
        case Mode::ReadWrite:
            m_attribute = check_hdf5(H5Aopen(location, name.c_str(), H5P_DEFAULT));
            break;
        case Mode::Create:
        {
            if (exists(location, name))
            {
                m_attribute = check_hdf5(H5Aopen(location, name.c_str(), H5P_DEFAULT));
            }
            else
            {
                m_attribute = check_hdf5(H5Acreate2(location, name.c_str(), datatype, dataspace,
                                                    H5P_DEFAULT, H5P_DEFAULT));
            }
            break;
        }
        case Mode::CreateExclusive:
            m_attribute = check_hdf5(
                H5Acreate2(location, name.c_str(), datatype, dataspace, H5P_DEFAULT, H5P_DEFAULT));
            break;
    }
}

void H5AttributeHandle::close () noexcept
{
    if (m_attribute >= 0)
    {
        H5Aclose(m_attribute);
        m_attribute = -1;
    }
}

bool H5AttributeHandle::exists (hid_t location, std::string const& name)
{
    htri_t exists = H5Aexists(location, name.c_str());
    return exists > 0;
}

namespace Impl
{
hid_t h5_type (float) { return H5T_NATIVE_FLOAT; };
hid_t h5_type (double) { return H5T_NATIVE_DOUBLE; };
} //namespace Impl

#else

[[noreturn]] void throw_hdf5_unavailable ()
{
    throw std::runtime_error("GEMPICX was built without HDF5 support");
}

#endif

} // namespace Gempic
