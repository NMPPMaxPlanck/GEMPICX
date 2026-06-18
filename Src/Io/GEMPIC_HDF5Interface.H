/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
/*
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
#ifndef GEMPIC_HDF5INTERFACE_H_
#define GEMPIC_HDF5INTERFACE_H_

#if GEMPIC_USE_HDF5
#include <array>
#include <hdf5.h>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>
#endif

namespace Gempic
{

#if GEMPIC_USE_HDF5

template <typename T>
auto check_hdf5 (T const& status)
{
    if (status < 0)
    {
        H5Eprint(H5E_DEFAULT, stderr);
        throw std::runtime_error("HDF5 Runtime error - More information on stderr.");
    }
    return status;
}

class H5FileHandle
{
public:
    enum class Mode
    {
        ReadOnly,
        ReadWrite,
        Create,
        CreateExclusive
    };
    inline static std::string const s_extension{".h5"};
    H5FileHandle(std::string filename, Mode mode, MPI_Comm comm = MPI_COMM_WORLD);
    ~H5FileHandle();

    H5FileHandle(H5FileHandle const&) = delete;
    H5FileHandle& operator=(H5FileHandle const&) = delete;
    H5FileHandle(H5FileHandle&& other) noexcept;
    H5FileHandle& operator=(H5FileHandle&& other) noexcept;

    hid_t h5id () const noexcept { return m_file; }
    Mode const& mode () const noexcept { return m_mode; }
    std::string const& name () const noexcept { return m_filename; }

private:
    void open(MPI_Comm comm);
    void close() noexcept;

    std::string m_filename;
    Mode m_mode;
    hid_t m_file = -1;
};

class H5GroupHandle
{
public:
    enum class Mode
    {
        ReadWrite,
        Create,
        CreateExclusive
    };

    H5GroupHandle(hid_t location, std::string groupName, Mode mode);
    ~H5GroupHandle();

    H5GroupHandle(H5GroupHandle const&) = delete;
    H5GroupHandle& operator=(H5GroupHandle const&) = delete;
    H5GroupHandle(H5GroupHandle&& other) noexcept;
    H5GroupHandle& operator=(H5GroupHandle&& other) noexcept;

    hid_t h5id () const noexcept { return m_group; }
    std::string const& name () const noexcept { return m_name; }

    static bool exists(hid_t location, std::string const& groupName);

private:
    void open(hid_t location, Mode mode);
    void close() noexcept;

    std::string m_name;
    hid_t m_group = -1;
};

// Forward declarations used by H5DataspaceHandle
class DiscreteGrid;

class H5DataspaceHandle
{
public:
    struct Scalar
    {
    };
    H5DataspaceHandle(Scalar);
    H5DataspaceHandle(std::vector<hsize_t> const& dims);
    H5DataspaceHandle(DiscreteGrid const& grid);
    H5DataspaceHandle(hid_t const&& dataspace);
    ~H5DataspaceHandle();

    H5DataspaceHandle(H5DataspaceHandle const&) = delete;
    H5DataspaceHandle& operator=(H5DataspaceHandle const&) = delete;
    H5DataspaceHandle(H5DataspaceHandle&& other) noexcept;
    H5DataspaceHandle& operator=(H5DataspaceHandle&& other) noexcept;

    hid_t h5id () const noexcept { return m_dataspace; }

private:
    void open(std::vector<hsize_t> const& dims);
    void open(Scalar);
    void close() noexcept;

    hid_t m_dataspace = -1;
};

class H5DatasetHandle
{
public:
    enum class Mode
    {
        ReadWrite,
        Create,
        CreateExclusive
    };

    H5DatasetHandle(hid_t location, std::string name, Mode mode, hid_t datatype, hid_t dataspace);
    ~H5DatasetHandle();

    H5DatasetHandle(H5DatasetHandle const&) = delete;
    H5DatasetHandle& operator=(H5DatasetHandle const&) = delete;
    H5DatasetHandle(H5DatasetHandle&& other) noexcept;
    H5DatasetHandle& operator=(H5DatasetHandle&& other) noexcept;

    hid_t h5id () const noexcept { return m_dataset; }
    std::string const& name () const noexcept { return m_name; }

    static bool exists(hid_t location, std::string const& name);

private:
    void open(hid_t location, std::string const& name, Mode mode, hid_t datatype, hid_t dataspace);
    void close() noexcept;

    std::string m_name;
    hid_t m_dataset = -1;
};

class H5AttributeHandle
{
public:
    enum class Mode
    {
        ReadWrite,
        Create,
        CreateExclusive
    };

    H5AttributeHandle(hid_t location, std::string name, Mode mode, hid_t datatype, hid_t dataspace);
    ~H5AttributeHandle();

    H5AttributeHandle(H5AttributeHandle const&) = delete;
    H5AttributeHandle& operator=(H5AttributeHandle const&) = delete;
    H5AttributeHandle(H5AttributeHandle&& other) noexcept;
    H5AttributeHandle& operator=(H5AttributeHandle&& other) noexcept;

    hid_t h5id () const noexcept { return m_attribute; }
    std::string const& name () const noexcept { return m_name; }

    static bool exists(hid_t location, std::string const& name);

private:
    void open(hid_t location, std::string const& name, Mode mode, hid_t datatype, hid_t dataspace);
    void close() noexcept;

    std::string m_name;
    hid_t m_attribute = -1;
};

namespace Impl
{
hid_t h5_type(float);
hid_t h5_type(double);
} // namespace Impl

#else

[[noreturn]] void throw_hdf5_unavailable();
struct H5FileHandle
{
};
struct H5GroupHandle
{
};

#endif

} /* namespace Gempic */
#endif
