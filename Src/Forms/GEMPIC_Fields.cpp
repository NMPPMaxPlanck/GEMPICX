/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <cmath>
#include <memory>

#include "GEMPIC_BoundaryConditions.H"
#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_FunctionParse.H"
#include "GEMPIC_HDF5Interface.H"
#include "GEMPIC_NumericalIntegrationDifferentiation.H"
#include "GEMPIC_Parameters.H"

namespace Gempic
{
namespace Impl
{
std::array<DiscreteField::DOFCategory, AMREX_SPACEDIM> convert_to_dof_category (
    std::array<int, AMREX_SPACEDIM> const& from)
{
    using DOF = DiscreteField::DOFCategory;
    return std::array<DOF, AMREX_SPACEDIM>{AMREX_D_DECL(
        static_cast<DOF>(from[0]), static_cast<DOF>(from[1]), static_cast<DOF>(from[2]))};
}
std::array<int, AMREX_SPACEDIM> convert_to_int (
    std::array<DiscreteField::DOFCategory, AMREX_SPACEDIM> const& from)
{
    return std::array<int, AMREX_SPACEDIM>{AMREX_D_DECL(
        static_cast<int>(from[0]), static_cast<int>(from[1]), static_cast<int>(from[2]))};
}
amrex::Box selected_ghost_box (DiscreteField const& df,
                               std::array<size_t, AMREX_SPACEDIM> const& ghostSize,
                               Direction const& dir,
                               Impl::GhostRegion const region)
{
    if (size_t(df.multi_fab().nGrow(dir)) < ghostSize[dir])
    {
        throw std::runtime_error (
            "Selected number of ghost cells are smaller than available number of ghost cells");
    }
    amrex::Box box{df.selected_box()};
    amrex::IntVect low{amrex::lbound(box)};
    low = low + df.multi_fab().nGrowVect();
    amrex::IntVect hi{amrex::ubound(box)};
    hi = hi - df.multi_fab().nGrowVect();
    amrex::IntVect lGhostSize{AMREX_D_DECL(static_cast<int>(ghostSize[0]),
                                           static_cast<int>(ghostSize[1]),
                                           static_cast<int>(ghostSize[2]))};
    switch (region)
    {
        case Impl::GhostRegion::low:
        {
            hi[dir] = low[dir] - 1;
            auto tmp{hi};
            hi = hi + lGhostSize;
            hi[dir] = tmp[dir];
            low = low - lGhostSize;
            return amrex::Box{low, hi};
        }
        case Impl::GhostRegion::up:
        {
            low[dir] = hi[dir] + 1;
            auto tmp{low};
            low = low - lGhostSize;
            low[dir] = tmp[dir];
            hi = hi + lGhostSize;
            return amrex::Box{low, hi};
        }
    }
    // Unreachable
    return amrex::Box{};
}
}; //namespace Impl

//!@cond EXCLUDE_API_DOC
DiscreteField::DiscreteField(std::string const& label,
                             Io::Parameters& params,
                             DiscreteGrid const& discreteGrid,
                             std::array<DOFCategory, AMREX_SPACEDIM> const& dofCategory,
                             Impl::BoundaryConditionConfiguration const& bcConf) :
    m_discreteGrid{discreteGrid}
{
    amrex::BoxArray boxArray{Impl::to_amrex_box(discrete_grid())};
    if (params.exists("ComputationalDomain.maxGridSize"))
    {
        amrex::IntVect maxGridSize;
        params.get("ComputationalDomain.maxGridSize", maxGridSize);
        boxArray.maxSize(maxGridSize);
    }
    if (boxArray.size() < amrex::ParallelContext::NProcsAll())
    {
        amrex::Abort("The DiscreteVectorField " + label + " is initialized with " +
                     std::to_string(boxArray.size()) + " boxes. Available are " +
                     std::to_string(amrex::ParallelContext::NProcsAll()) +
                     " MPI processes. We require #processes <= #boxes. "
                     "Reduce ComputationalDomain.maxGridSize or number of MPI processes");
    }
    amrex::DistributionMapping dm{boxArray};
    amrex::MultiFab data{boxArray, dm, 1, 0};
    amrex::GpuBndryFuncFab<Forms::BoundaryCondition> bc{Forms::BoundaryCondition{
        Impl::to_amrex_idx_type(discrete_grid()), bcConf.m_extrapolationDegree, bcConf.m_grid}};
    for (auto dir : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
    {
        if ((discrete_grid().is_periodic(dir) == true) and
            not(bcConf.m_bcRec.lo(dir) == amrex::BCType::int_dir or
                bcConf.m_bcRec.hi(dir) == amrex::BCType::int_dir))
        {
            throw std::invalid_argument (
                "ERROR: Boundary conditions are not periodic but domain is configured to be "
                "periodic!");
        }
    }
    amrex::PhysBCFunct<amrex::GpuBndryFuncFab<Forms::BoundaryCondition>> boundaryCondition{
        Impl::to_amrex_geometry(discrete_grid()), amrex::Vector<amrex::BCRec>{bcConf.m_bcRec}, bc};

    m_data = std::make_shared<Impl::DiscreteFieldData>(
        label, std::move (data), m_discreteGrid, Impl::convert_to_int(dofCategory),
        std::array<int, AMREX_SPACEDIM>{AMREX_D_DECL(0, 0, 0)}, boundaryCondition);
}

DiscreteField::DiscreteField(std::string const& label,
                             DiscreteGrid const& discreteGrid,
                             std::array<DOFCategory, AMREX_SPACEDIM> const& dofCategory,
                             amrex::BoxArray ba,
                             amrex::DistributionMapping dm,
                             Impl::BoundaryConditionConfiguration const& bcConf) :
    m_discreteGrid{discreteGrid}
{
    if (ba.size() < amrex::ParallelContext::NProcsAll())
    {
        amrex::Abort("The DiscreteVectorField " + label + " is initialized with " +
                     std::to_string(ba.size()) + " boxes. Available are " +
                     std::to_string(amrex::ParallelContext::NProcsAll()) +
                     " MPI processes. We require #processes <= #boxes. "
                     "Reduce ComputationalDomain.maxGridSize or number of MPI processes");
    }
    ba.convert(Impl::to_amrex_idx_type(m_discreteGrid));
    amrex::MultiFab data{ba, dm, 1, 0};
    amrex::GpuBndryFuncFab<Forms::BoundaryCondition> bc{Forms::BoundaryCondition{
        Impl::to_amrex_idx_type(discrete_grid()), bcConf.m_extrapolationDegree, bcConf.m_grid}};
    for (auto dir : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
    {
        if ((discrete_grid().is_periodic(dir) == true) and
            not(bcConf.m_bcRec.lo(dir) == amrex::BCType::int_dir or
                bcConf.m_bcRec.hi(dir) == amrex::BCType::int_dir))
        {
            throw std::invalid_argument (
                "ERROR: Boundary conditions are not periodic but domain is configured to be "
                "periodic!");
        }
    }
    amrex::PhysBCFunct<amrex::GpuBndryFuncFab<Forms::BoundaryCondition>> boundaryCondition{
        Impl::to_amrex_geometry(discrete_grid()), amrex::Vector<amrex::BCRec>{bcConf.m_bcRec}, bc};
    m_data = std::make_shared<Impl::DiscreteFieldData>(
        label, std::move (data), m_discreteGrid, Impl::convert_to_int(dofCategory),
        std::array<int, AMREX_SPACEDIM>{AMREX_D_DECL(0, 0, 0)}, boundaryCondition);
};

DiscreteField::DiscreteField(std::shared_ptr<Impl::DiscreteFieldData> const& data) :
    m_data{data}, m_discreteGrid{data->m_discreteGrid}
{
}

amrex::MultiFab const& DiscreteField::multi_fab() const { return m_data->m_data; }
amrex::MultiFab& DiscreteField::multi_fab() { return m_data->m_data; }
//! @endcond

amrex::Box const& DiscreteField::select_box(amrex::MFIter const& mfi)
{
    // Check normally done by AMReX when setting a view with mfi
    AMREX_ASSERT(this->multi_fab().DistributionMap() == mfi.DistributionMap());
    return this->select_box(mfi.index());
}
amrex::Box const& DiscreteField::select_box(int index)
{
    m_view = this->multi_fab().array(index);
    m_selectedBoxIdx = index;
    return this->multi_fab()[index].box();
}
amrex::Box const& DiscreteField::selected_box() const
{
    return m_data->m_data.get(m_selectedBoxIdx).box();
};

std::array<int, AMREX_SPACEDIM> DiscreteField::ghost_width() const { return m_data->m_ghostWidth; }

void DiscreteField::apply_boundary_conditions (std::array<size_t, AMREX_SPACEDIM> width)
{
    bool increaseGhost{false};
    for (auto dir : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
    {
        if (size_t(m_data->m_ghostWidth[dir]) < width[dir])
        {
            m_data->m_ghostWidth[dir] = width[dir];
            increaseGhost = true;
        }
    }
    amrex::IntVect ng;
    for (int i = 0; i < AMREX_SPACEDIM; i++)
    {
        ng[i] = m_data->m_ghostWidth[i];
    }
    if (increaseGhost)
    {
        amrex::BoxArray ba{this->multi_fab().boxArray()};
        amrex::DistributionMapping dm{this->multi_fab().DistributionMap()};
        amrex::MultiFab dataNew{ba, dm, 1, ng};
        dataNew.LocalCopy(this->multi_fab(), 0, 0, 1, amrex::IntVect{0});
        m_data->m_data = std::move(dataNew);
        m_view = amrex::Array4<amrex::Real>{};
        m_selectedBoxIdx = std::numeric_limits<int>::min();
    }
    this->multi_fab().FillBoundary(ng, Impl::to_amrex_periodicty(this->discrete_grid()));
    for (auto dir : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
    {
        if (not discrete_grid().is_periodic(dir))
        {
            amrex::Warning(
                "Warning: 'DiscreteField::apply_boundary_conditions()' Non periodic "
                "boundaries are not well tested! API and behavior might change");
            m_data->m_boundaryCondition(m_data->m_data, 0, 1, ng, 0.0, 0);
            break;
        }
    }
}

//! @cond EXCLUDE_API_DOC
DiscreteVectorField::DiscreteVectorField(
    std::string const& label,
    Io::Parameters& params,
    std::array<DiscreteGrid, 3> const& discreteGrid,
    std::array<std::array<DiscreteField::DOFCategory, AMREX_SPACEDIM>, 3> const& dofCategory,
    std::array<Impl::BoundaryConditionConfiguration, 3> const& bcConfig) :
    m_label{std::make_shared<std::string>(label)},
    m_fields{DiscreteField{label + "x", params, discreteGrid[Direction::xDir],
                           dofCategory[Direction::xDir], bcConfig[Direction::xDir]},
             DiscreteField{label + "y", params, discreteGrid[Direction::yDir],
                           dofCategory[Direction::yDir], bcConfig[Direction::yDir]},
             DiscreteField{label + "z", params, discreteGrid[Direction::zDir],
                           dofCategory[Direction::zDir], bcConfig[Direction::zDir]}}
{
}

DiscreteVectorField::DiscreteVectorField(
    std::string const& label,
    std::array<DiscreteGrid, 3> const discreteGrid,
    std::array<std::array<DiscreteField::DOFCategory, AMREX_SPACEDIM>, 3> const& dofCategory,
    amrex::BoxArray ba,
    amrex::DistributionMapping dm,
    std::array<Impl::BoundaryConditionConfiguration, 3> const& bcConfig) :
    m_label{std::make_shared<std::string>(label)},
    m_fields{DiscreteField{label + "x", discreteGrid[Direction::xDir], dofCategory[Direction::xDir],
                           ba, dm, bcConfig[Direction::xDir]},
             DiscreteField{label + "y", discreteGrid[Direction::yDir], dofCategory[Direction::yDir],
                           ba, dm, bcConfig[Direction::yDir]},
             DiscreteField{label + "z", discreteGrid[Direction::zDir], dofCategory[Direction::zDir],
                           ba, dm, bcConfig[Direction::zDir]}}
{
}

amrex::MultiFab const& DiscreteVectorField::multi_fab(Direction dir) const
{
    return m_fields[dir].multi_fab();
}
amrex::MultiFab& DiscreteVectorField::multi_fab(Direction dir) { return m_fields[dir].multi_fab(); }
//! @endcond

std::array<amrex::Box, 3> const DiscreteVectorField::select_box(amrex::MFIter const& mfi)
{
    for (Direction dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        // Check normally done by AMReX when setting a view with mfi
        AMREX_ASSERT(this->multi_fab(dir).DistributionMap() == mfi.DistributionMap());
    }
    return this->select_box(mfi.index());
}
std::array<amrex::Box, 3> const DiscreteVectorField::select_box(int index)
{
    std::array<amrex::Box, 3> boxes{};
    for (Direction dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        boxes[dir] = m_fields[dir].select_box(index);
    }
    return boxes;
}

amrex::Box const& DiscreteVectorField::selected_box(Direction dir) const
{
    return m_fields[dir].selected_box();
}

void DiscreteVectorField::apply_boundary_conditions (std::array<size_t, AMREX_SPACEDIM> width)
{
    for (Direction dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        m_fields[dir].apply_boundary_conditions(width);
    }
};

namespace Impl
{

struct H5Subspaces
{
    H5DataspaceHandle m_memory;
    H5DataspaceHandle m_file;
};

H5Subspaces create_subspaces (DiscreteField const& f, H5DatasetHandle const& dataset)
{
    amrex::Box const box{f.selected_box()};

    std::vector<hsize_t> count{};
    std::vector<hsize_t> boxSize{};
    for (auto axis : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        if (axis < AMREX_SPACEDIM)
        {
            if (axis == Direction::xDir) boxSize.push_back(static_cast<hsize_t>(length(box).x));
            if (axis == Direction::yDir) boxSize.push_back(static_cast<hsize_t>(length(box).y));
            if (axis == Direction::zDir) boxSize.push_back(static_cast<hsize_t>(length(box).z));
        }
        else
        {
            boxSize.push_back(1);
        }
    }
    for (auto axis : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        if (axis < AMREX_SPACEDIM)
        {
            if (axis == Direction::xDir)
            {
                count.push_back(boxSize[Direction::xDir] -
                                static_cast<hsize_t>(2 * f.ghost_width()[Direction::xDir]));
            }
            if (axis == Direction::yDir)
            {
                count.push_back(boxSize[Direction::yDir] -
                                static_cast<hsize_t>(2 * f.ghost_width()[Direction::yDir]));
            }
            if (axis == Direction::zDir)
            {
                count.push_back(boxSize[Direction::zDir] -
                                static_cast<hsize_t>(2 * f.ghost_width()[Direction::zDir]));
            }
        }
        else
        {
            count.push_back(1);
        }
    }
    std::reverse(count.begin(), count.end());
    std::array<hsize_t, 3> const block{1, 1, 1};
    std::array<hsize_t, 3> const stride{1, 1, 1};
    std::array<hsize_t, 3> offset{0, 0, 0};

    std::reverse(boxSize.begin(), boxSize.end());
    H5DataspaceHandle memoryDataspace{boxSize};
    offset = std::array<hsize_t, 3>{
        GEMPIC_D_PAD(static_cast<hsize_t>(f.ghost_width()[Direction::xDir]),
                     static_cast<hsize_t>(f.ghost_width()[Direction::yDir]),
                     static_cast<hsize_t>(f.ghost_width()[Direction::zDir]))};
    std::reverse(offset.begin(), offset.end());
    check_hdf5(H5Sselect_hyperslab(memoryDataspace.h5id(), H5S_SELECT_SET, offset.data(),
                                   stride.data(), count.data(), block.data()));

    offset = std::array<hsize_t, 3>{GEMPIC_D_PAD(
        static_cast<hsize_t>(box.smallEnd(Direction::xDir) + f.ghost_width()[Direction::xDir]),
        static_cast<hsize_t>(box.smallEnd(Direction::yDir) + f.ghost_width()[Direction::yDir]),
        static_cast<hsize_t>(box.smallEnd(Direction::zDir) + f.ghost_width()[Direction::zDir]))};
    std::reverse(offset.begin(), offset.end());
    H5DataspaceHandle fileSubspace{check_hdf5(H5Dget_space(dataset.h5id()))};
    check_hdf5(H5Sselect_hyperslab(fileSubspace.h5id(), H5S_SELECT_SET, offset.data(),
                                   stride.data(), count.data(), block.data()));

    return H5Subspaces{std::move(memoryDataspace), std::move(fileSubspace)};
}

void write_field_data (DiscreteField& f, H5DatasetHandle const& dataset)
{
    for (amrex::MFIter mfi{f.multi_fab()}; mfi.isValid(); ++mfi)
    {
        f.select_box(mfi);
        H5Subspaces hs = create_subspaces(f, dataset);

        hid_t parallelProperty = check_hdf5(H5Pcreate(H5P_DATASET_XFER));
        check_hdf5(H5Pset_dxpl_mpio(parallelProperty, H5FD_MPIO_COLLECTIVE));

        check_hdf5(H5Dwrite(dataset.h5id(), Impl::h5_type(amrex::Real{}), hs.m_memory.h5id(),
                            hs.m_file.h5id(), H5P_DEFAULT, f.view().dataPtr()));
        check_hdf5(H5Pclose(parallelProperty));
    }
}

void read_field_data (DiscreteField& f, H5DatasetHandle const& dataset)
{
    for (amrex::MFIter mfi{f.multi_fab()}; mfi.isValid(); ++mfi)
    {
        f.select_box(mfi);
        H5Subspaces hs = create_subspaces(f, dataset);

        hid_t parallelProperty = check_hdf5(H5Pcreate(H5P_DATASET_XFER));
        check_hdf5(H5Pset_dxpl_mpio(parallelProperty, H5FD_MPIO_COLLECTIVE));

        check_hdf5(H5Dread(dataset.h5id(), Impl::h5_type(amrex::Real{}), hs.m_memory.h5id(),
                           hs.m_file.h5id(), H5P_DEFAULT, f.view().dataPtr()));
        check_hdf5(H5Pclose(parallelProperty));
    }
}

void serialize_into (DiscreteField& f, H5GroupHandle const& fieldGroup, DiscreteTime const& t)
{
    auto grid = f.discrete_grid();
    if (not H5GroupHandle::exists(fieldGroup.h5id(), "grid"))
    {
        serialize("grid", grid, fieldGroup);
    }

    H5GroupHandle timeSeries{fieldGroup.h5id(), "TIMESERIES", H5GroupHandle::Mode::Create};
    H5GroupHandle timeGroup{timeSeries.h5id(), std::to_string(t.current_step()),
                            H5GroupHandle::Mode::CreateExclusive};
    serialize("simulationTime", t, timeGroup);

    H5DatasetHandle fileDataset{timeGroup.h5id(), "data", H5DatasetHandle::Mode::CreateExclusive,
                                Impl::h5_type(amrex::Real{}), H5DataspaceHandle{grid}.h5id()};
    H5AttributeHandle dofCategory{fileDataset.h5id(), "dofCategory",
                                  H5AttributeHandle::Mode::Create, H5T_NATIVE_INT,
                                  H5DataspaceHandle{std::vector<hsize_t>{AMREX_SPACEDIM}}.h5id()};
    std::array<DiscreteField::DOFCategory, AMREX_SPACEDIM> dofCategoryValue = f.dof_category();
    std::reverse(dofCategoryValue.begin(), dofCategoryValue.end());
    H5Awrite(dofCategory.h5id(), H5T_NATIVE_INT, &dofCategoryValue);

    write_field_data(f, fileDataset);
}

void deserialize_from (DiscreteField& f, H5GroupHandle const& fieldGroup, DiscreteTime const& t)
{
    auto grid = f.discrete_grid();
    DiscreteGrid storedGrid{};
    deserialize("grid", storedGrid, fieldGroup);
    if (storedGrid != grid)
    {
        throw std::runtime_error("H5ReadError: Grid in file does not match grid of discrete field");
    }

    H5GroupHandle timeSeries{fieldGroup.h5id(), "TIMESERIES", H5GroupHandle::Mode::ReadWrite};
    H5GroupHandle timeGroup{timeSeries.h5id(), std::to_string(t.current_step()),
                            H5GroupHandle::Mode::ReadWrite};

    H5DatasetHandle fileDataset{timeGroup.h5id(), "data", H5DatasetHandle::Mode::ReadWrite,
                                Impl::h5_type(amrex::Real{}), H5DataspaceHandle{grid}.h5id()};
    H5AttributeHandle dofCategory{fileDataset.h5id(), "dofCategory",
                                  H5AttributeHandle::Mode::Create, H5T_NATIVE_INT,
                                  H5DataspaceHandle{std::vector<hsize_t>{AMREX_SPACEDIM}}.h5id()};
    std::array<DiscreteField::DOFCategory, AMREX_SPACEDIM> dofCategoryValue;
    H5Aread(dofCategory.h5id(), H5T_NATIVE_INT, &dofCategoryValue);
    if (dofCategoryValue != f.dof_category())
    {
        throw std::runtime_error (
            "H5ReadError: DOF category in file does not match DOF category of discrete field");
    }

    read_field_data(f, fileDataset);
}
} //namespace Impl

void serialize (DiscreteField& f, H5FileHandle const& io, DiscreteTime const& t)
{
#ifdef GEMPIC_USE_HDF5
    H5GroupHandle fieldGroup{io.h5id(), f.label(), H5GroupHandle::Mode::Create};
    H5GroupHandle scalarGroup{fieldGroup.h5id(), "SCALAR", H5GroupHandle::Mode::Create};
    Impl::serialize_into(f, scalarGroup, t);
#else
    UNUSED(f);
    UNUSED(io);
    UNUSED(t);
    throw_hdf5_unavailable();
#endif
}

void deserialize (DiscreteField& f, H5FileHandle const& io, DiscreteTime const& t)
{
#ifdef GEMPIC_USE_HDF5
    H5GroupHandle fieldGroup{io.h5id(), f.label(), H5GroupHandle::Mode::ReadWrite};
    H5GroupHandle scalarGroup{fieldGroup.h5id(), "SCALAR", H5GroupHandle::Mode::ReadWrite};
    Impl::deserialize_from(f, scalarGroup, t);
#else
    UNUSED(f);
    UNUSED(io);
    UNUSED(t);
    throw_hdf5_unavailable();
#endif
}

void serialize (DiscreteVectorField& f, H5FileHandle const& io, DiscreteTime const& t)
{
#ifdef GEMPIC_USE_HDF5
    H5GroupHandle fieldGroup{io.h5id(), f.label(), H5GroupHandle::Mode::Create};
    for (Direction dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        H5GroupHandle componentGroup{fieldGroup.h5id(), direction_to_string(dir),
                                     H5GroupHandle::Mode::Create};
        Impl::serialize_into(f[dir], componentGroup, t);
    }
#else
    UNUSED(f);
    UNUSED(io);
    UNUSED(t);
    throw_hdf5_unavailable();
#endif
}

void deserialize (DiscreteVectorField& vf, H5FileHandle const& io, DiscreteTime const& t)
{
#ifdef GEMPIC_USE_HDF5
    H5GroupHandle fieldGroup{io.h5id(), vf.label(), H5GroupHandle::Mode::ReadWrite};
    for (Direction dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        H5GroupHandle componentGroup{fieldGroup.h5id(), direction_to_string(dir),
                                     H5GroupHandle::Mode::ReadWrite};
        Impl::deserialize_from(vf[dir], componentGroup, t);
    }
#else
    UNUSED(f);
    UNUSED(io);
    UNUSED(t);
    throw_hdf5_unavailable();
#endif
}

void fill_zero (DiscreteField& field)
{
    auto zero = [] AMREX_GPU_HOST_DEVICE(AMREX_D_DECL(amrex::Real /*x*/, amrex::Real /*y*/,
                                                      amrex::Real /*z*/)) -> amrex::Real
    { return 0.0; };
    Gempic::fill(field, zero);
}

void fill_zero (DiscreteVectorField& field)
{
    auto zero = [] AMREX_GPU_HOST_DEVICE(Direction /*dir*/,
                                         AMREX_D_DECL(amrex::Real /*x*/, amrex::Real /*y*/,
                                                      amrex::Real /*z*/)) -> amrex::Real
    { return 0.0; };
    Gempic::fill(field, zero);
}

DiscreteFieldFunctionParser::DiscreteFieldFunctionParser(std::string const& label,
                                                         Io::Parameters& params)
{
    std::string fullLabel{"FunctionParser." + label};
    auto [parser, function] = Utils::parse_function(fullLabel, params);
    m_parser = parser;
    m_function = function;
}

DiscreteVectorFieldFunctionParser::DiscreteVectorFieldFunctionParser(
    std::array<std::string, 3> const& label, Io::Parameters& params)
{
    std::array<std::string, 3> fullLabels{};
    for (auto const& dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        fullLabels[dir] = "FunctionParser." + label[dir];
    }
    auto [parser, function] = Utils::parse_functions(fullLabels, params);
    m_parser = parser;
    m_function = function;
}

bool is_nan (DiscreteField& a)
{
    auto ma = a.multi_fab().const_arrays();
    amrex::GpuTuple<int> isNan{};
    isNan = amrex::ParReduce(amrex::TypeList<amrex::ReduceOpLogicalOr>{}, amrex::TypeList<int>{},
                             a.multi_fab(),
                             [=] AMREX_GPU_HOST_DEVICE(int boxNo, int ix, int iy,
                                                       int iz) noexcept -> amrex::GpuTuple<int>
                             {
                                 auto aa = ma[boxNo];
                                 return {std::isnan(aa(ix, iy, iz))};
                             });
    //https://amrex-codes.github.io/amrex/docs_html/GPU.html#multifab-reductions
    //It should be noted that the reduction result of ParReduce is local and it is the user's
    //responsibility if MPI communication is needed
    MPI_Allreduce(MPI_IN_PLACE, &isNan, 1, MPI_INT, MPI_LOR,
                  amrex::ParallelContext::CommunicatorAll());
    return amrex::get<0>(isNan);
}

/**
 * Utility functions for DiscreteField and DiscreteVectorField
 */
void operator*=(DiscreteVectorField& field, amrex::Real const& scalar)
{
    for (Direction dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        for (amrex::MFIter mfi{field.multi_fab(dir)}; mfi.isValid(); ++mfi)
        {
            field.select_box(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_HOST_DEVICE(int ix, int iy, int iz)
                               { field(dir, ix, iy, iz) *= scalar; });
        }
    }
}
void operator+=(DiscreteVectorField& a, DiscreteVectorField const& b)
{
    for (Direction dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        for (amrex::MFIter mfi{a.multi_fab(dir)}; mfi.isValid(); ++mfi)
        {
            a.select_box(mfi);
            auto const& otherView{b.multi_fab(dir).array(mfi)};
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_HOST_DEVICE(int ix, int iy, int iz)
                               { a(dir, ix, iy, iz) += otherView(ix, iy, iz); });
        }
    }
}
amrex::Real l_inf_error (DiscreteField& a, DiscreteField& b)
{
    auto ma = a.multi_fab().const_arrays();
    auto mb = b.multi_fab().const_arrays();
    amrex::GpuTuple<amrex::Real, int> res{};
    res = amrex::ParReduce(
        amrex::TypeList<amrex::ReduceOpMax, amrex::ReduceOpLogicalOr>{},
        amrex::TypeList<amrex::Real, int>{}, a.multi_fab(),
        [=] AMREX_GPU_HOST_DEVICE(int boxNo, int ix, int iy,
                                  int iz) noexcept -> amrex::GpuTuple<amrex::Real, int>
        {
            auto aa = ma[boxNo];
            auto ba = mb[boxNo];
            amrex::Real ldiff{aa(ix, iy, iz) - ba(ix, iy, iz)};
            // NaN comparisons always evaluate to false, which is why we check on Nan manually.
            // https://stackoverflow.com/questions/38798791/nan-comparison-rule-in-c-c
            // https://rgambord.github.io/c99-doc/sections/7/12/14/index.html#id2
            return {std::abs(ldiff), static_cast<int>(std::isnan(ldiff))};
        });
    int isNan{amrex::get<1>(res)};
    amrex::Real norm{amrex::get<0>(res)};
    //https://amrex-codes.github.io/amrex/docs_html/GPU.html#multifab-reductions
    //It should be noted that the reduction result of ParReduce is local and it is the user's
    //responsibility if MPI communication is needed
    MPI_Allreduce(MPI_IN_PLACE, &isNan, 1, MPI_INT, MPI_LOR,
                  amrex::ParallelContext::CommunicatorAll());
    if (amrex::get<1>(res)) return std::numeric_limits<amrex::Real>::quiet_NaN();
    MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_MAX,
                  amrex::ParallelContext::CommunicatorAll());
    return norm;
}
std::array<bool, 3> is_nan (DiscreteVectorField& a)
{
    return std::array<bool, 3>{is_nan(a[Direction::xDir]), is_nan(a[Direction::yDir]),
                               is_nan(a[Direction::zDir])};
}
std::array<amrex::Real, 3> l_inf_error (DiscreteVectorField& a, DiscreteVectorField& b)
{
    return std::array<amrex::Real, 3>{l_inf_error(a[Direction::xDir], b[Direction::xDir]),
                                      l_inf_error(a[Direction::yDir], b[Direction::yDir]),
                                      l_inf_error(a[Direction::zDir], b[Direction::zDir])};
}

namespace Impl
{
amrex::Real dot (DiscreteField& a, DiscreteField& b)
{
    AMREX_ALWAYS_ASSERT(a.discrete_grid() == b.discrete_grid());
    AMREX_ALWAYS_ASSERT(a.multi_fab().nComp() == b.multi_fab().nComp());
    auto mask =
        amrex::OwnerMask(a.multi_fab(), Gempic::Impl::to_amrex_periodicty(a.discrete_grid()));
    return amrex::MultiFab::Dot (*mask, a.multi_fab(), 0, b.multi_fab(), 0, a.multi_fab().nComp(),
                                0);
}

amrex::Real dot (DiscreteVectorField& a, DiscreteVectorField& b)
{
    amrex::Real result = 0;
    for (Direction dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        result += dot(a[dir], b[dir]);
    }
    return result;
}
} //namespace Impl

namespace Forms
{
DeRhamComplex::DeRhamComplex(ComputationalDomain const& infra, int const hodgeDegree)
/*  :
  m_fieldRegistry(std::make_unique<FieldRegistry>())*/
{
    BL_PROFILE("Gempic::Forms::DeRhamComplex:: DeRhamComplex()");
    GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE(infra.geometry().Ok(), "Uninitialized geometry");

    m_grid = infra.m_grid;
    m_distriMap = infra.m_distriMap;
    m_geom = infra.m_geom;
    m_hodgeDegree = hodgeDegree;
    m_dx = infra.geometry().CellSizeArray();

    // set index: // 0 -> cell-centered | 1 -> node-centered
    m_primalNodeIndex = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 1, 1)});
    m_primalEdgeIndex[xDir] = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 1, 1)});
    m_primalEdgeIndex[yDir] = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 0, 1)});
    m_primalEdgeIndex[zDir] = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 1, 0)});
    m_primalFaceIndex[xDir] = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 0, 0)});
    m_primalFaceIndex[yDir] = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 1, 0)});
    m_primalFaceIndex[zDir] = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 0, 1)});
    m_primalCellIndex = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 0, 0)});
}

} //namespace Forms

} //namespace Gempic
