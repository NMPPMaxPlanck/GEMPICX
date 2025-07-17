#include <memory>

#include "GEMPIC_BoundaryConditions.H"
#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_NumericalIntegrationDifferentiation.H"
#include "GEMPIC_Parameters.H"

namespace Gempic
{

DiscreteField::DiscreteField(std::string const& label,
                             Io::Parameters& params,
                             DiscreteGrid const& discreteGrid,
                             Grid const& grid,
                             int const& boundaryExtrapolationDegree) :
    m_label{std::make_shared<std::string>(label)}, m_discreteGrid{discreteGrid}
{
    amrex::IntVect maxGridSize;
    params.get("ComputationalDomain.maxGridSize", maxGridSize);

    amrex::BoxArray boxArray{Impl::to_amrex_box(discrete_grid())};
    boxArray.maxSize(maxGridSize);
    if (boxArray.size() < amrex::ParallelContext::NProcsAll())
    {
        amrex::Abort("The DiscreteVectorField " + label + " is initialized with " +
                     std::to_string(boxArray.size()) + " boxes. Available are " +
                     std::to_string(amrex::ParallelContext::NProcsAll()) +
                     " MPI processes. We require #processes <= #boxes. "
                     "Reduce ComputationalDomain.maxGridSize or number of MPI processes");
    }
    amrex::DistributionMapping dm{boxArray};
    m_data = std::make_shared<amrex::MultiFab>(boxArray, dm, 1, 0);
    amrex::GpuBndryFuncFab<Forms::BoundaryCondition> bc{Forms::BoundaryCondition{
        Impl::to_amrex_idx_type(discrete_grid()), boundaryExtrapolationDegree, grid}};
    amrex::Vector<amrex::BCRec> bcRec =
        Forms::load_boundary_condition(Impl::to_amrex_geometry(discrete_grid()), "Default");
    m_boundaryCondition = amrex::PhysBCFunct<amrex::GpuBndryFuncFab<Forms::BoundaryCondition>>{
        Impl::to_amrex_geometry(discrete_grid()), bcRec, bc};
}

amrex::MultiFab const& DiscreteField::multi_fab() const { return *m_data; }
amrex::MultiFab& DiscreteField::multi_fab() { return *m_data; }

amrex::Box const& DiscreteField::select_box(amrex::MFIter const& mfi)
{
    m_view = this->multi_fab().array(mfi);
    m_boxStatus = Impl::BoxStatus::boxSelected;
    m_selectedBoxIdx = mfi.index();
    return this->multi_fab()[mfi].box();
}
amrex::Box const& DiscreteField::selected_box() const
{
    return m_data->get(m_selectedBoxIdx).box();
};
void DiscreteField::set_ghost_size (int width, Direction dir)
{
    if (m_haloWidth[dir] < width)
    {
        m_haloWidth[dir] = width;
        amrex::BoxArray ba{this->multi_fab().boxArray()};
        amrex::DistributionMapping dm{this->multi_fab().DistributionMap()};
        amrex::IntVect ng;
        for (int i = 0; i < AMREX_SPACEDIM; i++)
        {
            ng[i] = m_haloWidth[i];
        }
        amrex::MultiFab mDataNew{ba, dm, this->multi_fab().nComp(), ng};
        mDataNew.ParallelCopy(this->multi_fab());
        m_data = std::make_shared<amrex::MultiFab>(std::move(mDataNew));
        m_view = amrex::Array4<amrex::Real>{};
        m_boxStatus = Impl::BoxStatus::boxChanged;
        m_selectedBoxIdx = std::numeric_limits<int>::min();
    }
};

DiscreteVectorField::DiscreteVectorField(std::string const& label,
                                         Io::Parameters& params,
                                         std::array<DiscreteGrid, 3> const& discreteGrid,
                                         Grid const& grid,
                                         int const& boundaryExtrapolationDegree) :
    m_label{std::make_shared<std::string>(label)}, m_discreteGrid{discreteGrid}
{
    amrex::IntVect maxGridSize;
    params.get("ComputationalDomain.maxGridSize", maxGridSize);

    std::array<amrex::MultiFab, 3> data{};
    for (int dir{0}; dir < 3; dir++)
    {
        amrex::BoxArray boxArray{Impl::to_amrex_box(discrete_grid(static_cast<Direction>(dir)))};
        boxArray.maxSize(maxGridSize);
        if (boxArray.size() < amrex::ParallelContext::NProcsAll())
        {
            amrex::Abort("The DiscreteVectorField " + label + " is initialized with " +
                         std::to_string(boxArray.size()) + " boxes. Available are " +
                         std::to_string(amrex::ParallelContext::NProcsAll()) +
                         " MPI processes. We require #processes <= #boxes. "
                         "Reduce ComputationalDomain.maxGridSize or number of MPI processes");
        }
        amrex::DistributionMapping dm{boxArray};
        data[dir] = amrex::MultiFab{boxArray, dm, 1, 0};
    }
    m_data = std::make_shared<std::array<amrex::MultiFab, 3>>(std::move(data));
    for (size_t dir{0}; dir < 3; dir++)
    {
        amrex::GpuBndryFuncFab<Forms::BoundaryCondition> bc{Forms::BoundaryCondition{
            Impl::to_amrex_idx_type(discrete_grid(static_cast<Direction>(dir))),
            boundaryExtrapolationDegree, grid}};
        amrex::Vector<amrex::BCRec> bcRec = Forms::load_boundary_condition(
            Impl::to_amrex_geometry(discrete_grid(static_cast<Direction>(dir))), "Default");
        m_boundaryCondition[dir] =
            amrex::PhysBCFunct<amrex::GpuBndryFuncFab<Forms::BoundaryCondition>>{
                Impl::to_amrex_geometry(discrete_grid(static_cast<Direction>(dir))), bcRec, bc};
    }
}

DiscreteGrid const& DiscreteVectorField::discrete_grid(Direction dir) const
{
    return m_discreteGrid[dir];
}
amrex::MultiFab const& DiscreteVectorField::multi_fab(Direction dir) const
{
    return m_data->operator[](dir);
}
amrex::MultiFab& DiscreteVectorField::multi_fab(Direction dir) { return m_data->operator[](dir); }

std::array<amrex::Box, 3> const DiscreteVectorField::select_box(amrex::MFIter const& mfi)
{
    std::array<amrex::Box, 3> boxes{};
    for (int dir = 0; dir < 3; dir++)
    {
        m_view[dir] = this->multi_fab(static_cast<Direction>(dir)).array(mfi);
        boxes[dir] = this->multi_fab(static_cast<Direction>(dir))[mfi].box();
    }
    m_boxStatus = Impl::BoxStatus::boxSelected;
    m_selectedBoxIdx = mfi.index();
    return boxes;
}

amrex::Box const& DiscreteVectorField::selected_box(Direction dir) const
{
    return m_data->operator[](dir)[m_selectedBoxIdx].box();
}

void DiscreteVectorField::set_ghost_size (int width, Direction dir)
{
    if (m_haloWidth[dir] < width)
    {
        m_haloWidth[dir] = width;
        std::array<amrex::MultiFab, 3> dataNew{};
        for (size_t dir = 0; dir < 3; dir++)
        {
            amrex::BoxArray ba{this->multi_fab(static_cast<Direction>(dir)).boxArray()};
            amrex::DistributionMapping dm{
                this->multi_fab(static_cast<Direction>(dir)).DistributionMap()};
            amrex::IntVect ng;
            for (int i = 0; i < AMREX_SPACEDIM; i++)
            {
                ng[i] = m_haloWidth[i];
            }
            dataNew[dir].define(ba, dm, this->multi_fab(static_cast<Direction>(dir)).nComp(), ng);
            dataNew[dir].ParallelCopy(this->multi_fab(static_cast<Direction>(dir)));
        }
        m_data.reset();
        m_data = std::make_shared<std::array<amrex::MultiFab, 3>>(std::move(dataNew));
        for (int dir = 0; dir < 3; dir++)
        {
            m_view[dir] = amrex::Array4<amrex::Real>{};
        }
        m_boxStatus = Impl::BoxStatus::boxChanged;
        m_selectedBoxIdx = std::numeric_limits<int>::min();
    }
};

void operator*=(DiscreteVectorField& field, amrex::Real const& scalar)
{
    for (Direction dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        for (amrex::MFIter mfi{field.multi_fab(dir)}; mfi.isValid(); ++mfi)
        {
            field.select_box(mfi);
            amrex::ParallelFor(mfi.growntilebox(), field.multi_fab(dir).nComp(),
                               [=] AMREX_GPU_HOST_DEVICE(int ix, int iy, int iz, int n)
                               { field(dir, ix, iy, iz, n) *= scalar; });
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
            amrex::ParallelFor(mfi.growntilebox(), a.multi_fab(dir).nComp(),
                               [=] AMREX_GPU_HOST_DEVICE(int ix, int iy, int iz, int n)
                               { a(dir, ix, iy, iz, n) += otherView(ix, iy, iz, n); });
        }
    }
}

amrex::Real l_inf_error (DiscreteField& a, DiscreteField& b)
{
    Io::Parameters param{};
    // The last two parameters are not relevant.
    DiscreteField tmp{"tmp", param, a.discrete_grid(), Grid::primal, 0};
    for (amrex::MFIter mfi{a.multi_fab()}; mfi.isValid(); ++mfi)
    {
        a.select_box(mfi);
        b.select_box(mfi);
        tmp.select_box(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_HOST_DEVICE(int ix, int iy, int iz)
                           { tmp(ix, iy, iz) = std::abs(a(ix, iy, iz) - b(ix, iy, iz)); });
    }
    return tmp.multi_fab().norminf();
}

std::array<amrex::Real, 3> l_inf_error (DiscreteVectorField& a, DiscreteVectorField& b)
{
    std::array<amrex::Real, 3> maxError{0.0};
    Io::Parameters param{};
    // The last two parameters are not relevant.
    DiscreteVectorField tmp{"tmp",
                            param,
                            {a.discrete_grid(xDir), a.discrete_grid(yDir), a.discrete_grid(zDir)},
                            Grid::primal,
                            0};
    for (auto dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        for (amrex::MFIter mfi{a.multi_fab(dir)}; mfi.isValid(); ++mfi)
        {
            a.select_box(mfi);
            b.select_box(mfi);
            tmp.select_box(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_HOST_DEVICE(int ix, int iy, int iz)
                { tmp(dir, ix, iy, iz) = std::abs(a(dir, ix, iy, iz) - b(dir, ix, iy, iz)); });
        }
        maxError[dir] = tmp.multi_fab(dir).norminf();
    }
    return maxError;
}

namespace Impl
{
void fill_boundary (DiscreteField& field)
{
    field.multi_fab().FillBoundary(Impl::to_amrex_periodicty(field.discrete_grid()));
}

void fill_boundary (DiscreteVectorField& field)
{
    field.multi_fab(xDir).FillBoundary(Impl::to_amrex_periodicty(field.discrete_grid(xDir)));
    field.multi_fab(yDir).FillBoundary(Impl::to_amrex_periodicty(field.discrete_grid(yDir)));
    field.multi_fab(zDir).FillBoundary(Impl::to_amrex_periodicty(field.discrete_grid(zDir)));
}
} //namespace Impl

} //namespace Gempic
