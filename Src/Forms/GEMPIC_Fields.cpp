#include <memory>

#include "GEMPIC_BoundaryConditions.H"
#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_NumericalIntegrationDifferentiation.H"
#include "GEMPIC_Parameters.H"

namespace Gempic
{

DiscreteField::DiscreteField (std::string const &label,
                              Io::Parameters &params,
                              DiscreteGrid const &discreteGrid,
                              Grid const &grid,
                              int const &boundaryExtrapolationDegree) :
    m_label{std::make_shared<std::string>(label)}, m_discreteGrid{discreteGrid}
{
    amrex::Vector<int> maxGridSizeTmp;
    amrex::IntVect maxGridSize;
    params.get("ComputationalDomain.maxGridSize", maxGridSizeTmp);
    for (int i{0}; i < AMREX_SPACEDIM; ++i) maxGridSize[i] = maxGridSizeTmp[i];

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

amrex::MultiFab const &DiscreteField::multiFab () const { return *m_data; }
amrex::MultiFab &DiscreteField::multiFab () { return *m_data; }

amrex::Box const &DiscreteField::select_box (amrex::MFIter const &mfi)
{
    m_view = this->multiFab().array(mfi);
    m_boxStatus = Impl::BoxStatus::boxSelected;
    m_selectedBoxIdx = mfi.index();
    return this->multiFab()[mfi].box();
}
amrex::Box const &DiscreteField::selected_box () const
{
    return m_data->get(m_selectedBoxIdx).box();
};
void DiscreteField::set_ghost_size (int width, Direction dir)
{
    if (m_haloWidth[dir] < width)
    {
        m_haloWidth[dir] = width;
        amrex::BoxArray ba{this->multiFab().boxArray()};
        amrex::DistributionMapping dm{this->multiFab().DistributionMap()};
        amrex::IntVect ng;
        for (int i = 0; i < AMREX_SPACEDIM; i++)
        {
            ng[i] = m_haloWidth[static_cast<Direction>(i)];
        }
        amrex::MultiFab m_dataNew{ba, dm, this->multiFab().nComp(), ng};
        m_dataNew.ParallelCopy(this->multiFab());
        m_data = std::make_shared<amrex::MultiFab>(std::move(m_dataNew));
        //m_view = this->multiFab()[0].array();
        m_boxStatus = Impl::BoxStatus::boxChanged;
        //m_selectedBoxIdx = 0;
    }
};

DiscreteVectorField::DiscreteVectorField (
    std::string const &label,
    Io::Parameters &params,
    std::array<DiscreteGrid, AMREX_SPACEDIM> const &discreteGrid,
    Grid const &grid,
    int const &boundaryExtrapolationDegree) :
    m_label{std::make_shared<std::string>(label)}, m_discreteGrid{discreteGrid}
{
    amrex::Vector<int> maxGridSizeTmp;
    amrex::IntVect maxGridSize;
    params.get("ComputationalDomain.maxGridSize", maxGridSizeTmp);
    for (int i{0}; i < AMREX_SPACEDIM; ++i) maxGridSize[i] = maxGridSizeTmp[i];

    std::array<amrex::MultiFab, 3> data{};
    for (int dir{0}; dir < AMREX_SPACEDIM; dir++)
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
    for (size_t dir{0}; dir < AMREX_SPACEDIM; dir++)
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

DiscreteGrid const &DiscreteVectorField::discrete_grid (Direction dir) const
{
    return m_discreteGrid[dir];
}
amrex::MultiFab const &DiscreteVectorField::multiFab (Direction dir) const
{
    return m_data->operator[](dir);
}
amrex::MultiFab &DiscreteVectorField::multiFab (Direction dir) { return m_data->operator[](dir); }

std::array<amrex::Box, 3> const DiscreteVectorField::select_box (amrex::MFIter const &mfi)
{
    std::array<amrex::Box, 3> boxes{};
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        m_view[dir] = this->multiFab(static_cast<Direction>(dir)).array(mfi);
        boxes[dir] = this->multiFab(static_cast<Direction>(dir))[mfi].box();
    }
    m_boxStatus = Impl::BoxStatus::boxSelected;
    m_selectedBoxIdx = mfi.index();
    return boxes;
}

amrex::Box const &DiscreteVectorField::selected_box (Direction dir) const
{
    AMREX_ASSERT(dir <= AMREX_SPACEDIM);
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
            amrex::BoxArray ba{this->multiFab(static_cast<Direction>(dir)).boxArray()};
            amrex::DistributionMapping dm{
                this->multiFab(static_cast<Direction>(dir)).DistributionMap()};
            amrex::IntVect ng;
            for (int i = 0; i < AMREX_SPACEDIM; i++)
            {
                ng[i] = m_haloWidth[static_cast<Direction>(i)];
            }
            dataNew[dir].define(ba, dm, this->multiFab(static_cast<Direction>(dir)).nComp(), ng);
            dataNew[dir].ParallelCopy(this->multiFab(static_cast<Direction>(dir)));
        }
        m_data.reset();
        m_data = std::make_shared<std::array<amrex::MultiFab, 3>>(std::move(dataNew));
        //for (int dir = 0; dir < 3; dir++)
        //{
        //    m_view[dir] = this->multiFab(static_cast<Direction>(dir)).array(0);
        //}
        m_boxStatus = Impl::BoxStatus::boxChanged;
        //m_selectedBoxIdx = 0;
    }
};

void operator*=(DiscreteVectorField& field, amrex::Real const &scalar)
{
    for (Direction dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        for (amrex::MFIter mfi{field.multiFab(dir)}; mfi.isValid(); ++mfi)
        {
            field.select_box(mfi);
            amrex::ParallelFor(
                mfi.validbox(), field.multiFab(dir).nComp(),
                [=] AMREX_GPU_HOST_DEVICE (int ix, int iy, int iz, int n) {
                    field(ix, iy, iz, dir, n) *= scalar;
                });
        }
    }
}

void operator+=(DiscreteVectorField &a, DiscreteVectorField const &b)
{
    for (Direction dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        for (amrex::MFIter mfi{a.multiFab(dir)}; mfi.isValid(); ++mfi)
        {
            a.select_box(mfi);
            auto const &otherView{b.multiFab(dir).array(mfi)};
            amrex::ParallelFor(
                mfi.validbox(), a.multiFab(dir).nComp(),
                [=] AMREX_GPU_HOST_DEVICE (int ix, int iy, int iz, int n) {
                    a(ix, iy, iz, dir, n) += otherView(ix, iy, iz, n);
                });
        }
    }
}

amrex::Real L_inf_error (DiscreteField &a, DiscreteField &b)
{
    Io::Parameters param{};
    // The last to parameters are not relevant.
    DiscreteField tmp{"tmp", param, a.discrete_grid(), Grid::primal, 0};
    for (amrex::MFIter mfi{a.multiFab()}; mfi.isValid(); ++mfi)
    {
        a.select_box(mfi);
        b.select_box(mfi);
        tmp.select_box(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_HOST_DEVICE  (int ix, int iy, int iz)
                           { tmp(ix, iy, iz) = std::abs(a(ix, iy, iz) - b(ix, iy, iz)); });
    }
    return tmp.multiFab().norminf();
}

std::array<amrex::Real, AMREX_SPACEDIM> L_inf_error (DiscreteVectorField &a, DiscreteVectorField &b)
{
    std::array<amrex::Real, AMREX_SPACEDIM> maxError{0.0};
    Io::Parameters param{};
    // The last to parameters are not relevant.
    DiscreteVectorField tmp{"tmp",
                            param,
                            {AMREX_D_DECL(a.discrete_grid(xDir), a.discrete_grid(yDir), a.discrete_grid(zDir))},
                            Grid::primal,
                            0};
    for (auto dir : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
    {
        for (amrex::MFIter mfi{a.multiFab(dir)}; mfi.isValid(); ++mfi)
        {
            a.select_box(mfi);
            b.select_box(mfi);
            tmp.select_box(mfi);
            amrex::ParallelFor(
                mfi.validbox(),  [=] AMREX_GPU_HOST_DEVICE (int ix, int iy, int iz)
                { tmp(ix, iy, iz, dir) = std::abs(a(ix, iy, iz, dir) - b(ix, iy, iz, dir)); });
        }
        maxError[dir] = tmp.multiFab(dir).norminf();
    }
    return maxError;
}

namespace Impl
{
void fill_boundary (DiscreteField &field)
{
    field.multiFab().FillBoundary(Impl::to_amrex_periodicty(field.discrete_grid()));
}

void fill_boundary (DiscreteVectorField &field)
{
    field.multiFab(xDir).FillBoundary(Impl::to_amrex_periodicty(field.discrete_grid(xDir)));
    #if AMREX_SPACEDIM>1
    field.multiFab(yDir).FillBoundary(Impl::to_amrex_periodicty(field.discrete_grid(yDir)));
    #endif
    #if AMREX_SPACEDIM == 3
    field.multiFab(zDir).FillBoundary(Impl::to_amrex_periodicty(field.discrete_grid(zDir)));
    #endif
}
} //namespace Impl

} //namespace Gempic
