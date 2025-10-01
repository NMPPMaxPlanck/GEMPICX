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
void DiscreteField::set_ghost_size (std::array<size_t, AMREX_SPACEDIM> width)
{
    bool increaseHalo{false};
    for (auto dir : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
    {
        if (m_haloWidth[dir] < width[dir])
        {
            m_haloWidth[dir] = width[dir];
            increaseHalo = true;
        }
    }
    amrex::IntVect ng;
    for (int i = 0; i < AMREX_SPACEDIM; i++)
    {
        ng[i] = m_haloWidth[i];
    }
    if (increaseHalo)
    {
        int n{this->multi_fab().nComp()};
        amrex::BoxArray ba{this->multi_fab().boxArray()};
        amrex::DistributionMapping dm{this->multi_fab().DistributionMap()};
        amrex::MultiFab dataNew{ba, dm, n, ng};
        dataNew.LocalCopy(this->multi_fab(), 0, 0, n, amrex::IntVect{0});
        m_data = std::make_shared<amrex::MultiFab>(std::move(dataNew));
        m_view = amrex::Array4<amrex::Real>{};
        m_boxStatus = Impl::BoxStatus::boxChanged;
        m_selectedBoxIdx = std::numeric_limits<int>::min();
    }
    // Interface notes:
    // scomp is the starting index of the components copied
    // We always copy all components and do not treat it as a special parameter.
    // Parameter cross specifies whether all corners of the multifab should also be filled or not
    this->multi_fab().FillBoundary(0, this->multi_fab().nComp(), ng,
                                   Impl::to_amrex_periodicty(this->discrete_grid()), false);
}

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
    for (Direction dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        m_view[dir] = this->multi_fab(dir).array(mfi);
        boxes[dir] = this->multi_fab(dir)[mfi].box();
    }
    m_boxStatus = Impl::BoxStatus::boxSelected;
    m_selectedBoxIdx = mfi.index();
    return boxes;
}

amrex::Box const& DiscreteVectorField::selected_box(Direction dir) const
{
    return m_data->operator[](dir)[m_selectedBoxIdx].box();
}

void DiscreteVectorField::set_ghost_size (std::array<size_t, AMREX_SPACEDIM> width)
{
    bool increaseHalo{false};
    for (auto dir : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
    {
        if (m_haloWidth[dir] < width[dir])
        {
            m_haloWidth[dir] = width[dir];
            increaseHalo = true;
        }
    }
    amrex::IntVect ng;
    for (int i = 0; i < AMREX_SPACEDIM; i++)
    {
        ng[i] = m_haloWidth[i];
    }
    if (increaseHalo)
    {
        std::array<amrex::MultiFab, 3> dataNew{};
        for (Direction dir : {Direction::xDir, Direction::yDir, Direction::zDir})
        {
            amrex::BoxArray ba{this->multi_fab(dir).boxArray()};
            amrex::DistributionMapping dm{this->multi_fab(dir).DistributionMap()};
            amrex::IntVect ng;
            for (int i = 0; i < AMREX_SPACEDIM; i++)
            {
                ng[i] = m_haloWidth[i];
            }
            int n{this->multi_fab(dir).nComp()};
            dataNew[dir].define(ba, dm, n, ng);
            dataNew[dir].LocalCopy(this->multi_fab(dir), 0, 0, n, amrex::IntVect{0});
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
    // Interface notes:
    // scomp is the starting index of the components copied
    // We always copy all components and do not treat it as a special parameter.
    // Parameter cross specifies whether all corners of the multifab should also be filled or not
    for (auto dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        this->multi_fab(dir).FillBoundary(0, this->multi_fab(dir).nComp(), ng,
                                          Impl::to_amrex_periodicty(this->discrete_grid(dir)),
                                          false);
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
    //It should be noted that the reduction result of ParReduce is local and it is the user’s
    //responsibility if MPI communication is needed
    MPI_Allreduce(MPI_IN_PLACE, &isNan, 1, MPI_INT, MPI_LOR,
                  amrex::ParallelContext::CommunicatorAll());
    return amrex::get<0>(isNan);
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
            auto ba = ma[boxNo];
            amrex::Real ldiff{aa(ix, iy, iz) - ba(ix, iy, iz)};
            // NaN comparisons always evaluate to false, which is why we check on Nan manually.
            // https://stackoverflow.com/questions/38798791/nan-comparison-rule-in-c-c
            // https://rgambord.github.io/c99-doc/sections/7/12/14/index.html#id2
            return {ldiff, static_cast<int>(std::isnan(ldiff))};
        });
    int isNan{amrex::get<1>(res)};
    amrex::Real norm{amrex::get<0>(res)};
    //https://amrex-codes.github.io/amrex/docs_html/GPU.html#multifab-reductions
    //It should be noted that the reduction result of ParReduce is local and it is the user’s
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
    std::array<int, 3> isNan{};
    for (auto dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        auto ma = a.multi_fab(dir).const_arrays();
        amrex::GpuTuple<int> res{};
        res = amrex::ParReduce(amrex::TypeList<amrex::ReduceOpLogicalOr>{}, amrex::TypeList<int>{},
                               a.multi_fab(dir),
                               [=] AMREX_GPU_HOST_DEVICE(int boxNo, int ix, int iy,
                                                         int iz) noexcept -> amrex::GpuTuple<int>
                               {
                                   auto aa = ma[boxNo];
                                   return {std::isnan(aa(ix, iy, iz))};
                               });
        isNan[dir] = amrex::get<0>(res);
    }
    //https://amrex-codes.github.io/amrex/docs_html/GPU.html#multifab-reductions
    //It should be noted that the reduction result of ParReduce is local and it is the user’s
    //responsibility if MPI communication is needed
    MPI_Allreduce(MPI_IN_PLACE, &isNan[0], 3, MPI_INT, MPI_LOR,
                  amrex::ParallelContext::CommunicatorAll());
    return std::array<bool, 3>{static_cast<bool>(isNan[Direction::xDir]),
                               static_cast<bool>(isNan[Direction::yDir]),
                               static_cast<bool>(isNan[Direction::zDir])};
}

std::array<amrex::Real, 3> l_inf_error (DiscreteVectorField& a, DiscreteVectorField& b)
{
    std::array<amrex::Real, 3> maxError{0};
    std::array<int, 3> isNan{false, false, false};
    for (auto dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        auto ma = a.multi_fab(dir).const_arrays();
        auto mb = b.multi_fab(dir).const_arrays();
        amrex::GpuTuple<amrex::Real, int> res{};
        res = amrex::ParReduce(
            amrex::TypeList<amrex::ReduceOpMax, amrex::ReduceOpLogicalOr>{},
            amrex::TypeList<amrex::Real, int>{}, a.multi_fab(dir),
            [=] AMREX_GPU_HOST_DEVICE(int boxNo, int ix, int iy,
                                      int iz) noexcept -> amrex::GpuTuple<amrex::Real, int>
            {
                auto aa = ma[boxNo];
                auto ba = ma[boxNo];
                amrex::Real ldiff{aa(ix, iy, iz) - ba(ix, iy, iz)};
                // NaN comparisons always evaluate to false, which is why we check on Nan manually.
                // https://stackoverflow.com/questions/38798791/nan-comparison-rule-in-c-c
                // https://rgambord.github.io/c99-doc/sections/7/12/14/index.html#id2
                return {ldiff, static_cast<int>(std::isnan(ldiff))};
            });
        isNan[dir] = amrex::get<1>(res);
        maxError[dir] = amrex::get<0>(res);
    }
    MPI_Allreduce(MPI_IN_PLACE, &isNan[0], 3, MPI_INT, MPI_LOR,
                  amrex::ParallelContext::CommunicatorAll());
    for (auto dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        if (isNan[dir]) maxError[dir] = std::numeric_limits<amrex::Real>::quiet_NaN();
    }
    return maxError;
}

} //namespace Gempic
