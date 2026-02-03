#include <tuple>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Interpolation.H"
#include "GEMPIC_Parameters.H"

namespace Gempic
{

FiniteDifferenceDeRhamSpaces::FiniteDifferenceDeRhamSpaces(
    GaussLegendreQuadrature const& integrator) :
    m_integrator{integrator}
{
    Io::Parameters params{};
    m_grid = DiscreteGrid{params,
                          {AMREX_D_DECL(DiscreteGrid::Position::Cell, DiscreteGrid::Position::Cell,
                                        DiscreteGrid::Position::Cell)}};
    amrex::IntVect maxGridSize;
    params.get("ComputationalDomain.maxGridSize", maxGridSize);

    m_boxArray = amrex::BoxArray{Impl::to_amrex_box(m_grid)};
    m_boxArray.maxSize(maxGridSize);
    m_distributionMapping = amrex::DistributionMapping{m_boxArray};
};

FiniteDifferenceDeRhamSpaces::FiniteDifferenceDeRhamSpaces(
    GaussLegendreQuadrature const integrator,
    DiscreteGrid grid,
    amrex::BoxArray boxArray,
    amrex::DistributionMapping distributionMapping) :
    m_integrator{integrator},
    m_grid{grid},
    m_boxArray{boxArray},
    m_distributionMapping{distributionMapping}
{
}

FiniteDifferenceDeRhamSpaces::DOFCategories const FiniteDifferenceDeRhamSpaces::point_value() const
{
    DOFCategories valueType{AMREX_D_DECL(DiscreteField::DOFCategory::PointValue,
                                         DiscreteField::DOFCategory::PointValue,
                                         DiscreteField::DOFCategory::PointValue)};
    return valueType;
};
FiniteDifferenceDeRhamSpaces::DOFCategories const
FiniteDifferenceDeRhamSpaces::cell_centered_integral() const
{
    DOFCategories valueType{AMREX_D_DECL(DiscreteField::DOFCategory::CenteredLineIntegral,
                                         DiscreteField::DOFCategory::CenteredLineIntegral,
                                         DiscreteField::DOFCategory::CenteredLineIntegral)};
    return valueType;
};

std::array<FiniteDifferenceDeRhamSpaces::DOFCategories, 3> const
FiniteDifferenceDeRhamSpaces::edge_centered_integral() const
{
    std::array<FiniteDifferenceDeRhamSpaces::DOFCategories, 3> valueType{};
    for (int fieldDir = 0; fieldDir < 3; fieldDir++)
    {
        for (int gridDir = 0; gridDir < AMREX_SPACEDIM; gridDir++)
        {
            valueType[fieldDir][gridDir] = DiscreteField::DOFCategory::PointValue;
        }
    }
    for (int i = 0; i < AMREX_SPACEDIM; i++)
    {
        valueType[i][i] = DiscreteField::DOFCategory::CenteredLineIntegral;
    }
    return valueType;
};
std::array<FiniteDifferenceDeRhamSpaces::DOFCategories, 3> const
FiniteDifferenceDeRhamSpaces::face_centered_integral() const
{
    std::array<FiniteDifferenceDeRhamSpaces::DOFCategories, 3> valueType{};
    for (int fieldDir = 0; fieldDir < 3; fieldDir++)
    {
        for (int gridDir = 0; gridDir < AMREX_SPACEDIM; gridDir++)
        {
            valueType[fieldDir][gridDir] = DiscreteField::DOFCategory::CenteredLineIntegral;
        }
    }
    for (int i = 0; i < AMREX_SPACEDIM; i++)
    {
        valueType[i][i] = DiscreteField::DOFCategory::PointValue;
    }
    return valueType;
};

PrimalZeroForm FiniteDifferenceDeRhamSpaces::create_primal_zero_form (
    std::string const& label, Impl::BoundaryConditionConfiguration bcConf) const
{
    bcConf.m_grid = Grid::primal;
    DiscreteField df{label,      dof_on_node(m_grid),   point_value(),
                     m_boxArray, m_distributionMapping, bcConf};
    return PrimalZeroForm{std::move(df), m_integrator};
};
PrimalOneForm FiniteDifferenceDeRhamSpaces::create_primal_one_form (
    std::string const& label, std::array<Impl::BoundaryConditionConfiguration, 3> bcConf) const
{
    for (auto dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        bcConf[dir].m_grid = Grid::primal;
    }
    DiscreteVectorField df{label,      dof_on_edge(m_grid),   edge_centered_integral(),
                           m_boxArray, m_distributionMapping, bcConf};
    return PrimalOneForm{std::move(df), m_integrator};
};
PrimalTwoForm FiniteDifferenceDeRhamSpaces::create_primal_two_form (
    std::string const& label, std::array<Impl::BoundaryConditionConfiguration, 3> bcConf) const
{
    for (auto dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        bcConf[dir].m_grid = Grid::primal;
    }
    DiscreteVectorField df{label,      dof_on_face(m_grid),   face_centered_integral(),
                           m_boxArray, m_distributionMapping, bcConf};
    return PrimalTwoForm{std::move(df), m_integrator};
};
PrimalThreeForm FiniteDifferenceDeRhamSpaces::create_primal_three_form (
    std::string const& label, Impl::BoundaryConditionConfiguration bcConf) const
{
    bcConf.m_grid = Grid::primal;
    DiscreteField df{label,      dof_on_cell_center(m_grid), cell_centered_integral(),
                     m_boxArray, m_distributionMapping,      bcConf};
    return PrimalThreeForm{std::move(df), m_integrator};
};
DualZeroForm FiniteDifferenceDeRhamSpaces::create_dual_zero_form (
    std::string const& label, Impl::BoundaryConditionConfiguration bcConf) const
{
    bcConf.m_grid = Grid::dual;
    DiscreteField df{label,      dof_on_cell_center(m_grid), point_value(),
                     m_boxArray, m_distributionMapping,      bcConf};
    return DualZeroForm{std::move(df), m_integrator};
};
DualOneForm FiniteDifferenceDeRhamSpaces::create_dual_one_form (
    std::string const& label, std::array<Impl::BoundaryConditionConfiguration, 3> bcConf) const
{
    for (auto dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        bcConf[dir].m_grid = Grid::dual;
    }
    DiscreteVectorField df{label,      dof_on_face(m_grid),   edge_centered_integral(),
                           m_boxArray, m_distributionMapping, bcConf};
    return DualOneForm{std::move(df), m_integrator};
};
DualTwoForm FiniteDifferenceDeRhamSpaces::create_dual_two_form (
    std::string const& label, std::array<Impl::BoundaryConditionConfiguration, 3> bcConf) const
{
    for (auto dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        bcConf[dir].m_grid = Grid::dual;
    }
    DiscreteVectorField df{label,      dof_on_edge(m_grid),   face_centered_integral(),
                           m_boxArray, m_distributionMapping, bcConf};
    return DualTwoForm{std::move(df), m_integrator};
};
DualThreeForm FiniteDifferenceDeRhamSpaces::create_dual_three_form (
    std::string const& label, Impl::BoundaryConditionConfiguration bcConf) const
{
    bcConf.m_grid = Grid::dual;
    DiscreteField df{label,      dof_on_node(m_grid),   cell_centered_integral(),
                     m_boxArray, m_distributionMapping, bcConf};
    return DualThreeForm{std::move(df), m_integrator};
};

namespace Impl
{
std::array<amrex::Long, 3> stride (amrex::Array4<amrex::Real> const& view)
{
    return std::array<amrex::Long, 3>{1, view.jstride, view.kstride};
}
struct ForwardStencil
{
    AMREX_GPU_HOST_DEVICE amrex::Real operator()(amrex::Real const& f, size_t const& stride) const
    {
        amrex::Real const* fPtr{&f};
        return fPtr[stride] - fPtr[0];
    }
};
struct BackwardStencil
{
    AMREX_GPU_HOST_DEVICE amrex::Real operator()(amrex::Real const& f, size_t const& stride) const
    {
        amrex::Real const* fPtr{&f};
        return fPtr[0] - fPtr[-stride];
    }
};

template <typename Stencil>
void grad (DiscreteVectorField& of, DiscreteField& zf, Stencil const& stencil)
{
    for (auto dir : {Direction::xDir, Direction::yDir, Direction::zDir})
    {
        for (amrex::MFIter mfi{of.multi_fab(dir)}; mfi.isValid(); ++mfi)
        {
            of.select_box(mfi);
            zf.select_box(mfi);
            auto strides = stride(zf.view());
            if (dir < AMREX_SPACEDIM)
            {
                ParallelFor(mfi.validbox(), [=] AMREX_GPU_HOST_DEVICE(int ix, int iy, int iz)
                            { of(dir, ix, iy, iz) = stencil(zf(ix, iy, iz), strides[dir]); });
            }
            else
            {
                ParallelFor(mfi.validbox(), [=] AMREX_GPU_HOST_DEVICE(int ix, int iy, int iz)
                            { of(dir, ix, iy, iz) = 0.0; });
            }
        }
    }
}

template <typename Stencil>
void curl (DiscreteVectorField& tf, DiscreteVectorField& of, Stencil const& stencil)
{
    for (amrex::MFIter mfi(tf.multi_fab(Direction::xDir)); mfi.isValid(); ++mfi)
    {
        tf.select_box(mfi);
        of.select_box(mfi);
        std::array<std::array<amrex::Long, 3>, 3> strides{};
        for (auto dir : {Direction::xDir, Direction::yDir, Direction::zDir})
        {
            strides[dir] = Impl::stride(of.view(dir));
        }
        ParallelFor(mfi.validbox(),
                    [=] AMREX_GPU_DEVICE(int ix, int iy, int iz)
                    {
                        tf(Direction::xDir, ix, iy, iz) =
                            GEMPIC_D_ADD(0.0,
                                         stencil(of(Direction::zDir, ix, iy, iz),
                                                 strides[Direction::zDir][Direction::yDir]),
                                         -stencil(of(Direction::yDir, ix, iy, iz),
                                                  strides[Direction::yDir][Direction::zDir]));
                    });
    }

    for (amrex::MFIter mfi(tf.multi_fab(Direction::yDir)); mfi.isValid(); ++mfi)
    {
        tf.select_box(mfi);
        of.select_box(mfi);
        std::array<std::array<amrex::Long, 3>, 3> strides{};
        for (auto dir : {Direction::xDir, Direction::yDir, Direction::zDir})
        {
            strides[dir] = Impl::stride(of.view(dir));
        }
        ParallelFor(mfi.validbox(),
                    [=] AMREX_GPU_DEVICE(int ix, int iy, int iz)
                    {
                        tf(Direction::yDir, ix, iy, iz) =
                            GEMPIC_D_ADD(-stencil(of(Direction::zDir, ix, iy, iz),
                                                  strides[Direction::zDir][Direction::xDir]),
                                         0.0,
                                         stencil(of(Direction::xDir, ix, iy, iz),
                                                 strides[Direction::xDir][Direction::zDir]));
                    });
    }

    for (amrex::MFIter mfi(tf.multi_fab(Direction::zDir)); mfi.isValid(); ++mfi)
    {
        tf.select_box(mfi);
        of.select_box(mfi);
        std::array<std::array<amrex::Long, 3>, 3> strides{};
        for (auto dir : {Direction::xDir, Direction::yDir, Direction::zDir})
        {
            strides[dir] = Impl::stride(of.view(dir));
        }
        ParallelFor(mfi.validbox(),
                    [=] AMREX_GPU_DEVICE(int ix, int iy, int iz)
                    {
                        tf(Direction::zDir, ix, iy, iz) =
                            GEMPIC_D_ADD(stencil(of(Direction::yDir, ix, iy, iz),
                                                 strides[Direction::yDir][Direction::xDir]),
                                         -stencil(of(Direction::xDir, ix, iy, iz),
                                                  strides[Direction::xDir][Direction::yDir]),
                                         0.0);
                        ;
                    });
    }
}

template <typename Stencil>
void div (DiscreteField threeF, DiscreteVectorField twoF, Stencil const& stencil)
{
    for (amrex::MFIter mfi{threeF.multi_fab()}; mfi.isValid(); ++mfi)
    {
        threeF.select_box(mfi);
        twoF.select_box(mfi);
        std::array<std::array<amrex::Long, 3>, 3> strides{};
        for (auto dir : {Direction::xDir, Direction::yDir, Direction::zDir})
        {
            strides[dir] = Impl::stride(twoF.view(dir));
        }
        ParallelFor(mfi.validbox(),
                    [=] AMREX_GPU_DEVICE(int ix, int iy, int iz)
                    {
                        threeF(ix, iy, iz) =
                            GEMPIC_D_ADD(stencil(twoF(Direction::xDir, ix, iy, iz),
                                                 strides[Direction::xDir][Direction::xDir]),
                                         stencil(twoF(Direction::yDir, ix, iy, iz),
                                                 strides[Direction::yDir][Direction::yDir]),
                                         stencil(twoF(Direction::zDir, ix, iy, iz),
                                                 strides[Direction::zDir][Direction::zDir]));
                    });
    }
}

template <size_t n>
struct StridedArrayView
{
    StridedArrayView() = delete;
    constexpr StridedArrayView(amrex::Real& f, std::ptrdiff_t const& stride) :
        m_f{f}, m_stride{stride}
    {
    }
    amrex::Real& m_f;
    std::ptrdiff_t const m_stride;
    static constexpr size_t s_size{n};
};

template <size_t n>
struct StencilHelper
{
    template <int i = 0>
    static constexpr amrex::Real unroll (std::array<amrex::Real, n> coeff, StridedArrayView<n> view)
    {
        amrex::Real const* fPtr{&view.m_f};
        std::ptrdiff_t offset{static_cast<std::ptrdiff_t>(i - (n - 1) / 2) * view.m_stride};
        if constexpr (i < n - 1)
        {
            return coeff[i] * fPtr[offset] + StencilHelper<n>::unroll<i + 1>(coeff, view);
        }
        else
        {
            return coeff[n - 1] * fPtr[offset];
        }
    }
};

template <size_t n>
constexpr amrex::Real operator*(std::array<amrex::Real, n> stencil, StridedArrayView<n> const& view)
{
    static_assert(n % 2 == 1,
                  "constexpr amrex::Real operator*(std::array<amrex::Real, N> stencil, "
                  "StridedArrayView<N> const&) is only implemented for uneven stencils");
    return StencilHelper<n>::unroll(stencil, view);
}

template <typename T, size_t degree>
struct HodgeStencil
{
    constexpr HodgeStencil(std::array<T, degree - 1> coeff) : m_coeff{coeff} {};
    static_assert(degree % 2 == 0, "The stencil must be symmetric (N%2==0) but is not (N%2!=0)");
    static constexpr size_t s_width{degree - 1};
    static constexpr size_t s_halfWidth{(s_width - 1) / 2};
    std::array<T, s_width> m_coeff{};
};
constexpr HodgeStencil<amrex::Real, 2> hodgeFiniteDifferenceStencilDegree2{{1.0}};
constexpr HodgeStencil<amrex::Real, 4> hodgeFiniteDifferenceStencilDegree4NodeToCell{
    {1.0 / 24.0, 22.0 / 24.0, 1.0 / 24.0}};
constexpr HodgeStencil<amrex::Real, 4> hodgeFiniteDifferenceStencilDegree4CellToNode{
    {-1.0 / 24.0, 13.0 / 12.0, -1.0 / 24.0}};
constexpr HodgeStencil<amrex::Real, 6> hodgeFiniteDifferenceStencilDegree6NodeToCell{
    {-17.0 / 5760.0, 154.0 / 2880.0, 863.0 / 960.0, 154.0 / 2880.0, -17.0 / 5760.0}};
constexpr HodgeStencil<amrex::Real, 6> hodgeFiniteDifferenceStencilDegree6CellToNode{
    {3.0 / 640.0, -29.0 / 480.0, 1067.0 / 960.0, -29.0 / 480.0, 3.0 / 640.0}};
using HodgeFiniteDifferenceStencils = std::variant<HodgeStencil<amrex::Real, 2>,
                                                   HodgeStencil<amrex::Real, 4>,
                                                   HodgeStencil<amrex::Real, 6>>;
constexpr std::tuple<HodgeFiniteDifferenceStencils, HodgeFiniteDifferenceStencils>
select_hodge_stencil_finite_difference (size_t degree)
{
    Impl::HodgeFiniteDifferenceStencils nodeToCell{Impl::hodgeFiniteDifferenceStencilDegree2};
    Impl::HodgeFiniteDifferenceStencils cellToNode{Impl::hodgeFiniteDifferenceStencilDegree2};
    switch (degree)
    {
        case 2:
        {
            nodeToCell = Impl::hodgeFiniteDifferenceStencilDegree2;
            cellToNode = Impl::hodgeFiniteDifferenceStencilDegree2;
            return {nodeToCell, cellToNode};
        }
        case 4:
        {
            nodeToCell = Impl::hodgeFiniteDifferenceStencilDegree4NodeToCell;
            cellToNode = Impl::hodgeFiniteDifferenceStencilDegree4CellToNode;
            return {nodeToCell, cellToNode};
        }
        case 6:
        {
            nodeToCell = Impl::hodgeFiniteDifferenceStencilDegree6NodeToCell;
            cellToNode = Impl::hodgeFiniteDifferenceStencilDegree6CellToNode;
            return {nodeToCell, cellToNode};
        }
        default:
            throw std::runtime_error("Undefined Hodge degree given!");
    }
};
amrex::Real hodge_scale (DiscreteField const& dst, DiscreteField const& src)
{
    amrex::Real scale{1.0};
    for (auto gridDir : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
    {
        if (dst.dof_category(gridDir) == DiscreteField::DOFCategory::CenteredLineIntegral and
            src.dof_category(gridDir) == DiscreteField::DOFCategory::PointValue)
        {
            scale *= dst.discrete_grid().dx(gridDir);
        }
        else if (dst.dof_category(gridDir) == DiscreteField::DOFCategory::PointValue and
                 src.dof_category(gridDir) == DiscreteField::DOFCategory::CenteredLineIntegral)
        {
            scale *= 1.0 / dst.discrete_grid().dx(gridDir);
        }
    }
    return scale;
}

template <AMREX_D_DECL(typename StencilXdir, typename StencilYdir, typename StencilZdir)>
void hodge (DiscreteField& dst,
            DiscreteField& src,
            AMREX_D_DECL(StencilXdir stencilXdir, StencilYdir stencilYdir, StencilZdir stencilZdir))
{
    amrex::Real scale{hodge_scale(dst, src)};
    std::array<size_t, AMREX_SPACEDIM> ghostWidth{
        AMREX_D_DECL(stencilXdir.s_halfWidth, stencilYdir.s_halfWidth, stencilZdir.s_halfWidth)};
    src.apply_boundary_conditions(ghostWidth);
    for (amrex::MFIter mfi{src.multi_fab()}; mfi.isValid(); ++mfi)
    {
        src.select_box(mfi);
        dst.select_box(mfi);
        auto strides = stride(src.view());
        ParallelFor(mfi.validbox(),
                    [=] AMREX_GPU_DEVICE(int ix, int iy, int iz)
                    {
#if AMREX_SPACEDIM == 1
                        StridedArrayView<StencilXdir::s_width> tmp{src(ix, iy, iz),
                                                                   strides[Direction::xDir]};
                        dst(ix, iy, iz) = scale * (stencilXdir.m_coeff * tmp);
#elif AMREX_SPACEDIM == 2
                        std::array<amrex::Real, StencilXdir::s_width> tmp1D{};
                        int const &hw{static_cast<int>(stencilXdir.s_halfWidth)};
                        for(int sx=-hw;sx<=hw;sx++) {
                            StridedArrayView<StencilYdir::s_width> const tmp{src(ix + sx,iy,iz), strides[Direction::yDir]};
                            tmp1D[sx + hw] = stencilYdir.m_coeff*tmp;
                        };
                        StridedArrayView<StencilXdir::s_width> const tmp{tmp1D[hw], 1};
                        dst(ix,iy,iz) = scale * (stencilXdir.m_coeff*tmp);
#elif AMREX_SPACEDIM == 3
                        // No 2-D array is available. Is emulated manually using C-layout (LayoutRight).
                        std::array<amrex::Real, StencilXdir::s_width * StencilYdir::s_width> tmp2D{};
                        std::array<amrex::Real, StencilXdir::s_width> tmp1D{};
                        int const &hwx{static_cast<int>(stencilXdir.s_halfWidth)};
                        int const &hwy{static_cast<int>(stencilYdir.s_halfWidth)};
                        int const &w{static_cast<int>(stencilXdir.s_width)};
                        for(int sx=-hwx;sx<=hwx;sx++) {
                            for(int sy=-hwy;sy<=hwy;sy++) {
                               StridedArrayView<stencilZdir.s_width> const tmp{src(ix + sx,iy + sy,iz), strides[Direction::zDir]};
                               tmp2D[(sx + hwx) + (sy + hwy) * w] = stencilZdir.m_coeff * tmp;
                            }
                        }
                        for(int sx=0;sx<w;sx++) {
                            StridedArrayView<StencilYdir::s_width> const tmp{tmp2D[sx + hwy*w], w};
                            tmp1D[sx] = stencilYdir.m_coeff * tmp;
                        };
                        StridedArrayView<StencilXdir::s_width> const tmp{tmp1D[hwx], 1};
                        dst(ix,iy,iz) = scale * (stencilXdir.m_coeff * tmp);
#endif
                    });
    }
}

} //namespace Impl

void grad (PrimalOneForm& of, PrimalZeroForm& zf)
{
    BL_PROFILE("grad(PrimalZeroForm->PrimalOneForm)");
    Impl::grad(of, zf, Impl::ForwardStencil{});
}
void grad (DualOneForm& of, DualZeroForm& zf)
{
    BL_PROFILE("grad(DualZeroForm->DualOneForm)");
    zf.apply_boundary_conditions({AMREX_D_DECL(1, 1, 1)});
    Impl::grad(of, zf, Impl::BackwardStencil{});
}

void curl (PrimalTwoForm& tf, PrimalOneForm& of)
{
    BL_PROFILE("curl(PrimalOneForm->PrimalTwoForm)");
    Impl::curl(tf, of, Impl::ForwardStencil{});
}

void curl (DualTwoForm& tf, DualOneForm& of)
{
    BL_PROFILE("curl(DualOneForm->DualTwoForm)");
    of.apply_boundary_conditions({AMREX_D_DECL(1, 1, 1)});
    Impl::curl(tf, of, Impl::BackwardStencil{});
}

void div (PrimalThreeForm& threeF, PrimalTwoForm& twoF)
{
    BL_PROFILE("div(PrimalTwoForm->PrimalThreeForm)");
    Impl::div(threeF, twoF, Impl::ForwardStencil{});
}

void div (DualThreeForm& threeF, DualTwoForm& twoF)
{
    BL_PROFILE("div(DualTwoForm->DualThreeForm)");
    twoF.apply_boundary_conditions({AMREX_D_DECL(1, 1, 1)});
    Impl::div(threeF, twoF, Impl::BackwardStencil{});
}

namespace Impl
{
void finite_difference_hodge (PrimalZeroForm& p, DualThreeForm& d, size_t degree)
{
    auto [nodeToCell, cellToNode] = Impl::select_hodge_stencil_finite_difference(degree);
    std::visit([&] (auto const& chosenStencil)
               { Impl::hodge(p, d, AMREX_D_DECL(chosenStencil, chosenStencil, chosenStencil)); },
               cellToNode);
};
void finite_difference_hodge (PrimalOneForm& p, DualTwoForm& d, size_t degree)
{
    auto [nodeToCell, cellToNode] = Impl::select_hodge_stencil_finite_difference(degree);
    std::visit(
        [&] (auto const& chosenCellToNode, auto const& chosenNodeToCell)
        {
            Impl::hodge(p[Direction::xDir], d[Direction::xDir],
                        AMREX_D_DECL(chosenNodeToCell, chosenCellToNode, chosenCellToNode));
            Impl::hodge(p[Direction::yDir], d[Direction::yDir],
                        AMREX_D_DECL(chosenCellToNode, chosenNodeToCell, chosenCellToNode));
            Impl::hodge(p[Direction::zDir], d[Direction::zDir],
                        AMREX_D_DECL(chosenCellToNode, chosenCellToNode, chosenNodeToCell));
        },
        cellToNode, nodeToCell);
};
void finite_difference_hodge (PrimalTwoForm& p, DualOneForm& d, size_t degree)
{
    auto [nodeToCell, cellToNode] = Impl::select_hodge_stencil_finite_difference(degree);
    std::visit(
        [&] (auto const& chosenCellToNode, auto const& chosenNodeToCell)
        {
            Impl::hodge(p[Direction::xDir], d[Direction::xDir],
                        AMREX_D_DECL(chosenCellToNode, chosenNodeToCell, chosenNodeToCell));
            Impl::hodge(p[Direction::yDir], d[Direction::yDir],
                        AMREX_D_DECL(chosenNodeToCell, chosenCellToNode, chosenNodeToCell));
            Impl::hodge(p[Direction::zDir], d[Direction::zDir],
                        AMREX_D_DECL(chosenNodeToCell, chosenNodeToCell, chosenCellToNode));
        },
        cellToNode, nodeToCell);
};
void finite_difference_hodge (PrimalThreeForm& p, DualZeroForm& d, size_t degree)
{
    auto [nodeToCell, cellToNode] = Impl::select_hodge_stencil_finite_difference(degree);
    std::visit([&] (auto const& chosenStencil)
               { Impl::hodge(p, d, AMREX_D_DECL(chosenStencil, chosenStencil, chosenStencil)); },
               nodeToCell);
};
void finite_difference_hodge (DualZeroForm& d, PrimalThreeForm& p, size_t degree)
{
    auto [nodeToCell, cellToNode] = Impl::select_hodge_stencil_finite_difference(degree);
    std::visit([&] (auto const& chosenStencil)
               { Impl::hodge(d, p, AMREX_D_DECL(chosenStencil, chosenStencil, chosenStencil)); },
               cellToNode);
};
void finite_difference_hodge (DualOneForm& d, PrimalTwoForm& p, size_t degree)
{
    auto [nodeToCell, cellToNode] = Impl::select_hodge_stencil_finite_difference(degree);
    std::visit(
        [&] (auto const& chosenCellToNode, auto const& chosenNodeToCell)
        {
            Impl::hodge(d[Direction::xDir], p[Direction::xDir],
                        AMREX_D_DECL(chosenNodeToCell, chosenCellToNode, chosenCellToNode));
            Impl::hodge(d[Direction::yDir], p[Direction::yDir],
                        AMREX_D_DECL(chosenCellToNode, chosenNodeToCell, chosenCellToNode));
            Impl::hodge(d[Direction::zDir], p[Direction::zDir],
                        AMREX_D_DECL(chosenCellToNode, chosenCellToNode, chosenNodeToCell));
        },
        cellToNode, nodeToCell);
};
void finite_difference_hodge (DualTwoForm& d, PrimalOneForm& p, size_t degree)
{
    auto [nodeToCell, cellToNode] = Impl::select_hodge_stencil_finite_difference(degree);
    std::visit(
        [&] (auto const& chosenCellToNode, auto const& chosenNodeToCell)
        {
            Impl::hodge(d[Direction::xDir], p[Direction::xDir],
                        AMREX_D_DECL(chosenCellToNode, chosenNodeToCell, chosenNodeToCell));
            Impl::hodge(d[Direction::yDir], p[Direction::yDir],
                        AMREX_D_DECL(chosenNodeToCell, chosenCellToNode, chosenNodeToCell));
            Impl::hodge(d[Direction::zDir], p[Direction::zDir],
                        AMREX_D_DECL(chosenNodeToCell, chosenNodeToCell, chosenCellToNode));
        },
        cellToNode, nodeToCell);
};
void finite_difference_hodge (DualThreeForm& d, PrimalZeroForm& p, size_t degree)
{
    auto [nodeToCell, cellToNode] = Impl::select_hodge_stencil_finite_difference(degree);
    std::visit([&] (auto const& chosenStencil)
               { Impl::hodge(d, p, AMREX_D_DECL(chosenStencil, chosenStencil, chosenStencil)); },
               nodeToCell);
};
} // namespace Impl
void hodge (PrimalZeroForm& p, DualThreeForm& d)
{
    Io::Parameters params{};
    int degree{2};
    params.get_or_set("FiniteDifferenceDeRhamComplex.hodgeDegree", degree);
    Impl::finite_difference_hodge(p, d, degree);
};
void hodge (PrimalOneForm& p, DualTwoForm& d)
{
    Io::Parameters params{};
    int degree{2};
    params.get_or_set("FiniteDifferenceDeRhamComplex.hodgeDegree", degree);
    Impl::finite_difference_hodge(p, d, degree);
};
void hodge (PrimalTwoForm& p, DualOneForm& d)
{
    Io::Parameters params{};
    int degree{2};
    params.get_or_set("FiniteDifferenceDeRhamComplex.hodgeDegree", degree);
    Impl::finite_difference_hodge(p, d, degree);
};
void hodge (PrimalThreeForm& p, DualZeroForm& d)
{
    Io::Parameters params{};
    int degree{2};
    params.get_or_set("FiniteDifferenceDeRhamComplex.hodgeDegree", degree);
    Impl::finite_difference_hodge(p, d, degree);
};
void hodge (DualZeroForm& d, PrimalThreeForm& p)
{
    Io::Parameters params{};
    int degree{2};
    params.get_or_set("FiniteDifferenceDeRhamComplex.hodgeDegree", degree);
    Impl::finite_difference_hodge(d, p, degree);
};
void hodge (DualOneForm& d, PrimalTwoForm& p)
{
    Io::Parameters params{};
    int degree{2};
    params.get_or_set("FiniteDifferenceDeRhamComplex.hodgeDegree", degree);
    Impl::finite_difference_hodge(d, p, degree);
};
void hodge (DualTwoForm& d, PrimalOneForm& p)
{
    Io::Parameters params{};
    int degree{2};
    params.get_or_set("FiniteDifferenceDeRhamComplex.hodgeDegree", degree);
    Impl::finite_difference_hodge(d, p, degree);
};
void hodge (DualThreeForm& d, PrimalZeroForm& p)
{
    Io::Parameters params{};
    int degree{2};
    params.get_or_set("FiniteDifferenceDeRhamComplex.hodgeDegree", degree);
    Impl::finite_difference_hodge(d, p, degree);
};

} // namespace Gempic

namespace Gempic::Forms
{

FDDeRhamComplex::FDDeRhamComplex(ComputationalDomain const& infra,
                                 int const hodgeDegree,
                                 int const maxSplineDegree,
                                 HodgeScheme hodgeScheme,
                                 int nComp) :
    DeRhamComplex::DeRhamComplex{infra, hodgeDegree, maxSplineDegree}
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::FDDeRhamComplex()");
    // Parameters used in the projection and hodge
    for (size_t dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        m_dr[dir] = infra.geometry().CellSize(dir);
    }
    m_nGhost = DeRhamComplex::m_nGhost[xDir];
    m_hodgeScheme = hodgeScheme;

    // Read the scaling factors from input file and compute value for the Hodge operator
    Gempic::Io::Parameters parameters{"FDDeRhamComplex"};
    m_sV = 1.0;
    parameters.get_or_set("sV", m_sV);
    m_sOmega = 1.0;
    parameters.get_or_set("sOmega", m_sOmega);

    // There is only one components in each MultiFab as the different components of the forms are
    // centered differently
    m_tempPrimalZeroForm.define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 1, 1)})),
        m_distriMap, nComp, m_nGhost);
    m_tempDualZeroForm.define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 0, 0)})),
        m_distriMap, nComp, m_nGhost);

    m_tempPrimalOneForm[xDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 1, 1)})),
        m_distriMap, nComp, m_nGhost);
    m_tempPrimalOneForm[yDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 0, 1)})),
        m_distriMap, nComp, m_nGhost);
    m_tempPrimalOneForm[zDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 1, 0)})),
        m_distriMap, nComp, m_nGhost);

    m_tempDualOneForm[xDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 0, 0)})),
        m_distriMap, nComp, m_nGhost);
    m_tempDualOneForm[yDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 1, 0)})),
        m_distriMap, nComp, m_nGhost);
    m_tempDualOneForm[zDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 0, 1)})),
        m_distriMap, nComp, m_nGhost);

    m_tempPrimalTwoForm[xDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 0, 0)})),
        m_distriMap, nComp, m_nGhost);
    m_tempPrimalTwoForm[yDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 1, 0)})),
        m_distriMap, nComp, m_nGhost);
    m_tempPrimalTwoForm[zDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 0, 1)})),
        m_distriMap, nComp, m_nGhost);

    m_tempDualTwoForm[xDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 1, 1)})),
        m_distriMap, nComp, m_nGhost);
    m_tempDualTwoForm[yDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 0, 1)})),
        m_distriMap, nComp, m_nGhost);
    m_tempDualTwoForm[zDir].define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 1, 0)})),
        m_distriMap, nComp, m_nGhost);

    m_tempPrimalThreeForm.define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 0, 0)})),
        m_distriMap, nComp, m_nGhost);
    m_tempDualThreeForm.define(
        amrex::convert(m_grid, amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 1, 1)})),
        m_distriMap, nComp, m_nGhost);
}

FDDeRhamComplex::~FDDeRhamComplex() {}

amrex::Real FDDeRhamComplex::scaling_eto_d() { return 1 / (m_sOmega * m_sOmega); }

amrex::Real FDDeRhamComplex::scaling_dto_e() { return m_sOmega * m_sOmega; }

amrex::Real FDDeRhamComplex::scaling_bto_h() { return m_sV * m_sV / (m_sOmega * m_sOmega); }

} //namespace Gempic::Forms
