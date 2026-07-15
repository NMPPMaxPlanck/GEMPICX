/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include "GEMPIC_Amr.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"

using namespace Gempic;

namespace Gempic::Amr
{
amrex::Geometry create_geometry (Io::Parameters& params)
{
    std::array<DiscreteAxis::IndexPosition, AMREX_SPACEDIM> idxPosition;
    idxPosition.fill(DiscreteAxis::IndexPosition::Cell);
    DiscreteGrid grid(params, idxPosition);

    return Gempic::Impl::to_amrex_geometry(grid);
}

amrex::AmrInfo create_amr_info (Io::Parameters& params)
{
    amrex::AmrInfo amrInfo;
    params.get_or_set("MeshRefinement.maxLevel", amrInfo.max_level);
    params.get_or_set("MeshRefinement.blockingFactor", amrInfo.blocking_factor[0]);
    params.get_or_set("ComputationalDomain.maxGridSize", amrInfo.max_grid_size[0]);
    params.get_or_set("MeshRefinement.nErrorBuff", amrInfo.n_error_buf[0]);
    params.get_or_set("MeshRefinement.gridEff", amrInfo.grid_eff);
    params.get_or_set("MeshRefinement.nProper", amrInfo.n_proper);
    params.get_or_set("MeshRefinement.refRatio", amrInfo.ref_ratio[0]);
    return amrInfo;
}

GempicAmrCore::GempicAmrCore(Io::Parameters& params) :
    amrex::AmrCore(create_geometry(params), create_amr_info(params))
{
    std::string taggingFunction;
    params.get("MeshRefinement.taggingFunction", taggingFunction);
    m_taggingParser = amrex::Parser(taggingFunction);
    m_taggingParser.registerVariables({AMREX_D_DECL("x", "y", "z")});
    for (auto const& [name, value] : params.get_named_constants())
    {
        m_taggingParser.setConstant(name, value);
    }
    params.get("MeshRefinement.taggingLevels", m_taggingLevels);
}

void GempicAmrCore::ErrorEst (int lev, amrex::TagBoxArray& tags, amrex::Real /*time*/, int /*ngrow*/)
{
    if (lev >= m_taggingLevels.size()) return;

    int const tagval = amrex::TagBox::SET;

    auto const problo = Geom(lev).ProbLoArray();
    auto const dx = Geom(lev).CellSizeArray();

    for (amrex::MFIter mfi(tags, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.tilebox();
        auto const tagfab = tags.array(mfi);
        amrex::Real taggingLevel = m_taggingLevels[lev];
        auto f = m_taggingParser.compile<AMREX_SPACEDIM>();

        amrex::ParallelFor(bx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                           {
                               AMREX_D_TERM(amrex::Real x = problo[0] + (i + 0.5) * dx[0];
                                            , amrex::Real y = problo[1] + (j + 0.5) * dx[1];
                                            , amrex::Real z = problo[2] + (k + 0.5) * dx[2];)
                               if (f(AMREX_D_DECL(x, y, z)) > taggingLevel)
                               {
                                   tagfab(i, j, k) = tagval;
                               }
                           });
    }
}

MLFiniteDifferenceDeRhamSpaces::MLFiniteDifferenceDeRhamSpaces(
    Io::Parameters& params,
    std::shared_ptr<GempicAmrCore> amr,
    GaussLegendreQuadrature const& integrator) :
    m_amr{amr}
{
    for (int lev = 0; lev <= m_amr->finestLevel(); lev++)
    {
        DiscreteGrid grid{
            m_amr->Geom(lev),
            {AMREX_D_DECL(DiscreteAxis::IndexPosition::Cell, DiscreteAxis::IndexPosition::Cell,
                          DiscreteAxis::IndexPosition::Cell)}};
        m_deRham.push_back(FiniteDifferenceDeRhamSpaces(
            params, integrator, grid, amr->boxArray(lev), amr->DistributionMap(lev)));
    }
}

// TODO: Make creation of LevelInterface objects optional since they are not needed for all fields

MLPrimalZeroForm MLFiniteDifferenceDeRhamSpaces::create_primal_zero_form (
    std::string const& label,
    Gempic::Impl::BoundaryConditionConfiguration bcConf,
    bool buildLevelInterface,
    bool buildCoveredMask) const
{
    MLPrimalZeroForm form(m_amr, label);
    for (int lev = 0; lev <= m_amr->finestLevel(); lev++)
    {
        form.push_back(m_deRham[lev].create_primal_zero_form(label, bcConf));
    }
    if (buildLevelInterface)
    {
        form.m_levelInterfaces = make_level_interfaces(form, m_amr, buildCoveredMask);
    }
    return form;
}

MLPrimalOneForm MLFiniteDifferenceDeRhamSpaces::create_primal_one_form (
    std::string const& label,
    std::array<Gempic::Impl::BoundaryConditionConfiguration, 3> bcConf,
    bool buildLevelInterface,
    bool buildCoveredMask) const
{
    MLPrimalOneForm form(m_amr, label);
    for (int lev = 0; lev <= m_amr->finestLevel(); lev++)
    {
        form.push_back(m_deRham[lev].create_primal_one_form(label, bcConf));
    }
    if (buildLevelInterface)
    {
        form.m_levelInterfaces = make_level_interfaces(form, m_amr, buildCoveredMask);
    }
    return form;
}

MLPrimalTwoForm MLFiniteDifferenceDeRhamSpaces::create_primal_two_form (
    std::string const& label,
    std::array<Gempic::Impl::BoundaryConditionConfiguration, 3> bcConf,
    bool buildLevelInterface,
    bool buildCoveredMask) const
{
    MLPrimalTwoForm form(m_amr, label);
    for (int lev = 0; lev <= m_amr->finestLevel(); lev++)
    {
        form.push_back(m_deRham[lev].create_primal_two_form(label, bcConf));
    }
    if (buildLevelInterface)
    {
        form.m_levelInterfaces = make_level_interfaces(form, m_amr, buildCoveredMask);
    }
    return form;
}

MLPrimalThreeForm MLFiniteDifferenceDeRhamSpaces::create_primal_three_form (
    std::string const& label,
    Gempic::Impl::BoundaryConditionConfiguration bcConf,
    bool buildLevelInterface,
    bool buildCoveredMask) const
{
    MLPrimalThreeForm form(m_amr, label);
    for (int lev = 0; lev <= m_amr->finestLevel(); lev++)
    {
        form.push_back(m_deRham[lev].create_primal_three_form(label, bcConf));
    }
    if (buildLevelInterface)
    {
        form.m_levelInterfaces = make_level_interfaces(form, m_amr, buildCoveredMask);
    }
    return form;
}

MLDualZeroForm MLFiniteDifferenceDeRhamSpaces::create_dual_zero_form (
    std::string const& label,
    Gempic::Impl::BoundaryConditionConfiguration bcConf,
    bool buildLevelInterface,
    bool buildCoveredMask) const
{
    MLDualZeroForm form(m_amr, label);
    for (int lev = 0; lev <= m_amr->finestLevel(); lev++)
    {
        form.push_back(m_deRham[lev].create_dual_zero_form(label, bcConf));
    }
    if (buildLevelInterface)
    {
        form.m_levelInterfaces = make_level_interfaces(form, m_amr, buildCoveredMask);
    }
    return form;
}

MLDualOneForm MLFiniteDifferenceDeRhamSpaces::create_dual_one_form (
    std::string const& label,
    std::array<Gempic::Impl::BoundaryConditionConfiguration, 3> bcConf,
    bool buildLevelInterface,
    bool buildCoveredMask) const
{
    MLDualOneForm form(m_amr, label);
    for (int lev = 0; lev <= m_amr->finestLevel(); lev++)
    {
        form.push_back(m_deRham[lev].create_dual_one_form(label, bcConf));
    }
    if (buildLevelInterface)
    {
        form.m_levelInterfaces = make_level_interfaces(form, m_amr, buildCoveredMask);
    }
    return form;
}

MLDualTwoForm MLFiniteDifferenceDeRhamSpaces::create_dual_two_form (
    std::string const& label,
    std::array<Gempic::Impl::BoundaryConditionConfiguration, 3> bcConf,
    bool buildLevelInterface,
    bool buildCoveredMask) const
{
    MLDualTwoForm form(m_amr, label);
    for (int lev = 0; lev <= m_amr->finestLevel(); lev++)
    {
        form.push_back(m_deRham[lev].create_dual_two_form(label, bcConf));
    }
    if (buildLevelInterface)
    {
        form.m_levelInterfaces = make_level_interfaces(form, m_amr, buildCoveredMask);
    }
    return form;
}

MLDualThreeForm MLFiniteDifferenceDeRhamSpaces::create_dual_three_form (
    std::string const& label,
    Gempic::Impl::BoundaryConditionConfiguration bcConf,
    bool buildLevelInterface,
    bool buildCoveredMask) const
{
    MLDualThreeForm form(m_amr, label);
    for (int lev = 0; lev <= m_amr->finestLevel(); lev++)
    {
        form.push_back(m_deRham[lev].create_dual_three_form(label, bcConf));
    }
    if (buildLevelInterface)
    {
        form.m_levelInterfaces = make_level_interfaces(form, m_amr, buildCoveredMask);
    }
    return form;
}

amrex::Vector<amrex::MultiFab> make_diagnostic_multifab (std::shared_ptr<GempicAmrCore> const& amr,
                                                         int numComp)
{
    amrex::Vector<amrex::MultiFab> diagnostics;
    for (int lev = 0; lev <= amr->finestLevel(); lev++)
    {
        diagnostics.push_back(
            amrex::MultiFab(amr->boxArray(lev), amr->DistributionMap(lev), numComp, 0));
    }
    return diagnostics;
}

void write_plot_file (amrex::Vector<amrex::MultiFab>& diagnostics,
                      std::shared_ptr<GempicAmrCore> const& amr,
                      DiscreteTime const& time,
                      amrex::Vector<std::string> varNames,
                      std::string name)
{
    amrex::Vector<int> levelStep(amr->finestLevel() + 1, time.current_step());
    amrex::Vector<amrex::IntVect> refRatio(amr->finestLevel() + 1,
                                           amrex::IntVect(AMREX_D_DECL(2, 2, 2)));
    std::string filename = amrex::Concatenate(name, time.current_step());

    amrex::WriteMultiLevelPlotfile(filename, amr->finestLevel() + 1,
                                   amrex::GetVecOfConstPtrs(diagnostics), varNames, amr->Geom(),
                                   time.t(), levelStep, refRatio);
}

amrex::Real diagnostic_scale (DiscreteField const& field)
{
    amrex::Real scale{1.0};
    for (auto gridDir : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
    {
        if (field.dof_category(gridDir) == DiscreteField::DOFCategory::CenteredLineIntegral)
        {
            scale *= 1.0 / field.discrete_grid().dx(gridDir);
        }
    }
    return scale;
}
} // namespace Gempic::Amr