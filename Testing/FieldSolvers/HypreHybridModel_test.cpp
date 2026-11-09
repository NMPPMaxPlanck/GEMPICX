/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_Fields.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_HypreUtils.H"
#include "GEMPIC_QuasineutralSolver.H"
#include "GEMPIC_SplineClass.H"
#include "TestUtils/GEMPIC_TestUtils.H"

namespace
{
using namespace Gempic;
using namespace Forms;
using namespace FieldSolvers;

/**
 * @brief Tests the Poisson solver to obtain the
 * initial A-field for the quasineutral solver.
 */
template <typename HodgeDegreeStruct>
class HypreQuasineutralHybridModelTest : public testing::Test
{
public:
    static constexpr int s_vdim{3};
    static constexpr int s_ndata{1};
    static constexpr int s_degX{3};
    static constexpr int s_degY{3};
    static constexpr int s_degZ{3};
    static constexpr int s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};
    static constexpr int s_hodgeDegree{HodgeDegreeStruct::value};

    static int const s_nVar = AMREX_SPACEDIM + 1; // x, y, z, t

    amrex::Array<amrex::ParserExecutor<s_nVar>, 3> m_funcRHS;
    amrex::Array<amrex::Parser, 3> m_parserRHS;

    amrex::Array<amrex::ParserExecutor<s_nVar>, 3> m_funcBstarbyRho;
    amrex::Array<amrex::Parser, 3> m_parserBstarbyRho;

    amrex::Array<amrex::ParserExecutor<s_nVar>, 3> m_funcB;
    amrex::Array<amrex::Parser, 3> m_parserB;

    static void SetUpTestSuite ()
    {
        /* Initialize the infrastructure */
        amrex::ParmParse pp; // Used in lieu of input file

        amrex::Vector<amrex::Real> const domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
        pp.addarr("ComputationalDomain.domainLo", domainLo);

        amrex::Vector<amrex::Real> const domainHi{AMREX_D_DECL(2 * M_PI, 2 * M_PI, 2 * M_PI)};
        pp.addarr("ComputationalDomain.domainHi", domainHi);

        // Grid parameters
        amrex::Vector<int> const maxGridSize{AMREX_D_DECL(40, 40, 40)};
        pp.addarr("ComputationalDomain.maxGridSize", maxGridSize);

        amrex::Vector<int> const isPeriodic{AMREX_D_DECL(1, 1, 1)};
        pp.addarr("ComputationalDomain.isPeriodic", isPeriodic);

        pp.add("hypre.verbose", 2);

        std::string hypreSolver = "GMRES";
        pp.add("hypre.hypre_solver", hypreSolver);

        std::string hyprePreconditioner = "euclid";
        pp.add("hypre.hypre_preconditioner", hyprePreconditioner);

        pp.add("hypre.num_krylov", 50);

        pp.add("hypre.max_iterations", 1000);

        pp.add("hypre.euclid_level", 2);

        pp.add("hypre.euclid_use_block_jacobi", 1);
    }

    // virtual void SetUp() will be called before each test is run.
    void SetUp () override
    {
        amrex::Array<std::string, 3> const analyticalRHS = {
            "sin(y) + 0.1*cos(z)*(sin(y) + 2.0*sin(z))",
            "sin(z) + 0.1*cos(x)*(sin(z) + 2.0*sin(x))",
            "sin(x) + 0.1*cos(y)*(sin(x) + 2.0*sin(y))"};
#if AMREX_SPACEDIM == 2
        amrex::Array<std::string, 3> const analyticalBstarbyRho = {"sin(y)", "sin(2*x)",
                                                                   "cos(x)*cos(y)"};
        amrex::Array<std::string, 3> const analyticalB = {"sin(y)", "4*sin(2*x)",
                                                          "2*cos(x)*cos(y)"};
#elif AMREX_SPACEDIM == 3
        amrex::Array<std::string, 3> const analyticalBstarbyRho = {"0.1*cos(x)", "0.1*cos(y)",
                                                                   "0.1*cos(z)"};
        amrex::Array<std::string, 3> const analyticalB = {"sin(y)", "sin(z)", "sin(x)"};
#endif

        for (int i = 0; i < 3; ++i)
        {
            m_parserRHS[i].define(analyticalRHS[i]);
            m_parserRHS[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            m_funcRHS[i] = m_parserRHS[i].compile<s_nVar>();

            m_parserBstarbyRho[i].define(analyticalBstarbyRho[i]);
            m_parserBstarbyRho[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            m_funcBstarbyRho[i] = m_parserBstarbyRho[i].compile<s_nVar>();

            m_parserB[i].define(analyticalB[i]);
            m_parserB[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            m_funcB[i] = m_parserB[i].compile<s_nVar>();
        }
    }

    amrex::Real hybridmodel_operator_solve (int n)
    {
        // Initialize computational_domain
        amrex::Vector<int> const nCell{AMREX_D_DECL(n, n, n)};

        Gempic::Io::Parameters parameters{};
        amrex::ParmParse pp;
        pp.addarr("ComputationalDomain.nCell", nCell);
        ComputationalDomain infra;

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(infra, s_hodgeDegree, s_maxSplineDegree,
                                                        HodgeScheme::FDHodge);

        amrex::Array<DeRhamField<Grid::primal, Space::node>, 3> rhs = {
            DeRhamField<Grid::primal, Space::node>(deRham, m_funcRHS[0]),
            DeRhamField<Grid::primal, Space::node>(deRham, m_funcRHS[1]),
            DeRhamField<Grid::primal, Space::node>(deRham, m_funcRHS[2])};

        amrex::Array<DeRhamField<Grid::primal, Space::node>, 3> bstarbyRho = {
            DeRhamField<Grid::primal, Space::node>(deRham, m_funcBstarbyRho[0]),
            DeRhamField<Grid::primal, Space::node>(deRham, m_funcBstarbyRho[1]),
            DeRhamField<Grid::primal, Space::node>(deRham, m_funcBstarbyRho[2])};

        amrex::Array<DeRhamField<Grid::primal, Space::node>, 3> B = {
            DeRhamField<Grid::primal, Space::node>(deRham),
            DeRhamField<Grid::primal, Space::node>(deRham),
            DeRhamField<Grid::primal, Space::node>(deRham)};

        amrex::Array<DeRhamField<Grid::primal, Space::node>, 3> bAn = {
            DeRhamField<Grid::primal, Space::node>(deRham, m_funcB[0]),
            DeRhamField<Grid::primal, Space::node>(deRham, m_funcB[1]),
            DeRhamField<Grid::primal, Space::node>(deRham, m_funcB[2])};

        auto const& nddom = amrex::surroundingNodes(infra.box());

        std::array<decltype((bstarbyRho[0].m_data).arrays()), s_vdim> bstarbyRhoarrs;
        for (int ndir = 0; ndir < s_vdim; ++ndir)
        {
            bstarbyRhoarrs[ndir] = (bstarbyRho[ndir].m_data).arrays();
        }

        amrex::GpuArray<amrex::Real, 3> dxi{GEMPIC_D_PAD_ONE(infra.geometry().InvCellSize(xDir),
                                                             infra.geometry().InvCellSize(yDir),
                                                             infra.geometry().InvCellSize(zDir))};

        amrex::GpuArray<HYPRE_Real, AMREX_SPACEDIM> fac{
            AMREX_D_DECL(static_cast<HYPRE_Real>(dxi[xDir] * dxi[xDir]),
                         static_cast<HYPRE_Real>(dxi[yDir] * dxi[yDir]),
                         static_cast<HYPRE_Real>(dxi[zDir] * dxi[zDir]))};

        HYPRE_Real fac0 = HYPRE_Real(-2.) * (AMREX_D_TERM(fac[xDir], +fac[yDir], +fac[zDir]));

        // Is variable n at (i,j,k) in Box boxno (local index) valid?
        // (i.e., not exactly on Dirichlet boundary)
        auto marker = [=] AMREX_GPU_DEVICE(int /*boxno*/, int i, int j, int k, int /*n*/) -> bool
        { return nddom.contains(i, j, k); };

        amrex::Real dt = 1.0;

        // For variable n at (i,j,k) in Box boxno (local index), fill its row in
        // the matrix.
        // [in ] gid : gid[n] is the id for variable n at (i,j,k)
        // [out] ncols: # of columns in this row.
        // [out] cols: column indices in this row.
        // [out] mat : matrix elements in this row.
        auto filler = [=] AMREX_GPU_DEVICE(int boxno, int i, int j, int k, int ndir,
                                           amrex::Array4<HYPRE_Int const> const* gid,
                                           HYPRE_Int& ncols, HYPRE_Int* cols, HYPRE_Real* mat)
        {
            int n[AMREX_SPACEDIM], in[AMREX_SPACEDIM], jn[AMREX_SPACEDIM], kn[AMREX_SPACEDIM];
            for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
            {
                n[dir] = (ndir + dir) % AMREX_SPACEDIM;
                in[dir] = (n[dir] == 0);
                jn[dir] = (n[dir] == 1);
                kn[dir] = (n[dir] == 2);
            }

            ncols = 0;

            if (ndir == xDir)
            {
                // X-field coefficients
                cols[ncols] = gid[xDir](i, j, k);
                mat[ncols] = 1.0;
                ++ncols;

                cols[ncols] = gid[xDir](i, j + 1, k + 1);
                mat[ncols] = 0.25 * dxi[yDir] * dxi[zDir] *
                             (-bstarbyRhoarrs[xDir][boxno](i, j + 1, k) +
                              bstarbyRhoarrs[xDir][boxno](i, j, k + 1));
                ++ncols;

                cols[ncols] = gid[xDir](i, j + 1, k - 1);
                mat[ncols] = 0.25 * dxi[yDir] * dxi[zDir] *
                             (+bstarbyRhoarrs[xDir][boxno](i, j + 1, k) -
                              bstarbyRhoarrs[xDir][boxno](i, j, k - 1));
                ++ncols;

                cols[ncols] = gid[xDir](i, j - 1, k + 1);
                mat[ncols] = 0.25 * dxi[yDir] * dxi[zDir] *
                             (+bstarbyRhoarrs[xDir][boxno](i, j - 1, k) -
                              bstarbyRhoarrs[xDir][boxno](i, j, k + 1));
                ++ncols;

                cols[ncols] = gid[xDir](i, j - 1, k - 1);
                mat[ncols] = 0.25 * dxi[yDir] * dxi[zDir] *
                             (-bstarbyRhoarrs[xDir][boxno](i, j - 1, k) +
                              bstarbyRhoarrs[xDir][boxno](i, j, k - 1));
                ++ncols; //*/

                // Y-field coefficients
                cols[ncols] = gid[yDir](i, j, k);
                mat[ncols] = 0.25 * dxi[zDir] * dxi[zDir] *
                             (bstarbyRhoarrs[zDir][boxno](i, j, k + 1) +
                              bstarbyRhoarrs[zDir][boxno](i, j, k - 1));
                ++ncols;

                cols[ncols] = gid[yDir](i, j, k + 2);
                mat[ncols] =
                    0.25 * dxi[zDir] * dxi[zDir] * (-bstarbyRhoarrs[zDir][boxno](i, j, k + 1));
                ++ncols;

                cols[ncols] = gid[yDir](i, j, k - 2);
                mat[ncols] =
                    0.25 * dxi[zDir] * dxi[zDir] * (-bstarbyRhoarrs[zDir][boxno](i, j, k - 1));
                ++ncols; //*/

                //
                cols[ncols] = gid[yDir](i, j + 1, k + 1);
                mat[ncols] =
                    0.25 * dxi[yDir] * dxi[zDir] * (-bstarbyRhoarrs[yDir][boxno](i, j + 1, k));
                ++ncols;

                cols[ncols] = gid[yDir](i, j + 1, k - 1);
                mat[ncols] =
                    0.25 * dxi[yDir] * dxi[zDir] * (bstarbyRhoarrs[yDir][boxno](i, j + 1, k));
                ++ncols;

                cols[ncols] = gid[yDir](i, j - 1, k + 1);
                mat[ncols] =
                    0.25 * dxi[yDir] * dxi[zDir] * (bstarbyRhoarrs[yDir][boxno](i, j - 1, k));
                ++ncols;

                cols[ncols] = gid[yDir](i, j - 1, k - 1);
                mat[ncols] =
                    0.25 * dxi[yDir] * dxi[zDir] * (-bstarbyRhoarrs[yDir][boxno](i, j - 1, k));
                ++ncols; //*/

                //
                cols[ncols] = gid[yDir](i + 1, j, k + 1);
                mat[ncols] =
                    0.25 * dxi[xDir] * dxi[zDir] * (-bstarbyRhoarrs[xDir][boxno](i, j, k + 1));
                ++ncols;

                cols[ncols] = gid[yDir](i - 1, j, k + 1);
                mat[ncols] =
                    0.25 * dxi[xDir] * dxi[zDir] * (bstarbyRhoarrs[xDir][boxno](i, j, k + 1));
                ++ncols;

                cols[ncols] = gid[yDir](i + 1, j, k - 1);
                mat[ncols] =
                    0.25 * dxi[xDir] * dxi[zDir] * (bstarbyRhoarrs[xDir][boxno](i, j, k - 1));
                ++ncols;

                cols[ncols] = gid[yDir](i - 1, j, k - 1);
                mat[ncols] =
                    0.25 * dxi[xDir] * dxi[zDir] * (-bstarbyRhoarrs[xDir][boxno](i, j, k - 1));
                ++ncols; //*/

                // Z-field coefficients
                cols[ncols] = gid[zDir](i, j, k);
                mat[ncols] = 0.25 * dxi[yDir] * dxi[yDir] *
                             (-bstarbyRhoarrs[yDir][boxno](i, j + 1, k) -
                              bstarbyRhoarrs[yDir][boxno](i, j - 1, k));
                ++ncols;

                cols[ncols] = gid[zDir](i, j + 2, k);
                mat[ncols] =
                    0.25 * dxi[yDir] * dxi[yDir] * bstarbyRhoarrs[yDir][boxno](i, j + 1, k);
                ++ncols;

                cols[ncols] = gid[zDir](i, j - 2, k);
                mat[ncols] =
                    0.25 * dxi[yDir] * dxi[yDir] * bstarbyRhoarrs[yDir][boxno](i, j - 1, k);
                ++ncols; //*/

                //
                cols[ncols] = gid[zDir](i, j + 1, k + 1);
                mat[ncols] =
                    0.25 * dxi[zDir] * dxi[yDir] * (bstarbyRhoarrs[zDir][boxno](i, j, k + 1));
                ++ncols;

                cols[ncols] = gid[zDir](i, j - 1, k + 1);
                mat[ncols] =
                    0.25 * dxi[zDir] * dxi[yDir] * (-bstarbyRhoarrs[zDir][boxno](i, j, k + 1));
                ++ncols;

                cols[ncols] = gid[zDir](i, j + 1, k - 1);
                mat[ncols] =
                    0.25 * dxi[zDir] * dxi[yDir] * (-bstarbyRhoarrs[zDir][boxno](i, j, k - 1));
                ++ncols;

                cols[ncols] = gid[zDir](i, j - 1, k - 1);
                mat[ncols] =
                    0.25 * dxi[zDir] * dxi[yDir] * (bstarbyRhoarrs[zDir][boxno](i, j, k - 1));
                ++ncols; //*/

                //
                cols[ncols] = gid[zDir](i + 1, j + 1, k);
                mat[ncols] =
                    0.25 * dxi[xDir] * dxi[yDir] * (bstarbyRhoarrs[xDir][boxno](i, j + 1, k));
                ++ncols;

                cols[ncols] = gid[zDir](i - 1, j + 1, k);
                mat[ncols] =
                    0.25 * dxi[xDir] * dxi[yDir] * (-bstarbyRhoarrs[xDir][boxno](i, j + 1, k));
                ++ncols;

                cols[ncols] = gid[zDir](i + 1, j - 1, k);
                mat[ncols] =
                    0.25 * dxi[xDir] * dxi[yDir] * (-bstarbyRhoarrs[xDir][boxno](i, j - 1, k));
                ++ncols;

                cols[ncols] = gid[zDir](i - 1, j - 1, k);
                mat[ncols] =
                    0.25 * dxi[xDir] * dxi[yDir] * (bstarbyRhoarrs[xDir][boxno](i, j - 1, k));
                ++ncols; //*/
            }

            if (ndir == yDir)
            {
                // Y-field coefficients
                cols[ncols] = gid[yDir](i, j, k);
                mat[ncols] = 1.0;
                ++ncols;

                cols[ncols] = gid[yDir](i + 1, j, k + 1);
                mat[ncols] = 0.25 * dxi[zDir] * dxi[xDir] *
                             (-bstarbyRhoarrs[yDir][boxno](i, j, k + 1) +
                              bstarbyRhoarrs[yDir][boxno](i + 1, j, k));
                ++ncols;

                cols[ncols] = gid[yDir](i - 1, j, k + 1);
                mat[ncols] = 0.25 * dxi[zDir] * dxi[xDir] *
                             (+bstarbyRhoarrs[yDir][boxno](i, j, k + 1) -
                              bstarbyRhoarrs[yDir][boxno](i - 1, j, k));
                ++ncols;

                cols[ncols] = gid[yDir](i + 1, j, k - 1);
                mat[ncols] = 0.25 * dxi[zDir] * dxi[xDir] *
                             (+bstarbyRhoarrs[yDir][boxno](i, j, k - 1) -
                              bstarbyRhoarrs[yDir][boxno](i + 1, j, k));
                ++ncols;

                cols[ncols] = gid[yDir](i - 1, j, k - 1);
                mat[ncols] = 0.25 * dxi[zDir] * dxi[xDir] *
                             (-bstarbyRhoarrs[yDir][boxno](i, j, k - 1) +
                              bstarbyRhoarrs[yDir][boxno](i - 1, j, k));
                ++ncols; //*/

                // Z-field coefficients
                cols[ncols] = gid[zDir](i, j, k);
                mat[ncols] = 0.25 * dxi[xDir] * dxi[xDir] *
                             (bstarbyRhoarrs[xDir][boxno](i + 1, j, k) +
                              bstarbyRhoarrs[xDir][boxno](i - 1, j, k));
                ++ncols;

                cols[ncols] = gid[zDir](i + 2, j, k);
                mat[ncols] =
                    0.25 * dxi[xDir] * dxi[xDir] * (-bstarbyRhoarrs[xDir][boxno](i + 1, j, k));
                ++ncols;

                cols[ncols] = gid[zDir](i - 2, j, k);
                mat[ncols] =
                    0.25 * dxi[xDir] * dxi[xDir] * (-bstarbyRhoarrs[xDir][boxno](i - 1, j, k));
                ++ncols; //*/

                //
                cols[ncols] = gid[zDir](i + 1, j, k + 1);
                mat[ncols] =
                    0.25 * dxi[zDir] * dxi[xDir] * (-bstarbyRhoarrs[zDir][boxno](i, j, k + 1));
                ++ncols;

                cols[ncols] = gid[zDir](i - 1, j, k + 1);
                mat[ncols] =
                    0.25 * dxi[zDir] * dxi[xDir] * (bstarbyRhoarrs[zDir][boxno](i, j, k + 1));
                ++ncols;

                cols[ncols] = gid[zDir](i + 1, j, k - 1);
                mat[ncols] =
                    0.25 * dxi[zDir] * dxi[xDir] * (bstarbyRhoarrs[zDir][boxno](i, j, k - 1));
                ++ncols;

                cols[ncols] = gid[zDir](i - 1, j, k - 1);
                mat[ncols] =
                    0.25 * dxi[zDir] * dxi[xDir] * (-bstarbyRhoarrs[zDir][boxno](i, j, k - 1));
                ++ncols; //*/

                //
                cols[ncols] = gid[zDir](i + 1, j + 1, k);
                mat[ncols] =
                    0.25 * dxi[yDir] * dxi[xDir] * (-bstarbyRhoarrs[yDir][boxno](i + 1, j, k));
                ++ncols;

                cols[ncols] = gid[zDir](i + 1, j - 1, k);
                mat[ncols] =
                    0.25 * dxi[yDir] * dxi[xDir] * (bstarbyRhoarrs[yDir][boxno](i + 1, j, k));
                ++ncols;

                cols[ncols] = gid[zDir](i - 1, j + 1, k);
                mat[ncols] =
                    0.25 * dxi[yDir] * dxi[xDir] * (bstarbyRhoarrs[yDir][boxno](i - 1, j, k));
                ++ncols;

                cols[ncols] = gid[zDir](i - 1, j - 1, k);
                mat[ncols] =
                    0.25 * dxi[yDir] * dxi[xDir] * (-bstarbyRhoarrs[yDir][boxno](i - 1, j, k));
                ++ncols; //*/

                // X-field coefficients
                cols[ncols] = gid[xDir](i, j, k);
                mat[ncols] = 0.25 * dxi[zDir] * dxi[zDir] *
                             (-bstarbyRhoarrs[zDir][boxno](i, j, k + 1) -
                              bstarbyRhoarrs[zDir][boxno](i, j, k - 1));
                ++ncols;

                cols[ncols] = gid[xDir](i, j, k + 2);
                mat[ncols] =
                    0.25 * dxi[zDir] * dxi[zDir] * bstarbyRhoarrs[zDir][boxno](i, j, k + 1);
                ++ncols;

                cols[ncols] = gid[xDir](i, j, k - 2);
                mat[ncols] =
                    0.25 * dxi[zDir] * dxi[zDir] * bstarbyRhoarrs[zDir][boxno](i, j, k - 1);
                ++ncols; //*/

                //
                cols[ncols] = gid[xDir](i + 1, j, k + 1);
                mat[ncols] =
                    0.25 * dxi[xDir] * dxi[zDir] * (bstarbyRhoarrs[xDir][boxno](i + 1, j, k));
                ++ncols;

                cols[ncols] = gid[xDir](i + 1, j, k - 1);
                mat[ncols] =
                    0.25 * dxi[xDir] * dxi[zDir] * (-bstarbyRhoarrs[xDir][boxno](i + 1, j, k));
                ++ncols;

                cols[ncols] = gid[xDir](i - 1, j, k + 1);
                mat[ncols] =
                    0.25 * dxi[xDir] * dxi[zDir] * (-bstarbyRhoarrs[xDir][boxno](i - 1, j, k));
                ++ncols;

                cols[ncols] = gid[xDir](i - 1, j, k - 1);
                mat[ncols] =
                    0.25 * dxi[xDir] * dxi[zDir] * (bstarbyRhoarrs[xDir][boxno](i - 1, j, k));
                ++ncols; //*/

                //
                cols[ncols] = gid[xDir](i, j + 1, k + 1);
                mat[ncols] =
                    0.25 * dxi[yDir] * dxi[zDir] * (bstarbyRhoarrs[yDir][boxno](i, j, k + 1));
                ++ncols;

                cols[ncols] = gid[xDir](i, j - 1, k + 1);
                mat[ncols] =
                    0.25 * dxi[yDir] * dxi[zDir] * (-bstarbyRhoarrs[yDir][boxno](i, j, k + 1));
                ++ncols;

                cols[ncols] = gid[xDir](i, j + 1, k - 1);
                mat[ncols] =
                    0.25 * dxi[yDir] * dxi[zDir] * (-bstarbyRhoarrs[yDir][boxno](i, j, k - 1));
                ++ncols;

                cols[ncols] = gid[xDir](i, j - 1, k - 1);
                mat[ncols] =
                    0.25 * dxi[yDir] * dxi[zDir] * (bstarbyRhoarrs[yDir][boxno](i, j, k - 1));
                ++ncols; //*/
            }

            if (ndir == zDir)
            {
                // Z-field coefficients
                cols[ncols] = gid[zDir](i, j, k);
                mat[ncols] = 1.0;
                ++ncols;

                cols[ncols] = gid[zDir](i + 1, j + 1, k);
                mat[ncols] = 0.25 * dxi[xDir] * dxi[yDir] *
                             (-bstarbyRhoarrs[zDir][boxno](i + 1, j, k) +
                              bstarbyRhoarrs[zDir][boxno](i, j + 1, k));
                ++ncols;

                cols[ncols] = gid[zDir](i + 1, j - 1, k);
                mat[ncols] = 0.25 * dxi[xDir] * dxi[yDir] *
                             (+bstarbyRhoarrs[zDir][boxno](i + 1, j, k) -
                              bstarbyRhoarrs[zDir][boxno](i, j - 1, k));
                ++ncols;

                cols[ncols] = gid[zDir](i - 1, j + 1, k);
                mat[ncols] = 0.25 * dxi[xDir] * dxi[yDir] *
                             (+bstarbyRhoarrs[zDir][boxno](i - 1, j, k) -
                              bstarbyRhoarrs[zDir][boxno](i, j + 1, k));
                ++ncols;

                cols[ncols] = gid[zDir](i - 1, j - 1, k);
                mat[ncols] = 0.25 * dxi[xDir] * dxi[yDir] *
                             (-bstarbyRhoarrs[zDir][boxno](i - 1, j, k) +
                              bstarbyRhoarrs[zDir][boxno](i, j - 1, k));
                ++ncols; //*/

                // X-field coefficients
                cols[ncols] = gid[xDir](i, j, k);
                mat[ncols] = 0.25 * dxi[yDir] * dxi[yDir] *
                             (bstarbyRhoarrs[yDir][boxno](i, j + 1, k) +
                              bstarbyRhoarrs[yDir][boxno](i, j - 1, k));
                ++ncols;

                cols[ncols] = gid[xDir](i, j + 2, k);
                mat[ncols] =
                    0.25 * dxi[yDir] * dxi[yDir] * (-bstarbyRhoarrs[yDir][boxno](i, j + 1, k));
                ++ncols;

                cols[ncols] = gid[xDir](i, j - 2, k);
                mat[ncols] =
                    0.25 * dxi[yDir] * dxi[yDir] * (-bstarbyRhoarrs[yDir][boxno](i, j - 1, k));
                ++ncols; //*/

                //
                cols[ncols] = gid[xDir](i + 1, j + 1, k);
                mat[ncols] =
                    0.25 * dxi[xDir] * dxi[yDir] * (-bstarbyRhoarrs[xDir][boxno](i + 1, j, k));
                ++ncols;

                cols[ncols] = gid[xDir](i + 1, j - 1, k);
                mat[ncols] =
                    0.25 * dxi[xDir] * dxi[yDir] * (bstarbyRhoarrs[xDir][boxno](i + 1, j, k));
                ++ncols;

                cols[ncols] = gid[xDir](i - 1, j + 1, k);
                mat[ncols] =
                    0.25 * dxi[xDir] * dxi[yDir] * (bstarbyRhoarrs[xDir][boxno](i - 1, j, k));
                ++ncols;

                cols[ncols] = gid[xDir](i - 1, j - 1, k);
                mat[ncols] =
                    0.25 * dxi[xDir] * dxi[yDir] * (-bstarbyRhoarrs[xDir][boxno](i - 1, j, k));
                ++ncols; //*/

                //
                cols[ncols] = gid[xDir](i, j + 1, k + 1);
                mat[ncols] =
                    0.25 * dxi[zDir] * dxi[yDir] * (-bstarbyRhoarrs[zDir][boxno](i, j + 1, k));
                ++ncols;

                cols[ncols] = gid[xDir](i, j + 1, k - 1);
                mat[ncols] =
                    0.25 * dxi[zDir] * dxi[yDir] * (bstarbyRhoarrs[zDir][boxno](i, j + 1, k));
                ++ncols;

                cols[ncols] = gid[xDir](i, j - 1, k + 1);
                mat[ncols] =
                    0.25 * dxi[zDir] * dxi[yDir] * (bstarbyRhoarrs[zDir][boxno](i, j - 1, k));
                ++ncols;

                cols[ncols] = gid[xDir](i, j - 1, k - 1);
                mat[ncols] =
                    0.25 * dxi[zDir] * dxi[yDir] * (-bstarbyRhoarrs[zDir][boxno](i, j - 1, k));
                ++ncols; //*/

                // Y-field coefficients
                cols[ncols] = gid[yDir](i, j, k);
                mat[ncols] = 0.25 * dxi[xDir] * dxi[xDir] *
                             (-bstarbyRhoarrs[xDir][boxno](i + 1, j, k) -
                              bstarbyRhoarrs[xDir][boxno](i - 1, j, k));
                ++ncols;

                cols[ncols] = gid[yDir](i + 2, j, k);
                mat[ncols] =
                    0.25 * dxi[xDir] * dxi[xDir] * bstarbyRhoarrs[xDir][boxno](i + 1, j, k);
                ++ncols;

                cols[ncols] = gid[yDir](i - 2, j, k);
                mat[ncols] =
                    0.25 * dxi[xDir] * dxi[xDir] * bstarbyRhoarrs[xDir][boxno](i - 1, j, k);
                ++ncols; //*/

                //
                cols[ncols] = gid[yDir](i + 1, j + 1, k);
                mat[ncols] =
                    0.25 * dxi[yDir] * dxi[xDir] * (bstarbyRhoarrs[yDir][boxno](i, j + 1, k));
                ++ncols;

                cols[ncols] = gid[yDir](i - 1, j + 1, k);
                mat[ncols] =
                    0.25 * dxi[yDir] * dxi[xDir] * (-bstarbyRhoarrs[yDir][boxno](i, j + 1, k));
                ++ncols;

                cols[ncols] = gid[yDir](i + 1, j - 1, k);
                mat[ncols] =
                    0.25 * dxi[yDir] * dxi[xDir] * (-bstarbyRhoarrs[yDir][boxno](i, j - 1, k));
                ++ncols;

                cols[ncols] = gid[yDir](i - 1, j - 1, k);
                mat[ncols] =
                    0.25 * dxi[yDir] * dxi[xDir] * (bstarbyRhoarrs[yDir][boxno](i, j - 1, k));
                ++ncols; //*/

                //
                cols[ncols] = gid[yDir](i + 1, j, k + 1);
                mat[ncols] =
                    0.25 * dxi[zDir] * dxi[xDir] * (bstarbyRhoarrs[zDir][boxno](i + 1, j, k));
                ++ncols;

                cols[ncols] = gid[yDir](i + 1, j, k - 1);
                mat[ncols] =
                    0.25 * dxi[zDir] * dxi[xDir] * (-bstarbyRhoarrs[zDir][boxno](i + 1, j, k));
                ++ncols;

                cols[ncols] = gid[yDir](i - 1, j, k + 1);
                mat[ncols] =
                    0.25 * dxi[zDir] * dxi[xDir] * (-bstarbyRhoarrs[zDir][boxno](i - 1, j, k));
                ++ncols;

                cols[ncols] = gid[yDir](i - 1, j, k - 1);
                mat[ncols] =
                    0.25 * dxi[zDir] * dxi[xDir] * (bstarbyRhoarrs[zDir][boxno](i - 1, j, k));
                ++ncols; //*/
            }
        };

        constexpr int maxStencilSize = 27;

        amrex::Real relTol{1.e-15}, absTol{1.e-15}, amrex::Real accTol{1.e-12};
        amrex::Real maxiter{1000};

        amrex::HypreSolver<maxStencilSize> hypreSolver(
            {B[xDir].m_data.boxArray().ixType(), B[yDir].m_data.boxArray().ixType(),
             B[zDir].m_data.boxArray().ixType()},
            rhs[xDir].m_deRham->get_n_ghost(), infra.geometry(), infra.m_grid, infra.m_distriMap,
            marker, filler, amrex::system::verbose);

        hypreSolver.solve(amrex::Vector<amrex::MultiFab*>{&(B[xDir].m_data), &(B[yDir].m_data),
                                                          &(B[zDir].m_data)},
                          amrex::Vector<amrex::MultiFab const*>{
                              &(rhs[xDir].m_data), &(rhs[yDir].m_data), &(rhs[zDir].m_data)},
                          relTol, absTol, maxiter);
        Gempic::Utils::check_solver_convergence(hypreSolver, maxiter, accTol);

        B[xDir] -= bAn[xDir];
        B[yDir] -= bAn[yDir];
        B[zDir] -= bAn[zDir];

        return (Utils::gempic_norm(B[xDir].m_data, infra, 1) +
                Utils::gempic_norm(B[yDir].m_data, infra, 1) +
                Utils::gempic_norm(B[zDir].m_data, infra, 1));
    }
};

using MyTypes = ::testing::Types<std::integral_constant<int, 2>>;
TYPED_TEST_SUITE(HypreQuasineutralHybridModelTest, MyTypes);

TYPED_TEST(HypreQuasineutralHybridModelTest, HypreQuasineutralHybridModelConvergence)
{
#if AMREX_SPACEDIM == 2
    int const coarse = 16, fine = 32;
#elif AMREX_SPACEDIM == 3
    int const coarse = 12, fine = 24;
#endif
    amrex::Real errorCoarse, errorFine;
    amrex::Real tol = 0.1;
    amrex::Real rateOfConvergence;
    constexpr int hodgeDegree = TestFixture::s_hodgeDegree;

    errorCoarse = this->hybridmodel_operator_solve(coarse);
    amrex::Print() << "errorCoarse: " << errorCoarse << "\n";

    errorFine = this->hybridmodel_operator_solve(fine);
    amrex::Print() << "errorFine: " << errorFine << "\n";

    rateOfConvergence = std::log2(errorCoarse / errorFine);

    amrex::Print() << "rate_of_convergence_" << hodgeDegree << ':' << rateOfConvergence << "\n";
    EXPECT_NEAR(rateOfConvergence, hodgeDegree, tol);
}
} // namespace
