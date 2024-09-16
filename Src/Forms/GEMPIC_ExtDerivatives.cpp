#include "GEMPIC_Fields.H"

using namespace Gempic::Forms;

/**
 * @brief Computes the discrete curl \f$\mathbb{C}\f$ on the primal grid
 *
 * Description:
 * Using the geometric degrees of freedom. No approximations involved.
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * @param oneForm : DeRhamField<Grid::primal, Space::edge>, 1-form \f$u^1\f$ holding the edge
 * integrals
 * @param twoForm : DeRhamField<Grid::primal, Space::face>, 2-form \f$\mathbb{C} u^1\f$ holding the
 * resulting face integrals
 *
 * @return void
 */
void DeRhamComplex::curl (const DeRhamField<Grid::primal, Space::edge> &oneForm,
                         DeRhamField<Grid::primal, Space::face> &twoForm)
{
    BL_PROFILE("Gempic::Forms::DeRhamComplex::curl(primal,edge,primal,face)");
    int nComps{oneForm.m_data[xDir].nComp()};
    // Component-0 of curl
    for (amrex::MFIter mfi(twoForm.m_data[xDir]); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        amrex::Array4<amrex::Real> const &twoForm0 = (twoForm.m_data[xDir])[mfi].array();
#if (GEMPIC_SPACEDIM > 2)
        amrex::Array4<amrex::Real const> const &oneForm1 =
            (oneForm.m_data[yDir])[mfi].const_array();
#endif
#if (GEMPIC_SPACEDIM > 1)
        amrex::Array4<amrex::Real const> const &oneForm2 =
            (oneForm.m_data[zDir])[mfi].const_array();
#endif
        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        twoForm0(i, j, k, n) =
                            GEMPIC_D_ADD(0., oneForm2(i, j + 1, k, n) - oneForm2(i, j, k, n),
                                         -oneForm1(i, j, k + 1, n) + oneForm1(i, j, k, n));
                    });
    }
    twoForm.m_data[xDir].AverageSync(m_geom.periodicity());
    twoForm.m_data[xDir].FillBoundary_nowait(m_geom.periodicity());

    // Component-1 of curl
    for (amrex::MFIter mfi(twoForm.m_data[yDir]); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        amrex::Array4<amrex::Real> const &twoForm1 = (twoForm.m_data[yDir])[mfi].array();
        amrex::Array4<amrex::Real const> const &oneForm2 =
            (oneForm.m_data[zDir])[mfi].const_array();
#if (GEMPIC_SPACEDIM > 2)
        amrex::Array4<amrex::Real const> const &oneForm0 =
            (oneForm.m_data[xDir])[mfi].const_array();
#endif
        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        twoForm1(i, j, k, n) =
                            GEMPIC_D_ADD(-oneForm2(i + 1, j, k, n) + oneForm2(i, j, k, n), 0.,
                                         oneForm0(i, j, k + 1, n) - oneForm0(i, j, k, n));
                    });
    }
    twoForm.m_data[yDir].AverageSync(m_geom.periodicity());
    twoForm.m_data[yDir].FillBoundary_nowait(m_geom.periodicity());

    // Component-2 of curl
    for (amrex::MFIter mfi(twoForm.m_data[zDir]); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        amrex::Array4<amrex::Real> const &twoForm2 = (twoForm.m_data[zDir])[mfi].array();
#if (GEMPIC_SPACEDIM > 1)
        amrex::Array4<amrex::Real const> const &oneForm0 =
            (oneForm.m_data[xDir])[mfi].const_array();
#endif
        amrex::Array4<amrex::Real const> const &oneForm1 =
            (oneForm.m_data[yDir])[mfi].const_array();
        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        twoForm2(i, j, k, n) =
                            GEMPIC_D_ADD(oneForm1(i + 1, j, k, n) - oneForm1(i, j, k, n),
                                         -oneForm0(i, j + 1, k, n) + oneForm0(i, j, k, n), 0.);
                        ;
                    });
    }
    twoForm.m_data[zDir].AverageSync(m_geom.periodicity());
    twoForm.m_data[zDir].FillBoundary_nowait(m_geom.periodicity());

    // Wait for completed communication of guard data
    twoForm.m_data[xDir].FillBoundary_finish();
    twoForm.m_data[yDir].FillBoundary_finish();
    twoForm.m_data[zDir].FillBoundary_finish();
}

/**
 * @brief Adds dt times the discrete curl \f$\mathbb{C}\f$ on the primal grid
 *
 * Description:
 * Using the geometric degrees of freedom. No approximations involved.
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * @param oneForm : DeRhamField<Grid::primal, Space::edge>, 1-form \f$u^1\f$ holding the edge
 * integrals
 * @param twoForm : DeRhamField<Grid::primal, Space::face>, 2-form \f$\mathbb{C} u^1\f$ holding the
 * resulting face integrals
 * @param dt : time step by which curl is multiplied. Can be negative for subtraction.
 *
 * @return void
 */
void DeRhamComplex::add_dt_curl (const DeRhamField<Grid::primal, Space::edge> &oneForm,
                                DeRhamField<Grid::primal, Space::face> &twoForm,
                                amrex::Real dt)
{
    BL_PROFILE("Gempic::Forms::DeRhamComplex::add_dt_curl(primal,edge,primal,face)");
    int nComps{oneForm.m_data[xDir].nComp()};
    // Component-0 of curl
    for (amrex::MFIter mfi(twoForm.m_data[xDir]); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        amrex::Array4<amrex::Real> const &twoForm0 = (twoForm.m_data[xDir])[mfi].array();
#if (GEMPIC_SPACEDIM > 2)
        amrex::Array4<amrex::Real const> const &oneForm1 =
            (oneForm.m_data[yDir])[mfi].const_array();
#endif
#if (GEMPIC_SPACEDIM > 1)
        amrex::Array4<amrex::Real const> const &oneForm2 =
            (oneForm.m_data[zDir])[mfi].const_array();
#endif
        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        twoForm0(i, j, k, n) +=
                            dt * GEMPIC_D_ADD(0., oneForm2(i, j + 1, k, n) - oneForm2(i, j, k, n),
                                              -oneForm1(i, j, k + 1, n) + oneForm1(i, j, k, n));
                    });
    }
    twoForm.m_data[xDir].AverageSync(m_geom.periodicity());
    twoForm.m_data[xDir].FillBoundary_nowait(m_geom.periodicity());

    // Component-1 of curl
    for (amrex::MFIter mfi(twoForm.m_data[yDir]); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        amrex::Array4<amrex::Real> const &twoForm1 = (twoForm.m_data[yDir])[mfi].array();
        amrex::Array4<amrex::Real const> const &oneForm2 =
            (oneForm.m_data[zDir])[mfi].const_array();
#if (GEMPIC_SPACEDIM > 2)
        amrex::Array4<amrex::Real const> const &oneForm0 =
            (oneForm.m_data[xDir])[mfi].const_array();
#endif
        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        twoForm1(i, j, k, n) +=
                            dt * GEMPIC_D_ADD(-oneForm2(i + 1, j, k, n) + oneForm2(i, j, k, n), 0.,
                                              oneForm0(i, j, k + 1, n) - oneForm0(i, j, k, n));
                    });
    }
    twoForm.m_data[yDir].AverageSync(m_geom.periodicity());
    twoForm.m_data[yDir].FillBoundary_nowait(m_geom.periodicity());

    // Component-2 of curl
    for (amrex::MFIter mfi(twoForm.m_data[zDir]); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        amrex::Array4<amrex::Real> const &twoForm2 = (twoForm.m_data[zDir])[mfi].array();
#if (GEMPIC_SPACEDIM > 1)
        amrex::Array4<amrex::Real const> const &oneForm0 =
            (oneForm.m_data[xDir])[mfi].const_array();
#endif
        amrex::Array4<amrex::Real const> const &oneForm1 =
            (oneForm.m_data[yDir])[mfi].const_array();
        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        twoForm2(i, j, k, n) +=
                            dt * GEMPIC_D_ADD(oneForm1(i + 1, j, k, n) - oneForm1(i, j, k, n),
                                              -oneForm0(i, j + 1, k, n) + oneForm0(i, j, k, n), 0.);
                        ;
                    });
    }
    twoForm.m_data[zDir].AverageSync(m_geom.periodicity());
    twoForm.m_data[zDir].FillBoundary_nowait(m_geom.periodicity());

    // Wait for completed communication of guard data
    twoForm.m_data[xDir].FillBoundary_finish();
    twoForm.m_data[yDir].FillBoundary_finish();
    twoForm.m_data[zDir].FillBoundary_finish();
}

/**
 * @brief Computes the discrete curl \f$\mathbb{C}\f$ on the dual grid
 *
 * Description:
 * Using the geometric degrees of freedom. No approximations involved.
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * @param oneForm : DeRhamField<Grid::dual, Space::edge>, 1-form \f$\tilde{u}^1\f$ holding the edge
 * integrals
 * @param twoForm : DeRhamField<Grid::dual, Space::face>, 2-form \f$\tilde{C} \tilde{u}^1\f$ holding
 * the resulting face integrals
 *
 * @return void
 */
void DeRhamComplex::curl (const DeRhamField<Grid::dual, Space::edge> &oneForm,
                         DeRhamField<Grid::dual, Space::face> &twoForm)
{
    BL_PROFILE("Gempic::Forms::DeRhamComplex::curl(dual,edge,dual,face)");
    int nComps{oneForm.m_data[xDir].nComp()};
    // Component-0 of curl
    for (amrex::MFIter mfi(twoForm.m_data[xDir]); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        amrex::Array4<amrex::Real> const &twoForm0 = (twoForm.m_data[xDir])[mfi].array();
#if (GEMPIC_SPACEDIM > 2)
        amrex::Array4<amrex::Real const> const &oneForm1 =
            (oneForm.m_data[yDir])[mfi].const_array();
#endif
#if (GEMPIC_SPACEDIM > 1)
        amrex::Array4<amrex::Real const> const &oneForm2 =
            (oneForm.m_data[zDir])[mfi].const_array();
#endif
        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        twoForm0(i, j, k, n) =
                            GEMPIC_D_ADD(0., oneForm2(i, j, k, n) - oneForm2(i, j - 1, k, n),
                                         -oneForm1(i, j, k, n) + oneForm1(i, j, k - 1, n));
                    });
    }

    twoForm.m_data[xDir].AverageSync(m_geom.periodicity());
    twoForm.m_data[xDir].FillBoundary_nowait(m_geom.periodicity());

    // Component-1 of curl
    for (amrex::MFIter mfi(twoForm.m_data[yDir]); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        amrex::Array4<amrex::Real> const &twoForm1 = (twoForm.m_data[yDir])[mfi].array();
        amrex::Array4<amrex::Real const> const &oneForm2 =
            (oneForm.m_data[zDir])[mfi].const_array();
#if (GEMPIC_SPACEDIM > 2)
        amrex::Array4<amrex::Real const> const &oneForm0 =
            (oneForm.m_data[xDir])[mfi].const_array();
#endif
        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        twoForm1(i, j, k, n) =
                            GEMPIC_D_ADD(-oneForm2(i, j, k, n) + oneForm2(i - 1, j, k, n), 0.,
                                         oneForm0(i, j, k, n) - oneForm0(i, j, k - 1, n));
                    });
    }

    twoForm.m_data[yDir].AverageSync(m_geom.periodicity());
    twoForm.m_data[yDir].FillBoundary_nowait(m_geom.periodicity());

    // Component-2 of curl
    for (amrex::MFIter mfi(twoForm.m_data[zDir]); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        amrex::Array4<amrex::Real> const &twoForm2 = (twoForm.m_data[zDir])[mfi].array();
#if (GEMPIC_SPACEDIM > 1)
        amrex::Array4<amrex::Real const> const &oneForm0 =
            (oneForm.m_data[xDir])[mfi].const_array();
#endif
        amrex::Array4<amrex::Real const> const &oneForm1 =
            (oneForm.m_data[yDir])[mfi].const_array();

        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        twoForm2(i, j, k, n) =
                            GEMPIC_D_ADD(oneForm1(i, j, k, n) - oneForm1(i - 1, j, k, n),
                                         -oneForm0(i, j, k, n) + oneForm0(i, j - 1, k, n), 0.);
                    });
    }

    twoForm.m_data[zDir].AverageSync(m_geom.periodicity());
    twoForm.m_data[zDir].FillBoundary_nowait(m_geom.periodicity());

    // Wait for completed communication of guard data
    twoForm.m_data[xDir].FillBoundary_finish();
    twoForm.m_data[yDir].FillBoundary_finish();
    twoForm.m_data[zDir].FillBoundary_finish();
}

/**
 * @brief Adds dt times the discrete curl \f$\mathbb{C}\f$ on the dual grid
 *
 * Description:
 * Using the geometric degrees of freedom. No approximations involved.
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * @param oneForm : DeRhamField<Grid::dual, Space::edge>, 1-form \f$\tilde{u}^1\f$ holding the edge
 * integrals
 * @param twoForm : DeRhamField<Grid::dual, Space::face>, 2-form \f$\tilde{C} \tilde{u}^1\f$ holding
 * the resulting face integrals
 * @param dt : time step by which curl is multiplied. Can be negative for subtraction.
 *
 * @return void
 */
void DeRhamComplex::add_dt_curl (const DeRhamField<Grid::dual, Space::edge> &oneForm,
                                DeRhamField<Grid::dual, Space::face> &twoForm,
                                amrex::Real dt)
{
    BL_PROFILE("Gempic::Forms::DeRhamComplex::add_dt_curl(dual,edge,dual,face)");
    int nComps{oneForm.m_data[xDir].nComp()};
    // Component-0 of curl
    for (amrex::MFIter mfi(twoForm.m_data[xDir]); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        amrex::Array4<amrex::Real> const &twoForm0 = (twoForm.m_data[xDir])[mfi].array();
#if (GEMPIC_SPACEDIM > 2)
        amrex::Array4<amrex::Real const> const &oneForm1 =
            (oneForm.m_data[yDir])[mfi].const_array();
#endif
#if (GEMPIC_SPACEDIM > 1)
        amrex::Array4<amrex::Real const> const &oneForm2 =
            (oneForm.m_data[zDir])[mfi].const_array();
#endif
        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        twoForm0(i, j, k, n) +=
                            dt * GEMPIC_D_ADD(0., oneForm2(i, j, k, n) - oneForm2(i, j - 1, k, n),
                                              -oneForm1(i, j, k, n) + oneForm1(i, j, k - 1, n));
                    });
    }

    twoForm.m_data[xDir].AverageSync(m_geom.periodicity());
    twoForm.m_data[xDir].FillBoundary_nowait(m_geom.periodicity());

    // Component-1 of curl
    for (amrex::MFIter mfi(twoForm.m_data[yDir]); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        amrex::Array4<amrex::Real> const &twoForm1 = (twoForm.m_data[yDir])[mfi].array();
        amrex::Array4<amrex::Real const> const &oneForm2 =
            (oneForm.m_data[zDir])[mfi].const_array();
#if (GEMPIC_SPACEDIM > 2)
        amrex::Array4<amrex::Real const> const &oneForm0 =
            (oneForm.m_data[xDir])[mfi].const_array();
#endif
        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        twoForm1(i, j, k, n) +=
                            dt * GEMPIC_D_ADD(-oneForm2(i, j, k, n) + oneForm2(i - 1, j, k, n), 0.,
                                              oneForm0(i, j, k, n) - oneForm0(i, j, k - 1, n));
                    });
    }

    twoForm.m_data[yDir].AverageSync(m_geom.periodicity());
    twoForm.m_data[yDir].FillBoundary_nowait(m_geom.periodicity());

    // Component-2 of curl
    for (amrex::MFIter mfi(twoForm.m_data[zDir]); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        amrex::Array4<amrex::Real> const &twoForm2 = (twoForm.m_data[zDir])[mfi].array();
#if (GEMPIC_SPACEDIM > 1)
        amrex::Array4<amrex::Real const> const &oneForm0 =
            (oneForm.m_data[xDir])[mfi].const_array();
#endif
        amrex::Array4<amrex::Real const> const &oneForm1 =
            (oneForm.m_data[yDir])[mfi].const_array();

        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        twoForm2(i, j, k, n) +=
                            dt * GEMPIC_D_ADD(oneForm1(i, j, k, n) - oneForm1(i - 1, j, k, n),
                                              -oneForm0(i, j, k, n) + oneForm0(i, j - 1, k, n), 0.);
                    });
    }

    twoForm.m_data[zDir].AverageSync(m_geom.periodicity());
    twoForm.m_data[zDir].FillBoundary_nowait(m_geom.periodicity());

    // Wait for completed communication of guard data
    twoForm.m_data[xDir].FillBoundary_finish();
    twoForm.m_data[yDir].FillBoundary_finish();
    twoForm.m_data[zDir].FillBoundary_finish();
}

/**
 * @brief Computes the discrete gradient \f$\mathbb{G}\f$ on the primal grid
 *
 * Description:
 * Using the geometric degrees of freedom. No approximations involved.
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * @param zeroForm : DeRhamField<Grid::primal, Space::node>, 0-form \f$u^0\f$ holding the node
 * values
 * @param oneForm : DeRhamField<Grid::primal, Space::edge>, 1-form \f$\mathbb{G} u^0\f$ holding the
 * resulting edge integrals
 *
 * @return void
 */
void DeRhamComplex::grad (const DeRhamField<Grid::primal, Space::node> &zeroForm,
                         DeRhamField<Grid::primal, Space::edge> &oneForm)
{
    BL_PROFILE("Gempic::Forms::DeRhamComplex::grad(primal,node,primal,edge)");
    int nComps{zeroForm.m_data.nComp()};
    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(oneForm.m_data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();

            amrex::Array4<amrex::Real const> const &zeroFormMF =
                (zeroForm.m_data)[mfi].const_array();
            amrex::Array4<amrex::Real> const &oneFormMF = (oneForm.m_data[comp])[mfi].array();

            // Calculate gradient component by component
            if (comp == xDir)
            {
                ParallelFor(bx, nComps,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                            {
                                oneFormMF(i, j, k, n) =
                                    zeroFormMF(i + 1, j, k, n) - zeroFormMF(i, j, k, n);
                            });
            }

            if (comp == yDir)
            {
                ParallelFor(bx, nComps,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                            {
                                oneFormMF(i, j, k, n) = GEMPIC_D_ADD(
                                    0., zeroFormMF(i, j + 1, k, n) - zeroFormMF(i, j, k, n), 0.);
                            });
            }

            if (comp == zDir)
            {
                ParallelFor(bx, nComps,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                            {
                                oneFormMF(i, j, k, n) = GEMPIC_D_ADD(
                                    0., 0., zeroFormMF(i, j, k + 1, n) - zeroFormMF(i, j, k, n));
                            });
            }
        }
        oneForm.m_data[comp].AverageSync(m_geom.periodicity());
        oneForm.m_data[comp].FillBoundary_nowait(m_geom.periodicity());
    }
    oneForm.m_data[xDir].FillBoundary_finish();
    oneForm.m_data[yDir].FillBoundary_finish();
    oneForm.m_data[zDir].FillBoundary_finish();
}

/**
 * @brief Computes a * grad phi on the primal grid for some given constant a
 *
 * Description:
 * Using the geometric degrees of freedom. No approximations involved.
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * @param zeroForm : DeRhamField<Grid::primal, Space::node>, 0-form \f$u^0\f$ holding the node
 * values
 * @param oneForm : DeRhamField<Grid::primal, Space::edge>, 1-form \f$\mathbb{G} u^0\f$ holding the
 * resulting edge integrals
 * @param a : amrex::Real, constant to be multiplied with the gradient
 *
 * @return void
 */
void DeRhamComplex::a_times_grad (const DeRhamField<Grid::primal, Space::node> &zeroForm,
                                 DeRhamField<Grid::primal, Space::edge> &oneForm,
                                 amrex::Real a)
{
    BL_PROFILE("Gempic::Forms::DeRhamComplex::a_times_grad(primal,node,primal,edge)");
    int nComps{zeroForm.m_data.nComp()};
    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(oneForm.m_data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();

            amrex::Array4<amrex::Real const> const &zeroFormMF =
                (zeroForm.m_data)[mfi].const_array();
            amrex::Array4<amrex::Real> const &oneFormMF = (oneForm.m_data[comp])[mfi].array();

            // Calculate gradient component by component
            if (comp == xDir)
            {
                ParallelFor(bx, nComps,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                            {
                                oneFormMF(i, j, k, n) =
                                    a * zeroFormMF(i + 1, j, k, n) + zeroFormMF(i, j, k, n);
                            });
            }

            if (comp == yDir)
            {
                ParallelFor(bx, nComps,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                            {
                                oneFormMF(i, j, k, n) = GEMPIC_D_ADD(
                                    0., a * zeroFormMF(i, j + 1, k, n) + zeroFormMF(i, j, k, n),
                                    0.);
                            });
            }

            if (comp == zDir)
            {
                ParallelFor(bx, nComps,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                            {
                                oneFormMF(i, j, k, n) = GEMPIC_D_ADD(
                                    0., 0.,
                                    a * zeroFormMF(i, j, k + 1, n) + zeroFormMF(i, j, k, n));
                            });
            }
        }
        oneForm.m_data[comp].AverageSync(m_geom.periodicity());
        oneForm.m_data[comp].FillBoundary_nowait(m_geom.periodicity());
    }
    oneForm.m_data[xDir].FillBoundary_finish();
    oneForm.m_data[yDir].FillBoundary_finish();
    oneForm.m_data[zDir].FillBoundary_finish();
}

/**
 * @brief Computes the discrete gradient \f$\mathbb{G}\f$ on the dual grid
 *
 * Description:
 * Using the geometric degrees of freedom. No approximations involved.
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * @param zeroForm : DeRhamField<Grid::dual, Space::node>, 0-form \f$\tilde{u}^0\f$ holding the node
 * values
 * @param oneForm : DeRhamField<Grid::dual, Space::edge>, 1-form \f$\tilde{G} \tilde{u}^0\f$ holding
 * the resulting edge integrals
 *
 * @return void
 */
void DeRhamComplex::grad (const DeRhamField<Grid::dual, Space::node> &zeroForm,
                         DeRhamField<Grid::dual, Space::edge> &oneForm)
{
    BL_PROFILE("Gempic::Forms::DeRhamComplex::grad(dual,node,dual,edge)");
    int nComps{zeroForm.m_data.nComp()};
    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(oneForm.m_data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();

            amrex::Array4<amrex::Real const> const &zeroFormMF =
                (zeroForm.m_data)[mfi].const_array();
            amrex::Array4<amrex::Real> const &oneFormMF = (oneForm.m_data[comp])[mfi].array();

            // Calculate gradient component by component
            if (comp == xDir)
            {
                ParallelFor(bx, nComps,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                            {
                                oneFormMF(i, j, k, n) =
                                    zeroFormMF(i, j, k, n) - zeroFormMF(i - 1, j, k, n);
                            });
            }

            if (comp == yDir)
            {
                ParallelFor(bx, nComps,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                            {
                                oneFormMF(i, j, k, n) = GEMPIC_D_ADD(
                                    0., zeroFormMF(i, j, k, n) - zeroFormMF(i, j - 1, k, n), 0.);
                            });
            }

            if (comp == zDir)
            {
                ParallelFor(bx, nComps,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                            {
                                oneFormMF(i, j, k, n) = GEMPIC_D_ADD(
                                    0., 0., zeroFormMF(i, j, k, n) - zeroFormMF(i, j, k - 1, n));
                            });
            }
        }
        oneForm.m_data[comp].AverageSync(m_geom.periodicity());
        oneForm.m_data[comp].FillBoundary_nowait(m_geom.periodicity());
    }

    oneForm.m_data[xDir].FillBoundary_finish();
    oneForm.m_data[yDir].FillBoundary_finish();
    oneForm.m_data[zDir].FillBoundary_finish();
}

/**
 * @brief Computes a times the discrete gradient \f$a \mathbb{G}\f$ on the dual grid
 *
 * Description:
 * Using the geometric degrees of freedom. No approximations involved.
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * @param zeroForm : DeRhamField<Grid::dual, Space::node>, 0-form \f$\tilde{u}^0\f$ holding the node
 * values
 * @param oneForm : DeRhamField<Grid::dual, Space::edge>, 1-form \f$\tilde{G} \tilde{u}^0\f$ holding
 * the resulting edge integrals
 * @param a : multiplication factor
 *
 * @return void
 */
void DeRhamComplex::a_times_grad (const DeRhamField<Grid::dual, Space::node> &zeroForm,
                                 DeRhamField<Grid::dual, Space::edge> &oneForm,
                                 amrex::Real a)
{
    BL_PROFILE("Gempic::Forms::DeRhamComplex::a_times_grad(dual,node,dual,edge)");
    int nComps{zeroForm.m_data.nComp()};
    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(oneForm.m_data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();

            amrex::Array4<amrex::Real const> const &zeroFormMF =
                (zeroForm.m_data)[mfi].const_array();
            amrex::Array4<amrex::Real> const &oneFormMF = (oneForm.m_data[comp])[mfi].array();

            // Calculate gradient component by component
            if (comp == xDir)
            {
                ParallelFor(bx, nComps,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                            {
                                oneFormMF(i, j, k, n) =
                                    a * zeroFormMF(i, j, k, n) + zeroFormMF(i - 1, j, k, n);
                            });
            }

            if (comp == yDir)
            {
                ParallelFor(bx, nComps,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                            {
                                oneFormMF(i, j, k, n) = GEMPIC_D_ADD(
                                    0., a * zeroFormMF(i, j, k, n) + zeroFormMF(i, j - 1, k, n),
                                    0.);
                            });
            }

            if (comp == zDir)
            {
                ParallelFor(bx, nComps,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                            {
                                oneFormMF(i, j, k, n) = GEMPIC_D_ADD(
                                    0., 0.,
                                    a * zeroFormMF(i, j, k, n) + zeroFormMF(i, j, k - 1, n));
                            });
            }
        }
        oneForm.m_data[comp].AverageSync(m_geom.periodicity());
        oneForm.m_data[comp].FillBoundary_nowait(m_geom.periodicity());
    }

    oneForm.m_data[xDir].FillBoundary_finish();
    oneForm.m_data[yDir].FillBoundary_finish();
    oneForm.m_data[zDir].FillBoundary_finish();
}

/**
 * @brief Computes the discrete divergence \f$\mathbb{D}\f$ on the primal grid
 *
 * Description:
 * Using the geometric degrees of freedom. No approximations involved.
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * @param twoForm : DeRhamField<Grid::primal, Space::face>, 2-form \f$u^2\f$ holding the face
 * integrals
 * @param threeForm : DeRhamField<Grid::primal, Space::cell>, 3-form \f$\mathbb{D} u^2\f$ holding
 * the resulting cell integrals
 *
 * @return void
 */
void DeRhamComplex::div (const DeRhamField<Grid::primal, Space::face> &twoForm,
                        DeRhamField<Grid::primal, Space::cell> &threeForm)
{
    BL_PROFILE("Gempic::Forms::DeRhamComplex::div(primal,face,primal,cell)");
    int nComps{twoForm.m_data[xDir].nComp()};
    for (amrex::MFIter mfi(threeForm.m_data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        amrex::Array4<amrex::Real const> const &twoFormMF0 =
            (twoForm.m_data[xDir])[mfi].const_array();
#if (GEMPIC_SPACEDIM > 1)
        amrex::Array4<amrex::Real const> const &twoFormMF1 =
            (twoForm.m_data[yDir])[mfi].const_array();
#endif
#if (GEMPIC_SPACEDIM > 2)
        amrex::Array4<amrex::Real const> const &twoFormMF2 =
            (twoForm.m_data[zDir])[mfi].const_array();
#endif
        amrex::Array4<amrex::Real> const &threeFormMF = (threeForm.m_data)[mfi].array();

        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        threeFormMF(i, j, k, n) =
                            GEMPIC_D_ADD(twoFormMF0(i + 1, j, k, n) - twoFormMF0(i, j, k, n),
                                         twoFormMF1(i, j + 1, k, n) - twoFormMF1(i, j, k, n),
                                         twoFormMF2(i, j, k + 1, n) - twoFormMF2(i, j, k, n));
                    });
    }
    threeForm.average_sync();
    threeForm.fill_boundary();
}

/**
 * @brief Computes the discrete divergence \f$\mathbb{D}\f$ on the dual grid
 *
 * Description:
 * Using the geometric degrees of freedom. No approximations involved.
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * @param twoForm : DeRhamField<Grid::primal, Space::dual>, 2-form \f$\tilde{u}^2\f$ holding the
 * face integrals
 * @param threeForm : DeRhamField<Grid::primal, Space::dual>, 3-form \f$\tilde{D} \tilde{u}^2\f$
 * holding the resulting cell integrals
 *
 * @return void
 */
void DeRhamComplex::div (const DeRhamField<Grid::dual, Space::face> &twoForm,
                        DeRhamField<Grid::dual, Space::cell> &threeForm)
{
    BL_PROFILE("Gempic::Forms::DeRhamComplex::div(dual,face,dual,cell)");
    int nComps{twoForm.m_data[xDir].nComp()};
    for (amrex::MFIter mfi(threeForm.m_data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        amrex::Array4<amrex::Real const> const &twoFormMF0 =
            (twoForm.m_data[xDir])[mfi].const_array();
#if (GEMPIC_SPACEDIM > 1)
        amrex::Array4<amrex::Real const> const &twoFormMF1 =
            (twoForm.m_data[yDir])[mfi].const_array();
#endif
#if (GEMPIC_SPACEDIM > 2)
        amrex::Array4<amrex::Real const> const &twoFormMF2 =
            (twoForm.m_data[zDir])[mfi].const_array();
#endif
        amrex::Array4<amrex::Real> const &threeFormMF = (threeForm.m_data)[mfi].array();

        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        threeFormMF(i, j, k, n) =
                            GEMPIC_D_ADD(twoFormMF0(i, j, k, n) - twoFormMF0(i - 1, j, k, n),
                                         twoFormMF1(i, j, k, n) - twoFormMF1(i, j - 1, k, n),
                                         twoFormMF2(i, j, k, n) - twoFormMF2(i, j, k - 1, n));
                    });
    }
    threeForm.average_sync();
    threeForm.fill_boundary();
}
