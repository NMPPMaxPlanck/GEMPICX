#include "GEMPIC_RungeKutta.H"

using namespace Gempic::TimeLoop;
using namespace Gempic::Forms;
using namespace Gempic::ParticleMeshCoupling;

/**
 * @brief Low storage Runge-Kutta method
 * The one stage method is explicit Euler
 * We use 2N storage Runge-Kutta methods (Williamson 1980, Carpenter-Kennedy 1984)
 * have a Butcher tableau which can be replaced by just two vectors of coefficients A and B
 * This allows to store only one array in addition to the solution array
 */
RungeKutta::RungeKutta(ComputationalDomain infra,
                       std::shared_ptr<Gempic::Forms::FDDeRhamComplex> deRham) :
    m_infra(infra), m_deRham(deRham), m_s2D(deRham), m_s2B(deRham)
{
    Io::Parameters params("RungeKutta");
    params.get_or_set("stages", m_stages);
    amrex::Print() << "LSRK method with stages = " << m_stages << std::endl;

    switch (m_stages)
    {
        case 1: // Explicit Euler
            m_coeffsA = {0.0};
            m_coeffsB = {1.0};
            // m_dtSequence = {1.0};
            break;
        case 2: //Heun
            // m_coeffsA = {0.0, -1.0};
            // m_coeffsB = {1.0, 0.5};
            //!!Heun method can not be used for accumulate j
            // m_dt_sequence = {1.0, 0};
            // Ralston
            m_coeffsA = {0.0, -5. / 9.};
            m_coeffsB = {2. / 3., 3. / 4.};
            // m_dtSequence = {2. / 3., 1. / 3.};
            break;
        case 3: // coef from Carpenter - Kennedy 1994 report (Williamson 1980)
                // m_coeffsA = {0.0, 0.755726351946097, 0.386954477304099};
                // m_coeffsB = {0.245170287303492, 0.184896052186740, 0.569933660509768};
            m_coeffsA = {0.0, -5. / 9., -153. / 128};
            m_coeffsB = {1. / 3., 15. / 16., 8. / 15.};
            // m_dtSequence = {1. / 3., 5. / 12., 1. / 4.};
            break;
        case 5: // coef from Carpenter - Kennedy 1994
            m_coeffsA = {0.0, -0.417890474499852, -1.19215169464268, -1.69778469247153,
                         -1.51418344425716};
            m_coeffsB = {0.149659021999229, 0.379210312999627, 0.822955029386982, 0.699450455949122,
                         0.153057247968152};
            // m_dtSequence = {0.149659021999229, 0.220741935364975, 0.251854805770237,
            // 0.336026367540249, 0.041717869325309};
            break;
        default:
            AMREX_ALWAYS_ASSERT("Order of Runge-Kutta method not implemented.");
            break;
    }
}

void RungeKutta::lsrk_stage_ampere (DeRhamField<Grid::dual, Space::face>& D,
                                   DeRhamField<Grid::dual, Space::edge> const& H,
                                   DeRhamField<Grid::dual, Space::face> const& J,
                                   amrex::Real dt,
                                   int stageIndex)
{
    BL_PROFILE("GEMPIC::RungeKutta::lsrk_stage_ampere()");
    amrex::Real const coeffsAStageIndex = m_coeffsA[stageIndex];
    amrex::Real const coeffsBStageIndex = m_coeffsB[stageIndex];
    // Component-0 of Ampere's equation
    for (amrex::MFIter mfi(D.m_data[xDir]); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.validbox();

        amrex::Array4<amrex::Real> const& dxArray = (D.m_data[xDir])[mfi].array();
        amrex::Array4<amrex::Real> const& mS2DxArray = (m_s2D.m_data[xDir])[mfi].array();
        AMREX_D_TERM(
            amrex::Array4<amrex::Real const> const& jxArray = (J.m_data[xDir])[mfi].array();
            , amrex::Array4<amrex::Real const> const& hzArray = (H.m_data[zDir])[mfi].const_array();
            ,
            amrex::Array4<amrex::Real const> const& hyArray = (H.m_data[yDir])[mfi].const_array();)

        ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        mS2DxArray(i, j, k) =
                            coeffsAStageIndex * mS2DxArray(i, j, k) +
                            dt * (GEMPIC_D_ADD(0., hzArray(i, j, k) - hzArray(i, j - 1, k),
                                               -hyArray(i, j, k) + hyArray(i, j, k - 1)) -
                                  jxArray(i, j, k));
                        dxArray(i, j, k) += coeffsBStageIndex * mS2DxArray(i, j, k);
                    });
    }
    D.m_data[xDir].FillBoundary_nowait(m_infra.geometry().periodicity());
    m_s2D.m_data[xDir].FillBoundary_nowait(m_infra.geometry().periodicity());

    // Component-1 of Ampere's equation
    for (amrex::MFIter mfi(D.m_data[yDir]); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.validbox();

        amrex::Array4<amrex::Real> const& dyArray = (D.m_data[yDir])[mfi].array();
        amrex::Array4<amrex::Real> const& mS2DyArray = (m_s2D.m_data[yDir])[mfi].array();
        amrex::Array4<amrex::Real const> const& jyArray = (J.m_data[yDir])[mfi].array();

        amrex::Array4<amrex::Real const> const& hzArray = (H.m_data[zDir])[mfi].const_array();
#if (AMREX_SPACEDIM > 2)
        amrex::Array4<amrex::Real const> const& hxArray = (H.m_data[xDir])[mfi].const_array();
#endif
        ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        mS2DyArray(i, j, k) =
                            coeffsAStageIndex * mS2DyArray(i, j, k) +
                            dt * (GEMPIC_D_ADD(-hzArray(i, j, k) + hzArray(i - 1, j, k), 0.,
                                               hxArray(i, j, k) - hxArray(i, j, k - 1)) -
                                  jyArray(i, j, k));
                        dyArray(i, j, k) += coeffsBStageIndex * mS2DyArray(i, j, k);
                    });
    }
    D.m_data[yDir].FillBoundary_nowait(m_infra.geometry().periodicity());
    m_s2D.m_data[yDir].FillBoundary_nowait(m_infra.geometry().periodicity());

    // Component-2 of Ampere's equation
    for (amrex::MFIter mfi(D.m_data[zDir]); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.validbox();

        amrex::Array4<amrex::Real> const& dzArray = (D.m_data[zDir])[mfi].array();
        amrex::Array4<amrex::Real> const& mS2DzArray = (m_s2D.m_data[zDir])[mfi].array();
        amrex::Array4<amrex::Real const> const& jzArray = (J.m_data[zDir])[mfi].array();
#if (AMREX_SPACEDIM > 1)
        amrex::Array4<amrex::Real const> const& hxArray = (H.m_data[xDir])[mfi].const_array();
#endif
        amrex::Array4<amrex::Real const> const& hyArray = (H.m_data[yDir])[mfi].const_array();
        ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        mS2DzArray(i, j, k) =
                            coeffsAStageIndex * mS2DzArray(i, j, k) +
                            dt * (GEMPIC_D_ADD(hyArray(i, j, k) - hyArray(i - 1, j, k),
                                               -hxArray(i, j, k) + hxArray(i, j - 1, k), 0.) -
                                  jzArray(i, j, k));
                        dzArray(i, j, k) += coeffsBStageIndex * mS2DzArray(i, j, k);
                    });
    }
    D.m_data[zDir].FillBoundary_nowait(m_infra.geometry().periodicity());
    m_s2D.m_data[zDir].FillBoundary_nowait(m_infra.geometry().periodicity());

    // Wait for completed communication of guard data
    D.m_data[xDir].FillBoundary_finish();
    D.m_data[yDir].FillBoundary_finish();
    D.m_data[zDir].FillBoundary_finish();
    m_s2D.m_data[xDir].FillBoundary_finish();
    m_s2D.m_data[yDir].FillBoundary_finish();
    m_s2D.m_data[zDir].FillBoundary_finish();
}

void RungeKutta::lsrk_stage_faraday (DeRhamField<Grid::primal, Space::face>& B,
                                    DeRhamField<Grid::primal, Space::edge> const& E,
                                    amrex::Real dt,
                                    int stageIndex)
{
    BL_PROFILE("GEMPIC::RungeKutta::lsrk_stage_faraday()");
    amrex::Real const coeffsAStageIndex = m_coeffsA[stageIndex];
    amrex::Real const coeffsBStageIndex = m_coeffsB[stageIndex];
    // Component-0 of Faraday's equation
    for (amrex::MFIter mfi(B.m_data[xDir]); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.validbox();

        amrex::Array4<amrex::Real> const& s2BxArray = (m_s2B.m_data[xDir])[mfi].array();
        AMREX_D_TERM(
            amrex::Array4<amrex::Real> const& bxArray = (B.m_data[xDir])[mfi].array();
            , amrex::Array4<amrex::Real const> const& ezArray = (E.m_data[zDir])[mfi].const_array();
            ,
            amrex::Array4<amrex::Real const> const& eyArray = (E.m_data[yDir])[mfi].const_array();)
        ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        s2BxArray(i, j, k) =
                            coeffsAStageIndex * s2BxArray(i, j, k) -
                            dt * (GEMPIC_D_ADD(0., ezArray(i, j + 1, k) - ezArray(i, j, k),
                                               -eyArray(i, j, k + 1) + eyArray(i, j, k)));
                        bxArray(i, j, k) += coeffsBStageIndex * s2BxArray(i, j, k);
                    });
    }
    B.m_data[xDir].FillBoundary_nowait(m_infra.geometry().periodicity());
    m_s2B.m_data[xDir].FillBoundary_nowait(m_infra.geometry().periodicity());

    // Component-1 of Faraday's equation
    for (amrex::MFIter mfi(B.m_data[yDir]); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.validbox();

        amrex::Array4<amrex::Real> const& s2ByArray = (m_s2B.m_data[yDir])[mfi].array();
        amrex::Array4<amrex::Real> const& byArray = (B.m_data[yDir])[mfi].array();
        amrex::Array4<amrex::Real const> const& ezArray = (E.m_data[zDir])[mfi].const_array();
#if (AMREX_SPACEDIM > 2)
        amrex::Array4<amrex::Real const> const& exArray = (E.m_data[xDir])[mfi].const_array();
#endif
        ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        s2ByArray(i, j, k) =
                            coeffsAStageIndex * s2ByArray(i, j, k) -
                            dt * (GEMPIC_D_ADD(-ezArray(i + 1, j, k) + ezArray(i, j, k), 0.,
                                               exArray(i, j, k + 1) - exArray(i, j, k)));
                        byArray(i, j, k) += coeffsBStageIndex * s2ByArray(i, j, k);
                    });
    }
    B.m_data[yDir].FillBoundary_nowait(m_infra.geometry().periodicity());
    m_s2B.m_data[yDir].FillBoundary_nowait(m_infra.geometry().periodicity());

    // Component-2 of Faraday's equation
    for (amrex::MFIter mfi(B.m_data[zDir]); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.validbox();

        amrex::Array4<amrex::Real> const& s2BzArray = (m_s2B.m_data[zDir])[mfi].array();
        amrex::Array4<amrex::Real> const& bzArray = (B.m_data[zDir])[mfi].array();
#if (AMREX_SPACEDIM > 1)
        amrex::Array4<amrex::Real const> const& exArray = (E.m_data[xDir])[mfi].const_array();
#endif
        amrex::Array4<amrex::Real const> const& eyArray = (E.m_data[yDir])[mfi].const_array();

        ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        s2BzArray(i, j, k) =
                            coeffsAStageIndex * s2BzArray(i, j, k) -
                            dt * (GEMPIC_D_ADD(eyArray(i + 1, j, k) - eyArray(i, j, k),
                                               -exArray(i, j + 1, k) + exArray(i, j, k), 0.));
                        bzArray(i, j, k) += coeffsBStageIndex * s2BzArray(i, j, k);
                    });
    }
    B.m_data[zDir].FillBoundary_nowait(m_infra.geometry().periodicity());
    m_s2B.m_data[zDir].FillBoundary_nowait(m_infra.geometry().periodicity());

    // Wait for completed communication of guard data
    B.m_data[xDir].FillBoundary_finish();
    B.m_data[yDir].FillBoundary_finish();
    B.m_data[zDir].FillBoundary_finish();
    m_s2B.m_data[xDir].FillBoundary_finish();
    m_s2B.m_data[yDir].FillBoundary_finish();
    m_s2B.m_data[zDir].FillBoundary_finish();
}
