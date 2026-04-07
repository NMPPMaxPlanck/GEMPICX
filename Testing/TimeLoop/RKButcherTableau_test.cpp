/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_NumTools.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_RungeKutta.H"
#include "GEMPIC_Solvers.H"
#include "TestUtils/GEMPIC_TestUtils.H"

namespace
{
using namespace Gempic;
using namespace Forms;
using namespace TimeLoop;

/**
 * Handler structure for the Maxwell physical state
 *
 * @param Derived
 */
template <int nVar>
struct TestFieldsHandlerStruct : BaseFieldsHandlerStruct<TestFieldsHandlerStruct<nVar>>
{
    // and the variables
    amrex::GpuArray<amrex::Real, nVar> m_q;

    /**
     * the assign method
     *
     * @param other : the source of the assign
     */
    void assign (TestFieldsHandlerStruct const& other) // now we can use the = operator of the
                                                       // base class to call the assign...
    {
        for (int ivar = 0; ivar < nVar; ++ivar)
        {
            m_q[ivar] = other.m_q[ivar];
        }
    }

    /**
     * the mult_and_add method
     *
     * @param alpha : real scaling parameter
     * @param other : the source of the assign
     */
    void mult_and_add (amrex::Real alpha, std::shared_ptr<TestFieldsHandlerStruct> const& other)
    {
        for (int ivar = 0; ivar < nVar; ++ivar)
        {
            m_q[ivar] += alpha * other->m_q[ivar];
        }
    } //

    /**
     * the fill_boundary method
     */
    void fill_boundary () { return; }

    /**
     * Explicitly define copy assignment operator to avoid default deletion (from DeRhamFields)
     *
     * @param other : source
     */
    TestFieldsHandlerStruct& operator=(TestFieldsHandlerStruct const& other)
    {
        if (this != &other)
        {                  // Prevent self-assignment
            assign(other); // Use assign method to copy assignable members
        }
        return *this;
    }

    /**
     * Constructor
     */
    TestFieldsHandlerStruct(std::shared_ptr<FDDeRhamComplex>& /*drc*/) {}

    // this is necessary to define array of this classes: first define the array, then construct
    // component by component.
    TestFieldsHandlerStruct() = default; // Default constructor

    ~TestFieldsHandlerStruct() override
    { // Destructor called when deleted via base class pointer
      // Cleanup code for TestFieldsHandlerStruct
    }

private:
    static constexpr int s_nFields{nVar};
};

ComputationalDomain get_compdom ()
{
    // Cells of size 1x1x1
    std::array<amrex::Real, AMREX_SPACEDIM> const domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
    std::array<amrex::Real, AMREX_SPACEDIM> const domainHi{AMREX_D_DECL(3.0, 3.0, 3.0)};
    amrex::IntVect const nCell{AMREX_D_DECL(3, 3, 3)};
    amrex::IntVect const maxGridSize{AMREX_D_DECL(10, 10, 10)};
    std::array<int, AMREX_SPACEDIM> const isPeriodic{AMREX_D_DECL(0, 0, 0)};

    return ComputationalDomain(domainLo, domainHi, nCell, maxGridSize, isPeriodic);
}
/**
 * Numerical Scheme Class.
 *  - fields: in and out state.
 *  - fields_dt_ex: unknown time-derivative (to be computed).
 *  - fields_tmp_ex: auxiliary temporary state.
 */
template <int nVar>
class TestNumericalScheme : public NumericalScheme<TestFieldsHandlerStruct<nVar>>
{
public:
    int m_nmaxSteps{1000};
    static constexpr amrex::Real s_y0{3.0};
    static constexpr int s_hodgeDegree{2};
    static constexpr int s_maxSplineDegree{0};
    Gempic::Io::Parameters m_parameters{};
    ComputationalDomain m_compDom;
    // temporary deRham complex only because needed for the RK constructors
    std::shared_ptr<FDDeRhamComplex> m_drc;
    std::shared_ptr<TestFieldsHandlerStruct<nVar>> m_fields;
    std::shared_ptr<TestFieldsHandlerStruct<nVar>> m_fieldsQExplicit;
    std::shared_ptr<TestFieldsHandlerStruct<nVar>> m_fieldsDtEx;
    std::shared_ptr<TestFieldsHandlerStruct<nVar>> m_fieldsQStar;
    std::shared_ptr<TestFieldsHandlerStruct<nVar>> m_fieldsDtIm;
    std::shared_ptr<TestFieldsHandlerStruct<nVar>> m_fieldsNew;

    int m_timeStep;
    amrex::Real m_time;
    amrex::Real m_tend;
    amrex::Real m_tini;
    amrex::Real m_dt;

    void reset_time (amrex::Real dtIn)
    {
        m_time = 0.0;
        m_timeStep = 0;
        m_dt = dtIn;
    }

    amrex::Real reference_solution (amrex::Real time, int ivar)
    {
        // y(t) = y0 + t^(ivar) => dydt = ivar*t^(ivar-1)
        return s_y0 + std::pow(time, ivar);
    }

    void set_reference_data (TestFieldsHandlerStruct<nVar>& inputField, amrex::Real time)
    {
        for (int ivar = 0; ivar < nVar; ++ivar)
        {
            inputField.m_q[ivar] = reference_solution(time, ivar);
        }
    }

    /**
     * @brief get the pointer to the field used as initial condition for the explicit time-step, by
     * reference
     * @return shared pointer of FieldsHandlerStruct
     */
    std::shared_ptr<TestFieldsHandlerStruct<nVar>> const& get_pointer_initial_field_explicit ()
        override
    {
        return m_fieldsQExplicit;
    }

    std::shared_ptr<TestFieldsHandlerStruct<nVar>> const& get_pointer_initial_field_implicit ()
        override
    {
        return m_fieldsQStar;
    }

    std::shared_ptr<TestFieldsHandlerStruct<nVar>> const& get_pointer_time_derivative_explicit ()
        override
    {
        return m_fieldsDtEx;
    }

    std::shared_ptr<TestFieldsHandlerStruct<nVar>> const& get_pointer_time_derivative_implicit ()
        override
    {
        return m_fieldsDtIm;
    }

    /**
     * set the initial condition.
     */
    void set_initial_condition () override
    {
        // set initial condition
        m_time = 0.0;
        m_timeStep = 0;
        set_reference_data(*m_fields, 0.0);
        *m_fieldsQExplicit = *m_fields;
        *m_fieldsQStar = *m_fields;
        *m_fieldsNew = *m_fields;
        m_fieldsDtIm->m_q.fill(0.0);
        m_fieldsDtEx->m_q.fill(0.0);
    }

    // here we should add probably an input time, the time of the RK stage! (for this test, at
    // least)
    void one_step_solve_dqdt_explicit (amrex::Real rkTime) override
    {
        // compute explicit time derivative m_fieldsDtEx
        m_fieldsDtEx->m_q.fill(0.0);
        for (int ivar = 1; ivar < nVar; ++ivar)
        {
            m_fieldsDtEx->m_q[ivar] = static_cast<amrex::Real>(ivar) * std::pow(rkTime, ivar - 1);
        }
    }

    /**
     * update old state
     */
    void update_qold_implicit ()
    {
        *m_fieldsQStar = *m_fields;
        return;
    }

    /**
     * solve one implicit (Euler) step
     */
    void one_step_solve_q_implicit (amrex::Real rkTime, amrex::Real dtLoc) override
    {
        // identity
        // dydt = ivar*time^{ivar-1}
        m_fieldsNew->m_q[0] = m_fieldsQStar->m_q[0];
        for (int ivar = 1; ivar < nVar; ++ivar)
        {
            m_fieldsNew->m_q[ivar] = m_fieldsQStar->m_q[ivar];

            m_fieldsNew->m_q[ivar] = m_fieldsQStar->m_q[ivar] + dtLoc *
                                                                    static_cast<amrex::Real>(ivar) *
                                                                    std::pow(rkTime, ivar - 1);
        }
    }

    /**
     * compute time derivative for the semi-implicit solver.
     *
     * @param dtloc : time step
     */
    void one_step_solve_dqdt_semiimplicit (amrex::Real timeI,
                                           amrex::Real dtI,
                                           amrex::Real timeE,
                                           amrex::Real /*dtE*/) override
    {
        one_step_solve_dqdt_explicit(timeE); // this compute dQdt_ex... going in the rhs of the
                                             // implicit
                                             // system.
        one_step_solve_q_implicit(
            timeI, dtI); // using dQdt_ex and q_tmp_im to solve for q_i and evaluate dqdt_implicit
        one_step_eval_dqdt_implicit(
            dtI); // using dQdt_ex and q_tmp_im to solve for q_i and evaluate dqdt_implicit
    }

    /**
     * compute time derivative for the implicit solver.
     *
     * @param dtloc : time step
     */
    void one_step_eval_dqdt_implicit (amrex::Real dtLoc) override
    {
        for (int ivar = 0; ivar < nVar; ++ivar)
        {
            m_fieldsDtIm->m_q[ivar] = (m_fieldsNew->m_q[ivar] - m_fieldsQStar->m_q[ivar]) / dtLoc;
        }
        return;
    }

    /**
     * @brief Check if the final simulation time has been reached.
     *
     * Compares the current simulation time (`m_time`) to the specified end time (`m_tend`).
     *
     * @return `true` if the current time is greater than or equal to the final time, otherwise
     * `false`.
     */
    bool is_finaltime_reached () override { return m_time >= m_tend; }

    /**
     * initialize new time step.
     */
    void init_new_timestep () override
    {
        if (m_time + m_dt > m_tend + 1e-14)
        {
            m_dt = m_tend - m_time + 1e-14;
        }
        m_timeStep++;
    }

    /**
     * finalize current time step.
     */
    void finalize_new_timestep () override { m_time += m_dt; }

    /**
     * get the number of components.
     */
    int get_ncomp () override { return 1; }

    /**
     * get periodicity.
     */
    amrex::Periodicity get_periodicity () override
    {
        amrex::Periodicity tmpPeriodicity;
        return tmpPeriodicity;
    }

    TestNumericalScheme() :
        m_compDom{get_compdom()},
        m_drc{
            std::make_shared<FDDeRhamComplex>(m_compDom,
                                              s_hodgeDegree,
                                              s_maxSplineDegree,
                                              //HodgeScheme::GDECLumpHodge);  //
                                              HodgeScheme::GDECHodge)}, //HodgeScheme::FDHodge)}, //
        m_fields(std::make_shared<TestFieldsHandlerStruct<nVar>>(m_drc)),
        m_fieldsQExplicit(std::make_shared<TestFieldsHandlerStruct<nVar>>(m_drc)),
        m_fieldsDtEx(std::make_shared<TestFieldsHandlerStruct<nVar>>(m_drc)),
        m_fieldsQStar(std::make_shared<TestFieldsHandlerStruct<nVar>>(m_drc)),
        m_fieldsDtIm(std::make_shared<TestFieldsHandlerStruct<nVar>>(m_drc)),
        m_fieldsNew(std::make_shared<TestFieldsHandlerStruct<nVar>>(m_drc))
    {
        m_timeStep = 0;
        m_tend = 10.0;
        m_tini = 0.0;
        m_dt = 0.0;
        m_fields->m_q.fill(0.0);
        m_fieldsQExplicit->m_q.fill(0.0);
        m_fieldsQStar->m_q.fill(0.0);
        m_fieldsDtIm->m_q.fill(0.0);
        m_fieldsDtEx->m_q.fill(0.0);
        m_fieldsNew->m_q.fill(0.0);
    }
};

/**
 * @brief Tests the explicit and the imex RK integrators against scalar ODEs.
 */

template <typename RkTagWrapper> //, typename intRkType>
class RkButcherTableauTest : public testing::Test
{
public:
    static constexpr Gempic::TimeLoop::RungeKuttaTag s_rkTag{RkTagWrapper::value};
    static constexpr int s_rKstages{
        Gempic::TimeLoop::RKTypeSelector<s_rkTag>::s_nStages}; // polynomial degree of the
                                                               // reconstruction.
    static constexpr int s_rKorder{Gempic::TimeLoop::RKTypeSelector<s_rkTag>::s_order};

    static constexpr int s_nVarPde{s_rKstages + 3};

    // the Maxwell numerical scheme class
    TestNumericalScheme<s_nVarPde> m_numericalScheme;

    using RKTBtype = typename Gempic::TimeLoop::RKTypeSelector<s_rkTag>::selected_RK_ButcherTableau;
    // Create an instance of the selected type
    RKTBtype m_rKreference;

    template <typename RkClass>
    void rk (RkClass& rk)
    {
        amrex::Real tol{1e-10};
        amrex::Real tolOrder{1e-1};
        int nMaxSteps{m_numericalScheme.m_nmaxSteps};
        // set initial condition
        int nTimeSteps{10};
        TestFieldsHandlerStruct<s_nVarPde> referenceSol(m_numericalScheme.m_drc);
        m_numericalScheme.set_reference_data(referenceSol, m_numericalScheme.m_tend);
        constexpr int nIter = 4;
        std::array<TestFieldsHandlerStruct<s_nVarPde>, nIter - 1> convergence;
        std::array<TestFieldsHandlerStruct<s_nVarPde>, nIter> error;
        // Initialization of std::array elements
        for (int i = 0; i < nIter - 1; ++i)
        {
            convergence[i] = TestFieldsHandlerStruct<s_nVarPde>(m_numericalScheme.m_drc);
        }
        for (int i = 0; i < nIter; ++i)
        {
            error[i] = TestFieldsHandlerStruct<s_nVarPde>(m_numericalScheme.m_drc);
        }
        amrex::Real refinement{2.0};
        for (int iter = 0; iter < nIter; ++iter)
        {
            nTimeSteps *= refinement;
            amrex::Real dtTmp{(m_numericalScheme.m_tend - m_numericalScheme.m_tini) / nTimeSteps};
            m_numericalScheme.reset_time(dtTmp);
            m_numericalScheme.set_initial_condition();

            for (int tStep = 0; tStep < nMaxSteps; tStep++)
            {
                // eventually recompute dt:
                bool breakLoop = m_numericalScheme.is_finaltime_reached();
                if (breakLoop) break;
                m_numericalScheme.init_new_timestep();
                rk.integrate_step(m_numericalScheme.m_time, m_numericalScheme.m_dt);
                m_numericalScheme.finalize_new_timestep();
            }
            for (int ivar = 0; ivar < s_nVarPde; ++ivar)
            {
                error[iter].m_q[ivar] =
                    std::abs(referenceSol.m_q[ivar] - m_numericalScheme.m_fields->m_q[ivar]);
            }
            if (iter > 0)
            {
                for (int ivar = 0; ivar < s_nVarPde; ++ivar)
                {
                    if (error[iter - 1].m_q[ivar] < 1e-12 || error[iter].m_q[ivar] < 1e-12)
                    {
                        convergence[iter - 1].m_q[ivar] = 0;
                    }
                    else
                    {
                        convergence[iter - 1].m_q[ivar] =
                            -std::log(error[iter].m_q[ivar] / error[iter - 1].m_q[ivar]) /
                            std::log(refinement);
                    }
                    bool nearZero = std::abs(convergence[iter - 1].m_q[ivar]) < tol;
                    bool nearP = convergence[iter - 1].m_q[ivar] >
                                 static_cast<amrex::Real>(s_rKorder) - tolOrder;

                    EXPECT_TRUE(nearZero || nearP)
                        << "Value " << convergence[iter - 1].m_q[ivar] << " not near 0.0 or "
                        << s_rKorder << " within tolerance " << tol;
                }
            }
        }
    }

    RkButcherTableauTest() : m_numericalScheme() {}
};

template <typename RkTagWrapper> //, typename intRkType>
class RkButcherTableauExplicitTest : public RkButcherTableauTest<RkTagWrapper>
{
    using Parent = RkButcherTableauTest<RkTagWrapper>;

public:
    // the Explicit RK class
    Gempic::TimeLoop::
        ExplicitRK<TestFieldsHandlerStruct<Parent::s_nVarPde>, Parent::s_rkTag, FDDeRhamComplex>
            m_explicitRk;

    void rk_explicit () { this->rk(m_explicitRk); }

    RkButcherTableauExplicitTest() :
        m_explicitRk(this->m_numericalScheme.m_fields,
                     this->m_numericalScheme,
                     this->m_numericalScheme.m_drc,
                     this->m_rKreference)
    {
    }
};

template <typename RkTagWrapper> //, typename intRkType>
class RkButcherTableauImexTest : public RkButcherTableauTest<RkTagWrapper>
{
    using Parent = RkButcherTableauTest<RkTagWrapper>;

public:
    // the IMEX RK class
    Gempic::TimeLoop::
        ImexRk<TestFieldsHandlerStruct<Parent::s_nVarPde>, Parent::s_rkTag, FDDeRhamComplex>
            m_implicitExplicitRk;

    void rk_implicit_explicit () { this->rk(m_implicitExplicitRk); }

    RkButcherTableauImexTest() :
        m_implicitExplicitRk(this->m_numericalScheme.m_fields,
                             this->m_numericalScheme,
                             this->m_numericalScheme.m_drc,
                             this->m_rKreference)
    {
    }
};

using rkTypesExplicit = ::testing::Types<
    std::integral_constant<Gempic::TimeLoop::RungeKuttaTag,
                           Gempic::TimeLoop::RungeKuttaTag::ExplicitEuler>,
    std::integral_constant<Gempic::TimeLoop::RungeKuttaTag, Gempic::TimeLoop::RungeKuttaTag::RK2>,
    std::integral_constant<Gempic::TimeLoop::RungeKuttaTag, Gempic::TimeLoop::RungeKuttaTag::RK3>,
    std::integral_constant<Gempic::TimeLoop::RungeKuttaTag, Gempic::TimeLoop::RungeKuttaTag::RK4>,
    std::integral_constant<Gempic::TimeLoop::RungeKuttaTag,
                           Gempic::TimeLoop::RungeKuttaTag::SSPRK3>>;

// test explicit RK
TYPED_TEST_SUITE(RkButcherTableauExplicitTest, rkTypesExplicit);

TYPED_TEST(RkButcherTableauExplicitTest, RkExplicit) { this->rk_explicit(); }

using rkTypesImEx =
    ::testing::Types<std::integral_constant<Gempic::TimeLoop::RungeKuttaTag,
                                            Gempic::TimeLoop::RungeKuttaTag::ImexEuler>,
                     std::integral_constant<Gempic::TimeLoop::RungeKuttaTag,
                                            Gempic::TimeLoop::RungeKuttaTag::SASPImExRK3>>;

// test ImEx RK
TYPED_TEST_SUITE(RkButcherTableauImexTest, rkTypesImEx);

TYPED_TEST(RkButcherTableauImexTest, RkImplicitExplicit) { this->rk_implicit_explicit(); }

} // namespace
