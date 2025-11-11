#ifndef GEMPIC_MAXWELLPDE_H_
#define GEMPIC_MAXWELLPDE_H_

#include "GEMPIC_FieldMethods.H"
#include "GEMPIC_MaxwellInit.H"
#include "GEMPIC_MaxwellStructDefinitions.H"
#include "GEMPIC_NumTools.H"
#include "GEMPIC_Solvers.H"

/**
 * Maxwell MatOp functor class,
 * to solve the Schur complement of the homogeneous Maxwell equations.
 * (Faraday in strong form, solve for E)
 */
class GempicMaxwellMatOp
{
public:
    std::shared_ptr<Gempic::Forms::FDDeRhamComplex> m_drc;
    amrex::Real m_dtLocal{0.0};

    typedef Gempic::Forms::DeRhamField<Grid::primal, Space::edge> V1field;
    typedef Gempic::Forms::DeRhamField<Grid::primal, Space::face> V2field;
    typedef Gempic::Forms::DeRhamField<Grid::dual, Space::face> V1field_dual;
    typedef Gempic::Forms::DeRhamField<Grid::dual, Space::edge> V2field_dual;

    V1field_dual m_estar;
    V2field_dual m_bstar;
    V2field m_v2fieldTmp;

    /**
     * compute right hand side for Maxwell.
     *
     * @param[out] rhs : the right hand side
     * @param E : electric field as V1field
     * @param B : magnetic field as V2field
     * @param dtin : time step
     *
     * @return : $RETURN
     */
    void rhs_op (V1field_dual& rhs, V1field& E, V2field& B, amrex::Real dtin)
    {
        // e.g.
        //  B = Bold - dt curl E
        //  M1 E  - dt curlT M2 B = M1 Eold
        //  M1 E + dt^2 curlT M2 curl E = M1 Eold +dt curlT M2 Bold
        amrex::Real const dt{dtin};
        hodge(m_bstar, B);
        hodge(rhs, E); // ... todo
        add_dt_curl(rhs, m_bstar, dt);
    }

    /**
     * update B.
     *
     * @param [out] B : magnetic field as V2field
     * @param E : electric field as V1field
     * @param dtin : time step
     *
     * @return : $RETURN
     */
    void update_b (V1field& E, V2field& B, amrex::Real dtin)
    {
        amrex::Real const mdt{-dtin};
        add_dt_curl(B, E, mdt);
    }

    /**
     * update Matop.
     *
     * @param dtin : time step
     *
     */
    void update (amrex::Real dtIn) { m_dtLocal = dtIn; }

    /**
     * MatOp operator: compute matrix vector product.
     *
     * @param[out] : matrix vector product
     * @param : x vector
     *
     */
    void operator()(V1field_dual& Ax, V1field& x)
    {
        // e.g.
        //  B = Bold - dt curl E
        //  M1 E  - dt curlT M2 B = M1 Eold
        //  M1 E + dt^2 curlT M2 curl E = M1 Eold + dt curlT M2 Bold
        // curl E
        amrex::Real dt{m_dtLocal};
        curl(m_v2fieldTmp, x);

        // hodge M2 curl E
        hodge(m_bstar, m_v2fieldTmp);
        amrex::Real alpha{dt * dt};
        // hodge M1 E = M1*x ... todo
        hodge(Ax, x); // ... todo
        add_dt_curl(Ax, m_bstar, alpha);
    }

    /**
     *
     * @param TypeDiscPde: Numerical discretization type
     *
     */
    GempicMaxwellMatOp(MaxwellDiscretization& disc) :
        m_drc(disc.m_drc), m_estar(disc.m_drc), m_bstar(disc.m_drc), m_v2fieldTmp(disc.m_drc)
    {
    }
};

/**
 * Maxwell Numerical Scheme Class.
 *  - m_disc.myfields: in and out state.
 *  - m_disc.myfields_dt_ex: unknown time-derivative (to be computed).
 *  - m_disc.myfields_tmp_ex: auxiliary temporary state.
 */
class MaxwellNumericalScheme : public NumericalScheme<MaxwellFieldsHandlerStruct>
{
public:
    typedef Gempic::Forms::DeRhamField<Grid::primal, Space::edge> V1field;
    typedef Gempic::Forms::DeRhamField<Grid::dual, Space::face> V1field_dual;

    MaxwellDiscretization m_disc;
    GempicMaxwellMatOp m_maxwell;

    std::unique_ptr<
        Gempic::FieldSolvers::ConjugateGradient<V1field_dual, V1field, GempicMaxwellMatOp>>
        m_myCg2;

    V1field_dual m_rhs;
    V1field m_enew;

    int m_nmaxSteps{1000};

    /**
     * @brief get the pointer to the field used as initial condition for the explicit time-step, by
     * reference
     * @return shared pointer of FieldsHandlerStruct
     */
    std::shared_ptr<MaxwellFieldsHandlerStruct> const& get_pointer_initial_field_explicit ()
        override
    {
        return m_disc.m_myfieldsTmpEx;
    }

    std::shared_ptr<MaxwellFieldsHandlerStruct> const& get_pointer_initial_field_implicit ()
        override
    {
        return m_disc.m_myfieldsTmpIm;
    }

    std::shared_ptr<MaxwellFieldsHandlerStruct> const& get_pointer_time_derivative_explicit ()
        override
    {
        return m_disc.m_myfieldsDtEx;
    }

    std::shared_ptr<MaxwellFieldsHandlerStruct> const& get_pointer_time_derivative_implicit ()
        override
    {
        return m_disc.m_myfieldsDtIm;
    }

    void set_initial_condition () override
    {
        m_disc.m_time = 0.0;
        m_disc.m_timeStep = 0;
        m_disc.m_iopar.m_timeIo = m_disc.m_time + m_disc.m_iopar.m_dtIo;
        m_nmaxSteps = m_disc.m_iopar.m_nmaxSteps;
        maxwell_initial_condition(m_disc);
    }

    void one_step_solve_dqdt_explicit (amrex::Real time) override { return; }

    /**
     * update old state
     */
    void update_qold_implicit ()
    {
        m_disc.m_myfieldsTmpIm = m_disc.m_myfields;
        return;
    }

    /**
     * solve one implicit (Euler) step
     */
    void one_step_solve_q_implicit (amrex::Real time, amrex::Real dtLoc) override
    {
        m_myCg2->m_operator.update(dtLoc); // => update the matop A(Q,dt)
        // in the RHS we should consider:
        // - q_tmp_im: as initial state;
        // - dqdt_ex: as explicit terms;
        // - an input timestep, that can be rescaled by the RK coefficient! (because we first solve
        // for q, then evaluate dqdt)
        // FIRST: fill boundary
        m_disc.m_myfieldsTmpIm->fill_boundary();
        m_maxwell.rhs_op(m_rhs, m_disc.m_myfieldsTmpIm->m_e, m_disc.m_myfieldsTmpIm->m_b, dtLoc);
        m_myCg2->solve(m_disc.m_myfieldsNew->m_e, m_rhs);

        if (Gempic::Utils::Verbosity::level() > 0)
        {
            amrex::Print() << "        | CG tol:" << m_myCg2->get_tol()
                           << "CG iter:" << m_myCg2->get_iter() << std::endl;
        }

        copy(m_disc.m_myfieldsNew->m_b, m_disc.m_myfieldsTmpIm->m_b);
        m_maxwell.update_b(m_disc.m_myfieldsNew->m_e, m_disc.m_myfieldsNew->m_b, dtLoc);
    }

    /**
     * compute time derivative for the semi-implicit solver.
     *
     * @param dtloc : time step
     */
    void one_step_solve_dqdt_semiimplicit (amrex::Real timeI,
                                           amrex::Real dtI,
                                           amrex::Real timeE,
                                           amrex::Real dtE) override
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
        for (int idir = 0; idir < 3; ++idir)
        {
            for (amrex::MFIter mfi(m_disc.m_myfieldsDtIm->m_b.m_data[idir], true); mfi.isValid();
                 ++mfi)
            {
                amrex::Box const& bx = mfi.tilebox(); // OPENMP valid box is for real cells.
                // creating pointer to data for GPU:
                amrex::Array4<amrex::Real> const& dBdt =
                    (m_disc.m_myfieldsDtIm->m_b.m_data[idir])[mfi].array();
                amrex::Array4<amrex::Real> const& bnew =
                    (m_disc.m_myfieldsNew->m_b.m_data[idir])[mfi].array();
                amrex::Array4<amrex::Real> const& bold =
                    (m_disc.m_myfieldsTmpIm->m_b.m_data[idir])[mfi].array();
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                            { dBdt(i, j, k) = (bnew(i, j, k) - bold(i, j, k)) / dtLoc; });
            }
        }
        for (int idir = 0; idir < 3; ++idir)
        {
            for (amrex::MFIter mfi(m_disc.m_myfieldsDtIm->m_e.m_data[idir], true); mfi.isValid();
                 ++mfi)
            {
                amrex::Box const& bx = mfi.tilebox(); // OPENMP valid box is for real cells.
                // creating pointer to data for GPU:
                amrex::Array4<amrex::Real> const& dEdt =
                    (m_disc.m_myfieldsDtIm->m_e.m_data[idir])[mfi].array();
                amrex::Array4<amrex::Real> const& enew =
                    (m_disc.m_myfieldsNew->m_e.m_data[idir])[mfi].array();
                amrex::Array4<amrex::Real> const& eold =
                    (m_disc.m_myfieldsTmpIm->m_e.m_data[idir])[mfi].array();
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                            { dEdt(i, j, k) = (enew(i, j, k) - eold(i, j, k)) / dtLoc; });
            }
        }
        return;
    }

    /**
     * @brief return `true` if the current time is greater than or equal to the final time,
     * otherwise `false`.
     */
    bool is_finaltime_reached () override { return m_disc.m_time >= m_disc.m_tend; }

    /**
     * initialize new time step.
     */
    void init_new_timestep () override
    {
        if (m_disc.m_time + m_disc.m_dt > m_disc.m_tend + 1e-14)
        {
            m_disc.m_dt = m_disc.m_tend - m_disc.m_time + 1e-14;
        }
        m_disc.m_timeStep++;
    }

    /**
     * finalize current time step.
     */
    void finalize_new_timestep () override { m_disc.m_time += m_disc.m_dt; }

    /**
     * print now.
     */
    void print_now ()
    {
        m_disc.m_myfields->fill_boundary();
        m_disc.m_diagn.compute_and_write_to_file(m_disc.m_timeStep, m_disc.m_time);
    }

    /**
     * check and eventually print.
     */
    void check_and_print (int timeStep)
    {
        if (m_disc.m_iopar.m_printOutNSteps && timeStep % m_disc.m_iopar.m_nStepsIo == 0)
        {
            print_now();
        }
        if (m_disc.m_iopar.m_printOutTime && m_disc.m_time >= m_disc.m_iopar.m_timeIo)
        {
            while (m_disc.m_time >= m_disc.m_iopar.m_timeIo)
            {
                m_disc.m_iopar.m_timeIo += m_disc.m_iopar.m_dtIo;
            }
            print_now();
        }
    }

    /**
     * get the number of components.
     */
    int get_ncomp () override { return 1; }

    /**
     * get periodicity.
     */
    amrex::Periodicity get_periodicity () override { return m_disc.m_compDom.m_geom.periodicity(); }

    MaxwellNumericalScheme(std::array<std::string, 2> const& outputFields) :
        m_disc(outputFields),
        m_maxwell(m_disc),
        m_myCg2{std::make_unique<
            Gempic::FieldSolvers::ConjugateGradient<V1field_dual, V1field, GempicMaxwellMatOp>>(
            m_disc.m_drc, GempicMaxwellMatOp(m_disc))},
        m_rhs(m_disc.m_drc),
        m_enew(m_disc.m_drc)
    {
        Gempic::Io::Parameters params("TimeLoop");
        amrex::Real dt;
        amrex::Real tend;
        params.get("dt", dt);
        m_disc.m_dt = dt;
        params.get("tend", tend);
        m_disc.m_tend = tend;
    }
};

#endif