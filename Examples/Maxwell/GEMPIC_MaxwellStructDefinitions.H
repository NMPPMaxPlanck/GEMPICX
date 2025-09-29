#ifndef GEMPIC_MAXWELLSTRUCTDEFINITIONS_H_
#define GEMPIC_MAXWELLSTRUCTDEFINITIONS_H_

#include <iostream>
#include <random>
#include <string>
#include <type_traits>

#include <AMReX.H>
#include <AMReX_MFIter.H>
#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLNodeLaplacian.H>
#include <AMReX_Print.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Diagnostics.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_NumTools.H"
#include "GEMPIC_ParticleGroups.H"
#include "GEMPIC_RungeKutta.H"

/**
 * auxiliary parameter struct;
 */
struct IOParameters
{
    amrex::Real m_tEnd;
    int m_nmaxSteps, m_nStepsIo;
    amrex::Real m_dtIo, m_timeIo;
    bool m_printOutNSteps, m_printOutTime;

    IOParameters()
    {
        Gempic::Io::Parameters paramsTIME("TimeLoop");
        paramsTIME.get("tend", m_tEnd);
        paramsTIME.get("nmaxSteps", m_nmaxSteps);
        Gempic::Io::Parameters paramsIO("FullDiagnostics");
        paramsIO.get("save", m_printOutNSteps);
        paramsIO.get("saveInterval", m_nStepsIo);
        paramsIO.get("saveTime", m_printOutTime);
        paramsIO.get("saveTimeInterval", m_dtIo);
    }
};

/**
 * auxiliary discretization struct;
 */
struct StructDefinitions
{
    // Initialize computational_domain
    static constexpr int s_vdim{1};
    static constexpr int s_numspec{0};
    static constexpr int s_ndata{1};

    // Node spline degrees (smoothing spline degree is one less in each direction)
    static constexpr int s_degx{0};
    static constexpr int s_degy{0};
    static constexpr int s_degz{0};
    static constexpr int s_maxSplineDegree{std::max(std::max(s_degx, s_degy), s_degz)};

    static constexpr int s_hodgeDegree{4};
    static constexpr int s_nghost{s_hodgeDegree / 2}; //hodgeDegree / 2 - 1 + 1

    Gempic::ComputationalDomain m_compDom;

    // still needed for diagnostic?
    std::vector<std::shared_ptr<Gempic::Particle::ParticleGroups<s_vdim>>> m_particles;
};

/**
 * Handler structure for the Maxwell physical state
 *
 * @param Derived
 */
struct MaxwellFieldsHandlerStruct
    : Gempic::TimeLoop::BaseFieldsHandlerStruct<MaxwellFieldsHandlerStruct>
{
    // and the variables
    Gempic::Forms::DeRhamField<Grid::primal, Space::face> m_b;
    Gempic::Forms::DeRhamField<Grid::primal, Space::edge> m_e;

    /**
     * the assign method
     *
     * @param other : the source of the assign
     */
    void assign (MaxwellFieldsHandlerStruct const& other) // now we can use the = operator of the
                                                          // base class to call the assign...
    {
        copy(m_b, other.m_b);
        copy(m_e, other.m_e);
    }

    /**
     * the mult_and_add method
     *
     * @param alpha : real scaling parameter
     * @param other : the source of the assign
     */
    void mult_and_add (amrex::Real alpha, std::shared_ptr<MaxwellFieldsHandlerStruct> const& other)
    {
        //Q = Q + alpha * other.Q;
        //B = B + alpha * other.B;
        multiply_and_add(m_b, alpha, other->m_b);
        multiply_and_add(m_e, alpha, other->m_e);
    } //

    /**
     * the fill_boundary method
     */
    void fill_boundary ()
    {
        m_b.fill_boundary_nowait();
        m_e.fill_boundary_nowait();
        m_b.fill_boundary_finish();
        m_e.fill_boundary_finish();
    }

    /**
     * Explicitly define copy assignment operator to avoid default deletion (from DeRhamFields)
     *
     * @param other : source
     */
    MaxwellFieldsHandlerStruct& operator=(MaxwellFieldsHandlerStruct const& other)
    {
        if (this != &other)
        {                  // Prevent self-assignment
            assign(other); // Use assign method to copy assignable members
        }
        return *this;
    }

    /**
     * Constructor, overloaded for variables initialized with the GEMPIC diagnostic
     */
    MaxwellFieldsHandlerStruct(std::shared_ptr<Gempic::Forms::FDDeRhamComplex>& drc,
                               int nVar,
                               std::array<std::string, 2> strings) :
        m_b(drc, strings[0]), m_e(drc, strings[1]), m_nComp{nVar}, m_nGhost{drc->get_n_ghost()}
    {
    }

    /**
     * Constructor
     */
    MaxwellFieldsHandlerStruct(std::shared_ptr<Gempic::Forms::FDDeRhamComplex>& drc, int nVar) :
        m_b(drc), m_e(drc), m_nComp{nVar}, m_nGhost{AMREX_D_DECL(0, 0, 0)} //{drc->get_n_ghost()}
    {
    }

    ~MaxwellFieldsHandlerStruct() override
    { // Destructor called when deleted via base class pointer
      // Cleanup code for MaxwellFieldsHandlerStruct
    }

private:
    int m_nComp;
    amrex::IntVect m_nGhost;
    static constexpr int s_nFields{2};
};

/**
 * Definitions for the discretization of Maxwell
 */
struct MaxwellDiscretization : public StructDefinitions
{
    typedef Gempic::Forms::DeRhamField<Grid::primal, Space::edge> V1field;
    typedef Gempic::Forms::DeRhamField<Grid::primal, Space::face> V2field;
    typedef Gempic::Forms::DeRhamField<Grid::dual, Space::face> V1field_dual;
    typedef Gempic::Forms::DeRhamField<Grid::dual, Space::edge> V2field_dual;

    // Initialize the De Rham Complex
    std::shared_ptr<Gempic::Forms::FDDeRhamComplex> m_drc; //  HodgeScheme::GDECHodge); //

    // auxiliary state variables for the time integrators
    std::shared_ptr<MaxwellFieldsHandlerStruct> m_myfields;
    std::shared_ptr<MaxwellFieldsHandlerStruct> m_myfieldsDtEx;
    std::shared_ptr<MaxwellFieldsHandlerStruct> m_myfieldsDtIm;
    std::shared_ptr<MaxwellFieldsHandlerStruct> m_myfieldsTmpEx;
    std::shared_ptr<MaxwellFieldsHandlerStruct> m_myfieldsTmpIm;
    std::shared_ptr<MaxwellFieldsHandlerStruct> m_myfieldsNew;

    amrex::Real m_time;
    amrex::Real m_dt;
    amrex::Real m_tend;
    int m_timeStep;

    Gempic::Io::Diagnostics<s_degx, s_degy, s_degz, s_vdim, s_ndata> m_diagn;

    IOParameters m_iopar;

    MaxwellDiscretization(std::array<std::string, 2> const& outputFields) :
        m_drc{std::make_shared<Gempic::Forms::FDDeRhamComplex>(
            m_compDom, s_hodgeDegree, s_maxSplineDegree, Gempic::Forms::HodgeScheme::GDECHodge)},
        m_myfields(std::make_shared<MaxwellFieldsHandlerStruct>(m_drc, 1, outputFields)),
        m_myfieldsDtEx(std::make_shared<MaxwellFieldsHandlerStruct>(m_drc, 1)),
        m_myfieldsDtIm(std::make_shared<MaxwellFieldsHandlerStruct>(m_drc, 1)),
        m_myfieldsTmpEx(std::make_shared<MaxwellFieldsHandlerStruct>(m_drc, 1)),
        m_myfieldsTmpIm(std::make_shared<MaxwellFieldsHandlerStruct>(m_drc, 1)),
        m_myfieldsNew(std::make_shared<MaxwellFieldsHandlerStruct>(m_drc, 1)),
        m_diagn(m_compDom, m_drc, m_particles)
    {
        m_time = 0.0;
    }
};

#endif