/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
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
#include "GEMPIC_FieldRegistry.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_NumTools.H"
#include "GEMPIC_Particle.H"
#include "GEMPIC_RungeKutta.H"

constexpr Gempic::Forms::HodgeScheme maxwellHodgeScheme = Gempic::Forms::HodgeScheme::GDECHodge;
using MaxwellDeRham = Gempic::Forms::DeRhamComplexMF<maxwellHodgeScheme>;
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
    std::vector<std::shared_ptr<Gempic::Particle::ParticleSpecies<s_vdim>>> m_particles;
};

/**
 * Handler structure for the Maxwell physical state
 *
 * @param Derived
 */
struct MaxwellFieldsHandlerStruct
    : Gempic::TimeLoop::BaseFieldsHandlerStruct<MaxwellFieldsHandlerStruct>
{
    static constexpr int s_nVar = 1;
    // and the variables
    MaxwellDeRham::Primal2Form m_b;
    MaxwellDeRham::Primal1Form m_e;

    /**
     * the assign method
     *
     * @param other the source of the assign
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
     * @param alpha real scaling parameter
     * @param other the source of the assign
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
     * @param other source
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
     * Constructor
     */
    MaxwellFieldsHandlerStruct(std::shared_ptr<MaxwellDeRham>& drc) :
        m_b(drc), m_e(drc), m_nGhost{AMREX_D_DECL(0, 0, 0)}
    {
    }

    ~MaxwellFieldsHandlerStruct() override
    { // Destructor called when deleted via base class pointer
      // Cleanup code for MaxwellFieldsHandlerStruct
    }

private:
    amrex::IntVect m_nGhost;
    static constexpr int s_nFields{2};
};

/**
 * Definitions for the discretization of Maxwell
 */
struct MaxwellDiscretization : public StructDefinitions
{
    using V1field = MaxwellDeRham::Primal1Form;
    using V2field = MaxwellDeRham::Primal2Form;
    using V1field_dual = MaxwellDeRham::Dual2Form;
    using V2field_dual = MaxwellDeRham::Dual1Form;

    // Initialize the De Rham Complex
    std::shared_ptr<MaxwellDeRham> m_drc;

    // auxiliary state variables for the time integrators
    std::shared_ptr<MaxwellFieldsHandlerStruct> m_fields;
    std::shared_ptr<MaxwellFieldsHandlerStruct> m_fieldsDtEx;
    std::shared_ptr<MaxwellFieldsHandlerStruct> m_fieldsDtIm;
    std::shared_ptr<MaxwellFieldsHandlerStruct> m_fieldsTmpEx;
    std::shared_ptr<MaxwellFieldsHandlerStruct> m_fieldsTmpIm;
    std::shared_ptr<MaxwellFieldsHandlerStruct> m_fieldsNew;

    amrex::Real m_time;
    amrex::Real m_dt;
    amrex::Real m_tend;
    int m_timeStep;
    Gempic::Io::FieldRegistry m_fieldRegistry;

    Gempic::Io::Diagnostics<s_degx, s_degy, s_degz, s_vdim, s_ndata> m_diagn;

    IOParameters m_iopar;

    Gempic::Io::FieldRegistry maxwell_field_registration ()
    {
        Gempic::Io::FieldRegistry fieldRegistry;
        // labels for the output state in the diagnostics
        Gempic::Io::register_form(fieldRegistry, "B", m_fields->m_b);
        Gempic::Io::register_form(fieldRegistry, "E", m_fields->m_e);

        return fieldRegistry;
    }

    MaxwellDiscretization() :
        m_drc{std::make_shared<MaxwellDeRham>(m_compDom, s_hodgeDegree, s_maxSplineDegree)},
        m_fields(std::make_shared<MaxwellFieldsHandlerStruct>(m_drc)), // this has to be registered
        m_fieldsDtEx(std::make_shared<MaxwellFieldsHandlerStruct>(m_drc)),
        m_fieldsDtIm(std::make_shared<MaxwellFieldsHandlerStruct>(m_drc)),
        m_fieldsTmpEx(std::make_shared<MaxwellFieldsHandlerStruct>(m_drc)),
        m_fieldsTmpIm(std::make_shared<MaxwellFieldsHandlerStruct>(m_drc)),
        m_fieldsNew(std::make_shared<MaxwellFieldsHandlerStruct>(m_drc)),
        m_fieldRegistry{maxwell_field_registration()},
        m_diagn(m_compDom, m_drc, m_fieldRegistry, m_particles)
    {
    }
};

#endif