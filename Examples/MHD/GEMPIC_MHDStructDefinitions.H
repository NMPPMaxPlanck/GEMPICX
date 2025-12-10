#ifndef GEMPIC_MHDSTRUCTDEFINITIONS_H_
#define GEMPIC_MHDSTRUCTDEFINITIONS_H_

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
#include "GEMPIC_CustomDiagnosticStrategies.H"
#include "GEMPIC_Diagnostics.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_NumTools.H"
#include "GEMPIC_Particle.H"
#include "GEMPIC_RungeKutta.H"

constexpr int nVarMHD = 8;          // compile time constant
constexpr int polyrecDegreeMHD = 2; // polynomial degree of the WENO-FV (V3) reconstruction. //
                                    // even: we only have 3 stencils: Left-Central-Right.

static constexpr Gempic::TimeLoop::RungeKuttaTag myRKTag = Gempic::TimeLoop::RungeKuttaTag::RK3;
//constexpr int timeDegree =
//    2; // polynomial degree of the time discretization. // check availability in GEMPIC_NumTools.H

static constexpr int rhoIdx{0};
static constexpr int vxIdx{1};
static constexpr int vyIdx{2};
static constexpr int vzIdx{3};
static constexpr int mxIdx{1};
static constexpr int myIdx{2};
static constexpr int mzIdx{3};
static constexpr int pIdx{4};
static constexpr int eIdx{4};
static constexpr int bxIdx{5};
static constexpr int byIdx{6};
static constexpr int bzIdx{7};

//constexpr int myRKstages = timeDegree + 1; // polynomial degree of the reconstruction.

constexpr int nDof1D{polyrecDegreeMHD + 1};
//constexpr int volumeDof{AMREX_D_PICK(nDof_1D, nDof_1D* nDof_1D, nDof1D* nDof1D* nDof1D)};
constexpr int faceDof{AMREX_D_PICK(1, nDof1D, nDof1D* nDof1D)};
constexpr int edgeDof{AMREX_D_PICK(1, 1, nDof1D)};

/**
 * auxiliary parameter struct;
 */
struct IOParameters
{
    amrex::Real m_dtMax, m_tIni, m_tEnd;
    amrex::Real m_cfl;
    int m_nVar, m_nmaxSteps, m_nStepsIo;
    amrex::Real m_dtIo, m_timeIo;
    bool m_printOutNSteps, m_printOutTime;

    IOParameters()
    {
        Gempic::Io::Parameters paramsTIME("TimeLoop");
        paramsTIME.get("CFL", m_cfl);
        paramsTIME.get("dt_max", m_dtMax);
        paramsTIME.get("tend", m_tEnd);
        paramsTIME.get("tini", m_tIni);
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

    static constexpr int s_nDof1D{polyrecDegreeMHD + 1};
    static constexpr int s_hodgeDegree{s_nDof1D};
    static constexpr int s_hodgeOrder{s_nDof1D};
    //static constexpr int s_nghost{s_hodgeDegree / 2 -
    //                              1}; //s_hodgeDegree / 2 }; //hodgeDegree / 2 - 1 + 1
    static constexpr int s_nghost{s_nDof1D}; //s_hodgeOrder / 2}; //hodgeDegree / 2 - 1 + 1

    Gempic::ComputationalDomain m_compDom;

    // still needed for diagnostic?
    std::vector<std::shared_ptr<Gempic::Particle::ParticleSpecies<s_vdim>>> m_particles;
};

/**
 * Handler structure for the MHD physical state
 *
 * @param Derived
 */
struct MHDFieldsHandlerStruct : Gempic::TimeLoop::BaseFieldsHandlerStruct<MHDFieldsHandlerStruct>
{
public:
    // Member variables
    Gempic::Forms::DeRhamField<Grid::primal, Space::cell> m_q;
    Gempic::Forms::DeRhamField<Grid::primal, Space::face> m_b;

    // Implement the assign method
    void assign (MHDFieldsHandlerStruct const& other) // now we can use the = operator of the base
                                                      // class to call the assign...
    {
        copy(m_q, other.m_q);
        //copy(B, other.B);
    }

    /**
     * the mult_and_add method
     *
     * @param alpha : real scaling parameter
     * @param other : the source of the assign
     */
    void mult_and_add (amrex::Real alpha, std::shared_ptr<MHDFieldsHandlerStruct> const& other)
    {
        //Q = Q + alpha * other.Q;
        //B = B + alpha * other.B;
        multiply_and_add(m_q, alpha, other->m_q);
        //multiply_and_add(B, alpha, other->B);
    }

    /**
     * the fill_boundary method
     */
    void fill_boundary ()
    {
        m_q.fill_boundary_nowait();
        //B.fill_boundary_nowait();
        m_q.fill_boundary_finish();
        //B.fill_boundary_finish();
    }

    /**
     * the get_nfields method
     */
    int get_nfields () { return s_nFields; }

    /**
     * Explicitly define copy assignment operator to avoid default deletion (from DeRhamFields)
     *
     * @param other : source
     */
    MHDFieldsHandlerStruct& operator=(MHDFieldsHandlerStruct const& other)
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
    MHDFieldsHandlerStruct(std::shared_ptr<Gempic::Forms::FDDeRhamComplex>& drc,
                           int nVar,
                           std::array<std::string, 2> strings) :
        m_q(drc, strings[0], nVar),
        m_b(drc, strings[1]),
        m_nComp{nVar},
        m_nGhost{drc->get_n_ghost()}
    {
    }

    /**
     * Constructor
     */
    MHDFieldsHandlerStruct(std::shared_ptr<Gempic::Forms::FDDeRhamComplex>& drc, int nVar) :
        m_q(drc, nVar),
        m_b(drc),
        m_nComp{nVar},
        m_nGhost{AMREX_D_DECL(0, 0, 0)} //{drc->get_n_ghost()}
    {
    }

    ~MHDFieldsHandlerStruct() override
    { // Destructor called when deleted via base class pointer
      // Cleanup code for MHDFieldsHandlerStruct
    }

private:
    int m_nComp;
    amrex::IntVect m_nGhost;
    static constexpr int s_nFields{2};
};

/**
 * Definitions for the discretization of Maxwell
 */
struct MHDDiscretization : public StructDefinitions
{
public:
    typedef Gempic::Forms::DeRhamField<Grid::primal, Space::node> V0field;
    typedef Gempic::Forms::DeRhamField<Grid::primal, Space::edge> V1field;
    typedef Gempic::Forms::DeRhamField<Grid::primal, Space::face> V2field;
    typedef Gempic::Forms::DeRhamField<Grid::primal, Space::cell> V3field;

    // Initialize the De Rham Complex
    std::shared_ptr<Gempic::Forms::FDDeRhamComplex> m_drc; //  HodgeScheme::GDECHodge); //

    //Io::Parameters parameters{};
    // Initialize the input parameters
    IOParameters m_iopar;

    // auxiliary state variables for the EXPLICIT time integrators
    std::shared_ptr<MHDFieldsHandlerStruct> m_myfields;
    //std::shared_ptr<MHDFieldsHandlerStruct> m_myfieldsDtEx;
    std::shared_ptr<MHDFieldsHandlerStruct> m_myfieldsTmpEx;
    // auxiliary state variables for the IMPLICIT time integrators
    std::shared_ptr<MHDFieldsHandlerStruct> m_myfieldsDtEx;
    std::shared_ptr<MHDFieldsHandlerStruct> m_myfieldsDtIm;
    std::shared_ptr<MHDFieldsHandlerStruct> m_myfieldsTmpIm;
    std::shared_ptr<MHDFieldsHandlerStruct> m_myfieldsNew;

    std::shared_ptr<MHDFieldsHandlerStruct> m_myfieldsTilde;

    static constexpr int s_volumeDof{
        AMREX_D_PICK(s_nDof1D, s_nDof1D* s_nDof1D, s_nDof1D* s_nDof1D* s_nDof1D)};
    static constexpr int s_faceDof{AMREX_D_PICK(1, s_nDof1D, s_nDof1D* s_nDof1D)};
    static constexpr int s_edgeDof{AMREX_D_PICK(1, 1, s_nDof1D)};

    amrex::IndexType m_primalCellIndex{amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 0, 0)})};
    amrex::GpuArray<amrex::IndexType, AMREX_SPACEDIM> m_primalFaceIndex{
        AMREX_D_DECL(amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 0, 0)}),
                     amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 1, 0)}),
                     amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 0, 1)}))};

    amrex::GpuArray<amrex::IndexType, AMREX_SPACEDIM> m_primalEdgeIndex{
        AMREX_D_DECL(amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 1, 1)}),
                     amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 0, 1)}),
                     amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 1, 0)}))};

    amrex::Real m_time;
    amrex::Real m_dt;
    amrex::Real m_tend;
    int m_timeStep;

    Gempic::Io::Diagnostics<s_degx, s_degy, s_degz, s_vdim, s_ndata> m_diagn;

    amrex::Real m_dtLocal;
    amrex::Real m_dtOld;
    amrex::Real m_dtNew;

    amrex::GpuArray<amrex::Real, 3> m_dr;
    amrex::GpuArray<amrex::Real, 3> m_idr;
    amrex::GpuArray<amrex::Real, 3> m_r0;
    amrex::GpuArray<amrex::Real, 3> m_dS;
    amrex::Real m_dVol;
    amrex::Real m_idVol;

    Gempic::ComputationalDomain m_discretedomainIo;

    amrex::IntVect m_nGhosts{0};
    amrex::IntVect m_nGhostsVolumeDof{AMREX_D_DECL(0, 0, 0)};
    amrex::IntVect m_nGhostsFaceDof{AMREX_D_DECL(0, 0, 0)};
    amrex::Vector<int> m_nDofVector{AMREX_D_DECL(1, 1, 1)};

    // FV explicit
    //V3field Q;
    //V3field dQdt;
    V3field m_v;
    V2field m_fluxes;
    V2field m_qlRec;
    V2field m_qrRec;
    // mimetic FV explicit
    V2field m_bold;
    V1field m_e;

    amrex::Array<amrex::FabArray<amrex::BaseFab<amrex::GpuArray<amrex::Real, s_faceDof>>>,
                 AMREX_SPACEDIM>
        m_qlRecFab;
    amrex::Array<amrex::FabArray<amrex::BaseFab<amrex::GpuArray<amrex::Real, s_faceDof>>>,
                 AMREX_SPACEDIM>
        m_qrRecFab;
#ifdef AMREX_USE_GPU
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_qlRecFab2mfab;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_qrRecFab2mfab;
#endif
    amrex::Array<amrex::FabArray<amrex::BaseFab<amrex::GpuArray<amrex::Real, 3>>>, AMREX_SPACEDIM>
        m_wWeno; // default number of stencils is 3
    amrex::Array<amrex::FabArray<amrex::BaseFab<amrex::GpuArray<amrex::Real, s_edgeDof>>>,
                 AMREX_SPACEDIM>
        m_bEdge; // default number of stencils is 3

    MHDDiscretization(std::array<std::string, 2> const& outputFields) :
        m_drc{
            std::make_shared<Gempic::Forms::FDDeRhamComplex>(m_compDom,
                                                             s_hodgeDegree,
                                                             s_maxSplineDegree,
                                                             //HodgeScheme::GDECLumpHodge);  //
                                                             Gempic::Forms::HodgeScheme::FDHodge)},
        m_iopar(),
        m_myfields(std::make_shared<MHDFieldsHandlerStruct>(m_drc, nVarMHD, outputFields)),
        m_myfieldsTmpEx(std::make_shared<MHDFieldsHandlerStruct>(m_drc, nVarMHD)),
        m_myfieldsDtEx(std::make_shared<MHDFieldsHandlerStruct>(m_drc, nVarMHD)),
        m_myfieldsDtIm(std::make_shared<MHDFieldsHandlerStruct>(m_drc, nVarMHD)),
        m_myfieldsTmpIm(std::make_shared<MHDFieldsHandlerStruct>(m_drc, nVarMHD)),
        m_myfieldsNew(std::make_shared<MHDFieldsHandlerStruct>(m_drc, nVarMHD)),
        m_myfieldsTilde(std::make_shared<MHDFieldsHandlerStruct>(m_drc, nVarMHD)),
        m_diagn(m_compDom, m_drc, m_particles),
        m_nGhosts{m_drc->get_n_ghost()},
        m_v(m_drc, nVarMHD),
        m_fluxes(m_drc, nVarMHD),
        m_qlRec(m_drc, nVarMHD),
        m_qrRec(m_drc, nVarMHD),
        m_bold(m_drc),
        m_e(m_drc)
    {
        // default values
        m_time = m_iopar.m_tIni;
        m_dt = m_iopar.m_dtMax;
        m_dtLocal = m_dt;
        m_dtOld = m_dt;
        m_dtNew = m_dt;
        for (int idim = 0; idim < 3; idim++)
        {
            m_dr[idim] = 1.0;
            m_idr[idim] = 1.0;
            m_r0[idim] = 0.0;
        }
        for (int idim = 0; idim < AMREX_SPACEDIM; idim++)
        {
            m_dr[idim] = m_compDom.m_geom.CellSize()[idim];
            m_idr[idim] = 1.0 / m_compDom.m_geom.CellSize()[idim];
            m_r0[idim] = m_compDom.m_geom.ProbLo()[idim];
            //dr = {infra.m_geom.CellSize()[xDir], infra.m_geom.CellSize()[yDir],
            //      infra.m_geom.CellSize()[zDir]};
        }
        m_dVol = GEMPIC_D_MULT(m_dr[xDir], m_dr[yDir], m_dr[zDir]);
        m_idVol = 1.0 / m_dVol;
        for (int idim = 0; idim < AMREX_SPACEDIM; idim++)
        {
            m_dS[idim] = m_dVol * m_idr[idim];
        }

        for (int iDim = 0; iDim < AMREX_SPACEDIM; ++iDim)
        {
            m_qrRecFab[iDim] =
                amrex::FabArray<amrex::BaseFab<amrex::GpuArray<amrex::Real, s_faceDof>>>(
                    amrex::convert(m_compDom.m_grid, m_primalFaceIndex[iDim]),
                    m_compDom.m_distriMap, nVarMHD, {m_drc->get_n_ghost()});
            m_qlRecFab[iDim] =
                amrex::FabArray<amrex::BaseFab<amrex::GpuArray<amrex::Real, s_faceDof>>>(
                    amrex::convert(m_compDom.m_grid, m_primalFaceIndex[iDim]),
                    m_compDom.m_distriMap, nVarMHD, {m_drc->get_n_ghost()});
#ifdef AMREX_USE_GPU
            m_qrRecFab2mfab[iDim] =
                amrex::MultiFab(amrex::convert(m_compDom.m_grid, m_primalFaceIndex[iDim]),
                                m_compDom.m_distriMap, nVarMHD * faceDof, {m_drc->get_n_ghost()});
            m_qlRecFab2mfab[iDim] =
                amrex::MultiFab(amrex::convert(m_compDom.m_grid, m_primalFaceIndex[iDim]),
                                m_compDom.m_distriMap, nVarMHD * faceDof, {m_drc->get_n_ghost()});
#endif
            m_wWeno[iDim] = amrex::FabArray<amrex::BaseFab<amrex::GpuArray<amrex::Real, 3>>>(
                amrex::convert(m_compDom.m_grid, m_primalEdgeIndex[iDim]), m_compDom.m_distriMap, 1,
                {m_drc->get_n_ghost()});
            m_bEdge[iDim] =
                amrex::FabArray<amrex::BaseFab<amrex::GpuArray<amrex::Real, s_edgeDof>>>(
                    amrex::convert(m_compDom.m_grid, m_primalEdgeIndex[iDim]),
                    m_compDom.m_distriMap, 1, {m_drc->get_n_ghost()});
        }
    }
};

#endif