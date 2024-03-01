#ifndef GEMPIC_POISSONSOLVER_H_
#define GEMPIC_POISSONSOLVER_H_

#include <AMReX.H>
#include <AMReX_MFIter.H>
#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLNodeLaplacian.H>
#include <AMReX_Print.H>

#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"

namespace GEMPIC_PoissonSolver
{
using Grid = GEMPIC_FDDeRhamComplex::Grid;
using Space = GEMPIC_FDDeRhamComplex::Space;

class PoissonSolver
{
private:
    int m_maxIter;
    int m_maxCoarseningLevel;
    int m_maxFmgIter;
    int m_mgBottomMaxIter;
    int m_verbose;
    int m_bottomVerbose;
    amrex::Real m_reltol{1.e-11};
    amrex::Real m_maxsteps{100};
    std::shared_ptr<GEMPIC_Fields::DeRhamComplex> m_deRham;
    GEMPIC_FDDeRhamComplex::DeRhamField<Grid::primal, Space::edge> m_residual;

    // for Poisson
    amrex::Array<amrex::Real, GEMPIC_SPACEDIM> m_sigma;

public:
    PoissonSolver(std::shared_ptr<GEMPIC_Fields::DeRhamComplex> deRham);

    ~PoissonSolver();

    void solve(const Gempic::CompDom::ComputationalDomain& infra,
               GEMPIC_FDDeRhamComplex::DeRhamField<Grid::dual, Space::cell>& rho,
               GEMPIC_FDDeRhamComplex::DeRhamField<Grid::primal, Space::node>& phi);

    void subtract_constant_part(const Gempic::CompDom::ComputationalDomain& infra,
                                GEMPIC_FDDeRhamComplex::DeRhamField<Grid::dual, Space::cell>& rho,
                                const int nGhost);

    template <int hodgeDegree, GEMPIC_FDDeRhamComplex::HodgeScheme hodgeScheme>
    void solve_hodge_cg (
        GEMPIC_FDDeRhamComplex::DeRhamField<Grid::primal, Space::edge> primalEdgeField,
        GEMPIC_FDDeRhamComplex::DeRhamField<Grid::dual, Space::face> dualFaceField)
    {
        //    DeRhamField<Grid::primal, Space::edge> residual(deRham);  //, d(m_deRham),
        //    z(m_deRham);
        m_deRham->hodge(dualFaceField, m_residual);

        // for (amrex::MFIter mfi(primalEdgeField.data); mfi.isValid(); ++mfi)
        // {
        //     const amrex::Box &bx = mfi.validbox();
        //     amrex::Array4<amrex::Real> const &residualArr = (residual.data)[mfi].array();
        //     amrex::Array4<amrex::Real> const &zArr = (z.data)[mfi].array();
        //     amrex::Array4<amrex::Real> const &bArr = (primalEdgeField.data)[mfi].array();
        //     //amrex::Array4<amrex::Real> const &dArr = (b.data)[mfi].array();

        //     ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        //         {
        //             rArr(i,j,k) = bArr(i,j,k) - zArr(i,j,k);
        //             dArr(i,j,k) = rArr(i,j,k);
        //         });
        // }
        for (int step = 0; step < m_maxsteps; step++)
        {
        }
    }

    template <int hodgeDegree, GEMPIC_FDDeRhamComplex::HodgeScheme hodgeScheme>
    void apply_poisson_operator (
        Gempic::CompDom::ComputationalDomain compDom,
        GEMPIC_FDDeRhamComplex::DeRhamField<Grid::dual, Space::cell>& dualCellOut,
        GEMPIC_FDDeRhamComplex::DeRhamField<Grid::primal, Space::node>& primalNodeIn)
    {
        auto deRham{primalNodeIn.m_deRham};
        GEMPIC_FDDeRhamComplex::DeRhamField<Grid::primal, Space::edge> tempPrimalEdge(deRham);
        GEMPIC_FDDeRhamComplex::DeRhamField<Grid::dual, Space::face> tempDualFace(deRham);
        deRham->grad(primalNodeIn, tempPrimalEdge);
        solve_hodge_cg<hodgeDegree, hodgeScheme>(tempPrimalEdge, tempDualFace);
        deRham->div(tempDualFace, dualCellOut);
    }
};

}  // namespace GEMPIC_PoissonSolver

#endif
