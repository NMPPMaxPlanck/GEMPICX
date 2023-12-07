#ifndef GEMPIC_POISSONSOLVER_H_
#define GEMPIC_POISSONSOLVER_H_

#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_MFIter.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_MLNodeLaplacian.H>
#include <GEMPIC_Fields.H>
#include <GEMPIC_FDDeRhamComplex.H>


namespace GEMPIC_PoissonSolver
{
using namespace GEMPIC_Fields;
using namespace GEMPIC_FDDeRhamComplex;
    class PoissonSolver
    {
        private:
            int m_maxIter;
            int m_maxCoarseningLevel;
            int m_maxFmgIter;
            int m_mgBottomMaxIter;
            int m_verbose;
            int m_bottomVerbose;
            
            //for Poisson
            //amrex::Array<amrex::Real, GEMPIC_SPACEDIM> m_sigma;


        public:
            PoissonSolver();

            ~PoissonSolver();

            void solve(const Gempic::CompDom::computational_domain& infra, DeRhamField<Grid::dual, Space::cell>& rho,
                          DeRhamField<Grid::primal, Space::node>& phi);

            // The solver needs to subtract the kernel of the matrix to be invertible
            void subtractConstantPart(const Gempic::CompDom::computational_domain& infra,DeRhamField<Grid::dual, Space::cell>& rho, const int nGhost);
    };
}

#endif
