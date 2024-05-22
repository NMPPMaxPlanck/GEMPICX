#include <memory>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_PoissonSolver.H"

using namespace Gempic::FieldSolvers;

/**
 * @brief Implementation of Poisson solvers
 * @p Parameters
 * @p rho, which is a dual 3-form
 * @p phi, which is a primal 0-form
 *
 *
 */

PoissonSolver::PoissonSolver(std::shared_ptr<Forms::DeRhamComplex> deRham) :
    m_deRham{deRham}, m_residual{deRham}
{
    m_maxCoarseningLevel = 0;  // no multigrid (else 30 (from tutorial))
    m_maxIter = 100;
    m_mgBottomMaxIter = 100;
    m_maxFmgIter = 0;
    m_verbose = 0;
    m_bottomVerbose = 0;
    m_maxsteps = 100;
}

PoissonSolver::~PoissonSolver() {}

void PoissonSolver::solve (const ComputationalDomain& infra,
                          Forms::DeRhamField<Grid::dual, Space::cell>& rho,
                          Forms::DeRhamField<Grid::primal, Space::node>& phi)
{
    BL_PROFILE("Gempic::FieldSolvers::PoissonSolver::solve()");
    amrex::LPInfo lpInfo;
    lpInfo.setMaxCoarseningLevel(m_maxCoarseningLevel);

    // amrex::MLEBNodeFDLaplacian linop({params.geometry()}, {params.grid()}, {params.distriMap()},
    // lpInfo);

    amrex::MLNodeLaplacian linop({infra.m_geom}, {infra.m_grid}, {infra.m_distriMap}, lpInfo, {},
                                 1.0);

    // Set boundary conditions on linear operator for lower end and higher end
    linop.setDomainBC({AMREX_D_DECL(amrex::LinOpBCType::Periodic, amrex::LinOpBCType::Periodic,
                                    amrex::LinOpBCType::Periodic)},
                      {AMREX_D_DECL(amrex::LinOpBCType::Periodic, amrex::LinOpBCType::Periodic,
                                    amrex::LinOpBCType::Periodic)});

    // Additional parameters for Poisson
    // m_sigma = {AMREX_D_DECL(-1., -1., -1.)};
    // linop.setSigma( m_sigma);

    // Sum of rhs needs to be 0 in domain is periodic in all directions
    if (infra.m_geom.isAllPeriodic())
    {
        amrex::Real rhoSum = rho.m_data.sum_unique(0, false, infra.m_geom.periodicity());
        amrex::Print().SetPrecision(17)
            << " sum " << rhoSum << " " << rhoSum / (64 * 64 * 64) << std::endl;
        amrex::Real ninv =
            1.0 / GEMPIC_D_MULT(infra.m_nCell[xDir], infra.m_nCell[yDir], infra.m_nCell[zDir]);
        amrex::Real rhoSumNinv = rhoSum * ninv;
        rho -= rhoSumNinv;
        // for (amrex::MFIter mfi(rho.data); mfi.isValid(); ++mfi)
        // {
        //     const amrex::Box &bx = mfi.validbox();
        //     amrex::Array4<amrex::Real> const &rhoarr = (rho.data)[mfi].array();
        //     ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        //     {
        //         amrex::Print().SetPrecision(15) << " rho " << rhoarr(i,j,k) << std::endl;
        //         rhoarr(i, j, k) =  rhoarr(i, j, k) -rhoSum*Ninv;
        //     });
        // }
        rhoSum = rho.m_data.sum_unique(0, false, infra.m_geom.periodicity());
        amrex::Print().SetPrecision(15) << " sum2 " << rhoSum << std::endl;
    }

    // Initialize solver class
    amrex::MLMG mlmg(linop);

    // Configure solver class
    mlmg.setMaxIter(m_maxIter);
    mlmg.setMaxFmgIter(m_maxFmgIter);
    mlmg.setBottomMaxIter(m_mgBottomMaxIter);
    mlmg.setVerbose(m_verbose);
    mlmg.setBottomVerbose(m_bottomVerbose);
    mlmg.setBottomSolver(amrex::BottomSolver::cg);
    amrex::Real relTol = 1.e-11;
    amrex::Real absTol = 1.e-12;
    // Solve Poisson
    mlmg.solve({&phi.m_data}, {&rho.m_data}, relTol, absTol);
    // AMReX Poisson solver does not use Hodge. Need to rescale phi
    auto const dr = infra.m_dx;
    phi.m_data.mult(1 / GEMPIC_D_MULT(dr[xDir], dr[yDir], dr[zDir]));

    phi.average_sync();
    phi.fill_boundary();
}

void PoissonSolver::subtract_constant_part (const ComputationalDomain& infra,
                                           Forms::DeRhamField<Grid::dual, Space::cell>& rho,
                                           const int nGhost)
{
    BL_PROFILE("Gempic::FieldSolvers::PoissonSolver::subtract_constant_part()");
    // In the context of the Poisson solver rho has always one component.
    const int nComp = 1;
    // Calculates a nodal mask for rho
    std::unique_ptr<amrex::iMultiFab> nodalMask;
    nodalMask = std::make_unique<amrex::iMultiFab>(
        convert(infra.m_grid, amrex::IntVect::TheNodeVector()), infra.m_distriMap, nComp, nGhost);

    for (amrex::MFIter mfi(*nodalMask); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        amrex::IntVect hi = {bx.bigEnd()};

        amrex::Array4<int> const& maskArr = (*nodalMask)[mfi].array();
        ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        // if-loop to exclude ownership for the point that is at the upper
                        // boundary for nodal directions
                        if ((i <= (hi[xDir] - 1)) && (j <= (hi[yDir] - 1)) && (k <= (hi[zDir] - 1)))
                        {
                            maskArr(i, j, k) = 1.0;
                        }
                    });
    }

    amrex::Real nm1 = 0.0;
    int counter = 0;
    for (amrex::MFIter mfi(rho.m_data, true); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.growntilebox(0);
        auto const& a = rho.m_data.const_array(mfi);
        amrex::Array4<int const> const& mfab = nodalMask->const_array(mfi);
        AMREX_LOOP_3D(bx, i, j, k, {
            if (mfab(i, j, k))
            {
                nm1 += a(i, j, k, 0);
                counter++;
            }
        });
    }

    amrex::ParallelAllReduce::Sum(nm1, amrex::ParallelContext::CommunicatorSub());
    amrex::ParallelAllReduce::Sum(counter, amrex::ParallelContext::CommunicatorSub());
    rho -= nm1 / static_cast<amrex::Real>(counter);
}
