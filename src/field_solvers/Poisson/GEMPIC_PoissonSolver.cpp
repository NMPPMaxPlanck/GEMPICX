#include "GEMPIC_PoissonSolver.H"

using namespace GEMPIC_PoissonSolver;

/** 
 * @brief Implementation of Poisson solvers
 * @p Parameters
 * @p rho, which is a dual 3-form
 * @p phi, which is a primal 0-form
 * 
 * 
 */

PoissonSolver::PoissonSolver()
{
    m_maxCoarseningLevel = 3;
    m_maxIter = 1000;
    m_mgBottomMaxIter = 1000;
    m_maxFmgIter = 0;
    m_verbose = 0;
    m_bottomVerbose = 0;
}

PoissonSolver::~PoissonSolver() {}

void PoissonSolver::solve(Parameters params, DeRhamField<Grid::dual, Space::cell>& rho,
                          DeRhamField<Grid::primal, Space::node>& phi)
{
    amrex::LPInfo lpInfo;
    lpInfo.setMaxCoarseningLevel(m_maxCoarseningLevel);

    //amrex::MLEBNodeFDLaplacian linop({params.geometry()}, {params.grid()}, {params.distriMap()}, lpInfo);

    amrex::MLNodeLaplacian linop({params.geometry()}, {params.grid()}, {params.distriMap()}, lpInfo, {}, 1.0);

    // Set boundary conditions on linear operator for lower end and higher end
    linop.setDomainBC({AMREX_D_DECL(amrex::LinOpBCType::Periodic, amrex::LinOpBCType::Periodic, amrex::LinOpBCType::Periodic)},
                      {AMREX_D_DECL(amrex::LinOpBCType::Periodic, amrex::LinOpBCType::Periodic, amrex::LinOpBCType::Periodic)});

    // Additional parameters for Poisson
    //m_sigma = {AMREX_D_DECL(-1., -1., -1.)};
    //linop.setSigma( m_sigma);

    // Sum of rhs needs to be 0 in domain is periodic in all directions
    if (params.geometry().isAllPeriodic())
    {
    amrex::Real rhoSum = rho.data.sum_unique(0,false,params.geometry().periodicity());
    amrex::Print().SetPrecision(17) << " sum " << rhoSum << " " << rhoSum/(64*64*64) << std::endl;
    amrex::Real Ninv = 1.0/GEMPIC_D_MULT(params.nCell()[xDir],params.nCell()[yDir],params.nCell()[zDir]);
    amrex::Real rhoSumNinv = rhoSum *Ninv;
    rho.data.plus(-rhoSumNinv,0,1);
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
    rhoSum = rho.data.sum_unique(0,false,params.geometry().periodicity());
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
    mlmg.solve({&phi.data}, {&rho.data}, relTol, absTol);
    // AMReX Poisson solver does not use Hodge. Need to rescale phi
    auto const dr = params.dr();
    phi.data.mult(1/GEMPIC_D_MULT(dr[xDir],dr[yDir],dr[zDir]));

    phi.averageSync();
    phi.fillBoundary();
}

void PoissonSolver::subtractConstantPart(Parameters params, DeRhamField<Grid::dual, Space::cell>& rho, const int nGhost)
{
    const int nComp = 1;
    // Calculates a nodal mask for rho
    std::unique_ptr<amrex::iMultiFab> nodal_Mask;
    nodal_Mask.reset(new amrex::iMultiFab(convert(params.grid(), amrex::IntVect::TheNodeVector()),
                                              params.distriMap(), nComp, nGhost));

    for (amrex::MFIter mfi(*nodal_Mask); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        amrex::IntVect hi = {bx.bigEnd()};

        amrex::Array4<int> const &mask_arr = (*nodal_Mask)[mfi].array();
        ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        // if-loop to exclude ownership for the point that is at the upper
                        // boundary for nodal directions
                        if ((i <= (hi[xDir] - 1)) && (j <= (hi[yDir] - 1)) && (k <= (hi[zDir] - 1)))
                            mask_arr(i, j, k) = 1.0;
                    });
    }

    amrex::Real nm1 = 0.0;
    int counter = 0;
    for (amrex::MFIter mfi(rho.data, true); mfi.isValid(); ++mfi)
    {
        amrex::Box const &bx = mfi.growntilebox(0);
        auto const &a = rho.data.const_array(mfi);
        amrex::Array4<int const> const &mfab = nodal_Mask->const_array(mfi);
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
    rho.data.plus(-nm1 / ((double)counter), nGhost);
}
