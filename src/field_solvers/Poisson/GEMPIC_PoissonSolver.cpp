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
    m_maxCoarseningLevel = 30;
    m_maxIter = 1000;
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

    amrex::MLEBNodeFDLaplacian linop({params.geometry()}, {params.grid()}, {params.distriMap()}, lpInfo);

    // Set boundary conditions on linear operator for lower end and higher end
    linop.setDomainBC({AMREX_D_DECL(amrex::LinOpBCType::Periodic, amrex::LinOpBCType::Periodic, amrex::LinOpBCType::Periodic)},
                      {AMREX_D_DECL(amrex::LinOpBCType::Periodic, amrex::LinOpBCType::Periodic, amrex::LinOpBCType::Periodic)});

    // Additional parameters for Poisson
    m_sigma = {AMREX_D_DECL(-1., -1., -1.)};
    linop.setSigma(m_sigma);
    amrex::Real relTol = 1.e-11;

    // Initialize solver class
    amrex::MLMG mlmg(linop);
    
    // Configure solver class
    mlmg.setMaxIter(m_maxIter);
    mlmg.setMaxFmgIter(m_maxFmgIter);
    mlmg.setBottomMaxIter(m_mgBottomMaxIter);
    mlmg.setVerbose(m_verbose);
    mlmg.setBottomVerbose(m_bottomVerbose);
    mlmg.setBottomSolver(amrex::BottomSolver::cg);

    // Do we need to subtract the constant part if we have averageSync ?
    subtractConstantPart(params, rho, 4);
    
    // Solve Poisson
    mlmg.solve({&phi.data}, {&rho.data}, relTol, 0.0);

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
                        // boundary for nodal dimensions
                        if ((i <= (hi[0] - 1)) && (j <= (hi[1] - 1)) && (k <= (hi[2] - 1)))
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
