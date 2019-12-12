#include <AMReX.H>
#include <AMReX_MLNodeLaplacian.H>
#include <AMReX_MLMG.H>

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    {
        // ----------------------------------------------------------------------------------------------------
        // Variables
        int max_level = 1;
            int ref_ratio = 2;
            int n_cell[3] = {512, 32, 32};
            int max_grid_size = 64;

            bool composite_solve = true;

            // For MLMG solver
            int verbose = 2;
            int bottom_verbose = 0;
            int max_iter = 100;
            int max_fmg_iter = 0;
            amrex::Real reltol = 1.e-11;

            amrex::Vector<amrex::Geometry> geom;
            amrex::Vector<amrex::BoxArray> grids;
            amrex::Vector<amrex::DistributionMapping> dmap;

            amrex::Vector<amrex::MultiFab> solution;
            amrex::Vector<amrex::MultiFab> rhs;
            amrex::Vector<amrex::MultiFab> exact_solution;
            amrex::Vector<amrex::MultiFab> sigma;

        // ----------------------------------------------------------------------------------------------------
        // Init Data

        int nlevels = max_level + 1;
            geom.resize(nlevels);
            grids.resize(nlevels);
            dmap.resize(nlevels);

            solution.resize(nlevels);
            rhs.resize(nlevels);
            exact_solution.resize(nlevels);
            sigma.resize(nlevels);

            RealBox rb({AMREX_D_DECL(0.,0.,0.)}, {AMREX_D_DECL(1.,1.,1.)});
            Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0,0,0)};
            Geometry::Setup(&rb, 0, is_periodic.data());
            Box domain0(IntVect{AMREX_D_DECL(0,0,0)}, IntVect{AMREX_D_DECL(n_cell[0]-1,n_cell[1]-1,n_cell[2]-1)});
            Box domain = domain0;
            for (int ilev = 0; ilev < nlevels; ++ilev)
            {
                geom[ilev].define(domain);
                domain.refine(ref_ratio);
            }

            domain = domain0;
            for (int ilev = 0; ilev < nlevels; ++ilev)
            {
                grids[ilev].define(domain);
                grids[ilev].maxSize(max_grid_size);

                // AMREX-code (only makes sense if n_cell equal in all 3 directions)
                //domain.grow(-n_cell[0]/4);   // fine level cover the middle of the coarse domain
                // substituted by:
                domain.growLo(0, -n_cell[0]/4); //idir, n_cell
                domain.growHi(0, -n_cell[0]/4);
                domain.growLo(1, -n_cell[1]/4);
                domain.growHi(1, -n_cell[1]/4);
                domain.growLo(2, -n_cell[2]/4);
                domain.growHi(2, -n_cell[2]/4);

                domain.refine(ref_ratio);
            }

            for (int ilev = 0; ilev < nlevels; ++ilev)
            {
                dmap[ilev].define(grids[ilev]);
                const BoxArray& nba = amrex::convert(grids[ilev],IntVect::TheNodeVector());
                // These are nodal
                solution      [ilev].define(nba        , dmap[ilev], 1, 0);
                rhs           [ilev].define(nba        , dmap[ilev], 1, 0);
                exact_solution[ilev].define(nba        , dmap[ilev], 1, 0);
                // sigma is cell-centered.
                sigma         [ilev].define(grids[ilev], dmap[ilev], 1, 0);

                const auto dx = geom[ilev].CellSizeArray();

        #ifdef _OPENMP
        #pragma omp parallel if (Gpu::notInLaunchRegion())
        #endif
                for (MFIter mfi(rhs[ilev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    const Box& bx = mfi.tilebox();
                    Array4<Real> const phi = exact_solution[ilev].array(mfi);
                    Array4<Real> const rh  = rhs[ilev].array(mfi);
                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        constexpr Real pi = 3.1415926535897932;
                        constexpr Real tpi = 2.*pi;
                        constexpr Real fpi = 4.*pi;
                        constexpr Real fac = tpi*tpi*AMREX_SPACEDIM;

                        Real x = i*dx[0];
                        Real y = j*dx[1];
                        Real z = k*dx[2];

                        phi(i,j,k) = (std::cos(tpi*x) * std::cos(tpi*y) * std::cos(tpi*z))
                            + 0.25 * (std::cos(fpi*x) * std::cos(fpi*y) * std::cos(fpi*z));

                        rh(i,j,k) = -fac * (std::cos(tpi*x) * std::cos(tpi*y) * std::cos(tpi*z))
                            -        fac * (std::cos(fpi*x) * std::cos(fpi*y) * std::cos(fpi*z));
                    });
                }

                sigma[ilev].setVal(1.0);
            }

            //--------------------------------------------------------------------------------------------
            // Solve

            if (composite_solve)
                {
                    MLNodeLaplacian linop(geom, grids, dmap);

                    linop.setDomainBC({AMREX_D_DECL(LinOpBCType::Dirichlet,
                                                    LinOpBCType::Dirichlet,
                                                    LinOpBCType::Dirichlet)},
                                      {AMREX_D_DECL(LinOpBCType::Dirichlet,
                                                    LinOpBCType::Dirichlet,
                                                    LinOpBCType::Dirichlet)});

                    for (int ilev = 0; ilev <= max_level; ++ilev) {
                        linop.setSigma(ilev, sigma[ilev]);
                    }

                    MLMG mlmg(linop);
                    mlmg.setMaxIter(max_iter);
                    mlmg.setMaxFmgIter(max_fmg_iter);
                    mlmg.setVerbose(verbose);
                    mlmg.setBottomVerbose(bottom_verbose);

                    // solution is passed to MLMG::solve to provide an initial guess.
                    // Additionally it also provides boundary conditions for Dirichlet
                    // boundaries if there are any.
                    for (int ilev = 0; ilev <= max_level; ++ilev) {
                        //MultiFab::Copy(solution[ilev], exact_solution[ilev], 0, 0, 1, 0);
                        const Box& interior = amrex::surroundingNodes(
                            amrex::grow(geom[ilev].Domain(), -1));
                        // Usually we want the best initial guess.  For testing here,
                        // we set the domain boundaries to exact solution and zero out
                        // the interior.
                        solution[ilev].setVal(0.0, interior, 0, 1, 0);
                    }

                    mlmg.solve(GetVecOfPtrs(solution), GetVecOfConstPtrs(rhs), reltol, 0.0);
                }

    }

    amrex::Finalize();
}



