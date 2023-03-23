#include <AMReX.H>
#include <GEMPIC_parameters.H>
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_gempic_norm.H>
#include <GEMPIC_particle_mesh_coupling_C2.H>
#include <gtest/gtest.h>

//Basics first
namespace {
    template <int degree>
    void projection(amrex::Real (*function_to_project)(amrex::Real, amrex::Real, amrex::Real,
                                                       amrex::Real),
                    amrex::Real t, computational_domain ifr,
                    amrex::GpuArray<bool, GEMPIC_SPACEDIM> integrate, amrex::IndexType IndexVect,
                    amrex::MultiFab &MultiF)
    {
        std::vector<amrex::Real> stencil;
        switch (degree)
        {
            case 2:
                stencil = {1.0};
                break;
            case 4:
                stencil = {1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0};
                break;
            case 6:
                stencil = {1.0 / 12.0, 4.0 / 12.0, 2.0 / 12.0, 4.0 / 12.0, 1.0 / 12.0};
                break;
            default:
                amrex::Print() << "degree not implemented" << std::endl;
        }

        int x_len[GEMPIC_SPACEDIM];
        for (int i = 0; i < GEMPIC_SPACEDIM; i++) x_len[i] = (integrate[i] ? (degree - 1) : 1);
        for (amrex::MFIter mfi(MultiF); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.tilebox();
            amrex::Array4<amrex::Real> const &proj_arr = MultiF[mfi].array();
            // ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            amrex::Dim3 lo = lbound(bx);
            amrex::Dim3 hi = ubound(bx);
            for (int k = lo.z; k <= hi.z; ++k)
            {
                for (int j = lo.y; j <= hi.y; ++j)
                {
                    for (int i = lo.x; i <= hi.x; ++i)
                    {
                        double x_eval = 0, y_eval = 0, z_eval = 0;

                        amrex::GpuArray<amrex::Real, degree - 1> x;
                        amrex::Real x_midpoint;
                        amrex::GpuArray<amrex::Real, degree - 1> y;
                        amrex::Real y_midpoint;
#if (GEMPIC_SPACEDIM > 2)
                        amrex::GpuArray<amrex::Real, degree - 1> z;
                        amrex::Real z_midpoint;

                        z_midpoint =
                            ifr.plo[2] + ((double)k + 0.5 - 0.5 * (IndexVect)[2]) * ifr.dx[2];
                        for (int d = 0; d < x_len[2]; d++)
                            z[d] = z_midpoint + (d - ((x_len[2] + 1) / 2.0 - 1)) * ifr.dx[0] /
                                                    pow(2.0, (x_len[2] + 1) / 2.0 - 1);
#endif
                        y_midpoint =
                            ifr.plo[1] + ((double)j + 0.5 - 0.5 * (IndexVect)[1]) * ifr.dx[1];
                        for (int d = 0; d < x_len[1]; d++)
                            y[d] = y_midpoint + (d - ((x_len[1] + 1) / 2.0 - 1)) * ifr.dx[0] /
                                                    pow(2.0, (x_len[1] + 1) / 2.0 - 1);

                        x_midpoint =
                            ifr.plo[0] + ((double)i + 0.5 - 0.5 * (IndexVect)[0]) * ifr.dx[0];
                        for (int d = 0; d < x_len[0]; d++)
                            x[d] = x_midpoint + (d - ((x_len[0] + 1) / 2.0 - 1)) * ifr.dx[0] /
                                                    pow(2.0, (x_len[0] + 1) / 2.0 - 1);

                        amrex::Real val = 0.0;
#if (GEMPIC_SPACEDIM > 2)
                        for (int zd = 0; zd < x_len[2]; zd++)
#endif
                            for (int yd = 0; yd < x_len[1]; yd++)
                            {
                                for (int xd = 0; xd < x_len[0]; xd++)
                                {
                                    x_eval = x[xd];
                                    y_eval = y[yd];
#if (GEMPIC_SPACEDIM > 2)
                                    z_eval = z[zd];
#endif
                                    val += (integrate[0] ? stencil[xd] : 1) *
                                           (integrate[1] ? stencil[yd] : 1) *
#if (GEMPIC_SPACEDIM > 2)
                                           (integrate[2] ? stencil[zd] : 1) *
#endif
                                           (function_to_project(x_eval, y_eval, z_eval, t));
                                    ;
                                }
                            }
                        proj_arr(i, j, k) = val;
                    }
                }
            }
        }
        MultiF.FillBoundary(ifr.geom.periodicity());
    }

    // wave function
    AMREX_GPU_HOST_DEVICE amrex::Real funct_rho(amrex::Real x, amrex::Real y, amrex::Real z,
                                                amrex::Real t)
    {
        amrex::Real val = 0.0;
        return val;
    }
    // Test fixture
    class DepositRhoTest : public testing::Test {
        protected:

        static const int degx = 0;
        static const int degy = 0;
        static const int degz = 0;
        static const int numspec = 1;
        static const int vdim = 3;
        static const int degree = 2;

        double charge = 1;
        double mass = 1;

        computational_domain infra;
        amrex::GpuArray<std::unique_ptr<particle_groups<vdim>>, numspec> part_gr;
        amrex::MultiFab rho; 

        // virtual void SetUp() will be called before each test is run.
        void SetUp() override {
            /* Initialize the infrastructure */
            const amrex::RealBox realBox({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                            {AMREX_D_DECL(10.0, 10.0, 10.0)});
            const amrex::IntVect nCell = {AMREX_D_DECL(10, 10, 10)};
            const amrex::IntVect maxGridSize = {AMREX_D_DECL(10, 10, 10)};
            const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic = {1, 1, 1};

            // This class does the same as GEMPIC_parameters.H and needs to be urgently redesigned.
            // {1, 1, 1} represents periodicity, has different types than Params and gempic_parameters.
            infra.initialize_computational_domain(nCell, maxGridSize, {1, 1, 1}, realBox);

            projection<degree>(funct_rho, 0.0, infra, {AMREX_D_DECL(true, true, true)},
                                amrex::IndexType(amrex::IntVect::TheNodeVector()), rho);
            
            // particles
            for (int spec = 0; spec < numspec; spec++)
            {
                part_gr[spec] =
                    std::make_unique<particle_groups<vdim>>(charge, mass, infra);
            }

        }
    };

    TEST_F(DepositRhoTest, BasicAssertions) {
        // Adding particle to one cell
        for (amrex::MFIter mfi = part_gr[0]->MakeMFIter(0); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.validbox();
            amrex::IntVect lo = {bx.smallEnd()};
            if (lo[0] != infra.geom.ProbLo()[0] || lo[1] != infra.geom.ProbLo()[1] || lo[2] != infra.geom.ProbLo()[2]) continue;
            amrex::ParticleTile<0, 0, vdim + 1, 0>& particles =
                part_gr[0]->GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
            part_gr[0]->add_particle({AMREX_D_DECL(infra.geom.ProbLo()[0] + 0.5*infra.dx[0],
                                                   infra.geom.ProbLo()[1] + 0.5*infra.dx[1],
                                                   infra.geom.ProbLo()[2] + 0.5*infra.dx[2])},
                                    {0,0,0}, 1, particles);
        }
        EXPECT_EQ(0,gempic_norm(rho, infra, 2));
        rho.setVal(0.0);

        EXPECT_EQ(0,gempic_norm(rho, infra, 2));
        part_gr[0]->Redistribute();  // assign particles to the tile they are in
        // Particle iteration ... over one particle. Hopefully.
        for (amrex::ParIter<0, 0, vdim + 1, 0> pti(*part_gr[0], 0); pti.isValid(); ++pti)
        {
            const auto& particles = pti.GetArrayOfStructs();
            const auto partData = particles().data();

            amrex::Array4<amrex::Real> const& rhoarr = rho[pti].array();
            splines_at_particles<degx, degy, degz> spline;
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
            for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                position[d] = partData[0].pos(d);
            spline.init_particles(position, infra.plo, infra.dxi);
            // Needs at least max(degx, degy, degz) ghost cells
            gempic_deposit_rho_C3<degx, degy, degz>(
                spline, 0*infra.dxi[GEMPIC_SPACEDIM],
                rhoarr);
        }

        EXPECT_EQ(0,gempic_norm(rho, infra, 2));
    }
}