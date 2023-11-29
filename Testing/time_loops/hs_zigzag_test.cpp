#include <AMReX.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_FDDeRhamComplex.H>
#include <GEMPIC_Fields.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_mesh_coupling.H>
#include <GEMPIC_hs_zigzag.H>
#include <GEMPIC_Spline_Class.H>
#include "gtest/gtest.h"
#include "test_utils/GEMPIC_test_utils.H"
#include "gmock/gmock.h"

#include "GEMPIC_Splitting.H"

using namespace Particles;
using namespace GEMPIC_FDDeRhamComplex;
using namespace GEMPIC_Fields;
using namespace Time_Loop;

using ::testing::Mock;
using ::testing::Exactly;
using ::testing::_;

namespace {
    template<int vDim, int numspec, int degX, int degY, int degZ, int degmw, int ndata>
    class MockHSZigZagC2 : public Time_Loop::HSZigZagC2<vDim, numspec, degX, degY, degZ, degmw, ndata> {
        public:
    };

    template<int degX, int degY, int degZ>
    class MockSpline : public Spline::SplineBase<degX, degY, degZ> {
        public:
        MockSpline(amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const &position,
                   amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const &plo,
                   amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM + 1> const &dxInverse) :
                   Spline::SplineBase<degX, degY, degZ>(position, plo, dxInverse) {}

        template<Field form, unsigned int vDim>
        AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, vDim> evalSplineField (const amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> fieldArray) const
        {
            amrex::GpuArray<amrex::Real, vDim> fields;
            for (int comp = 0; comp < vDim; comp++)
            {
                fields[comp] = 0.;
            }

            return fields;
        }
    };

    class HSZigZagC2Test : public testing::Test {
        protected:

        static const int degX{1};
        static const int degY{1};
        static const int degZ{1};
        static const int degmw{2};

        static const int numSpec{1};
        static const int vDim{3};
        static const int spec{0};
        static const int nData{1};
        const int Nghost{GEMPIC_TestUtils::initNGhost(degX, degY, degZ)};
        Parameters params;

        double charge{1};
        double mass{1};

        computational_domain infra{false}; // "uninitialized" computational domain
        amrex::GpuArray<std::unique_ptr<particle_groups<vDim>>, numSpec> particleGroup;

        MockHSZigZagC2<vDim, numSpec, degX, degY, degZ, degmw, nData> mockHSZigZagC2;

        static void SetUpTestSuite()
        {
            GTEST_SKIP() << "hs_zigzag tests not implemented";
            /* Initialize the infrastructure */
            //const amrex::RealBox realBox({AMREX_D_DECL(0.0, 0.0, 0.0)},
            //                             {AMREX_D_DECL(10.0, 10.0, 10.0)});
            amrex::Vector<amrex::Real> domain_lo{AMREX_D_DECL(0.0, 0.0, 0.0)};
            // 
            amrex::Vector<amrex::Real> k{AMREX_D_DECL(0.2*M_PI, 0.2*M_PI, 0.2*M_PI)};
            const amrex::Vector<int> nCell{AMREX_D_DECL(10, 10, 10)};
            const amrex::Vector<int> maxGridSize{AMREX_D_DECL(10, 10, 10)};
            const amrex::Vector<int> isPeriodic{AMREX_D_DECL(1, 1, 1)};


            amrex::ParmParse pp;
            pp.addarr("domain_lo", domain_lo);
            pp.addarr("k", k);
            pp.addarr("n_cell_vector", nCell);
            pp.addarr("max_grid_size_vector", maxGridSize);
            pp.addarr("is_periodic_vector", isPeriodic);

            // particle settings
            double charge{1};
            double mass{1};

            pp.add("particle.species0.charge", charge);
            pp.add("particle.species0.mass", mass);
        }

        // virtual void SetUp() will be called before each test is run.
        void SetUp() override {
            /* Initialize the infrastructure */
            infra = computational_domain{};

            // Initialize the De Rham Complex
            deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, maxSplineDegree);

            // particles
            for (int spec{0}; spec < numSpec; spec++)
            {
                particleGroup[spec] =
                    std::make_unique<particle_groups<vDim>>(spec, infra);
            }

        }
    };
}

namespace Gempic::Particles {
    // You cannot do partial template specialization for functions, so here is an explicit specialization for a special case
    template <>
    AMREX_GPU_HOST_DEVICE void accumulate_J_integrate_B<xDir, MockSpline<1, 1, 1>,4>(
        MockSpline<1, 1, 1> &spline,
        amrex::Real weight,
        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const dx,
        amrex::GpuArray<amrex::Array4<amrex::Real>, int(4 / 2.5) * 2 + 1> const &bArray,
        amrex::GpuArray<amrex::Array4<amrex::Real>, 4> const &jArray,
        amrex::GpuArray<amrex::Real, 2> &fields)
    {
        
    }
}

namespace {
    TEST_F(HSZigZagC2Test , ApplyHEParticleTest) {
        // Adding particle to one cell
        const int numParticles{1};
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{*infra.geom.ProbLo()};
        amrex::Array<amrex::Real, numParticles> weights{1};
        GEMPIC_TestUtils::addSingleParticles(particleGroup, infra, weights, positions);

        // (default) charge correctly transferred from addSingleParticles
        EXPECT_EQ(1, particleGroup[0]->getCharge());

        // Parse analytical fields and initialize parserEval. Has to be the same as Bx,By,Bz and Ex,
        // Ey, Ez
        const amrex::Array<std::string, 3> analyticalFuncE{"0.0", "0.0", "0.0"};

        const int nVar{GEMPIC_SPACEDIM + 1};  // x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, 3> funcE;
        amrex::Array<amrex::Parser, 3> parser;

        for (int i{0}; i < 3; ++i)
        {
            parser[i].define(analyticalFuncE[i]);
            parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            funcE[i] = parser[i].compile<nVar>();
        }

        // Initialize the De Rham Complex
        auto deRham{std::make_shared<FDDeRhamComplex>(params)};

        DeRhamField<Grid::primal, Space::edge> E(deRham, funcE);

        particleGroup[0]->Redistribute();  // assign particles to the tile they are in
        // Particle iteration ... over one particle. Hopefully.

        bool particleLoopRun{false};
        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[spec], 0); pti.isValid(); ++pti)
        {
            particleLoopRun = true;

            MockSpline<degX, degY, degZ> mockSpline({AMREX_D_DECL(1., 1., 1.)}, {AMREX_D_DECL(1., 1., 1.)}, {AMREX_D_DECL(1., 1., 1.), 1.});

            const long np{pti.numParticles()};
            EXPECT_EQ(1, np); // Only one particle added by addSingleParticles

            amrex::Particle<0, 0>* AMREX_RESTRICT particles =
                &(pti.GetArrayOfStructs()[0]);

            amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> eArray;
            for (int cc{0}; cc < vDim; cc++) eArray[cc] = (E.data[cc])[pti].array();

            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
            for (unsigned int d{0}; d < GEMPIC_SPACEDIM; ++d)
                position[d] = particles[0].pos(d);

            auto particle_attributes = &pti.GetStructOfArrays();
            amrex::ParticleReal* const AMREX_RESTRICT velx =
                particle_attributes->GetRealData(0).data();
            amrex::ParticleReal* const AMREX_RESTRICT vely =
                particle_attributes->GetRealData(1).data();
            amrex::ParticleReal* const AMREX_RESTRICT velz =
                particle_attributes->GetRealData(2).data();

            amrex::GpuArray<amrex::Real, 3> efield({1, 1, 1});

            EXPECT_EQ(0, velx[0]);
            EXPECT_EQ(0, vely[0]);
            EXPECT_EQ(0, velz[0]);
        }
        ASSERT_TRUE(particleLoopRun);
    }
}