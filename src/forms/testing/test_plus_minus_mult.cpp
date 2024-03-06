#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Parameters.H"

using namespace Gempic::Forms;

int main (int argc, char *argv[])
{
    amrex::Initialize(argc, argv);
    Gempic::Io::Parameters parameters{};
    {
        // const amrex::RealBox realBox({AMREX_D_DECL(0, 0, 0)},{AMREX_D_DECL( 10, 10, 10)});
        const amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
        const amrex::Vector<amrex::Real> k{AMREX_D_DECL(0.2 * M_PI, 0.2 * M_PI, 0.2 * M_PI)};
        const amrex::Vector<int> nCell{AMREX_D_DECL(10, 10, 10)};
        const amrex::Vector<int> maxGridSize{AMREX_D_DECL(10, 10, 10)};
        const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic{AMREX_D_DECL(1, 1, 1)};
        const int hodgeDegree = 2;
        const int maxSplineDegree = 1;

        parameters.set("domain_lo", domainLo);
        parameters.set("k", k);
        parameters.set("n_cell_vector", nCell);
        parameters.set("max_grid_size_vector", maxGridSize);
        parameters.set("is_periodic_vector", isPeriodic);

        // Initialize computational_domain
        Gempic::ComputationalDomain infra;

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, maxSplineDegree,
                                                        HodgeScheme::FDHodge);

        // Declare the fields
        DeRhamField<Grid::primal, Space::edge> e1(deRham);
        DeRhamField<Grid::primal, Space::edge> e2(deRham);
        DeRhamField<Grid::primal, Space::edge> e3(deRham);

        const amrex::Array<std::string, 3> analyticalFuncE1 = {
            "x - y - z",
            "x - y - z",
            "x - y - z",
        };

        const amrex::Array<std::string, 3> analyticalFuncE2 = {
            "x - y - z",
            "x - y - z",
            "x - y - z",
        };

        const amrex::Array<std::string, 3> analyticalFuncE3 = {
            "1.0",
            "1.0",
            "1.0",
        };

        const int nVar = GEMPIC_SPACEDIM + 1;  // x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, 3> func;
        amrex::Array<amrex::Parser, 3> parser;
        for (int i = 0; i < 3; ++i)
        {
            parser[i].define(analyticalFuncE1[i]);
            parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            func[i] = parser[i].compile<nVar>();
        }

        deRham->projection(func, 0.0, e1);

        for (int i = 0; i < 3; ++i)
        {
            parser[i].define(analyticalFuncE2[i]);
            parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            func[i] = parser[i].compile<nVar>();
        }

        deRham->projection(func, 0.0, e2);

        for (int i = 0; i < 3; ++i)
        {
            parser[i].define(analyticalFuncE3[i]);
            parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            func[i] = parser[i].compile<nVar>();
        }

        deRham->projection(func, 0.0, e3);

        amrex::Real scalar = 2.;

        e2 += e1;
        e3 -= e1;
        e2 *= scalar;

        // Visualize fields
        for (int comp = 0; comp < 3; ++comp)
        {
            for (amrex::MFIter mfi(e1.m_data[comp]); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();
                const auto lo = lbound(bx);
                const auto hi = ubound(bx);

                amrex::Array4<amrex::Real> const &e1Mf = (e1.m_data[comp])[mfi].array();
                amrex::Array4<amrex::Real> const &e2Mf = (e2.m_data[comp])[mfi].array();
                amrex::Array4<amrex::Real> const &e3Mf = (e3.m_data[comp])[mfi].array();

                GEMPIC_D_LOOP_BEGIN(for (int i = lo.x; i <= hi.x; ++i),
                                    for (int j = lo.y; j <= hi.y; ++j),
                                    for (int k = lo.z; k <= hi.z; ++k))
#if (GEMPIC_SPACEDIM == 3)
                    amrex::Print()
                        << "(" << i << "," << j << "," << k << ") E1: " << e1Mf(i, j, k)
                        << " E2: " << e2Mf(i, j, k) << " E3: " << e3Mf(i, j, k) << std::endl;
#endif
                GEMPIC_D_LOOP_END
            }
        }
    }
    amrex::Finalize();
}
