#include <GEMPIC_Fields.H>
#include <GEMPIC_parameters.H>
#include <GEMPIC_FDDeRhamComplex.H>

using namespace GEMPIC_Fields;
using namespace GEMPIC_FDDeRhamComplex;

int main (int argc, char *argv[]) 
{
    const bool build_parm_parse = true;
	amrex::Initialize(argc, argv, build_parm_parse, MPI_COMM_WORLD); 
    Parameters parameters{};
{

    //const amrex::RealBox realBox({AMREX_D_DECL(-M_PI,-M_PI,-M_PI)},{AMREX_D_DECL(M_PI, M_PI, M_PI)});
    const amrex::Vector<amrex::Real> domain_lo{AMREX_D_DECL(-M_PI, -M_PI, -M_PI)};
    const amrex::Vector<amrex::Real> k{AMREX_D_DECL(1.0, 1.0, 1.0)};
	const amrex::Vector<int> nCell{AMREX_D_DECL(8, 8, 8)};
    const amrex::Vector<int> maxGridSize{AMREX_D_DECL(4, 4, 4)};
    const amrex::Vector<int> isPeriodic{AMREX_D_DECL(1, 1, 1)};
    const int hodgeDegree = 2;
    const int maxSplineDegree = 1;

    const amrex::Array<std::string, 3> analyticalE = {"0.0", 
                                                      "0.0",
                                                      "cos (x)"};
    parameters.set("domain_lo", domain_lo);
    parameters.set("k", k);
    parameters.set("n_cell_vector", nCell);
    parameters.set("max_grid_size_vector", maxGridSize);
    parameters.set("is_periodic_vector", isPeriodic);
	const int nVar = GEMPIC_SPACEDIM;
    amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> func; 
    amrex::Array<amrex::Parser, GEMPIC_SPACEDIM> parser;
    
    for (int comp{0}; comp < GEMPIC_SPACEDIM; ++comp)
    {
        parser[comp].define(analyticalE[comp]);
        parser[comp].registerVariables({AMREX_D_DECL("x", "y", "z")});
        func[comp] = parser[comp].compile<nVar>();
    }

    // Initialize computational_domain
    Gempic::CompDom::computational_domain infra;

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, maxSplineDegree, HodgeScheme::FDHodge);

	DeRhamField<Grid::primal, Space::edge> field(deRham);

    // Write contents of field
    for (int comp = 0; comp < GEMPIC_SPACEDIM; ++comp)
    {
        for (amrex::MFIter mfi(field.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            const amrex::IntVect lbound = bx.smallEnd();
            const amrex::IntVect ubound = bx.bigEnd();
            amrex::Array4<amrex::Real> const &oneForm = (field.data[comp])[mfi].array();

            const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> dr = infra.dx;
            const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = infra.geom.ProbLoArray(); // Put ProbLo in parameters as a getter. Domainbounds...

            // Do the actual writing here
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {

                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r =
                {
                 AMREX_D_DECL(r0[xDir] + i*dr[xDir],
                 r0[yDir] + j*dr[yDir],
                 r0[zDir] + k*dr[zDir])
                };

                oneForm(i, j, k) = func[comp](AMREX_D_DECL(r[xDir], r[yDir], r[zDir]));
            });

            // Modify MF boundaries to see whether the value is averaged
            // There are 4 values intersecting 1, 1, 1.3, 1.7 (the 5th 4,0,0 does not refer to the same centeredness
            // => oneForm(4, 0, 0)[2] will be rounded to (1.3 + 1.7 + 1 + 1) / 4
            if (comp == zDir && mfi.index() == 0)
                oneForm(4, 0, 0) = 1.3;
            if (comp == zDir && mfi.index() == 1)
                oneForm(4, 0, 0) = 1.7;
        }
    }

    field.averageSync();

    // Calculate the averages manually
    const amrex::Real aux = (1.0 + 1.0 + 1.3 + 1.7)/4;
    bool passed = true;
    for (amrex::MFIter mfi(field.data[zDir]); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        const auto lo = lbound(bx);
        const auto hi = ubound(bx);
        passed = (aux == (field.data[zDir])[0].array()(4,0,0)) && (aux == field.data[zDir][1].array()(4,0,0));
    }

    if (passed == true)
    {
        amrex::PrintToFile("test_boundaries_sync.output") << std::endl;
        amrex::PrintToFile("test_boundaries_sync.output") << true << std::endl;
    }
    else
    {
        amrex::PrintToFile("test_boundaries_sync.output") << std::endl;
        amrex::PrintToFile("test_boundaries_sync.output") << false << std::endl;
    }


    if (amrex::ParallelDescriptor::MyProc() == 0)
        std::rename("test_boundaries_sync.output.0", "test_boundaries_sync.output");
    amrex::Print() << "IOProcessorNumber " << amrex::ParallelDescriptor::IOProcessorNumber() << std::endl;
}    
    amrex::Finalize();
}

