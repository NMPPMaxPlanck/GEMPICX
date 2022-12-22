#include <GEMPIC_Fields.H>
#include <GEMPIC_Params.H>
#include <GEMPIC_FDDeRhamComplex.H>

using namespace GEMPIC_Fields;
using namespace GEMPIC_FDDeRhamComplex;

int main (int argc, char *argv[]) 
{
    const bool build_parm_parse = true;
	amrex::Initialize(argc, argv, build_parm_parse, MPI_COMM_WORLD); 

    const amrex::RealBox realBox({AMREX_D_DECL(-M_PI,-M_PI,-M_PI)},{AMREX_D_DECL(M_PI, M_PI, M_PI)});
	const amrex::IntVect nCell = {AMREX_D_DECL(8, 8, 8)};
    const amrex::IntVect maxGridSize = {AMREX_D_DECL(4, 4, 4)};
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic = {1, 1, 1};
    const int degree = 2;

    const amrex::Array<std::string, 3> analyticalE = {"0.0", 
                                                      "0.0",
                                                      "cos (x)"};
	const int nVar = 3;
    amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> func; 
    amrex::Parser parser;
    
    parser.define(analyticalE[0]);
    parser.define(analyticalE[1]);
    parser.define(analyticalE[2]);
    parser.registerVariables({"x"});
    func[0] = parser.compile<nVar>();
    func[1] = parser.compile<nVar>();
    func[2] = parser.compile<nVar>();

    Parameters params(realBox, nCell, maxGridSize, isPeriodic, degree);
    const amrex::Geometry geom = params.geometry();

    auto deRham = std::make_shared<FDDeRhamComplex>(params);
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

            const amrex::RealVect dr = params.dr();
            const amrex::GpuArray<amrex::Real, 3> r0 = params.geometry().ProbLoArray(); // Put ProbLo in parameters as a getter. Domainbounds...

            // Do the actual writing here
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {

                amrex::GpuArray<amrex::Real, 3> r =
                {
                 r0[0] + i*dr[0],
                 r0[1] + j*dr[1],
                 r0[2] + k*dr[2]
                };

                oneForm(i, j, k) = func[comp](r[0], r[1], r[2]);
            });

            // Modify MF boundaries to see whether the value is averaged
            // There are 4 values intersecting 1, 1, 1.3, 1.7 (the 5th 4,0,0 does not refer to the same centeredness
            // => oneForm(4, 0, 0)[2] will be rounded to (1.3 + 1.7 + 1 + 1) / 4
            if (comp == 2 && mfi.index() == 0)
                oneForm(4, 0, 0) = 1.3;
            if (comp == 2 && mfi.index() == 1)
                oneForm(4, 0, 0) = 1.7;
        }
    }

    field.averageSync();

    // Calculate the averages manually
    const amrex::Real aux = (1.0 + 1.0 + 1.3 + 1.7)/4;
    bool passed = true;
    for (amrex::MFIter mfi(field.data[2]); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        const auto lo = lbound(bx);
        const auto hi = ubound(bx);
        passed = (aux == (field.data[2])[0].array()(4,0,0)) && (aux == field.data[2][1].array()(4,0,0));
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
    
    amrex::Finalize();
}

