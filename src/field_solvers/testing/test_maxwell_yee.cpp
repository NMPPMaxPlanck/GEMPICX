#include <AMReX.H>
#include <AMReX_Print.H>

int main(int argc, char* argv[])
{
    std::cout << "Hello" << std::endl;
    amrex::Initialize(argc,argv);
    amrex::Print() << "Hello world from AMReX version "
                   << amrex::Version() << "\n";
    amrex::Finalize();
}

