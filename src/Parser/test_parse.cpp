#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include <GpuParser.H>
#include <WarpXParser.H>
#include <WarpXParserWrapper.H>
#include <WarpXUtil.H>

void main_main ()
{
    std::cout << "Parsing test for parsing functions: " << std::endl;
   
    std::ofstream ofs("test_parse.output", std::ofstream::out);
    amrex::Print(ofs) << "Parsing" << std::endl;
    
    //---------------------------------------------------------------------------
    amrex::ParmParse pp;
    std::string str_function;
    
    pp.get("function_sin", str_function);
    
    const int N = 3;
    std::unique_ptr<ParserWrapper<N> > function_parser;
    function_parser.reset(new ParserWrapper<N>(makeParser(str_function,{"x", "y", "z"})));
    ParserWrapper<3> *fct_par = function_parser.get();

    amrex::Real value = 0.0;
    //std::cout << "sin(" << value << ") is " << (*fct_par)(value) << std::endl;
    //---------------------------------------------------------------------------
    
    ofs.close();
    
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main();

    amrex::Finalize();
}



