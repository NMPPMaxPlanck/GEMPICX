#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

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
    
    pp.get("function_sin(x)", str_function);
    
    std::unique_ptr<ParserWrapper<1> > function_parser;
    //function_parser.reset(new ParserWrapper<1>(makeParser(str_function,{"x"})));
    
    //---------------------------------------------------------------------------
    
    ofs.close();
    
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main();

    amrex::Finalize();
}



