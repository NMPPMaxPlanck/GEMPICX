/*****************************************************************/
/*
\brief Check correctnes of parameters from ParamBSL6D and if no 
       error can be found write the correct parameters into the 
       config structures

\author Nils Schild
\department NMPP
\email  nils.schild@ipp.mpg.de
\date   10.05.2021
*/
/*****************************************************************/

#include <string>
#include "config.hpp"


#include <fstream>
#include "io_param.hpp"

#include <stdexcept>

namespace io {

void createGempicConfigStructure(ParamGempic const param,
                                 Gempic::ConfigGempic& config,
                                 std::ofstream& ofs){
  
  int err = 0;
  int warn = 0;

  ofs << "/***************************************************/\n";
  ofs << "            Create Configuration Structure           \n";
  ofs << "/***************************************************/\n";
  ofs << "\n";
  ofs << "/***************************/\n";
  ofs << " Check propagator structure: \n";
  ofs << "   - Parameter n_steps: " << param.n_steps << "\n";
  if(param.n_steps<0){
    ofs << "     ERROR:\n";
    ofs << "     n_steps has to be at least zero! \n";
    err++;
  } else if(param.n_steps == 0) {
    config.propagator.n_steps = param.n_steps;
    ofs << "       -> Use param n_steps \n";
    ofs << "     WARNING:\n";
    ofs << "     n_steps is zero and only initilization is done! \n";
    warn++;
  } else {
    config.propagator.n_steps = param.n_steps;
    ofs << "       -> Use param n_steps \n";
  }
  ofs << "\n";
  ofs << "\n";
  
  if(warn>0) {

    ofs << "/***************************************************/\n";
    ofs << "            WARNINGS                                 \n";
    ofs << "/***************************************************/\n";
    ofs << " " + std::to_string(warn) +" configuration warnings!\n";
    ofs << " Parameters do not contain optimal arguments. Warnings\n";
    ofs << " are listed in the 'Create Configuration Structure'";
    ofs << " section!";
    ofs << "\n";
    ofs << "\n";
  }
  if(err>0) {
    ofs << "/***************************************************/\n";
    ofs << "            ERRORS                                   \n";
    ofs << "/***************************************************/\n";
    ofs << " " + std::to_string(err) +" configuration errors!\n";
    ofs << " Parameters do not contain valid arguments. Errors are \n";
    ofs << " listed in the 'Create Configuration Structure' section! \n";
    ofs << "\n";
    ofs << "\n";
    ofs.flush();
    throw std::invalid_argument(std::to_string(err)+" parameters in the config file do not contain valid values. Details are listed in 'gempic.out'.\n");
  }
  ofs.flush();

};

}
