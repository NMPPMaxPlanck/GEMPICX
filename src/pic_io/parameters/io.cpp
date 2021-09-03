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

void createBSL6DConfigStructure(ParamBSL6D const param,
                                bsl6d::ConfigBSL6D& config, 
                                std::ofstream& ofs){
  
  int err = 0;
  int warn = 0;

  ofs << "/***************************************************/\n";
  ofs << "            Create Configuration Structure           \n";
  ofs << "/***************************************************/\n";
  ofs << "\n";
  ofs << "/***************************/\n";
  ofs << " Check Test Strucutre: \n";
  ofs << "   - Parameter testKey: " << param.testKey << "\n";
  if(param.testKey<0){
    ofs << "     ERROR:\n";
    ofs << "     testKey has to be at least zero! \n";
    err++;
  } else if(param.testKey == 0) {
    config.test.testKey = param.testKey;
    ofs << "       -> Use param testKey \n";
    ofs << "     WARNING:\n";
    ofs << "     testKey is recommended to be larger than zero! \n";
    warn++;
  } else {
    config.test.testKey = param.testKey;
    ofs << "       -> Use param testKey \n";
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
    throw std::invalid_argument(std::to_string(err)+" parameters in the config file do not contain valid values. Details are listed in 'bsl6d.out'.\n");
  }
  ofs.flush();

};

}
