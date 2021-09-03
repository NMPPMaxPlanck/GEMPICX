/*****************************************************************/
/*
\brief Source file to io_param.hpp
       Read the configuration file and store read parameters into
       ParamBSL6D-Class.

\author Nils Schild
\department NMPP
\email  nils.schild@ipp.mpg.de
\date   01.09.2021
*/
/*****************************************************************/


#include <string>
#include <map>
#include "keys.hpp"

#include <fstream>
#include "io_param.hpp"

namespace io{
  
void ParamBSL6D::getDataFromConfigFile(std::string const fileName){
  
  /***************************************************************/
  /* Get Data from config file */
//  IConfFile readConfig(fileName,getLoggedBSL6DParams());

  /***************************************************************/
  /* Store data into parameters */
//  testKey = readConfig.getParam_int("testKey",0);


}

void ParamBSL6D::printParams(std::ofstream& ofs) {

  /***************************************************************/
  ofs << "/***************************************************/\n";
  ofs << " Parameters in configuration file: \n";
  ofs << "\n";
  ofs << "  Test Parameters:\n";
  ofs << "   - testKey: " << testKey << "\n";
  ofs << "\n";
  ofs << "\n";
  ofs.flush();

}

}
