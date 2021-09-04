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
#include "GEMPIC_parameter_keys.hpp"

#include <fstream>
#include "io_param.hpp"

#include <AMReX.H>
#include <AMReX_ParmParse.H>

namespace io{
  
void ParamGempic::getDataFromConfigFile( ){
  
  /***************************************************************/
  /* Get Data from config file */
  amrex::ParmParse pp;
	//  IConfFile readConfig(fileName,getLoggedBSL6DParams());

  /***************************************************************/
  /* Store data into parameters */
  /* Propagator parameters */
  pp.query("n_steps", n_steps);


}

void ParamGempic::printParams(std::ofstream& ofs) {

  /***************************************************************/
  ofs << "/***************************************************/\n";
  ofs << " Parameters in configuration file: \n";
  ofs << "\n";
  ofs << "  Propagator parameters:\n";
  ofs << "   - n_steps: " << n_steps << "\n";
  ofs << "\n";
  ofs << "\n";
  ofs.flush();

}

}
