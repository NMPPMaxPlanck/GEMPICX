/*****************************************************************/
/*
\brief Source file to config.hpp
       Create configuration structures for different components
       of the bsl6d-Framework.

\author Nils Schild
\department NMPP
\email  nils.schild@ipp.mpg.de
\date   01.09.2021
*/
/*****************************************************************/

#include <string>
#include <fstream>

#include "GEMPIC_parameter_config.hpp"

namespace Gempic{

void gempic_param_config::print_gempic_param_config(std::ofstream& ofs){
  
  ofs << "/***************************************************/\n";
  ofs << "            Configuration structure                  \n";
  ofs << "/***************************************************/\n";
  ofs << "\n";
  ofs << "  Propagator: \n";
  ofs << "   - n_steps: " << propagator.n_steps << "\n";
  ofs << "\n";
  ofs << "\n";
 
};

}
