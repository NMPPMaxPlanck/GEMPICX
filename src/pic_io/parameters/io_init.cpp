/*****************************************************************/
/*
\brief Initialize all io streans

\author Nils Schild
\department NMPP
\email  nils.schild@ipp.mpg.de
\date   03.09.2021
*/
/*****************************************************************/

#include <string>
#include <fstream>

#include "io_init.hpp"

namespace Gempic{

std::ofstream initMasterOut(std::string fileName) {
  
  std::ofstream ofs(fileName);
  if(!ofs) 
    throw std::ios_base::failure("Could not open file " + fileName);
  

  ofs << "ToDo: Set basic output configuration (especially float precision)\n";
  ofs << "/***************************************************/\n";
  ofs << "            Welcome to your bsl6d calculation        \n";
  ofs << "/***************************************************/\n";
  ofs << "\n";
  ofs << "\n";
  ofs << "/***************************/\n";
  ofs << " General runtime informations: \n";
  ofs << " Add loaded libraries; add git hash and git tag; add run command\n";
  ofs.flush();

  return ofs;

}

}
