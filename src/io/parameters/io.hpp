/*****************************************************************/
/*
\brief Header file to io.cpp
       

\author Nils Schild
\department NMPP
\email  nils.schild@ipp.mpg.de
\date   01.09.2021
*/
/*****************************************************************/


namespace io{
  void createGempicConfigStructure(param_gempic const param,
                                   Gempic::ConfigGempic& config,
                                   std::ofstream& ofs);
}
