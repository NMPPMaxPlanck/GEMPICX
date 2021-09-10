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
  void create_gempic_param_config_structure(param_gempic const param,
                                   Gempic::gempic_param_config& config,
                                   std::ofstream& ofs);
}
