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
  void createBSL6DConfigStructure(ParamBSL6D const param,
                                  bsl6d::ConfigBSL6D& config, 
                                  std::ofstream& ofs);
}
