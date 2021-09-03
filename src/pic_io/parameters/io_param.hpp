/*****************************************************************/
/*
\brief Header file to io_param.hpp
       Read the configuration file and store read parameters into
       ParamBSL6D-Class.

\author Nils Schild
\department NMPP
\email  nils.schild@ipp.mpg.de
\date   01.09.2021
*/
/*****************************************************************/
#ifndef IO_PARAM_H
#define IO_PARAM_H

namespace io{

class ParamGempic{
  
  public:
    ParamGempic() = default;
    ~ParamGempic() = default;
    void getDataFromConfigFile();
    void printParams(std::ofstream& ofs);

    /*************************************************************/
    /*
    Section with read parameters
    */
    /*************************************************************/

    /* Propagator parameters */
    int n_steps = 1;

};

}
#endif /*IO_PARAM_H*/
