/*****************************************************************/
/*
\brief This source-file contains all keys that can be used on the
       BSL6D-Code.
       The different parameters are logged in a map. The key of the
       is the identifier of the parameter.
       The key points to a string which contains a description of
       the parameter which contains the following information:
       "<Data-Type>:<Default-Value>:<Level>:<Description>"
       
       -<Data-Type>:
          int,double,std::string
       -<Default-Value>: 
          All valid values for int, double, std::string
       -<Level>:
          B - Basic
          I - Intermediate
          E - Expert
       -<Description>:
          Short description of the functionality of the parameter.
       

\author Nils Schild
\department NMPP
\email  nils.schild@ipp.mpg.de
\date   31.08.2021
*/
/*****************************************************************/
#include <map>
#include <string>
#include "keys.hpp"

namespace io {

  std::map<std::string,std::string> loggedParams {
    {"testKey","0:E:Integer:This keyword just tests the functionality of IConfigFile in io_file.cpp"}
  };


  std::map<std::string,std::string> getLoggedBSL6DParams(){
    return loggedParams;
  };

}

