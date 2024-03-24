#include "history/value.h"
#include <iostream>
#include <sstream>
#include <functional>

std::string organisation::history::value::serialise()
{
    std::string result("H ");

    result += position.serialise() + " ";
    result += data.serialise() + " ";
    result += std::to_string(sequence) + " ";
    result += std::to_string(client) + " ";
    result += std::to_string(epoch);
    result += "\n";

    return result;
}

void organisation::history::value::deserialise(std::string source)
{
    std::stringstream ss(source);
    std::string str;

    value value;
    int index = 0;

    while(std::getline(ss,str,' '))
    {        
        if(index == 0)
        {
            if(str.compare("H")!=0) return;    
            value.clear();
        }
        else if(index == 1)
        {
            value.position.deserialise(str);            
        }        
        else if(index == 2)
        {
            value.data.deserialise(str);
        }
        else if(index == 3)        
        {
            value.sequence = std::atoi(str.c_str());         
        }
        else if(index == 4)        
        {
            value.client = std::atoi(str.c_str());
        }
        else if(index == 5)
        {
            value.epoch = std::atoi(str.c_str());
            *this = value;
        }        

        ++index;
    };
}
