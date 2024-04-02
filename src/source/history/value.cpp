#include "history/value.h"
#include <iostream>
#include <sstream>
#include <functional>

std::string organisation::history::value::serialise()
{
    std::string result("H ");

    result += std::to_string((stationary == true) ? 1 : 0) + " Pos=";
    result += position.serialise() + " Data=";
    result += data.serialise() + " Nxt=";
    //result += collision.serialise() + " Nxt=";
    result += next.serialise() + " Seq=";
    result += std::to_string(sequence) + " Cli=";
    result += std::to_string(client) + " E=";
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
            value.stationary = std::atoi(str.c_str()) == 1 ? true : false;
        }
        else if(index == 2)
        {
            value.position.deserialise(str);            
        }        
        else if(index == 3)
        {
            value.data.deserialise(str);
        }
        else if(index == 4)
        {
            value.sequence = std::atoi(str.c_str());         
        }
        else if(index == 5)        
        {
            value.client = std::atoi(str.c_str());
        }
        else if(index == 6)
        {
            value.epoch = std::atoi(str.c_str());
            *this = value;
        }        

        ++index;
    };
}
