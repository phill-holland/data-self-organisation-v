#include "history/stream.h"
#include <fstream>
#include <sstream>

bool organisation::history::stream::get(value &destination, int index)
{
    if((index < 0)||(index >= data.size())) return false;

    return false;
}
            
bool organisation::history::stream::save(std::string filename)
{
    std::fstream output(filename, std::fstream::out | std::fstream::binary);

    if(output.is_open())
    {
        for(auto &it: data)
        {
            std::string data = it.serialise();
            output.write(data.c_str(), data.size());
        }
    }

    output.close();

    return true;
}

bool organisation::history::stream::load(std::string filename)
{
    clear();
    
    std::ifstream source(filename);
    if(source.is_open())
    {
        for(std::string line; getline(source, line); )
        {
            std::stringstream stream(line);
		    std::string type;
	            
            if(stream >> type)
            {                
                value temp;
                if(type == "H") 
                {
                    temp.deserialise(line);                
                    data.push_back(temp);
                }
            }
        }
    }

    source.close();

    return true;
}