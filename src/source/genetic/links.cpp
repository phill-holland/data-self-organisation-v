#include "genetic/links.h"
#include <sstream>
#include <functional>
#include <iostream>

std::mt19937_64 organisation::genetic::links::generator(std::random_device{}());

std::string organisation::genetic::links::serialise()
{
    std::string result;

    int index = 0;
    for(auto &it: values)
    {     
        result += "L " + std::to_string(index) + " " + it.serialise() + "\n";
        ++index;
    }

    return result;                    
}

void organisation::genetic::links::deserialise(std::string source)
{
    std::stringstream ss(source);
    std::string str;

    point value(-1,-1,-1);
    int index = 0;
    int offset = 0;

    while(std::getline(ss,str,' '))
    {
        if(index == 0)
        {
            if(str.compare("L")!=0) return;        
        }
        else if(index == 1)
        {          
            offset = std::atoi(str.c_str());  
        }
        else if(index == 2)
        {            
            value.deserialise(str);
            values[offset] = value;
        }

        ++index;
    };
}

bool organisation::genetic::links::validate(data &source)
{    
    if(values.size() != source.maximum())
    { 
        std::cout << "links::validate(false): values.size(" << values.size() << ") != data.size(" << source.maximum() << ")\r\n"; 
        return false; 
    }

    for(auto &it: values)
    {
        point value = it;
        int coordinates[] = { value.x, value.y, value.z };

        for(int i = 0; i < 3; ++i)
        {
            if(coordinates[i] != -1)
            {
                if(source.map(coordinates[i]).empty()) 
                { 
                    std::cout << "links::validate(false): map empty [" << coordinates[i] << "]\r\n"; 
                    return false; 
                }
            }
        }     
    }
    
    return true;
}

void organisation::genetic::links::generate(data &source, inputs::input &epochs)
{
    clear();
                
    std::vector<int> raw = source.all();

    for(auto &it: raw)
    {
        point value;
        value.generate2(raw,_max_cache_dimension);
        values[it] = value;
    };
}

bool organisation::genetic::links::mutate(data &source, inputs::input &epochs)
{
    const int COUNTER = 15;

    if(values.empty()) return false;
    std::vector<int> all = source.outputs(epochs);

    int offset = (std::uniform_int_distribution<int>{0, (int)(values.size() - 1)})(generator);

    int counter = 0;

    point old = values[offset];
    point value = old;

    while((old==value)&&(counter++<COUNTER))
    {
        value = old;
        value.mutate(all,_max_cache_dimension);
    }

    if(value == old) return false;

    values[offset] = value;
    
    return true;
}

void organisation::genetic::links::append(genetic *source, int src_start, int src_end)
{
    links *s = dynamic_cast<links*>(source);

    for(int i = src_start; i < src_end; ++i)
    {
        values[i] = s->values[i];
    }    
}

bool organisation::genetic::links::get(organisation::point &result, int idx)
{    
    if((idx < 0)||(idx >= size())) return false;
    result = values[idx];
    return true;
}

bool organisation::genetic::links::set(organisation::point source, int idx)
{
    if((idx < 0)||(idx >= size())) return false;
    values[idx] = source;
    return true;
}

void organisation::genetic::links::copy(const links &source)
{
    clear();

    for(int i = 0; i < size(); ++i)
    {
        values[i] = source.values[i];
    }
}

bool organisation::genetic::links::equals(const links &source)
{
    if(values.size() != source.values.size()) 
        return false;

    for(int i = 0; i < values.size(); ++i)
    {
        point a = values[i];
        point b = source.values[i];

        if(a != b) 
            return false;
    }

    return true;
}