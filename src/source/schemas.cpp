#include "schemas.h"
#include <iostream>
#include <functional>
#include <limits>
#include <iostream>

void organisation::schemas::reset(parameters &settings)
{
    init = false; cleanup();
    this->length = settings.population;
    
    data.resize(this->length);

    for(int i = 0; i < this->length; ++i) { data[i] = NULL; }

    for(int i = 0; i < this->length; ++i)   
    { 
        data[i] = new organisation::schema(settings);
        if(data[i] == NULL) return;
    }

    init = true;
}

bool organisation::schemas::clear()
{
    for(int i = 0; i < length; ++i)
    {
        data[i]->clear();
    }

    return true;
}

bool organisation::schemas::generate(organisation::data &source, inputs::input &epochs)
{
    for(int i = 0; i < length; ++i)
    {
        data[i]->generate(source, epochs);
    }

    return true;
}

bool organisation::schemas::load(std::string directory, int start, int len, organisation::data &original, organisation::data &remapped)
{
    for(int i = 0; i < len; ++i)
    {
        int offset = i + start;
        if(offset < length)
        {
            std::string filename = directory + "/run" + std::to_string(i) + ".txt";
            data[i]->clear();
            if(!data[i]->prog.load(filename)) 
                return false;
            else data[i]->prog.remap(original, remapped);
        }
    }

    return true;
}

void organisation::schemas::makeNull() 
{ 
}

void organisation::schemas::cleanup() 
{ 
    if(data.size() > 0)
    {
        for(int i = length - 1; i >= 0; i--)
        { 
            if(data[i] != NULL) delete data[i];
        }
    }
}