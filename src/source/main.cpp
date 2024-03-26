#include "population.h"
#include "data.h"
#include "general.h"
#include <iostream>
#include <string.h>

#include "fifo.h"
#include "schema.h"
#include "vector.h"
#include "input.h"
#include "output.h"
#include "dictionary.h"

#include "templates/programs.h"

#include "history/stream.h"

#include "parallel/device.hpp"
#include "parallel/queue.hpp"
#include "parallel/program.hpp"

using namespace std;

//const organisation::dictionary dictionary;

const int width = 6, height = 6, depth = 6; //6,6,6
const int device_idx = 0;
const int generations = 500;

organisation::parameters get_parameters()//organisation::data &mappings)
{
    organisation::parameters parameters(width, height, depth);

    parameters.dim_clients = organisation::point(10,10,10);
    parameters.iterations = 30;
    parameters.max_values = 100;
    parameters.max_cache = parameters.max_values;// / 2;
        
    parameters.population = parameters.clients() * 4;//4;//4;//8;//4;

    parameters.output_stationary_only = true;
    
    parameters.width = width;
    parameters.height = height;
    parameters.depth = depth;
    //parameters.mappings = mappings;        

    // ***    
    parameters.min_movement_patterns = 2;//7;
    parameters.max_movement_patterns = 2;//7;
    parameters.max_insert_delay = 5; //7
    parameters.scores.max_collisions = 0;//2;//0;//2;

    parameters.max_cache_dimension = 3;
    
    parameters.min_insert_words = 1;
    parameters.max_insert_words = 3;

    parameters.max_movements = 5;
    // ***

/*
    std::string input1("daisy daisy give me your answer do");
    std::string expected1("I'm half crazy for the love of you");

    std::string input2("it won't be a stylish marriage");
    std::string expected2("I can't afford a carriage");
*/


    std::string input1("daisy give");
    std::string expected1("I'm half");

    //std::string input2("daisy answer");
    std::string input2("banana answer");
    std::string expected2("love you");

    std::string input3("bicycle two");
    std::string expected3("made for");

    std::string input4("bucket face");
    std::string expected4("fancy marriage");

    organisation::inputs::epoch epoch1(input1, expected1);
    organisation::inputs::epoch epoch2(input2, expected2);
    organisation::inputs::epoch epoch3(input3, expected3);
    organisation::inputs::epoch epoch4(input4, expected4);
    
    parameters.input.push_back(epoch1);
    parameters.input.push_back(epoch2);
    //parameters.input.push_back(epoch3);
    //parameters.input.push_back(epoch4);
    
    organisation::dictionary words;
    words.push_back(parameters.input);
    auto strings = words.get();
    organisation::data mappings(strings);
    parameters.mappings = mappings;        

    for(int i = 0; i < parameters.input.size(); ++i)
    {        
        organisation::inputs::epoch temp;
        if(parameters.input.get(temp, i))
            std::cout << "input: \"" << temp.input << "\" expected: \"" << temp.expected << "\"\r\n";
    }

    return parameters;
}

bool run(organisation::templates::programs *program, organisation::parameters &parameters, organisation::schema &result)
{         	
    organisation::populations::population p(program, parameters);
    if(!p.initalised()) return false;

    int actual = 0;

    p.clear();
    p.generate();
    
    result.copy(p.go(actual, generations));

    if(actual <= generations) 
    {
        std::string filename("output/run.txt");
        result.prog.save(filename);
    }
    
    return true;
}

/*
bool single(organisation::templates::programs *program, organisation::parameters &parameters)
{
    organisation::history::stream stream;

    parameters.dim_clients = organisation::point(1,1,1);
    parameters.history = &stream;

    organisation::schema s1(parameters);

    if(!s1.prog.load("data/run.txt")) return false;
        
    std::vector<organisation::schema*> source = { &s1 };
    
    program->copy(source.data(), source.size());
    program->set(parameters.mappings, parameters.input);
    program->run(parameters.mappings);

    if(!stream.save("data/trace.txt")) return false;

    std::vector<organisation::outputs::output> results = program->get(parameters.mappings);

    return true;    
}
*/
bool single()
{
    organisation::parameters parameters = get_parameters();
    organisation::history::stream stream;

    parameters.dim_clients = organisation::point(1,1,1);
    parameters.history = &stream;

	::parallel::device device(device_idx);
	::parallel::queue queue(device);

    parallel::mapper::configuration mapper;

    organisation::schema result(parameters);   
    organisation::parallel::program program(device, &queue, mapper, parameters);

    if(!program.initalised()) return false;
    
    organisation::schema s1(parameters);

    if(!s1.prog.load("data/two_epochs.txt")) return false;
        
    std::vector<organisation::schema*> source = { &s1 };
    
    program.copy(source.data(), source.size());
    program.set(parameters.mappings, parameters.input);
    program.run(parameters.mappings);

    if(!stream.save("data/trace.txt")) return false;

   std::vector<organisation::outputs::output> results = program.get(parameters.mappings);

    return true;    
}

int main(int argc, char *argv[])
{  
    single();
    return 0;
    //auto strings = dictionary.get();
    //organisation::data mappings(strings);
    
    organisation::parameters parameters = get_parameters();//mappings);

	::parallel::device device(device_idx);
	::parallel::queue queue(device);

    parallel::mapper::configuration mapper;

    organisation::schema result(parameters);   
    organisation::parallel::program program(device, &queue, mapper, parameters);

    if(program.initalised())
    {
        run(&program, parameters, result);
        //single(&program, parameters);
    }
       
    return 0;
}
