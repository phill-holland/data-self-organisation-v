#include <gtest/gtest.h>
#include "parallel/program.hpp"
#include "parallel/map/configuration.hpp"
#include "parallel/value.hpp"
#include "general.h"
#include "data.h"
#include "schema.h"
#include "kdpoint.h"
#include "population.h"
#include <unordered_map>

organisation::schema getSchema(organisation::parameters &parameters, 
                               organisation::vector direction,
                               organisation::vector rebound,
                               organisation::point wall,
                               organisation::point value,
                               int delay)
{
    organisation::point starting(parameters.width / 2, parameters.height / 2, parameters.depth / 2);

    organisation::schema s1(parameters);

    organisation::genetic::movements::movement movement(parameters.min_movements, parameters.max_movements);
    movement.directions = { direction };

    organisation::genetic::inserts::insert insert(parameters);
    organisation::genetic::inserts::value a(delay, organisation::point(starting.x,starting.y,starting.z), movement);

    insert.values = { a };
    
    organisation::genetic::cache cache(parameters);    
    cache.set(value, wall);

    organisation::genetic::collisions collisions(parameters);

    int offset = 0;
    for(int i = 0; i < parameters.mappings.maximum(); ++i)
    {        
        collisions.set(rebound.encode(), offset + direction.encode());
        offset += parameters.max_collisions;
    }

    s1.prog.set(cache);
    s1.prog.set(insert);
    s1.prog.set(collisions);

    return s1;
}

TEST(BasicPopulationTestParallel, BasicAssertions)
{    
    GTEST_SKIP();

    const int width = 20, height = 20, depth = 20;

    std::string values1("daisy give me your answer do .");
    std::string input1("daisy");
    std::string expected1("daisy");

    std::vector<std::string> strings = organisation::split(values1);
    organisation::data mappings(strings);

	::parallel::device device(0);
	::parallel::queue queue(device);

    organisation::parameters parameters(width, height, depth);
    
    parameters.dim_clients = organisation::point(2,3,1);
    parameters.iterations = 9;
    parameters.mappings = mappings;

    organisation::inputs::epoch epoch1(input1, expected1);

    parameters.input.push_back(epoch1);

    parallel::mapper::configuration mapper;    
    organisation::parallel::program program(device, &queue, mapper, parameters);
        
    EXPECT_TRUE(program.initalised());
        
    organisation::schema s1 = getSchema(parameters, { 1, 0, 0 }, { 0, 1, 0 }, { 12,10,10 }, { 0, -1, -1 }, 1);
    organisation::schema s2 = getSchema(parameters, {-1, 0, 0 }, { 0,-1, 0 }, {  7,10,10 }, { 1, -1, -1 }, 1);
    organisation::schema s3 = getSchema(parameters, { 0, 1, 0 }, { 1, 0, 0 }, {10, 12,10 }, { 2, -1, -1 }, 1);
    organisation::schema s4 = getSchema(parameters, { 0,-1, 0 }, {-1, 0, 0 }, {10,  7,10 }, { 3, -1, -1 }, 1);
    organisation::schema s5 = getSchema(parameters, { 0, 0, 1 }, { 1, 0, 0 }, {10, 10,12 }, { 4, -1, -1 }, 1);
    organisation::schema s6 = getSchema(parameters, { 0, 0,-1 }, {-1, 0, 0 }, {10, 10, 7 }, { 5, -1, -1 }, 1);

    organisation::scores::score match;
    match.set(1.0f,0); match.set(1.0f,1); match.set(1.0f,2); match.set(0.0f,3);

    s1.scores[0] = match;

    std::vector<organisation::schema*> source = { &s1,&s2,&s3,&s4,&s5,&s6 };

    parameters.population = source.size() * 2;    
    
    organisation::populations::population population(&program, parameters);
    EXPECT_TRUE(population.initalised());

    population.clear();
    population.generate();

    for(int i = 0; i < source.size(); ++i)
    {
        EXPECT_TRUE(population.set(*source[i], i));
    }    

    int generation = 0;
    int generations = 1;

    organisation::schema result = population.go(generation, generations);

    EXPECT_TRUE(result.equals(s1));
    EXPECT_FLOAT_EQ(result.sum(), 0.75f);
}

TEST(BasicPopulationTestTwoEpochsParallel, BasicAssertions)
{    
    GTEST_SKIP();

    const int width = 20, height = 20, depth = 20;

    std::string values1("daisy give me your answer do .");
    std::string input("daisy");

    std::string expected1("daisy");
    std::string expected2("me");

    std::vector<float> expected_sums = {
        0.5f,
        0.25f,
        0.5f,
        0.25f,
        0.25f,
        0.25f
    };

    std::vector<std::string> strings = organisation::split(values1);
    organisation::data mappings(strings);

	::parallel::device device(0);
	::parallel::queue queue(device);

    organisation::parameters parameters(width, height, depth);
    
    parameters.dim_clients = organisation::point(2,3,1);
    parameters.iterations = 9;
    parameters.mappings = mappings;
    parameters.worst = false;

    organisation::inputs::epoch epoch1(input, expected1);
    organisation::inputs::epoch epoch2(input, expected2);

    parameters.input.push_back(epoch1);
    parameters.input.push_back(epoch2);

    parallel::mapper::configuration mapper;    
    organisation::parallel::program program(device, &queue, mapper, parameters);
        
    EXPECT_TRUE(program.initalised());
        
    organisation::schema s1 = getSchema(parameters, { 1, 0, 0 }, { 0, 1, 0 }, { 12,10,10 }, { 0, -1, -1 }, 1);
    organisation::schema s2 = getSchema(parameters, {-1, 0, 0 }, { 0,-1, 0 }, {  7,10,10 }, { 1, -1, -1 }, 1);
    organisation::schema s3 = getSchema(parameters, { 0, 1, 0 }, { 1, 0, 0 }, {10, 12,10 }, { 2, -1, -1 }, 1);
    organisation::schema s4 = getSchema(parameters, { 0,-1, 0 }, {-1, 0, 0 }, {10,  7,10 }, { 3, -1, -1 }, 1);
    organisation::schema s5 = getSchema(parameters, { 0, 0, 1 }, { 1, 0, 0 }, {10, 10,12 }, { 4, -1, -1 }, 1);
    organisation::schema s6 = getSchema(parameters, { 0, 0,-1 }, {-1, 0, 0 }, {10, 10, 7 }, { 5, -1, -1 }, 1);

    organisation::scores::score match;
    match.set(1.0f,0); match.set(1.0f,1); match.set(1.0f,2); match.set(0.0f,3);

    organisation::scores::score incorrect;
    incorrect.set(0.0f,0); incorrect.set(0.0f,1); incorrect.set(1.0f,2); incorrect.set(0.0f,3);

    s1.scores[0] = match; s1.scores[1] = incorrect;
    s2.scores[0] = incorrect; s2.scores[1] = incorrect;
    s3.scores[0] = incorrect; s3.scores[1] = match;
    s4.scores[0] = incorrect; s4.scores[1] = incorrect;
    s5.scores[0] = incorrect; s5.scores[1] = incorrect;
    s6.scores[0] = incorrect; s6.scores[1] = incorrect;

    std::vector<organisation::schema*> source = { &s1,&s2,&s3,&s4,&s5,&s6 };

    parameters.population = source.size() * 2;    
    
    organisation::populations::population population(&program, parameters);
    EXPECT_TRUE(population.initalised());

    population.clear();
    population.generate();

    for(int i = 0; i < source.size(); ++i)
    {
        EXPECT_TRUE(population.set(*source[i], i));
    }    

    int generation = 0;
    int generations = 2;

    population.go(generation, generations);

    std::vector<organisation::schema> result;    
    for(int i = 6; i < parameters.population; ++i)
    {
        organisation::schema temp(parameters);
        if(population.get(temp,i))
        {
            result.push_back(temp);
        }
    }

    EXPECT_EQ(result.size(), source.size());
    
    for(int i = 0; i < source.size(); ++i)
    {
        EXPECT_FLOAT_EQ(result[i].sum(), expected_sums[i]);
        EXPECT_TRUE(source[i]->equals(result[i]));
    }
}
