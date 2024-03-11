#include <gtest/gtest.h>
#include <string>
#include "program.h"
#include "data.h"
#include "vector.h"
#include "schema.h"
#include "general.h"
#include "point.h"
#include "parameters.h"
#include "genetic/cache.h"
#include "genetic/movement.h"
#include "genetic/collisions.h"
#include "genetic/insert.h"

/*
TEST(BasicSerialisationDeserialisation, BasicAssertions)
{
    GTEST_SKIP();

    const int width = 20, height = 20, depth = 20;
    organisation::point starting(width / 2, height / 2, depth / 2);

    std::vector<std::string> strings = { "daisy", "give" };
    organisation::data mappings(strings);

    organisation::parameters parameters(width, height, depth);
    parameters.mappings = mappings;

    organisation::program p1(parameters);
    organisation::program p2(parameters);

    organisation::genetic::cache cache(parameters);
    cache.set(organisation::point(0,-1,-1), starting);
    cache.set(organisation::point(0,1,-1), organisation::point(starting.x + 1, starting.y, starting.z));
    cache.set(organisation::point(0,2,3), organisation::point(starting.x + 2, starting.y, starting.z));
    
    organisation::genetic::inserts::insert insert(parameters);

    organisation::genetic::inserts::value a(1, organisation::point(starting.x,starting.y,starting.z), 0);
    organisation::genetic::inserts::value b(2, organisation::point(starting.x + 1,starting.y,starting.z), 1);
    organisation::genetic::inserts::value c(3, organisation::point(starting.x + 2,starting.y,starting.z), 0);

    insert.values = { a, b, c };

    organisation::genetic::movements::movement movement(parameters);
    std::vector<organisation::vector> m1 = { { 1,0,0 }, { 0,0,1 } };
    std::vector<organisation::vector> m2 = { { 0,1,0 }, { -1,0,0 }, { 0,0,-1} };

    movement.set(0,m1);
    movement.set(1,m2);    

    organisation::genetic::collisions collisions(parameters);

    organisation::vector up = { 1,0,0 };
    organisation::vector rebound = { 0,1,0 };

    collisions.set(rebound.encode(),up.encode());
    collisions.set(rebound.encode(),up.encode() + parameters.max_collisions);
    
    p1.set(cache);
    p1.set(insert);
    p1.set(movement);
    p1.set(collisions);

    std::string data = p1.serialise();
    p2.deserialise(data);

    EXPECT_TRUE(p1.equals(p2));
}
*/