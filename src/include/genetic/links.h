#include "genetic/templates/genetic.h"
#include "genetic/templates/serialiser.h"
#include "point.h"
#include "parameters.h"
#include "input.h"
#include <vector>
#include <unordered_map>
#include <random>

#ifndef _ORGANISATION_GENETIC_LINKS
#define _ORGANISATION_GENETIC_LINKS

namespace organisation
{
    namespace genetic
    {
        class links : public templates::genetic, public templates::serialiser
        {
            static std::mt19937_64 generator;

            int _max_cache_dimension;

        public:
            std::vector<point> values;            

        public:
            links(parameters &settings) 
            { 
                _max_cache_dimension = settings.max_cache_dimension;
            }

        public:
            size_t size() { return values.size(); }
            void clear() 
            { 
                values.clear(); 
            }

            bool empty()
            {
                return values.empty();
            }

            std::string serialise();
            void deserialise(std::string source);

            bool validate(data &source);

        public:
            void generate(data &source, inputs::input &epochs);
            bool mutate(data &source, inputs::input &epochs);
            void append(genetic *source, int src_start, int src_end);

        public:
            void copy(const links &source);
            bool equals(const links &source);
        };
    };
};

#endif