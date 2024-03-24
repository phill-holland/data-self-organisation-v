#include "point.h"

#ifndef _ORGANISATION_HISTORY
#define _ORGANISATION_HISTORY

namespace organisation
{
    namespace history
    {
        class value
        {
        public:
            point position;            
            point data;

            int sequence;
            int client;
            int epoch;

        public:
            value(point _position = point(0,0,0), point _data = point(0,0,0))
            {
                position = _position;
                data = _data;
                sequence = 0;
                client = 0;
                epoch = 0;
            }

            void clear()
            {
                position = point(0,0,0);
                data = point(0,0,0);
                
                sequence = 0;
                client = 0;
                epoch = 0;
            }

            std::string serialise();
            void deserialise(std::string source);
        };
    };
};

#endif
    