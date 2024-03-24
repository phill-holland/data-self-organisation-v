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
            point client;
            point data;
            int epoch;

        public:
            value(point _position = point(0,0,0), point _data = point(0,0,0))
            {
                position = _position;
                data = _data;
            }
        };
    };
};

#endif
    