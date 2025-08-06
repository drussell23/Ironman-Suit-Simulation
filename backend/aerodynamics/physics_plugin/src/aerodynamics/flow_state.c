#include <stdlib.h> // for malloc, free
#include <string.h> // for memset

#include "aerodynamics/flow_state.h"

int flow_state_init(FlowState *s, const Mesh *m)
{
    if (!s || !m)
    {
        return -1; // Invalid arguments
    }

    // How many nodes in this mesh?
    size_t n = mesh_get_num_nodes(m);
    s->num_nodes = n;

    // calloc zeroes all doubles to 0.0
    s->velocity = calloc(3 * n, sizeof(*s->velocity));
    s->pressure = calloc(n, sizeof(*s->pressure));
    s->turbulence_kinetic_energy = calloc(n, sizeof(*s->turbulence_kinetic_energy));
    s->turbulence_dissipation_rate = calloc(n, sizeof(*s->turbulence_dissipation_rate));

    // If any calloc failed, free what did succeed and bail
    if (!s->velocity || !s->pressure || !s->turbulence_kinetic_energy || !s->turbulence_dissipation_rate)
    {
        flow_state_destroy(s);
        return -1; // Allocation failure
    }

    return 0; // Success
}

void flow_state_destroy(FlowState *s)
{
    free(s->velocity);
    free(s->pressure);
    free(s->turbulence_kinetic_energy);
    free(s->turbulence_dissipation_rate);

    s->velocity = s->pressure = s->turbulence_kinetic_energy = s->turbulence_dissipation_rate = NULL;
    s->num_nodes = 0;
}
