#include <stdlib.h> // for malloc, free
#include <string.h> // for memset

#include "aerodynamics/flow_state.h"

int flow_state_init(FlowState *s, const Mesh *m)
{
    s->num_nodes = mesh_get_num_nodes(m);

    size_t n = s->num_nodes;

    s->velocity = calloc(n * 3, sizeof(double));
    s->pressure = calloc(n, sizeof(double));
    s->turbulence_kinetic_energy = calloc(n, sizeof(double));
    s->turbulence_dissipation_rate = calloc(n, sizeof(double));

    if (!s->velocity || !s->pressure || !s->turbulence_kinetic_energy || !s->turbulence_dissipation_rate)
    {
        flow_state_destroy(s);
        return -1; // Allocation failure
    }
    return 0; // Success.
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
