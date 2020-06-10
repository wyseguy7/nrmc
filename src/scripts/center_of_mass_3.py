import sys
import pickle


# sys.path.append('C:\\Users\\wyseg\\nonreversiblecodebase\\src\\')
sys.path.append('/home/grad/etw16/src') # TODO make this less garbage-y


import constructor
import districtingGraph
import initializer
import centerOfMassFlow
import metropolisHastings
import precinct_flow as pf
from networkx import draw
from matplotlib import pyplot as plt
import pandas as pd
import os


print('initializing run...')
state, args = initializer.setRunParametersFromCommandLine([])
state["step"] = 0
proposal, args = centerOfMassFlow.define(args)
info = args

info = initializer.fillMissingInfoFields(info)
state = initializer.determineStateInfo(state, info)
# state = constructor.contructPlan(state, info)
state = constructor.splitSquareLattice(state, info) ## terrible code; trying to move fast
print('run initialized...')


state_new = pf.State.from_state(state)


# Make the district into 3
state_new.color_to_node[2] = set()  # add new color
state_new.node_to_color[0] = 0
state_new.color_to_node[0].add(0)

for node in state_new.graph.nodes():

    centroid = state_new.graph.nodes()[node]['Centroid']
    if centroid[0] > 20:
        # between district 1 and 2
        if 40 - centroid[1] - centroid[0] < 0:
            district_id = 1
        else:
            district_id = 2
    else:
        # between district 0 and 2
        if centroid[1] - centroid[0] > 0:
            district_id = 0
        else:
            district_id = 2

    state_new.flip(node, district_id)
# end making the district into 3


# load in new 'boundary' attribute to nodes - required to check simply_connectedness
for node in state_new.graph.nodes():
    centroid = state_new.graph.nodes()[node]['Centroid']
    state_new.graph.nodes()[node]['boundary'] = (centroid[0] in (0, 40) or centroid[1] in (0, 40))
    # TODO update for new lattice size




process = pf.CenterOfMassFlow(state_new, beta=1, measure_beta=2, minimum_population=360) # TODO fill in args


try:

    date = str(pd.datetime.today().date())
    for i in range(10000000):
        process.step()

        if i % 10000 == 0:
            f = plt.figure(figsize=(15,8))
            draw(process.state.graph, pos={node_id: (process.state.graph.nodes()[node_id]['Centroid'][0],
                                                     process.state.graph.nodes()[node_id]['Centroid'][1]) for node_id in process.state.graph.nodes()},
                 node_color=[process.state.node_to_color[i] for i in process.state.graph.nodes()], node_size=100)


            filepath = os.path.join('gerry_pics', 'center_of_mass',
                                     '{}_beta=2_3_body_square_lattice_{}.png'.format(date, i))
            f.savefig(filepath)

            plt.close()


finally:

    # TODO put those heatmaps here so we don't have to do later

    with open('center_of_mass_3_body{}.pkl'.format(date), mode='wb') as f:
        pickle.dump(process, f)