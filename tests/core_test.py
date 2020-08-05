"""
Integration tests:

    Step forward 1000 iterations with single node flip
        Test contested edges
        Test that all proposals are represented

    Test that measure is respected on a 5x5 lattice (within certain tolerance?)

Edge case:

    Population boundary is still respected in 3-district case
    Non-connected is still respected in 3-district case


Unit tests:
    Test that we can correctly calculate dot product
    Test that we accurately update contested edges
    Test that we can


District-to-district:
    Changing the flow direction changes the involution state correctly


Center of mass:
    Flipping the flow direction changes the involution state
    Proposals respect involution state


"""


def test_core():

    from src.nrmc.core import  MetropolisProcess
    from src.nrmc.lattice import create_square_lattice
    from src.nrmc.updaters import contested_edges_naive

    state = create_square_lattice(n=40, num_districts=2)
    process = MetropolisProcess(state)

    for i in range(10000):
        process.step()

    contested_edges = contested_edges_naive(process.state)
    assert contested_edges == process.state.contested_edges

    proposals = process.get_proposals(process.state)
    # each proposal should be represented in the state
    for node_id, neighbor in contested_edges:

        old_color, new_color = process.state.node_to_color[node_id], process.state.node_to_color[neighbor]
        assert (node_id, old_color, new_color) in proposals # each contested edge should be represented here




