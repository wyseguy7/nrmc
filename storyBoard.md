# Epic 


# Theme 1
 
Numerically test skew detailed balance to understand the efficacy of different implementations

# Story 1.1  
square lattice example:
2 districts 
* [x] no tempering; no skew  
* [x] tempering no skew  
* [x] precinct flow (tempering)  
* [ ] COM flow (tempering)  [In progress, use my code]
* [ ] district flow (tempering)

* Make a heat map of which nodes have been flipped for all 7 runs
* Make a heat maps of which nodes are on the boundary for all 7 runs

* Make autocorrelation plots over several observables
    - put dems in lower half, reps in upper half; observable is % of dems in district 1
    - COM
    - orientation

Question -- after 1 million runs, what is the distribution of the front:
	i) for each node, find the number of times it has flipped
	ii) for each node, find the number samples it lies on the boundary

# Story 1.2

3 districts (need to think about energy)
(energy -- add population and isoperimetric)
_ precinct flow (tempering)
_ COM flow (tempering)
_ district-to-district flow (tempering)

Real world examples: Only worry about population and compactness
Alamance; House (2 districts)
x Single node flip
_ Single node flip with tempering
_ precinct flow (tempering)
_ COM flow (tempering)

Mecklenburg; Senate (5 districts)
x Single node flip
_ Single node flip with tempering
_ precinct flow (tempering)
_ COM flow (tempering)
_ district-to-district flow (tempering)

NC; Congressional (13 districts)
_ Single node flip
_ Single node flip with tempering
_ precinct flow (tempering)
_ COM flow (tempering)
_ district-to-district flow (tempering)

