import enumHelpers as eh
import numpy as np
import os
from glob import glob

planFiles = glob(os.path.join("popAndCompPaired", "*.csv"))

for planFile in planFiles:
    clusterName = eh.getClusterName(planFile.split("/")[-1])
    print(clusterName)

    with open(planFile) as pf:
        pfLines = pf.readlines()
    geoIDs = pfLines[0].rstrip().split(",")
    enumPlans = pfLines[1:]
    enumPlans = [p.rstrip().split(",") for p in enumPlans]

    energies = eh.getEnergies(clusterName, geoIDs, enumPlans)
    electionResults = eh.runElections(clusterName, geoIDs, enumPlans, 
                                                 "EL12G_PR")
    # print(electionResults)
    # exit()
    cumEnergy = [e for e in energies]
    for ii in range(1, len(energies)):
        cumEnergy[ii] += cumEnergy[ii-1]
    
    outPath = os.path.join("electionResults", clusterName)
    if not os.path.exists(outPath):
        os.makedirs(outPath)

    # with open(os.path.join(outPath, clusterName + "EL12G_PR.txt"), "w") as outf:
    #     for jj in range(len(electionResults)):
    #         elec = electionResults[jj]
    #         for r in elec:  
    #             outf.write("\t".join([str(v) for v in r]) + "\n")

    with open(os.path.join(outPath, clusterName + "EL12G_PR.txt"), "w") as outf:
        np.random.seed(238901283)
        for jj in range(20000):
            rn = np.random.random()*cumEnergy[-1]
            elecInd = min(np.searchsorted(cumEnergy, rn), len(cumEnergy)-1)
            elec = electionResults[elecInd]
            for r in elec:  
                outf.write("\t".join([str(v) for v in r]) + "\n")
                # print("\t".join([str(v) for v in r]))