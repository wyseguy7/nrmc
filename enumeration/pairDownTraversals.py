import enumHelpers as eh
import os
from glob import glob

planFiles = glob(os.path.join("popPaired", "*.csv"))

for planFile in planFiles:
    clusterName = eh.getClusterName(planFile.split("/")[-1])
    
    with open(planFile) as pf:
        pfLines = pf.readlines()
    geoIDs = pfLines[0].rstrip().split(",")
    enumPlans = pfLines[1:]
    enumPlans = [p.rstrip().split(",") for p in enumPlans]

    print("begin with", len(enumPlans))
    enumPlans = eh.pairDownTraversals(clusterName, geoIDs, enumPlans)
    print("after no traversals", len(enumPlans))

    outPath = os.path.join("popAndTraversals")
    if not os.path.exists(outPath):
        os.makedirs(outPath)
    with open(os.path.join(outPath, clusterName + ".csv"), "w") as outf:
        outf.write(",".join(geoIDs) + "\n")
        for p in enumPlans:
            outf.write(",".join(p) + "\n")

    # print("after pop, before pp", len(enumPlans))
    # enumPlans = eh.pairDownPolsbyPopper(geoIDs, enumPlans)
    # enumPlans = eh.pairDownReock(geoIDs, enumPlans)
    # enumPlans = eh.pairDownMCD(geoIDs, enumPlans)
    # exit()

