import enumHelpers as eh
import os
from glob import glob

planFiles = glob(os.path.join("popPaired", "*.csv"))

for planFile in planFiles:
    clusterName = eh.getClusterName(planFile.split("/")[-1])
    if "Duplin" not in clusterName:
        continue
    
    with open(planFile) as pf:
        pfLines = pf.readlines()
    geoIDs = pfLines[0].rstrip().split(",")
    enumPlans = pfLines[1:]
    enumPlans = [p.rstrip().split(",") for p in enumPlans]

    compactnesses = eh.determinePolsbyPopper(clusterName, geoIDs, enumPlans)
    
    outPath = os.path.join("compactnesses")
    if not os.path.exists(outPath):
        os.makedirs(outPath)
    with open(os.path.join(outPath, clusterName + ".csv"), "w") as outf:
        # outf.write(",".join(geoIDs) + "\n")
        for c in compactnesses:
            clist = list(c)
            clist.sort()
            clist = [str(comp) for comp in clist]
            for ce in range(len(clist)):
                outf.write(str(ce+1) + "\t" + str(clist[ce])[:15] + "\n")
    with open(os.path.join(outPath, clusterName + "2.csv"), "w") as outf:
        # outf.write(",".join(geoIDs) + "\n")
        for c in compactnesses:
            clist = list(c)
            clist.sort()
            clist = [str(comp)[:15] for comp in clist]
            outf.write("\t".join(clist) + "\n")

    # print("after pop, before pp", len(enumPlans))
    # enumPlans = eh.pairDownPolsbyPopper(geoIDs, enumPlans)
    # enumPlans = eh.pairDownReock(geoIDs, enumPlans)
    # enumPlans = eh.pairDownMCD(geoIDs, enumPlans)
    # exit()

