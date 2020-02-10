import enumHelpers as eh
import os
from glob import glob

# planFiles = glob(os.path.join("popAndCompPaired", "*.csv"))
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
    # enumPlans = [["1","1","2","1","1","1","1","1","2","2","2","1","2","2","2","2","1","2"]]

    electionResults = eh.runElections(clusterName, geoIDs, enumPlans, 
                                      "EL08G_GV")

    outPath = os.path.join("electionResultsNoDems", clusterName)
    if not os.path.exists(outPath):
        os.makedirs(outPath)
    with open(os.path.join(outPath, clusterName + "EL08G_GV.txt"), "w") as outf:
        for r in electionResults:
            noDems = sum([m>50 for m in r])
            outf.write(str(noDems) + "\n")
            # outf.write(str(ii+1) + "\t" + str(r[ii]) + "\n")
            # outf.write("\t".join([str(v) for v in r]) + "\n")



    # print("after pop, before pp", len(enumPlans))
    # enumPlans = eh.pairDownPolsbyPopper(geoIDs, enumPlans)
    # enumPlans = eh.pairDownReock(geoIDs, enumPlans)
    # enumPlans = eh.pairDownMCD(geoIDs, enumPlans)
    # exit()

