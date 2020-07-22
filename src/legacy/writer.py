import numpy as np
import os


def makeStateDataHeaders(state, info):
    stateDataHeaders = ["step"]

    noDistricts = info["parameters"]["districts"]
    # if "spanningTreeCounts" in state["extensions"]:
    #     for di in range(noDistricts):
    #         stateDataHeaders += ["spanningTreeCounts_dist" + str(di)]

    if "flowDir" in state:
        stateDataHeaders += ["flowDir"]

    return stateDataHeaders

def makeStateDataArray(step, state, info):
    stateDataArray = [step]

    noDistricts = info["parameters"]["districts"]
    if "spanningTreeCounts" in state["extensions"]:
        for di in range(noDistricts):
            noSpanningTrees = np.round(np.exp(state["spanningTreeCounts"][di]))
            stateDataArray += [int(noSpanningTrees)]

    if "flowDir" in state:
        stateDataArray += [state["flowDir"]]

    return stateDataArray

def setupOutputs(state, info, delim = '\t'):
    try:
        outDir = info["parameters"]["outDir"]
    except:
        return

    if not os.path.exists(outDir):
        os.makedirs(outDir)

    outDirSamples = os.path.join(outDir, "Samples")
    if not os.path.exists(outDirSamples):
        os.makedirs(outDirSamples)

    stateDataHeaders = makeStateDataHeaders(state, info)
    metaDataPath = os.path.join(outDir, "metaData.txt")
    with open(metaDataPath, "w") as mdf:
        mdf.write(delim.join([str(sdh) for sdh in stateDataHeaders]) + "\n")
    
def recordState(step, state, info, delim = '\t'):
    try:
        outDir = info["parameters"]["outDir"]
    except:
        return
    outDirSamples = os.path.join(outDir, "Samples")
    outPath = os.path.join(outDirSamples, str(step) + ".txt")
    outFile = open(outPath, "w")

    for di in state["districts"]:
        district = state["districts"][di]
        for n in district.nodes:
            outFile.write(delim.join([str(n), str(di)])+ "\n")
    outFile.close()

def recordStateData(step, state, info, delim = '\t'):
    try:
        outDir = info["parameters"]["outDir"]
    except:
        return

    stateDataArray = makeStateDataArray(step, state, info)
    metaDataPath = os.path.join(outDir, "metaData.txt")
    with open(metaDataPath, "a") as mdf:
        mdf.write(delim.join([str(sdh) for sdh in stateDataArray]) + "\n")

