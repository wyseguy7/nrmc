import networkx as nx
import os

################################################################################

def setMetaDataPath(path, metaDataFile):
    dataFiles = os.listdir(path)
    
    if metaDataFile == '':
        metaDataFiles = [f for f in dataFiles if "metadata" in f.lower()]
        if len(metaDataFiles) > 1:
            raise Exception("Ambiguous meta-data in path", path)
        metaDataFile = metaDataFiles[0]

    return os.path.join(path, metaDataFile)

################################################################################

def setMetaData(path, metaDataFile, borderKey, delim = '\t', auxFields = {}):
    
    metaDataPath = setMetaDataPath(path, metaDataFile)

    metaData = {}
    metaData["BorderKey"] = borderKey

    with open(metaDataPath) as mdp:
        for line in mdp:
            dat, file, ID, col = line.rstrip().split(delim)
            ID = [int(i) for i in ID.split(':')[-1].split(",")]
            col = [int(c) for c in col.split(':')[-1].split(",")]
            metaData[dat] = (os.path.join(path, file), ID, col)

    for auxField in auxFields:
        auxData = auxFields[auxField]
        filePath = os.path.join(path, auxData[0])
        if not os.path.exists(filePath):
            dataFiles = [f for f in os.listdir(path) if auxField in f]
            if len(dataFiles) > 1:
                raise Exception("Ambiguous auxiliary field", auxField, 
                                " -- found", dataFiles)
            filePath = os.path.join(path, dataFiles[0])
        metaData[auxField] = (filePath, auxData[1], auxData[2])

    return metaData

################################################################################

def initializeGraph(metaData, delim = '\t'):
    G = nx.Graph()
    path, [id1, id2], [col] = metaData["BorderLengths"]
    borderKey = metaData["BorderKey"]
    with open(path) as f:
        for line in f:
            L = line.strip().split(delim)
            if borderKey not in L and float(L[col]) > 0:
                G.add_edge(L[id1], L[id2])
                G.edges[L[id1], L[id2]]['BorderLength'] = float(L[col])
            else:
                fid = L[id2] if L[id1] == borderKey else L[id1]
                G.add_node(fid)
                G.nodes[fid]['BorderLength'] = float(L[col])
    del metaData["BorderLengths"]
    del metaData["BorderKey"]
    return G

################################################################################

def setNodeData(G, md, metaData, delim = '\t'):
    path, [ID], cols = metaData
    with open(path) as mdf:
        for line in mdf:
            L = line.rstrip().split(delim)
            try:
                fid, dat = L[ID], [float(L[c]) for c in cols]
            except:
                fid, dat = L[ID], [L[c] for c in cols]
            if len(dat) == 1:
                dat = dat[0]
            G.nodes[fid][md] = dat

################################################################################

def setGraphData(G, mdName, metaData, delim = '\t'):
    idList = metaData[1]
    if len(idList) == 1:
        setNodeData(G, mdName, metaData, delim)
    elif len(idList) == 2:
        setEdgeData(G, mdName, metaData, delim)
    else:
        raise Exception("Neither edge nor node data")


################################################################################

def setGraph(metaData, delim = '\t'):
    G = initializeGraph(metaData, delim)
    for md in metaData:
        setGraphData(G, md, metaData[md], delim)
    return G

################################################################################

def set(path, metaDataFile = '', delim = '\t', borderKey = '-1', 
        auxFields = {}):
    metaData = setMetaData(path, metaDataFile, borderKey, delim, auxFields)
    G = setGraph(metaData, delim)
    return G
