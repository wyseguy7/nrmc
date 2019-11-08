import networkx as nx
import numpy as np
import os

def saveGraph(g, nodeDataToIndex, edgeDataToIndex, pathToData):
    if not os.path.exists(pathToData):
        os.makedirs(pathToData)

    nodeDataPath = os.path.join(pathToData, "nodeData.txt")
    edgeDataPath = os.path.join(pathToData, "edgeData.txt")

    metaDataFile = open(os.path.join(pathToData, "metaData.txt"), "w")
    nodeIdInd = str(nodeDataToIndex["ID"])
    for nd in nodeDataToIndex:
        if nd == "ID":
            continue
        try:
            colsStr = ','.join([str(ii) for ii in list(nodeDataToIndex[nd])])
        except:
            colsStr = str(nodeDataToIndex[nd])
        toWrite = [nd, "nodeData.txt", "ID:" + nodeIdInd, "col:" + colsStr]
        metaDataFile.write('\t'.join(toWrite) + "\n")

    edgeIdInds = ','.join([str(ii) for ii in edgeDataToIndex["ID"]])
    for ed in edgeDataToIndex:
        if ed == "ID":
            continue
        try:
            colsStr = ','.join([str(ii) for ii in list(edgeDataToIndex[ed])])
        except:
            colsStr = str(edgeDataToIndex[ed])
        toWrite = [ed, "edgeData.txt", "ID:" + edgeIdInds, "col:" + colsStr]
        metaDataFile.write('\t'.join(toWrite) + "\n")
    metaDataFile.close()

    nodeDataFile = open(nodeDataPath, "w")
    for ni in range(len(g.nodes)):
        toWrite = []
        for nd in nodeDataToIndex:
            ind = nodeDataToIndex[nd]
            try:
                indList = list(ind)
                nDatList = list(g.nodes[ni][nd])
            except:
                try:
                    indList = [ind]
                    nDatList = [g.nodes[ni][nd]]
                except:
                    indList = [ind]
                    nDatList = [ni]
            for ii in range(len(indList)):
                while indList[ii] >= len(toWrite):
                    toWrite.append("")
                toWrite[indList[ii]] = str(nDatList[ii])
        nodeDataFile.write('\t'.join(toWrite) + "\n")
    nodeDataFile.close()

    edgeDataFile = open(edgeDataPath, "w")
    # print(g.edges)
    for ei in list(g.edges):
        # print("working on edge", ei)
        toWrite = []
        for ed in edgeDataToIndex:
            ind = edgeDataToIndex[ed]
            # print("adding info", ed, ind)
            try:
                # print("trying")
                indList = list(ind)
                eDatList = list(g.edges[ei[0], ei[1]][ed])
            except:
                # print("excepting")
                try:
                    indList = [ind]
                    eDatList = [g.edges[ei[0], ei[1]][ed]]
                except:
                    # print("excepting again")
                    indList = list(ind)
                    eDatList = [ei[0], ei[1]]
            # print(ed, ind, indList)
            for ii in range(len(indList)):
                while indList[ii] >= len(toWrite):
                    toWrite.append('')
                toWrite[indList[ii]] = str(eDatList[ii])
        # print("writing", toWrite, "about edge", ei)
        edgeDataFile.write('\t'.join(toWrite) + "\n")
    edgeDataFile.close()

    # edgeDataFile = open(edgeDataPath, "w")

def generateBigonDualGraphData(pathToData, pop = 1):
    geometryDesc = pathToData.split(os.sep)[-1]
    graphData = geometryDesc.split("_")
    bigons = 4
    extRegions = 4
    if len(geometryDesc) > 1:
        bigons = int(graphData[1])
    if len(geometryDesc) > 2:
        extRegions = int(graphData[2])

    dTheta = 2.0*np.pi/extRegions

    nodeDataToIndex = {"ID" : 0, "Centroid" : (1, 2), "Population" : 3}
    edgeDataToIndex = {"ID" : (0, 1), "BorderLengths" : 2}

    g = nx.Graph()
    extNodeIDs = list(range(extRegions))
    g.add_nodes_from(extNodeIDs)
    for ni in range(extRegions):
        theta = ni*dTheta + np.pi*0.5
        x = (np.cos(theta), np.sin(theta))
        g.nodes[ni]["Centroid"] = x
        g.nodes[ni]["Population"] = 1

    if bigons == 0:
        for ni in range(extRegions):
            nip = (ni+1)%extRegions
            g.add_edge(ni, nip)
            g.edges[ni, nip]["BorderLengths"] = 1
    else:
        curEdgeID = extRegions
        for ni in range(extRegions):
            nip = (ni+1)%extRegions
            
            theta = (ni + 0.5)*dTheta + np.pi*0.5
            x = (np.cos(theta), np.sin(theta))

            for nb in range(bigons):
                g.add_node(curEdgeID)
                g.add_edge(ni, curEdgeID)
                g.add_edge(nip, curEdgeID)
                g.nodes[curEdgeID]["Centroid"] = x
                g.nodes[curEdgeID]["Population"] = 1
                g.edges[ni, curEdgeID]["BorderLengths"] = 1
                g.edges[nip, curEdgeID]["BorderLengths"] = 1
                curEdgeID += 1
    saveGraph(g, nodeDataToIndex, edgeDataToIndex, pathToData)

def generateSquareLatticeData(pathToData, pop = 1):
    geometryDesc = pathToData.split(os.sep)[-1]
    graphData = geometryDesc.split("_")
    N_x = 40
    N_y = 40
    if len(geometryDesc) > 1:
        N_x = N_y = int(graphData[1])
    if len(geometryDesc) > 2:
        N_y = int(graphData[2])

    nodeDataToIndex = {"ID" : 0, "Centroid" : (1, 2), "Population" : 3}
    edgeDataToIndex = {"ID" : (0, 1), "BorderLengths" : 2}

    g = nx.Graph()
    for ix in range(N_x+1):
        for iy in range(N_y+1):
            ni = ix + (N_x+1)*iy
            g.add_node(ni)
            g.nodes[ni]["Centroid"] = (ix, iy)
            g.nodes[ni]["Population"] = 1

    for ix in range(N_x):
        for iy in range(N_y):
            ni = ix + (N_x+1)*iy
            nipx = (ix+1) + (N_x+1)*iy
            nipy = ix + (N_x+1)*(iy+1)
            # print(ix, iy, ni, nipx, nipy)
            g.add_edge(ni, nipx)
            g.add_edge(ni, nipy)
            g.edges[ni, nipx]["BorderLengths"] = 1
            g.edges[ni, nipy]["BorderLengths"] = 1
    for ix in range(N_x):
        ni = ix + (N_x+1)*N_y
        nipx = ix+1 + (N_x+1)*N_y
        g.add_edge(ni, nipx)
        g.edges[ni, nipx]["BorderLengths"] = 1

    for iy in range(N_y):
        ni = N_x + (N_x+1)*iy
        nipy = N_x + (N_x+1)*(iy+1)
        g.add_edge(ni, nipy)
        g.edges[ni, nipy]["BorderLengths"] = 1

    saveGraph(g, nodeDataToIndex, edgeDataToIndex, pathToData)
