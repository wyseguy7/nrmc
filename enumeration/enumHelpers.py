import geopandas as gpd
import numpy as np
import os
import smallestCircle

projectDir = os.path.join("..")
extData = os.path.join(projectDir, "ExtractedData", "ClusterHouse")
sfData = os.path.join(projectDir, "Shapefiles", "ClusterShapeFilesHouse")

energyWeights = {
    "Person_C_Vance_C_Granville_P_Warren_C_" : [100.0, .2, .0001], #pop, comp, mcd
    "Nash_P_Franklin_C_" : [50.0, 0.2, 0.0001],
    "Duplin_C_Onslow_P_" : [50.0, 0.2, 0.0005],
}

################################################################################

def getAllClusters():
    houseClusters = {c : {} for c in os.listdir(extData) if c[0] != "."}
    for c in houseClusters.keys():
        hc = c.replace("_P_", "_").replace("_C_", "_")
        hc = hc[:-1].split("_")
        houseClusters[c] = set(hc)
    return houseClusters

################################################################################

def getClusterName(enumName):
    name = enumName.replace("Precinct", "_").replace("County", "_")
    name = name.replace("_P_", "_").replace("_C_", "_")
    name = name.replace(".csv", "")[:-1].split("_")
    nameSet = set(name)

    houseClusters = getAllClusters()
    for c in houseClusters:
        if nameSet == houseClusters[c]:
            clusterName = c
    return clusterName

################################################################################

def mapGeoIDToFID(clusterName, geoIDs):
    geoIDPath = os.path.join(extData, clusterName, clusterName + "_GEOIDS.txt")
    geoIDsToFID = {}
    fidToGeoID = {}
    with open(geoIDPath) as gidf:
        for line in gidf:
            fid, geoID = line.rstrip().split("\t")[:2]
            geoIDsToFID[geoID] = fid
            fidToGeoID[fid] = geoID
    return geoIDsToFID, fidToGeoID

################################################################################

def getFidToPop(clusterName):
    fileName = clusterName + "_POPULATION.txt"
    popPath = os.path.join(extData, clusterName, fileName)
    fidToPop = {}
    with open(popPath) as popf:
        for line in popf:
            fid, pop = line.rstrip().split("\t")[:2]
            fidToPop[fid] = float(pop)
    return fidToPop

################################################################################

def getFidToData(clusterName, data, dtype = float):
    fileName = clusterName + "_" + data + ".txt"
    dataPath = os.path.join(extData, clusterName, fileName)
    fidToData = {}
    with open(dataPath) as dataf:
        for line in dataf:
            fid, d = line.rstrip().split("\t")[:2]
            fidToData[fid] = dtype(d)
    return fidToData

################################################################################

def getFidToDataList(clusterName, data, dtype = float):
    fileName = clusterName + "_" + data + ".txt"
    dataPath = os.path.join(extData, clusterName, fileName)
    fidToData = {}
    with open(dataPath) as dataf:
        for line in dataf:
            splitline = line.rstrip().split("\t")
            fid = splitline[0]
            dlist = [dtype(d) for d in splitline[1:]]
            fidToData[fid] = dlist
    return fidToData

################################################################################

def getFidPairToData(clusterName, data, dtype = float):
    fileName = clusterName + "_" + data + ".txt"
    dataPath = os.path.join(extData, clusterName, fileName)
    fidToData = {}
    with open(dataPath) as dataf:
        for line in dataf:
            fid1, fid2, d = line.rstrip().split("\t")[:3]
            if int(fid1) > int(fid2):
                key = (fid2, fid2)
            else:
                key = (fid1, fid2)
            fidToData[key] = dtype(d)
    return fidToData

################################################################################

def pairDownPop(clusterName, geoIDs, enumPlans, ideal = 79462.5, 
                maxPopDiv = 0.05):
    geoIDsToFID, fidToGeoID = mapGeoIDToFID(clusterName, geoIDs)
    fidToPop = getFidToPop(clusterName)

    pairedPlans = []
    for ii in range(len(enumPlans)):
        plan = enumPlans[ii]
        distPops = {}
        # print(geoIDs)
        # print(plan)
        for geoID, distAssign in zip(geoIDs, plan):
            fid = geoIDsToFID[geoID]
            pop = fidToPop[fid]
            if distAssign in distPops:
                distPops[distAssign] += pop
            else:
                distPops[distAssign] = pop
        totUnderThresh = 0
        for d in distPops:
            popDiv = np.abs(distPops[d]-ideal)/float(ideal)
            if popDiv <= maxPopDiv:
                totUnderThresh += 1
        if totUnderThresh > 0 and totUnderThresh == len(distPops):
            pairedPlans.append(enumPlans[ii])
    return pairedPlans

################################################################################

def pairDownPolsbyPopper(clusterName, geoIDs, enumPlans, 
                         minPolsbyPopper = 0.17952):
    geoIDsToFID, fidToGeoID = mapGeoIDToFID(clusterName, geoIDs)
    fidToArea = getFidToData(clusterName, "AREAS")
    fidsToBrdLen = getFidPairToData(clusterName, "BORDERLENGTHS")

    pairedPlans = []
    for ii in range(len(enumPlans)):
        plan = {geoIDsToFID[g] : d for g,d in zip(geoIDs, enumPlans[ii])}
        distAreas = {}
        for fid in plan:
            area = fidToArea[fid]
            dist = plan[fid]
            if dist in distAreas:
                distAreas[dist] += area
            else:
                distAreas[dist] = area
        distPerim = {}
        for fidpair in fidsToBrdLen:
            if fidpair[0] == "-1":
                dist = plan[fidpair[1]]
                if dist in distPerim:
                    distPerim[dist] += fidsToBrdLen[fidpair]
                else:
                    distPerim[dist] = fidsToBrdLen[fidpair]
            elif plan[fidpair[0]] != plan[fidpair[1]]:
                dist1 = plan[fidpair[0]]
                dist2 = plan[fidpair[1]]
                if dist1 in distPerim:
                    distPerim[dist1] += fidsToBrdLen[fidpair]
                else:
                    distPerim[dist1] = fidsToBrdLen[fidpair]
                if dist2 in distPerim:
                    distPerim[dist2] += fidsToBrdLen[fidpair]
                else:
                    distPerim[dist2] = fidsToBrdLen[fidpair]
        totUnderThresh = 0
        pp = []
        for d in distPerim:
            polsbyPopper = 4*np.pi*distAreas[d]/distPerim[d]**2
            pp.append(polsbyPopper)
            # print(clusterName, polsbyPopper)
            if polsbyPopper > minPolsbyPopper:
                totUnderThresh += 1
        print(pp)
        if totUnderThresh > 0 and totUnderThresh == len(distPerim):
            pairedPlans.append(enumPlans[ii])
    return pairedPlans

################################################################################

def determinePolsbyPopper(clusterName, geoIDs, enumPlans):
    geoIDsToFID, fidToGeoID = mapGeoIDToFID(clusterName, geoIDs)
    fidToArea = getFidToData(clusterName, "AREAS")
    fidsToBrdLen = getFidPairToData(clusterName, "BORDERLENGTHS")

    compset= set()
    for ii in range(len(enumPlans)):
        plan = {geoIDsToFID[g] : d for g,d in zip(geoIDs, enumPlans[ii])}
        distAreas = {}
        for fid in plan:
            area = fidToArea[fid]
            dist = plan[fid]
            if dist in distAreas:
                distAreas[dist] += area
            else:
                distAreas[dist] = area
        distPerim = {}
        for fidpair in fidsToBrdLen:
            if fidpair[0] == "-1":
                dist = plan[fidpair[1]]
                if dist in distPerim:
                    distPerim[dist] += fidsToBrdLen[fidpair]
                else:
                    distPerim[dist] = fidsToBrdLen[fidpair]
            elif plan[fidpair[0]] != plan[fidpair[1]]:
                dist1 = plan[fidpair[0]]
                dist2 = plan[fidpair[1]]
                if dist1 in distPerim:
                    distPerim[dist1] += fidsToBrdLen[fidpair]
                else:
                    distPerim[dist1] = fidsToBrdLen[fidpair]
                if dist2 in distPerim:
                    distPerim[dist2] += fidsToBrdLen[fidpair]
                else:
                    distPerim[dist2] = fidsToBrdLen[fidpair]
        totUnderThresh = 0
        pp = []
        for d in distPerim:
            polsbyPopper = 4*np.pi*distAreas[d]/distPerim[d]**2
            pp.append(polsbyPopper)
        pp.sort()
        pptuple = tuple(pp)
        if pptuple in compset:
            print("WARNING -- found duplicate", pptuple)
        compset.add(pptuple)
    return compset

################################################################################

def checkContiguityInCounty(cnty, distToNodes, countyToFid, nbrList, 
                            fidToCounty):
    nodesInDistAndCnty = distToNodes[cnty]
    nodesInCnty = set(countyToFid[cnty])

    # print(nodesInDistAndCnty)

    visted = set([nodesInDistAndCnty[0]])
    queue = [nodesInDistAndCnty[0]]

    while len(queue) > 0:
        fid = queue.pop() 
        for fidnbr in nbrList[fid]:
            if fidnbr in visted:
                continue
            if fidnbr not in nodesInDistAndCnty:
                continue
            # have not been visted, is in county
            visted.add(fidnbr)
            queue.append(fidnbr)
    return (len(visted) == len(nodesInDistAndCnty))

################################################################################

def pairDownTraversals(clusterName, geoIDs, enumPlans):
    geoIDsToFID, fidToGeoID = mapGeoIDToFID(clusterName, geoIDs)
    fidsToBrdLen = getFidPairToData(clusterName, "BORDERLENGTHS")
    fidToGEOIDAndCounty = getFidToDataList(clusterName, "GEOIDS", dtype = str)
    fidToCounty = {f : fidToGEOIDAndCounty[f][1] for f in fidToGEOIDAndCounty}
    countyToFid = {}
    for f in fidToCounty:
        cnty = fidToCounty[f]
        if cnty in countyToFid:
            countyToFid[cnty].append(f)
        else:
            countyToFid[cnty] = [f]

    nbrList = {}
    for fidpair in fidsToBrdLen:
        if fidpair[0] == "-1":
            continue
        fid1 = fidpair[0]
        fid2 = fidpair[1]
        if fid1 in nbrList:
            nbrList[fid1].add(fid2)
        else:
            nbrList[fid1] = set([fid2])
        if fid2 in nbrList:
            nbrList[fid2].add(fid1)
        else:
            nbrList[fid2] = set([fid1])

    pairedPlans = []
    for ii in range(len(enumPlans)):
        plan = {geoIDsToFID[g] : d for g,d in zip(geoIDs, enumPlans[ii])}
        dists = list(set(list(plan.values())))
        contig = []
        for dist in dists:
            distToNodes = {}
            for f in plan:
                cnty = fidToCounty[f]
                if dist == plan[f]:
                    if cnty in distToNodes:
                        distToNodes[cnty].append(f)
                    else:
                        distToNodes[cnty] = [f]
            for cnty in distToNodes:
                contigHuh = checkContiguityInCounty(cnty, distToNodes, 
                                                    countyToFid, 
                                                    nbrList, fidToCounty)
                contig.append(contigHuh)
        if False in contig:
            continue
        pairedPlans.append(enumPlans[ii])
    return pairedPlans

################################################################################

def pairDownReock(clusterName, geoIDs, enumPlans, minReock = 0.15):
    geoIDsToFID, fidToGeoID = mapGeoIDToFID(clusterName, geoIDs)
    clusterSF = gpd.read_file(os.path.join(sfData, clusterName))

    pairedPlans = []
    for ii in range(len(enumPlans)):
        plan = {geoIDsToFID[g] : d for g,d in zip(geoIDs, enumPlans[ii])}
        planArray = np.empty(len(plan), dtype = int)
        for key in plan:
            planArray[int(key)] = plan[key]
        clusterSF['plan'] = planArray
        distSF = clusterSF.dissolve(by='plan')
        dists = list(distSF.index)

        reockScores = []
        for jj in dists:
            geom = distSF.loc[jj]['geometry']
            area = geom.area
            hull = list(geom.convex_hull.exterior.coords)
            circle = smallestCircle.make_circle(hull)
            reockScore = area/(np.pi*circle[2]**2)
            reockScores.append(reockScore)
        
        totUnderThresh = 0
        for rs in reockScores:
            if rs > minReock:
                totUnderThresh += 1
        if totUnderThresh > 0 and totUnderThresh == len(reockScores):
            pairedPlans.append(enumPlans[ii])
    return pairedPlans

################################################################################

def runElections(clusterName, geoIDs, enumPlans, elec):
    geoIDsToFID, fidToGeoID = mapGeoIDToFID(clusterName, geoIDs)
    fidToVotes = getFidToDataList(clusterName,  elec)

    elecMargins = []
    for ii in range(len(enumPlans)):
        plan = {geoIDsToFID[g] : d for g,d in zip(geoIDs, enumPlans[ii])}
        distVotes = {}
        for fid in plan:
            votes = fidToVotes[fid]
            dist = plan[fid]
            # print(distVotes, votes)
            if dist in distVotes:
                distVotes[dist][0] += votes[0]
                distVotes[dist][1] += votes[1]
            else:
                distVotes[dist] = votes[:2]
        # print(distVotes)
        margins = []
        for k in distVotes:
            demVotes = distVotes[k][0]
            repVotes = distVotes[k][1]
            margin = 100.0*demVotes/(repVotes + demVotes)
            margins.append(margin)
        margins.sort()
        elec = []
        for jj,m in enumerate(margins):
            # elec.append([jj+1,m])
            elec.append(m)
        elecMargins.append(elec)
    return elecMargins

################################################################################

def getFidToDataDict(clusterName, data, dtypeKey = str, dtypeVal = float):
    fileName = clusterName + "_" + data + ".txt"
    dataPath = os.path.join(extData, clusterName, fileName)
    fidToData = {}
    with open(dataPath) as dataf:
        for line in dataf:
            splitline = line.rstrip().replace("[","").replace("]","")
            splitline = splitline.split("\t")
            fid = splitline[0]
            dataDict = {}
            for jj in range(int((len(splitline)-1)/2)):
                ind = 2*jj+1
                key = dtypeKey(splitline[ind])
                val = dtypeVal(splitline[ind+1])
                dataDict[key] = val
            fidToData[fid] = dataDict
    return fidToData

################################################################################

def getPopDivs(plan, fidToPop, idealPop):
    distPops = {}
    for fid in fidToPop:
        pop = fidToPop[fid]
        dist = plan[fid]
        if dist in distPops:
            distPops[dist] += pop
        else:
            distPops[dist] = pop
    popDivs = [np.abs(p-idealPop)/float(idealPop) for p in distPops.values()]
    return popDivs

################################################################################

def getPPScores(plan, fidToArea, fidsToBrdLen):
    distAreas = {}
    for fid in plan:
        area = fidToArea[fid]
        dist = plan[fid]
        if dist in distAreas:
            distAreas[dist] += area
        else:
            distAreas[dist] = area
    distPerim = {}
    for fidpair in fidsToBrdLen:
        if fidpair[0] == "-1":
            dist = plan[fidpair[1]]
            if dist in distPerim:
                distPerim[dist] += fidsToBrdLen[fidpair]
            else:
                distPerim[dist] = fidsToBrdLen[fidpair]
        elif plan[fidpair[0]] != plan[fidpair[1]]:
            dist1 = plan[fidpair[0]]
            dist2 = plan[fidpair[1]]
            if dist1 in distPerim:
                distPerim[dist1] += fidsToBrdLen[fidpair]
            else:
                distPerim[dist1] = fidsToBrdLen[fidpair]
            if dist2 in distPerim:
                distPerim[dist2] += fidsToBrdLen[fidpair]
            else:
                distPerim[dist2] = fidsToBrdLen[fidpair]
    distPPScore = {}
    for d in distPerim:
        polsbyPopper = 4*np.pi*distAreas[d]/distPerim[d]**2
        distPPScore[d] = polsbyPopper
    return list(distPPScore.values())

################################################################################

def getSmallMCDSplits(plan, fidsToMCDs):
    mcdToDistPops = {}
    for fid in fidsToMCDs:
        dist = plan[fid]
        for mcd in fidsToMCDs[fid]:
            mcdPop = fidsToMCDs[fid][mcd]
            if mcd in mcdToDistPops:
                if dist in mcdToDistPops[mcd]:
                    mcdToDistPops[mcd][dist] += mcdPop
                else:
                    mcdToDistPops[mcd][dist] = mcdPop
            else:
                mcdToDistPops[mcd] = {}
                mcdToDistPops[mcd][dist] = mcdPop
    numPeopleOut = []
    for mcd in mcdToDistPops:
        mcdSplit = list(mcdToDistPops[mcd].values())
        if len(mcdSplit) > 1:
            mcdSplit.sort()
            numPeopleOut.append(sum(mcdSplit[:-1]))
    return numPeopleOut

################################################################################

def getEnergies(clusterName, geoIDs, enumPlans, idealPop = 79462.5):
    geoIDsToFID, fidToGeoID = mapGeoIDToFID(clusterName, geoIDs)

    fidToArea = getFidToData(clusterName, "AREAS")
    fidToPop = getFidToData(clusterName, "POPULATION")
    fidsToBrdLen = getFidPairToData(clusterName, "BORDERLENGTHS")
    fidsToMCDs = getFidToDataDict(clusterName, "MCDS")

    weights = energyWeights[clusterName]

    energies = []
    for ii in range(len(enumPlans)):
        plan = {geoIDsToFID[g] : d for g,d in zip(geoIDs, enumPlans[ii])}
        popDivs = getPopDivs(plan, fidToPop, idealPop)
        polsPopScores = getPPScores(plan, fidToArea, fidsToBrdLen)
        isoPerim = [4.0*np.pi/float(p) for p in polsPopScores]
        mcdSplits = getSmallMCDSplits(plan, fidsToMCDs)
        Jpop = np.sqrt(sum([p**2 for p in popDivs]))
        Jcomp = sum(isoPerim)
        Jmcd = sum(mcdSplits)
        # Jplan = Jpop*weights[0] + Jcomp*weights[1] + Jmcd*weights[2]
        Jplan =  Jmcd*weights[2]
        # print(isoPerim)
        # print(ii, Jpop, Jcomp, Jmcd, Jplan)
        energies.append(np.exp(-Jplan))
    return energies

################################################################################
