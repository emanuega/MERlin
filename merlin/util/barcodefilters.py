import numpy as np
from scipy.spatial import cKDTree
import networkx as nx
import pandas as pd
from typing import List


def remove_zplane_duplicates_all_barcodeids(barcodes: pd.DataFrame,
                                            zPlanes: int,
                                            maxDist: float,
                                            allZPos: List) -> pd.DataFrame:
    """ Depending on the separation between z planes, spots from a single
        molecule may be observed in more than one z plane. These putative
        duplicates are removed based on supplied distance and z plane
        constraints. In evaluating this method, when z planes are separated
        by 1.5 µm the likelihood of finding a putative duplicate above or below
        the selected plane is ~5-10%, whereas the false-positive rate is closer
        to 1%, as determined by checking two planes above or below, or comparing
        barcodes of different identities but similar abundance between
        adjacent z planes.

    Args:
        barcodes: a pandas dataframe containing all the entries for a given
                  barcode identity
        zPlanes: number of planes above and below to consider when evaluating
                 potential duplicates
        maxDist: maximum euclidean distance allowed to separate centroids of
                 putative barcode duplicate, in pixels
    Returns:
        keptBarcodes: pandas dataframe where barcodes of the same identity that
                      fall within parameters of z plane duplicates have
                      been removed.
    """
    if len(barcodes) == 0:
        return barcodes
    else:
        barcodeGroups = barcodes.groupby('barcode_id')
        bcToKeep = []
        for bcGroup, bcData in barcodeGroups:
            bcToKeep.append(
                remove_zplane_duplicates_single_barcodeid(bcData, zPlanes,
                                                          maxDist, allZPos))
        mergedBC = pd.concat(bcToKeep, 0).reset_index(drop=True)
        mergedBC = mergedBC.sort_values(by=['barcode_id', 'z'])
        return mergedBC


def remove_zplane_duplicates_single_barcodeid(barcodes: pd.DataFrame,
                                              zPlanes: int,
                                              maxDist: float,
                                              allZPos: List) -> pd.DataFrame:
    """ Remove barcodes with a given barcode id that are putative z plane
        duplicates.

    Args:
        barcodes: a pandas dataframe containing all the entries for a given
                  barcode identity
        zPlanes: number of planes above and below to consider when evaluating
                 potential duplicates
        maxDist: maximum euclidean distance allowed to separate centroids of
                 putative barcode duplicate, in pixels
    Returns:
        keptBarcodes: pandas dataframe where barcodes of the same identity that
                      fall within parameters of z plane duplicates have
                      been removed.
    """
    barcodes.reset_index(drop=True, inplace=True)
    if not len(barcodes['barcode_id'].unique()) == 1:
        errorString = 'The method remove_zplane_duplicates_single_barcodeid ' +\
                      'should be given a dataframe containing molecules ' +\
                      'that all have the same barcode id. Please use ' +\
                      'remove_zplane_duplicates_all_barcodeids to handle ' +\
                      'dataframes containing multiple barcode ids'
        raise ValueError(errorString)
    graph = nx.Graph()
    zPos = sorted(allZPos)
    graph.add_nodes_from(barcodes.index.values.tolist())
    for z in range(0, len(zPos)):
        zToCompare = [pos for pos, otherZ in enumerate(zPos) if
                      (pos >= z - zPlanes) & (pos <= z + zPlanes) & ~(pos == z)]
        treeBC = barcodes[barcodes['z'] == z]
        if len(treeBC) == 0:
            pass
        else:
            tree = cKDTree(treeBC.loc[:, ['x', 'y']].values)
            for compZ in zToCompare:
                queryBC = barcodes[barcodes['z'] == compZ]
                if len(queryBC) == 0:
                    pass
                else:
                    dist, idx = tree.query(queryBC.loc[:, ['x', 'y']].values,
                                           k=1, distance_upper_bound=maxDist)
                    currentHits = treeBC.index.values[idx[np.isfinite(dist)]]
                    comparisonHits = queryBC.index.values[np.isfinite(dist)]
                    graph.add_edges_from(list(zip(currentHits, comparisonHits)))
        connectedComponents = [list(x) for x in
                               list(nx.connected_components(graph))]

    def choose_brighter_barcode(barcodes, indexes):
        sortedBC = barcodes.loc[indexes, :].sort_values(by='mean_intensity',
                                                        ascending=False)
        return sortedBC.index.values.tolist()[0]

    keptBarcodes = barcodes.loc[sorted([x[0] if len(x) == 1 else
                                        choose_brighter_barcode(barcodes, x)
                                        for x in connectedComponents]), :]
    return keptBarcodes
