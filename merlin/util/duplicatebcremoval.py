import numpy as np
from scipy.spatial import cKDTree
import networkx as nx
import pandas as pd
from random import choice

def cleanup_across_z(barcodes: pd.DataFrame, zPlanes: int, maxDist: float) \
        -> pd.DataFrame:
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
    barcodes.reset_index(drop = True, inplace = True)
    graph = nx.Graph()
    zPos = sorted(barcodes['z'].unique())
    graph.add_nodes_from(barcodes.index.values.tolist())
    for i in range(0, len(zPos)):
        z = zPos[i]
        zToCompare = [otherZ for pos, otherZ in enumerate(zPos) if
                      (pos >= i - zPlanes) & (pos <= i + zPlanes) & ~(pos == i)]
        treeBC = barcodes[barcodes['z'] == z]
        tree = cKDTree(treeBC.loc[:, ['x', 'y']].values)
        for compZ in zToCompare:
            queryBC = barcodes[barcodes['z'] == compZ]
            dist, idx = tree.query(queryBC.loc[:, ['x', 'y']].values, k=1,
                                   distance_upper_bound=maxDist)
            currentHits = treeBC.index.values[idx[np.isfinite(dist)]]
            comparisonHits = queryBC.index.values[np.isfinite(dist)]
            graph.add_edges_from(list(zip(currentHits, comparisonHits)))
    connectedComponents = [list(x) for x in list(nx.connected_components(graph))]
    keptBarcodes = barcodes.loc[sorted([x[0] if len(x) == 1 else choice(x) for x
                                        in connectedComponents]), :]
    return keptBarcodes
