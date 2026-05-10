#!/usr/bin/python
# -*- coding:utf-8 -*-
import networkx as nx
import numpy as np
from networkx.algorithms import isomorphism
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem.rdchem import Mol as RDKitMol
from typing import List, Tuple, Dict, Union


atomic_radii = dict(
    B=0.83, Br=1.21, C=0.68, Cl=0.99, F=0.64, H=0.23, I=1.40, N=0.68, O=0.68, P=1.05, S=1.02, Se=1.22, Si=1.20
)


def _mol_to_topology(mol: Union[RDKitMol, str], include_Hs: bool=False):
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
        Chem.Kekulize(mol, True)
    g = nx.Graph()
    in_graph = []
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        symbol = atom.GetSymbol()
        if symbol != 'H' or include_Hs:
            g.add_node(i, atom=symbol)
            in_graph.append(True)
        else:
            in_graph.append(False)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if in_graph[i] and in_graph[j]:
            g.add_edge(i, j, bond_type=bond.GetBondType())
    return g


def get_atom_map(g1: Union[nx.Graph, RDKitMol, str], g2: Union[nx.Graph, RDKitMol, str]) -> Dict:
    if not isinstance(g1, nx.Graph):
        g1 = _mol_to_topology(g1)
    if not isinstance(g2, nx.Graph):
        g2 = _mol_to_topology(g2)
    gm = isomorphism.GraphMatcher(g1, g2, node_match=lambda n1, n2: n1['atom'] == n2['atom'])
    assert gm.is_isomorphic(), f'g1 node {len(g1)}, g2 node {len(g2)}'
    return gm.mapping


def struct_to_topology(atoms: List[str], coordinates: List[Tuple[float, float, float]]) -> nx.Graph:
    node_ids = list(range(len(atoms)))
    coordinates = np.array(coordinates)
    dist = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    dist = np.linalg.norm(dist, axis=-1)

    radius = np.array([atomic_radii[atom] for atom in atoms])
    dist_bond = (radius[:, np.newaxis] + radius[np.newaxis, :]) * 1.3

    adj_mat = np.logical_and(0.1 < dist, dist_bond > dist)
    g = nx.Graph()
    for i in node_ids:
        g.add_node(i, atom=atoms[i])
    for i, j in zip(*np.nonzero(adj_mat)):
        g.add_edge(int(i), int(j))
    return g


def struct_to_bonds(atoms: List[str], coordinates: List[Tuple[float, float, float]], smiles: str, include_Hs: bool=False) -> Tuple[int, int, int]:
    bond2id = {
        BondType.SINGLE: 1,
        BondType.DOUBLE: 2,
        BondType.TRIPLE: 3,
        BondType.AROMATIC: 4
    }

    g1 = _mol_to_topology(smiles, include_Hs)
    g2 = struct_to_topology(atoms, coordinates)
    matching = get_atom_map(g1, g2)
    bonds = []
    for edge in g1.edges.data():
        i, j, attr = edge
        bonds.append((matching[i], matching[j], bond2id[attr['bond_type']]))
    return bonds
