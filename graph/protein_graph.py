# This script is to generate protein graphs that run GCNN
# Author: Changpeng Lu
# Usage:
# python protein_graph.py -o HCV_selector_10_ang_sin_single_pairwise_substrate_covalent -pr_path /projects/f_sdk94_1/EnzymeModelling/CompleteSilentFiles -class HCV.txt -index_p1 7 -prot HCV.pdb -d 10
# right now, only suitable for single protease 

# Load packages
import os
import pickle as pkl
import pyrosetta as pr
from pyrosetta import *
from pyrosetta.rosetta.core.scoring import *
from pyrosetta.rosetta.core.pose import get_chain_from_chain_id, center_of_mass
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector, \
ResidueIndexSelector, NeighborhoodResidueSelector
from pyrosetta.rosetta.core.select import get_residues_from_subset
import logging
import pandas as pd
import numpy as np
import argparse

# functions
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="Output name")
    parser.add_argument("-pr_path", "--protease_path", default = "/projects/f_sdk94_1/EnzymeModelling/CompleteSilentFiles", help="Path to silent pose directory for protease")
    parser.add_argument("-class", "--classification_file", default = "HCV.txt", help="Name of txt for sequences to use, must be in folder")
    parser.add_argument("-index_p1", "--index_p1", type=int, default = 7, help="Index of p1 in the pdb, starting from 1.")
    parser.add_argument("-ub", "--upstream_buffer", type=int, default = 6, help="Upstream buffer from p1, starting from 1")
    parser.add_argument("-db", "--downstream_buffer", type=int, default = -1, help="Downstream buffer from p1, starting from 1")
    parser.add_argument("-prot", "--protein_template", default=None, \
                        help="protein template pdb name. It only can be used when all graphs are the same size.")
    parser.add_argument("-subind", "--substrate_indices", default=None, \
                        help="Instead of selecting residues using distance threshold, directly give pose indices.")
    parser.add_argument("-intind", "--interface_indices", default=None, \
                        help="Instead of selecting residues using distance threshold, directly give pose indices.")
    parser.add_argument("-sc", "--substrate_chain_id", default=2, type=int,\
                        help="chain ID of the substrate")
    parser.add_argument("-d","--select_distance", type=int, default=10, help="Distance for NeighborSelector")
    parser.add_argument("-is", "--is_silent", action='store_true', \
                        help="if input is in silent file mode, otherwise, just ignore this flag")
    parser.add_argument("-test", "--testset", action='store_true', help="if needed to save generated test index. only applicable for first time generation")
    parser.add_argument("-val", "--valset", action='store_true', help="if needed to save generated validation index. only applicable for first time generation")
    return parser.parse_args()

def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def graph_list_pickle(graph_ls, label_ls, sequence_ls, dataset_name, destination_path, export_indices=False, testset=False, valset=False):
    """Takes in a list of graphs and labels and pickles them in proper format. It also puts an index file in the directory"""
    # find number of classifications possible
    #if args.classifiers is None:
    s = set()
    for el in label_ls:
        s.add(el)
    s = list(s)
    s.sort()
#    num_classifiers = len(s)
 #   else:
  #          num_classifiers = len(args.classifiers)

    # get population size, number of nodes, and number of node features
    population = len(graph_ls)
    F = graph_ls[0].V.shape[1]
    M = graph_ls[0].A.shape[2]
    if export_indices:
        N = max([graph_ls[i].V.shape[0] for i in range(population)])
        pose_indices = np.full(shape = (population, N), fill_value = -1, dtype=np.int64)
    else:
        N = graph_ls[0].V.shape[0]
    # generate feature matrices
    x = np.zeros(shape = (population, N, F))
    y = np.zeros(shape = (population, num_classifiers), dtype = np.int64)
    graph = np.zeros(shape = (population, N, N, M))

    # populate all elements
    for i in range(population):
        print(sequence_ls[i])
        if not export_indices:
            x[i,:,:] = graph_ls[i].V
            graph[i,:,:,:] = graph_ls[i].A
        else:
            n = len(graph_ls[i].pose_indices)
            x[i,0:n,:] = graph_ls[i].V
            graph[i, 0:n, 0:n, :] = graph_ls[i].A
            pose_indices[i, 0:n] = graph_ls[i].pose_indices
        one_hot_label_vec = np.zeros(num_classifiers)
        one_hot_label_vec[s.index(label_ls[i])] = 1
        y[i, :] = one_hot_label_vec
    # get indices of the test set
    idx = [x for x in range(population)]
    np.random.shuffle(idx)
    if testset and valset:
        cutoff = int(0.8 * len(idx))
        cutoff_2 = int(0.9 * len(idx))
        idx_test = idx[cutoff_2:]
        idx_train = idx[:cutoff]
        idx_val = idx[cutoff: cutoff_2]
    else:
        test_fraction = .3
        cutoff = int(len(idx) * test_fraction)
        idx_test = idx[:cutoff]
    
    # pickle everything
    pkl.dump(x, open( os.path.join(destination_path,\
            "ind.{}.x".format(dataset_name)), "wb"))
    pkl.dump(y, open( os.path.join(destination_path,\
            "ind.{}.y".format(dataset_name)), "wb"))
    pkl.dump(graph, open( os.path.join(destination_path,\
            "ind.{}.graph".format(dataset_name)), "wb"))
    pkl.dump(sequence_ls, open( os.path.join(destination_path,\
            "ind.{}.sequences".format(dataset_name)), "wb"))
    pkl.dump(s, open( os.path.join(destination_path,\
            "ind.{}.labelorder".format(dataset_name)), "wb"))
    # single proteases
    # pkl.dump([dataset_name] * x.shape[0], open(os.path.join(destination_path, \
    #         "ind.{}.proteases".format(dataset_name)), "wb"))
    if testset == True:
        # save test index
        np.savetxt(os.path.join(destination_path, \
                "ind.{}.trisplit.test.index".format(dataset_name)), idx_test, fmt='%d')
    if valset == True:
        np.savetxt(os.path.join(destination_path, \
                "ind.{}.trisplit.val.index".format(dataset_name)), idx_val, fmt='%d')
    if export_indices:
        df = pd.DataFrame(pose_indices, index=sequence_ls)
        df.to_csv(os.path.join(destination_path, "{}_graphs_pose_indices.csv".format(dataset_name)))

def get_silent_file(sequence, path_to_silent_files):
    """This just returns an absolute path to the silent file (windows specific possibly) false if not found"""
    silent_file = None
    for silent in os.listdir(path_to_silent_files):
        correct = True
        for counter, char in enumerate(silent):
            if char != sequence[counter] and char != "_":
                correct = False
                break
        if correct:
            silent_file = silent
            break
    if silent_file == None:
        print("Silent dir for {} not found in {}!".format(sequence, path_to_silent_files))
        return False
    silent_dir = os.path.join(path_to_silent_files, silent_file)
    silent_file_path = os.path.join(silent_dir, silent_file)
    
    if os.path.exists(silent_file_path):
        return silent_file_path
    else:
        print("Silent file for {} not found {}!".format(sequence, silent_file_path))
        return False

def generate_dummy_silent(sequence, path_to_silent_files):
    silent_file = get_silent_file(sequence, path_to_silent_files)
    if not silent_file:
            return "Error: No Silent"
    with open(silent_file) as f:
        lineList = f.readlines()
    tag_ending = "substrate.{}".format(sequence)
    found, done = (False, False)
    ind, start, end, last_score = (0,0,0,0)
    while ind < len(lineList) and not done:
        x = lineList[ind]
        if "SCORE" in x:
            last_score = ind
        if not found and "ANNOTATED_SEQUENCE: " in x and tag_ending in x:
            start = last_score
            found = True
        elif found and "ANNOTATED_SEQUENCE: " in x:
            end = last_score
            done = True
        ind += 1
    if not found:
        print("The requested sequence {} was not found in the silent file {} (Parsing Error)".format(sequence, silent_file))
        raise ValueError("The requested sequence {} was not found in the silent file {} (Parsing Error)".format(sequence, silent_file))
    if not done:
        end = len(lineList)
    
    filename = sequence + str(np.random.randint(10000, 100000))
    path_bin = os.path.join(os.getcwd(), "bin")
    makedirs(path_bin)
    filename = os.path.join(path_bin, filename)
    with open(filename, "w") as f:
        # add header
        for i in range(3):
            f.write(lineList[i])
        # add binary information
        for i in range(start, end):
            f.write(lineList[i])
    return filename

def get_pose_from_pdb(sequence, path):
    #for pdb in os.listdir(path):
    #    if pdb == sequence:
    try:
        ret = pose_from_pdb(os.path.join(path, sequence))
        return ret
    except:
        return "Error: PDB file not exist"

def get_pose(sequence, path, is_silent = True):
    if is_silent == True:
        try:
            filename = generate_dummy_silent(sequence, path)
            for pose in poses_from_silent(filename):
                ret = pose
            os.remove(filename)
            return ret
        except:
            return "Error: Invalid Silent"        
    else:
        ret = get_pose_from_pdb(sequence, path)
        return ret

def index_substrate(pose, chain_id=2):
    """Takes a pose and returns the indices of the substrate."""
    # get substrate with built in selector
    #num_chains = pose.num_chains()
    chain_name = get_chain_from_chain_id(chain_id, pose)
    sub_sel = ChainSelector(chain_name)
    v1 = sub_sel.apply(pose)
    substrate_indices = []
    for count,ele in enumerate(v1):
        if ele:
            substrate_indices.append(count + 1)
    return substrate_indices

def index_substrate_cut_site(pose, index_p1 = 7, upstream_buffer = 6, downstream_buffer = 1, substrate_chainID = 2, protease = None):
    """This function takes the ROSETTA INDEX of the P1 residue for a substrate within its chain, a pose, and
    the number of upstream and downstream residues to model, and returns the indices of the substrate. If the
    buffer actually goes OOB of the substrate, a None type for that ind is instead returned for 0 pad modelling"""
    ind_sub = index_substrate(pose, chain_id=substrate_chainID)
    ind_active = []
    for i in range(-upstream_buffer, downstream_buffer):
        index_interest = i + index_p1
        if index_interest < 0 or index_interest >= len(ind_sub):
            ind_active.append(None)
        else:
            ind_active.append(ind_sub[index_interest])
    return ind_active

def selector_to_list(pose, selector):
    """
    Produces a list of residues from a pose identified by a given selector
    """

    return list(get_residues_from_subset(selector.apply(pose)))

def index_interface(pose,
                    substrate_indices,
                    d=10):
    """This function takes a pose and a number of interface/substrate to consider and returns interface indices. The
    value k and pose are not used..."""
    
    # Selection for neighbor residues
    focus_res = ','.join([str(j) for j in substrate_indices])
    focus_selector = ResidueIndexSelector(focus_res)

    interface = NeighborhoodResidueSelector()
    interface.set_focus_selector(focus_selector)
    interface.set_distance(d)
    interface.set_include_focus_in_subset(False)
    interface_indices = selector_to_list(pose, interface)
    interface_indices.sort()
            
    return interface_indices

def get_ind_from_protease(protease_name, pdb_path, index_p1, ub, db, sc, dis, sfxn):
    # load default pose as original
    pose = pose_from_pdb(os.path.join(pdb_path, protease_name))
    sfxn.score(pose)
    #substrate_ind = index_substrate(pose) #the whole substrate
    cutsite_ind = index_substrate_cut_site(pose, index_p1, upstream_buffer=ub, downstream_buffer=db, substrate_chainID=sc) #p2-p6 on the substrate
    interface_ind = index_interface(pose, cutsite_ind, dis)
    return cutsite_ind, interface_ind

class protein_graph:
    """This class is going to hold a graphical representation of a protein. It can be generated from two sources:
    a pose object, or the file path of a pdb. Since we are attempting to model a substrate/protein complex,
    the substrate and interface indices are ROSETTA(starting at 1) based indexes. When specified, these indices are
    the indices that are used as nodes. When not supplied, all indices are used. It is assumed that:
    
    The substrate's indices are the last in the pdb/pose
    When supplied interface and substrate are non-zero length
    The intersection of substrate and interface indices is empty
    Only canonical amino acids are not supported
    Possible Values:
    energy_terms = [fa_intra_sol_xover4, fa_intra_rep, rama_prepro, omega, p_aa_pp, fa_dun, ref]
    energy_edge_terms = [pro_close, fa_atr, fa_rep, fa_sol, fa_elec, lk_ball_wtd]"""
    
    def __init__ (self, substrate_indices = None,
                  interface_indices = None,
                  pdb_file_path = None,
                  pose = None,
                  params = dict(),
                  sfxn = None):

        # assure user provided a source
        if pdb_file_path == None and pose == None:
            raise PathNotDeclaredError("No pose or pdb path provided")
        
        # make pose from pdb
        if pdb_file_path != None:
            try:
                cleanATOM(pdb_file_path)##### Need to fix this #####
                pose = pose_from_pdb(pdb_file_path)
            except:
                raise PathNotDeclaredError("Failed to generate pose, file path invalid or other issue")
        
        # if substrate or interface indices are given we will make vertice_arr specially tailored
        if substrate_indices == None:
            ls = interface_indices
        else:
            ls = substrate_indices + interface_indices
            substrate_indices = np.array(substrate_indices)
        self.pose_indices = ls
        vertice_arr = np.array(ls)
        interface_indices = np.array(interface_indices)
        
        # Get All Node Features
        if params["amino_acids"]: num_amino = 20
        else: num_amino = 0
        num_dim_sine = params["sinusoidal_encoding"]
        energy_terms = len(params["energy_terms"])
        if params["coordinates"]: num_coord = 3
        else: num_coord = 0
        
        # Make and Apply Score Function
        if sfxn == None: sfxn = get_fa_scorefxn()
        sfxn(pose)
        energies = pose.energies()
        
        # Determine N (number of residues)
        N = len(vertice_arr)
        
        # Determine F (number of node features)
        F = sum([num_amino, num_dim_sine, energy_terms, num_coord])
        if params["substrate_boolean"]: F += 1
        
        # Initialize V (Feature Tensor NxF)
        self.V = np.zeros(shape = (N, F))
        
        # Determine M (number of edge features)
        M = 0
#         if params["distance"]: M += 1
        M += len(params["energy_edge_terms"])
        if params["interface_edge"]: M += 1
        if params["covalent_edge"]: M += 1
        if params["hbond"]: M += 1
        if params["distance"]: M += 25
        #M += np.sum(params['distance'])
        # initialize A (Multiple Adj. Mat. NxNxM)
        self.A = np.zeros(shape = (N, N, M))
        counter_F = 0
        counter_M = 0

        # One Hot Vectors for Amino Acid Type
        if params["amino_acids"]:
            all_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            seq = pose.sequence()        
            # use the native ordering to generate features
            for i in range(len(vertice_arr)):
                i_ind = vertice_arr[i]
                if i_ind != None:
                    res = seq[i_ind - 1]
                    j = all_amino_acids.find(res)
                    self.V[i][j] = 1
            counter_F += 20
        
        # Sinusoidal Positional Encoding
        if num_dim_sine != 0:
            if not substrate_indices.any() and not interface_indices.any():
                n_position = N
                position_enc = np.array([
                    [pos / np.power(10000, 2*i/num_dim_sine) for i in range(num_dim_sine)]
                    if pos != 0 else np.zeros(num_dim_sine) for pos in range(n_position)])
                position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
                position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
                self.V[0:n_position,counter_F:(counter_F + num_dim_sine)] = position_enc
            elif substrate_indices.any() and interface_indices.any():
                # add substrates
                n_position = len(substrate_indices)
                position_enc = np.array([
                    [pos / np.power(10000, 2*i/num_dim_sine) for i in range(num_dim_sine)]
                    if pos != 0 else np.zeros(num_dim_sine) for pos in range(n_position)])
                position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
                position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
                
                for i in range(len(substrate_indices)):
                    if substrate_indices[i] != None:
                        self.V[i, counter_F:(counter_F + num_dim_sine)] = position_enc[i, :]
                # add interface
                n_position = len(pose.sequence()) - len(substrate_indices)
                position_enc = np.array([
                    [pos / np.power(10000, 2*i/num_dim_sine) for i in range(num_dim_sine)]
                    if pos != 0 else np.zeros(num_dim_sine) for pos in range(n_position)])
                position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
                position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
                for i in range(N - len(substrate_indices)):
                    if interface_indices[i] != None:
                        self.V[(len(substrate_indices) + i), counter_F:(counter_F + num_dim_sine)] = position_enc[i, :]
            else:
                # add substrates
                n_position = len(substrate_indices)
                position_enc = np.array([
                    [pos / np.power(10000, 2*i/num_dim_sine) for i in range(num_dim_sine)]
                    if pos != 0 else np.zeros(num_dim_sine) for pos in (substrate_indices - substrate_indices[0])])
                position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
                position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
                self.V[0:n_position,counter_F:(counter_F + num_dim_sine)] = position_enc
            counter_F += num_dim_sine

        # Single Body Energy Terms
        for counter, term in enumerate(params["energy_terms"], counter_F):
            for i in range(N):
                if vertice_arr[i] != None:
                    self.V[i, counter] = energies.residue_total_energies(vertice_arr[i])[term]
        counter_F += energy_terms

        if params["coordinates"]:
            for i in range(len(vertice_arr)):
                if vertice_arr[i] != None:
                    # N CA C O CB
                    for atm in ['N','CA','C','O','CB']:
                        try: 
                            coord = np.array(pose.residue(vertice_arr[i]).xyz(atm))
                        except RuntimeError:
                            CA_coord = np.array(pose.residue(vertice_arr[i]).xyz("CA"))
                            N_coord = np.array(pose.residue(vertice_arr[i]).xyz("N"))
                            C_coord = np.array(pose.residue(vertice_arr[i]).xyz("C"))
                            b = CA_coord - N_coord
                            c = C_coord - CA_coord
                            a = np.cross(b,c)
                            coord = -0.58273431*a + 0.56802827*b - 0.54067466*c + CA_coord
                        self.V[i, counter_F : (counter_F + 3)] = coord
                        counter_F += 3
            #print(self.V[i, counter_F : (counter_F + 3)])

        
        
        # New node feature
        """
        if params["new feature"]:
            self.V[:, counter_F] = whatever
            counter_F += 1
        """
        
        # Substrate boolean
        if params["substrate_boolean"]:
            self.V[0:len(substrate_indices),counter_F] = np.array([1 for x in range(len(substrate_indices))])

        # Total Two Body Energy and Energy Terms
        if len(params["energy_edge_terms"]) != 0:
            for i in range(len(vertice_arr)):
                for j in range(i, len(vertice_arr)):
                    if vertice_arr[i] != None and vertice_arr[j] != None:
                        if i != j:
                            rsd1 = pose.residue(vertice_arr[i])
                            rsd2 = pose.residue(vertice_arr[j])
                            emap = EMapVector()
                            sfxn.eval_ci_2b(rsd1, rsd2, pose, emap)
                            for counter, term in enumerate(params["energy_edge_terms"]):
                                self.A[i, j, counter_M + counter] = emap[term]
                                self.A[j, i, counter_M + counter] = emap[term]
            counter_M += len(params["energy_edge_terms"])

        # Hydrogen Bonding Energies
        if params["hbond"]:
            hbs=pose.get_hbonds()
            res_dict = dict()
            for res in vertice_arr:
                hbl = hbs.residue_hbonds(res)
                for hb in hbl:
                    residues = (hb.don_res(), hb.acc_res())
                    if residues[0] > residues[1]: residues = (hb.acc_res(), hb.don_res())
                    if residues[0] in vertice_arr and residues[1] in vertice_arr: res_dict[residues] = hb.energy()
            for residues in res_dict:
                for i in np.where(vertice_arr==residues[0])[0]:
                    for j in np.where(vertice_arr==residues[1])[0]:
                        self.A[i,j,counter_M] += self.A[i,j,counter_M] + res_dict[residues]
                        self.A[j,i,counter_M] += self.A[j,i,counter_M] + res_dict[residues]
            counter_M += 1

        # Protease - Substrate Interactions Boolean
        if params["interface_edge"]:
            self.A[0:len(substrate_indices), len(substrate_indices):len(vertice_arr), counter_M] = 1
            self.A[len(substrate_indices):len(vertice_arr), 0:len(substrate_indices), counter_M] = 1
            counter_M += 1

        # Covalent Bond Connection Boolean
        if params["covalent_edge"]:
            for i in range(len(vertice_arr) - 1):
                if vertice_arr[i + 1] - vertice_arr[i] == 1:
                    self.A[i, i + 1, counter_M] = 1
                    self.A[i + 1, i, counter_M] = 1
            counter_M += 1

        if params["distance"]:
            atom_types = np.array(["N","CA","C","O"])
            # first, get coordinates arrays
            rsd_coords = np.zeros((len(vertice_arr), 5, 3))
            for i in range(len(vertice_arr)):
                rsd = pose.residue(vertice_arr[i])
                for k,atm in enumerate(atom_types):
                    coord = np.array(pose.residue(vertice_arr[i]).xyz(atm))
                    rsd_coords[i, k,:] = coord
            for i in range(len(vertice_arr)):
                try:
                    CB_coord = np.array(pose.residue(vertice_arr[i]).xyz("CB"))
                except RuntimeError:
                    b = rsd_coords[i,1,:] - rsd_coords[i,0,:]
                    c = rsd_coords[i,2,:] - rsd_coords[i,1,:]
                    a = np.cross(b,c)
                    CB_coord = -0.58273431*a + 0.56802827*b - 0.54067466*c + rsd_coords[i,1,:]
                rsd_coords[i, 4, :] = CB_coord
            for ki in range(5):
                for kj in range(5):
                    for i in range(len(vertice_arr)):
                        for j in range(i, len(vertice_arr)):
                            if vertice_arr[i] != None and vertice_arr[j] != None:
                                if i != j and ki != kj:
                                    coord1 = rsd_coords[i,ki,:]
                                    coord2 = rsd_coords[j,kj,:]
                                    dis = np.linalg.norm(coord1 - coord2)
                                    self.A[i, j, counter_M] = dis
                                    self.A[j, i, counter_M] = dis 
                    counter_M += 1

# Goes from a sequence to a graph representation.
def generate_graph(seq, pr_path, substrate_ind, interface_ind, params, sfxn, is_silent=False):
    pose = get_pose(seq, pr_path, is_silent=is_silent)
    if type(pose) == type("string"):
        return pose
    g = protein_graph(pose = pose,
                       substrate_indices = substrate_ind,
                       interface_indices = interface_ind,
                       sfxn = sfxn,
                       params = params)
    return g

def main(args):
    #preset
    pr.init()
    sfxn = get_fa_scorefxn()
    classifier_path = "classifications/"
    data_path = "../data" 
    pdb_path = "crystal_structures"

    class_file = args.classification_file #list of samples
    output = args.output 
    pr_path = args.protease_path
    index_p1 = args.index_p1
    ub = args.upstream_buffer
    db = args.downstream_buffer
    sc = args.substrate_chain_id
    protein_template = args.protein_template
    dis = args.select_distance
    substrate_indices = args.substrate_indices
    interface_indices = args.interface_indices
    logger = get_logger(logpath=os.path.join(data_path, 'logs'), filepath=os.path.abspath(__file__))
    
    params = {"amino_acids":True,
                "sinusoidal_encoding":0,
                "coordinates": False,
                "distance": False, # N, CA,C,O,CB distance mask
                "substrate_boolean":True,
                "energy_terms":[fa_intra_sol_xover4, fa_intra_rep, rama_prepro, omega, p_aa_pp, fa_dun, ref],
                "energy_edge_terms":[fa_atr, fa_rep, fa_sol, fa_elec, lk_ball_wtd],
                "hbond": True,
                "interface_edge": True,
                "covalent_edge": True,}
    logger.info("Features Info: {}".format(params))
    if protein_template:
        if substrate_indices == None and interface_indices == None:
            cutsite_ind, interface_ind = get_ind_from_protease(protein_template, pdb_path, index_p1, ub, db, sc, dis, sfxn)
        else:
            cutsite_ind = [int(x) for x in substrate_indices.strip('][').split(', ')]
            interface_ind = [int(x) for x in interface_indices.strip('][').split(', ')]
        if cutsite_ind != None:
            logger.info("Focus substrate indices are {}".format(','.join([str(u) for u in cutsite_ind])))
        if interface_ind != None:
            logger.info("Neighbor residues indices are {}".format(','.join([str(q) for q in interface_ind])))
    else:
        cutsite_ind, interface_ind = [], []

    
    # Read in labels and sequences
    try:
        df = pd.read_csv(os.path.join(classifier_path, class_file), sep = "\t")
        labels = list(df["Result"])
        sequences = list(df["Sequence"])
    except:
        raise ValueError("Path either invalid to classsifications or not properly formatted. \
    Please check template sample.txt")
    
    # get all graphs into a list
    missed_sequences = []
    error_sequences = []
    seq_final = []
    label_final = []
    graphs = []
    for i in range(len(sequences)):
        seq = sequences[i]
        print(seq)
        if protein_template:
            graph = generate_graph(seq, pr_path, cutsite_ind, interface_ind, params, sfxn, is_silent=args.is_silent)
            V = graph.V
        else:
            cutsite_ind, interface_ind = get_ind_from_protease(seq, pr_path, index_p1, ub, db, sc, dis, sfxn)
            graph = generate_graph(seq, pr_path, cutsite_ind, interface_ind, params, sfxn, is_silent=args.is_silent)
        if graph == "Error: No Silent":
            missed_sequences.append(seq)
        elif graph == "Error: Invalid Silent":
            error_sequences.append(seq)
        else:
            seq_final.append(seq)
            graphs.append(graph)
            label_final.append(labels[i])
            logger.info("Graph for {} has been generated successfully.".format(seq)) 
    logger.info("There were {} poses which loaded".format(len(graphs)))
    logger.info("There were {} poses missing due to silent files.".format(len(missed_sequences)))
    logger.info("There were {} poses which failed to be loaded.".format(len(error_sequences)))
    if protein_template:
        graph_list_pickle(graphs, label_final, seq_final, output, data_path, testset=args.testset, valset=args.valset)
    else:
        graph_list_pickle(graphs, label_final, seq_final, output, data_path, testset=args.testset, valset=args.valset, export_indices=True)

    
    
if __name__ == '__main__':
    args = parse_args()
    logger.info(args)
    main(args)
