"""
A individual-based model of P. pacificus development and population dynamics.
"""

__author__ = 'Ata Kalirad'

__version__ = '1.0'

import numpy as np
import random as rnd
import pandas as pd
import os
import uuid
import pickle
from tqdm import tqdm 
from itertools import cycle

states = ('E', 'J', 'J2A', 'A', 'dead', 'dauer')

# import fecundity and MF data

file_names_unfiltered = [file for file in os.listdir('./sim_inputs/') if not file.startswith('.')]

file_names = [i for i in file_names_unfiltered if 'nv' not in i]

file_names_nv = [i for i in file_names_unfiltered if 'nv' in i]
    
fec_data = {i[:-4]: pd.read_csv('./sim_inputs/'+i, index_col=0) for i in file_names}

fec_data_nv = {i[:-4]: pd.read_csv('./sim_inputs/'+i, index_col=0) for i in file_names_nv}
    
MF_data = {i:list(pd.read_csv('./sim_inputs/MF_'+i+'.csv', index_col=0).iloc[:,0]) for i in ['P_OP50', 'MP_OP50', 'P_Novo', 'MP_Novo']}
    
def sigmoid(x, state, strain, diet):
    """_summary_

    Args:
        x (int): current age of the worm
        state (str): current state of 
        strain (str): strain of the worm
        diet (str): current diet

    Returns:
        float : transition probability
    """
    L = 1
    k = 0.7
    if state == 'E':
        x0 = 24
    elif state == 'J':
        if (strain == 'P' or strain == 'MP') and diet == 'Novo':
            x0 = 48
        elif strain == 'NP' and diet == 'Novo':
            x0 = 43
        else:
            x0 = 48
    elif state == 'J2A':
        x0 = 24
    elif state == 'A':
        x0 = 200
    elif state == 'dauer':
        x0 = 1000
    return np.divide(L, 1 + np.exp(-1*k*(x - x0)))

class Population(object):
    """Population object
    """
    
    def __init__(self, dim, resource_type='OP50', rw_l=5, walk_bias=(1, 1), recog_par=0.95, pred_par=0.1, mig_rate=0.05, egg_var=True, loc_save=False):
        """Initiali

        Args:
            dim (tuple): Dimensions of the lattice.
            resource_type (str, optional): The bacterial diet. Defaults to 'OP50'.
            rw_l (int, optional): Length of random walk at each step. Defaults to 5.
            walk_bias (tuple, optional): Bias of the random walk. Defaults to (1, 1).
            recog_par (float, optional): Fidelity of the self-recognition system. Defaults to 0.95.
            pred_par (float, optional): Predation probability. Defaults to 0.1.
            mig_rate (float, optional): Dauer larvae migration rate. Defaults to 0.05.
            egg_var (bool, optional): Variation of fecundity. Defaults to True.
            loc_save (bool, optional): Save the location of worms during the simulation. Defaults to False.
        """
        assert resource_type == 'OP50' or 'Novo'
        self.index = [(i,j) for i in range(dim[0]) for j in range(dim[1])]
        self.n_loc = len(self.index)
        self.top_layer = {}
        self.middle_layer = {}
        self.bottom_layer = {}
        self.eggs = {k:[] for k in self.index}
        self.funerary = []
        self.funerary_J2A_arrest = []
        self.occupancy_time = {'NP':{j:[] for i,j in enumerate(['E', 'J', 'J2A', 'A'])}, 'MP':{j:[] for i,j in enumerate(['E', 'J', 'J2A', 'A'])}, 'P':{j:[] for i,j in enumerate(['E', 'J', 'J2A', 'A'])}}
        self.resource_type = resource_type
        # Generate Moore neighborhood for egg-laying and random_walk
        dist=1
        self.moore_neighbor = {i:list(set([(x,y) for x in np.arange(i[0] - dist, i[0] + dist + 1) for y in np.arange(i[1] - dist, i[1] + dist + 1)]).intersection(set(self.index))) for i in self.index}
        self.rw_l=rw_l
        self.walk_bias = walk_bias
        self.recog_par = recog_par
        self.n_spots_t = []
        self.maternity_dic = {}
        if pred_par > 0:
            self.pred = True
        else:
            self.pred = False
        self.pred_par = pred_par
        self.mig_rate = mig_rate
        self.dauer_migrated = {'NP':[], 'MP':[], 'P':[]}
        self.func_response = {'NP':[], 'MP':[], 'P':[]}
        self.egg_var = egg_var
        self.loc_save = loc_save
        
    @property 
    def empty_loc_top(self):
        """Empty locations on the top lattice.

        Returns:
            set
        """
        return set(self.index).difference(self.top_layer.keys())
    
    @property
    def empty_loc_middle(self):
        """Empty locations on the middle lattice.

        Returns:
            set
        """
        return set(self.index).difference(self.middle_layer.keys())
        
    @property
    def empty_loc_bottom(self):
        """Empty locations on the bottom lattice.

        Returns:
            set
        """
        return set(self.index).difference(self.bottom_layer.keys())


    def random_walk_2d(self, loc, empty_spots, pot_prey_spots=False):
        """Simple random walk algorithm

        Args:
            loc (tuple): Current location of the worm.
            empty_spots (list): Available spots on the lattice.
            pot_prey_spots (bool, optional): List of preys in the population. Defaults to False.

        Returns:
            list: The trajectory of the random walk.
        """
        step_taken = 0
        trajectory = []
        curr_loc = loc
        trajectory.append(curr_loc)
        while step_taken < self.rw_l:
            curr_neighbors = list(set(self.moore_neighbor[curr_loc]).intersection(empty_spots))
            if len(curr_neighbors) == 0:
                break
            else:
                if not pot_prey_spots:
                    next_loc = curr_neighbors[rnd.randint(0, len(curr_neighbors) - 1)]
                else:
                    w_steps = [self.walk_bias[0] if i not in pot_prey_spots else self.walk_bias[1] for i in curr_neighbors]
                    if np.sum(w_steps) == 0:
                        break
                    else:
                        next_loc = rnd.choices(list(curr_neighbors), weights=w_steps)[0]
                curr_loc = next_loc
            trajectory.append(curr_loc)
            step_taken += 1
        return trajectory
    
    def move_worms(self):
        """Move juvenile and adult worms and trace killing events.
        """
        curr_pos_top = [*self.top_layer]
        curr_pos_mid = [(k[0], k[1], 'mid') for i, (k,v) in enumerate(self.middle_layer.items()) if v[0] == 'J']
        mixed_pos = curr_pos_top + curr_pos_mid
        J2A = [k for i, (k,v) in enumerate(self.bottom_layer.items()) if v[0] == 'J2A']
        rnd.shuffle(mixed_pos)
        for worm in mixed_pos:
            if len(worm) == 3 and (worm[0], worm[1]) in self.middle_layer:
                    trajectory = self.random_walk_2d((worm[0], worm[1]), self.empty_loc_middle)
                    new_pos = trajectory[-1]
                    if new_pos != (worm[0], worm[1]):
                        self.middle_layer[new_pos] = self.middle_layer[(worm[0], worm[1])]
                        del self.middle_layer[(worm[0], worm[1])]
            elif len(worm) == 2:
                if self.walk_bias[0] == self.walk_bias[1]:
                    trajectory = self.random_walk_2d(worm, self.empty_loc_top )
                else:
                    J_D = list(self.middle_layer.keys())
                    pre_adult_spots = J_D + J2A
                    trajectory = self.random_walk_2d(worm, self.empty_loc_top, pre_adult_spots)
                new_pos = trajectory[-1]
                if new_pos != worm:
                    self.top_layer[new_pos] = self.top_layer[worm]
                    del self.top_layer[worm]
                if self.pred and self.top_layer[new_pos][3]=='EU':
                    pot_preys = {**{(k, 'bottom'):v[2] for i, (k,v) in enumerate(self.bottom_layer.items()) if k in set(trajectory) and v[0] == 'J2A'}, **{(k, 'middle'):v[2] for i, (k,v) in enumerate(self.middle_layer.items()) if k in set(trajectory)}}
                    recognized_prob = np.random.binomial(1, self.recog_par, size=len(pot_preys))
                    recognized = {k:(v if recognized_prob[i] else 'MC') for i, (k,v) in enumerate(pot_preys.items())}
                    preys = [k for i, (k,v) in enumerate(recognized.items()) if v!= self.top_layer[new_pos][2] or v=='MC']
                    if len(preys) > 0:
                        killed = np.random.binomial(1, self.pred_par, size=len(preys))
                        killed = np.where(killed==1)[0]
                        if len(killed) > 0:
                            for i in killed:
                                if preys[i][1] == 'middle':
                                    del self.middle_layer[preys[i][0]]
                                else:
                                    del self.bottom_layer[preys[i][0]]
                                    J2A.remove(preys[i][0])
                                    
    def transfer_layers(self, top, mid, bottom):
        """Replace population lattices

        Args:
            top (dict)
            mid (dict)
            bottom (dict)
        """
        self.top_layer = top
        self.middle_layer = mid
        self.bottom_layer = bottom
        
    def add_worm(self, worm):
        """Add worm to the population

        Args:
            worm (tuple)
        """
        if not worm[4]:
            worm = (worm[0], worm[1], worm[2], worm[3], str(uuid.uuid4()))
        spots_top = list(set(self.index).difference(self.top_layer.keys()))
        spots_middle = list(set(self.index).difference(self.middle_layer.keys()))
        spots_bottom = list(set(self.index).difference(self.bottom_layer.keys()))
        if worm[0] == 'E' or worm[0] == 'J2A':
            if len(spots_bottom) > 0:
                rand_pos =  np.random.choice(range(len(spots_bottom)))
                self.bottom_layer[spots_bottom[rand_pos]] = worm
        elif worm[0] == 'J' or worm[0] == 'dauer':
            if len(spots_middle) > 0:
                rand_pos =  np.random.choice(range(len(spots_middle)))
                self.middle_layer[spots_middle[rand_pos]] = worm
        else:
            if len(spots_top) > 0:
                rand_pos =  np.random.choice(range(len(spots_top)))
                if worm[3] == 'ND':
                    if worm[2] == 'NP':
                        eu_mf = 1
                    else:
                        eu_mf_prob = np.random.choice(MF_data[worm[2] + '_' + self.resource_type])
                        eu_mf = np.random.binomial(1, eu_mf_prob)
                    worm_mf = 'ST'
                    if eu_mf:
                        worm_mf = 'EU'
                    self.top_layer[spots_top[rand_pos]] =  (worm[0], worm[1], worm[2], worm_mf, worm[4])
                else:
                    self.top_layer[spots_top[rand_pos]] = worm
        
    def lay_eggs(self, age_lim=120):
        """Lay eggs according to the fecundity model

        Args:
            age_lim (int, optional): The maximum age of egg-laying for an adult. Defaults to 120.
        """
        pot_mothers = [k for i, (k,v) in enumerate(self.top_layer.items()) if v[1] < age_lim]
        egg_laying_spots = {k:list(set(self.moore_neighbor[k]).intersection(self.empty_loc_bottom)) for k in pot_mothers}
        mothers = {k:v for i, (k,v) in enumerate(egg_laying_spots.items()) if len(v) > 0}
        spots_used = []
        rnd_order = np.random.choice(range(len(mothers)),size=len(mothers), replace=False)
        keys = list(mothers.keys())
        for i in rnd_order:
            available_spot = list(set(egg_laying_spots[keys[i]]).difference(set(spots_used)))
            n_available_spot = len(available_spot)
            ###
            self.n_spots_t.append(n_available_spot)
            ###
            if n_available_spot > 0:
                mom_id = self.top_layer[keys[i]][4]
                if mom_id not in self.maternity_dic:
                    if self.egg_var:
                        mat_key = self.top_layer[keys[i]][2] + '_' + self.resource_type
                        self.maternity_dic[mom_id] = list(fec_data[mat_key][str(np.random.randint(0, 10000))])
                    else:
                        mat_key = self.top_layer[keys[i]][2] + '_' + self.resource_type + '_nv'
                        self.maternity_dic[mom_id] = list(fec_data_nv[mat_key][str(np.random.randint(0, 10000))])
                curr_age = self.top_layer[keys[i]][1]
                n_eggs = self.maternity_dic[mom_id][curr_age]
                if n_eggs > 0:
                    if n_eggs > n_available_spot:
                        n_eggs = n_available_spot
                    for loc in range(n_eggs):
                        self.bottom_layer[available_spot[loc]] = ('E', 0, self.top_layer[keys[i]][2], 'ND', str(uuid.uuid4())) 
                        spots_used.append(available_spot[loc])
                if curr_age == (age_lim - 1):
                    del self.maternity_dic[mom_id]
                
    def develop_top_layer(self):
        transition_probs = {k:sigmoid(v[1], v[0], v[2], self.resource_type) for i, (k,v) in enumerate(self.top_layer.items())}
        if len(transition_probs) > 0:
            developed = {k:np.random.binomial(1, v) for i, (k,v) in enumerate(transition_probs.items())}
            developed = [k for i, (k,v) in enumerate(developed.items()) if v == 1]
            for i in developed:
                next_state = 'dead'
                curr_age = self.top_layer[i][1]
                self.occupancy_time[self.top_layer[i][2]]['A'].append(curr_age)
                self.top_layer[i] = (next_state, 0, self.top_layer[i][2], self.top_layer[i][3], self.top_layer[i][4])
                            
    def develop_middle_layer(self):
        transition_probs = {k:sigmoid(v[1], v[0], v[2], self.resource_type) for i, (k,v) in enumerate(self.middle_layer.items()) if v[0] != 'dauer'}
        transition_probs = {k:v for i, (k,v) in enumerate(transition_probs.items()) if np.round(v, decimals=3) > 0}
        if len(transition_probs) > 0:
            developed = {k:np.random.binomial(1, v) for i, (k,v) in enumerate(transition_probs.items())}
            developed = [k for i, (k,v) in enumerate(developed.items()) if v == 1]
            for i in developed:
                curr_state = self.middle_layer[i][0]
                self.occupancy_time[self.middle_layer[i][2]][curr_state].append(self.middle_layer[i][1])
                if curr_state == 'J':
                    if i not in self.top_layer:
                        next_state = 'A'
                        if self.middle_layer[i][2] == 'NP':
                            eu_mf = 1
                        else:
                            eu_mf_prob = np.random.choice(MF_data[self.middle_layer[i][2] + '_' + self.resource_type])
                            eu_mf = np.random.binomial(1, eu_mf_prob)
                        worm_mf = 'ST'
                        if eu_mf:
                            worm_mf = 'EU'
                        self.top_layer[i] = (next_state, 0, self.middle_layer[i][2], worm_mf, self.middle_layer[i][4])
                        del self.middle_layer[i]
                    else:
                        next_state = 'dauer'
                        self.middle_layer[i] = (next_state, 0, self.middle_layer[i][2], self.middle_layer[i][3], self.middle_layer[i][4])

    def develop_bottom_layer(self):
        transition_probs = {k:sigmoid(v[1], v[0], v[2], self.resource_type) for i, (k,v) in enumerate(self.bottom_layer.items())}
        transition_probs = {k:v for i, (k,v) in enumerate(transition_probs.items()) if np.round(v, decimals=3) > 0}
        if len(transition_probs) > 0:
            developed = {k:np.random.binomial(1, v) for i, (k,v) in enumerate(transition_probs.items())}
            developed = [k for i, (k,v) in enumerate(developed.items()) if v == 1]
            for i in developed:
                curr_state = self.bottom_layer[i][0]
                self.occupancy_time[self.bottom_layer[i][2]][curr_state].append(self.bottom_layer[i][1])
                if curr_state == 'E':
                    if i not in self.middle_layer:
                        next_state = 'J'
                        self.middle_layer[i] = (next_state, 0, self.bottom_layer[i][2], self.bottom_layer[i][3], self.bottom_layer[i][4])
                        del self.bottom_layer[i]
                    else:
                        next_state = 'J2A'
                        self.bottom_layer[i] = (next_state, 0, self.bottom_layer[i][2], self.bottom_layer[i][3], self.bottom_layer[i][4])
                else:
                    next_state = 'dead'
                    self.bottom_layer[i] = (next_state, 0, self.bottom_layer[i][2], self.bottom_layer[i][3], self.bottom_layer[i][4])
                                    
    def migrate_dauer(self):
        """Move dauer larvae according to the migration probability
        """
        P_migrants = 0
        NP_migrants = 0
        MP_migrants = 0
        possible_migrants = [k for i, (k,v)  in enumerate(self.middle_layer.items()) if v[0] == 'dauer']
        if len(possible_migrants) > 0:
            migrants_prob = np.random.binomial(1, self.mig_rate, size=len(possible_migrants))
            migrants_prob = np.where(migrants_prob==1)[0]
            if len(migrants_prob) > 0:
                for i in migrants_prob:
                    if self.middle_layer[possible_migrants[i]][2] == 'P':
                        P_migrants += 1
                    elif self.middle_layer[possible_migrants[i]][2] == 'NP': 
                        NP_migrants += 1
                    else:
                        MP_migrants += 1
                    del self.middle_layer[possible_migrants[i]]
        self.dauer_migrated['P'].append(P_migrants)
        self.dauer_migrated['NP'].append(NP_migrants)
        self.dauer_migrated['MP'].append(MP_migrants)
        
    def age_pop(self):
        """Increase the age of worms in the population by one
        """
        for i, (k, v) in enumerate(self.top_layer.items()):
            if v[0] != 'dead':
                self.top_layer[k] = (v[0], v[1]+1, v[2], v[3], v[4])
        for i, (k, v) in enumerate(self.middle_layer.items()):
            if v[0] != 'dead':
                self.middle_layer[k] = (v[0], v[1]+1, v[2], v[3], v[4])
        for i, (k, v) in enumerate(self.bottom_layer.items()):
            self.bottom_layer[k] = (v[0], v[1]+1, v[2], v[3], v[4])
    
    def take_a_step(self):
        """Take a single step of the simulations
        """
        # remove dead worms
        self.funerary += [k for i, (k,v)  in enumerate(self.top_layer.items()) if v[0] == 'dead']
        self.funerary_J2A_arrest += [k for i, (k,v)  in enumerate(self.bottom_layer.items()) if v[0] == 'dead']
        self.top_layer = {k:v for i, (k,v) in enumerate(self.top_layer.items()) if v[0] != 'dead'}
        self.bottom_layer = {k:v for i, (k,v) in enumerate(self.bottom_layer.items()) if v[0] != 'dead'}
        self.move_worms()
        self.lay_eggs()
        self.develop_bottom_layer()
        self.develop_middle_layer()
        self.develop_top_layer()
        self.migrate_dauer()
        self.age_pop()
        
    def init_history(self):
        """Initialize the history dictionary 
        """
        self.history = {'NP':{}, 'MP':{}, 'P':{}}
        for i in self.history.keys():
            self.history[i]['E'] = []
            self.history[i]['J'] = []
            self.history[i]['J2A'] = []
            self.history[i]['A'] = []
            self.history[i]['EU_prop'] = []
            self.history[i]['dauer'] = []
        self.dauer_migrated['P'].append(0)
        self.dauer_migrated['NP'].append(0)
        self.dauer_migrated['MP'].append(0)
        self.loc_history = {'NP':{}, 'MP':{}, 'P':{}}
        if self.loc_save:
            for i in self.loc_history.keys():
                self.loc_history[i]['A_locs'] = []
                self.loc_history[i]['J_locs'] = []
                self.loc_history[i]['D_locs'] = []
                self.loc_history[i]['E_locs'] = []
                self.loc_history[i]['J2A_locs'] = []
        
    def update_history(self):  
        """Update the history dictionary
        """
        for strain in self.history.keys(): 
            self.history[strain]['E'].append(len([i for i, (k,v) in enumerate(self.bottom_layer.items()) if v[2] == strain]))
            self.history[strain]['J'].append(len([i for i, (k,v) in enumerate(self.middle_layer.items()) if (v[0] == 'J' and v[2] == strain)]))
            self.history[strain]['J2A'].append(len([i for i, (k,v) in enumerate(self.bottom_layer.items()) if (v[0] == 'J2A' and v[2] == strain)]))
            n_A = [k for i, (k,v) in enumerate(self.top_layer.items()) if (v[0] == 'A' and v[2] == strain)]
            n_EU = [k for i, (k,v) in enumerate(self.top_layer.items()) if (v[0] == 'A' and v[2] == strain and v[3] == 'EU')]
            self.history[strain]['A'].append(len(n_A))
            eu_prop = 0
            if len(n_A) > 0:
                eu_prop = np.divide(len(n_EU), len(n_A))
            self.history[strain]['EU_prop'].append(eu_prop)
            self.history[strain]['dauer'].append(len([i for i, (k,v) in enumerate(self.middle_layer.items()) if (v[0] == 'dauer' and v[2] == strain)]))  
            if self.loc_save:
                self.loc_history[strain]['A_locs'].append(n_A)
                n_J = [k for i, (k,v) in enumerate(self.middle_layer.items()) if (v[0] == 'J' and v[2] == strain)]
                self.loc_history[strain]['J_locs'].append(n_J)
                n_D = [k for i, (k,v) in enumerate(self.middle_layer.items()) if (v[0] == 'dauer' and v[2] == strain)]
                self.loc_history[strain]['D_locs'].append(n_D)
                n_E = [k for i, (k,v) in enumerate(self.bottom_layer.items()) if (v[0] == 'E' and v[2] == strain)]
                self.loc_history[strain]['E_locs'].append(n_E)
                n_J2A = [k for i, (k,v) in enumerate(self.bottom_layer.items()) if (v[0] == 'J2A' and v[2] == strain)]
                self.loc_history[strain]['J2A_locs'].append(n_J2A)

    def save_stats(self, directory):
            """Save evolutionary history in pickle format.

            Parameters
            ----------
            directory : int
                A number added to the file name to avoid overwriting.
            """
            id_token = str(uuid.uuid4())
            if not os.path.exists(directory):
                os.makedirs(directory)
            occupancy_t = {}
            for j in ['E', 'J', 'J2A', 'A']:
                occupancy_t[j + '_NP'] = self.occupancy_time['NP'][j]
                occupancy_t[j + '_MP'] = self.occupancy_time['MP'][j]
                occupancy_t[j + '_P'] = self.occupancy_time['P'][j]
            occupancy_t = dict([(k,pd.Series(v, dtype='float64')) for k,v in occupancy_t.items()])
            occupancy_t = pd.DataFrame(occupancy_t)
            file = directory + "/octime_" + id_token
            occupancy_t.to_csv(file)
            temp = {}
            n_steps = len(self.history['NP']['E'])
            for j in ['E', 'J', 'J2A', 'A', 'dauer', 'dauer_mig']:
                if j == 'dauer_mig':
                    temp[j] = self.dauer_migrated['NP'] + self.dauer_migrated['MP'] + self.dauer_migrated['P']
                else:
                    temp[j] = self.history['NP'][j] + self.history['MP'][j] + self.history['P'][j]
                temp['Strain'] = ['NP' for k in range(n_steps)] + ['MP' for k in range(n_steps)] + ['P' for k in range(n_steps)]
            temp = pd.DataFrame(temp)
            file = directory + "/dynamic_" + id_token
            temp.to_csv(file)
            #save final pop structure
            file = open(directory + "/final_state_" + id_token, 'wb')
            pickle.dump({'top':self.top_layer, 'mid':self.middle_layer, 'bottom':self.bottom_layer}, file)
            file.close()
            #save loc data
            if self.loc_save:
                file = open(directory + "/loc_" + id_token, 'wb')
                pickle.dump(self.loc_history, file)
                file.close()
        
    def simulate(self, t, verbose=False):
        """Simulate the population model

        Args:
            t (int): The length of the simulation.
            verbose (bool, optional): If True, shows a progress bar using tqdm library. Defaults to False.
        """
        self.init_history()
        self.update_history()
        for i in tqdm(range(t), disable=not verbose):
            self.take_a_step()
            self.update_history()