"""
An individual-based model of P. pacificus development and population dynamics.
"""

__author__ = 'Ata Kalirad'

__version__ = '1.1'

import numpy as np
import random as rnd
import pandas as pd
import os
import uuid
import pickle
from tqdm import tqdm 
from itertools import cycle

# set random number generator seed

states = ('E', 'J', 'J2A', 'A', 'dead', 'dauer')

# import fecundity and MF data

file_names_unfiltered = [file for file in os.listdir('./sim_inputs/') if not file.startswith('.')]

file_names = [i for i in file_names_unfiltered if 'nv' not in i]

file_names_nv = [i for i in file_names_unfiltered if 'nv' in i]
    
fec_data = {i[:-4]: pd.read_csv('./sim_inputs/'+i, index_col=0) for i in file_names}

fec_data_nv = {i[:-4]: pd.read_csv('./sim_inputs/'+i, index_col=0) for i in file_names_nv}
    
MF_data = {i:list(pd.read_csv('./sim_inputs/MF_'+i+'.csv', index_col=0).iloc[:,0]) for i in ['P_OP50', 'MP_OP50', 'P_Novo', 'MP_Novo']}
    
def sigmoid(x, state, strain, diet):
    
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

# Metapop functions
def create_neighbor_dict(line_length, ring=False):
    neighbor_dict = {}
    for i in range(line_length):
        neighbors = []
        # Add the left neighbor (if it exists)
        if i > 0:
            neighbors.append(i - 1)
        elif ring:
            neighbors.append(line_length - 1)  # Wrap around to the last location
        # Add the right neighbor (if it exists)
        if i < line_length - 1:
            neighbors.append(i + 1)
        elif ring:
            neighbors.append(0)  # Wrap around to the first location
        neighbor_dict[i] = neighbors
    return neighbor_dict

def assign_items_to_neighbors(items, neighbors, curr):
    if len(neighbors) == 1:
        # If there is only one neighbor, assign all items to that neighbor
        assigned_items = []
        neighbor = neighbors[0]
        for item in items:
            assigned_items.append((item, curr, neighbor))
    elif len(neighbors) == 2:
        # If there are two neighbors, randomly assign items to either neighbor
        assigned_items = []
        for item in items:
            assigned_neighbor = rnd.choice(neighbors)
            assigned_items.append((item, curr, assigned_neighbor))
    else:
        # Handle the case where there are more than two neighbors (customize as needed)
        print("Cannot handle more than two neighbors.")
        return None
    return assigned_items

def migration(possible, neighbours, mig_rate=0.5):
    n_pop = len(neighbours.keys())
    migrants = {}
    for i in range(n_pop):
        if len(possible[i]) > 0:
            migrants_prob = np.random.binomial(1, mig_rate, size=len(possible[i]))
            migrants_prob = np.where(migrants_prob==1)[0]
            migrants[i] = [possible[i][j] for j in migrants_prob]
    temp = []
    for i in list(migrants.keys()):
        temp += assign_items_to_neighbors(migrants[i], neighbours[i], i)
    return temp
# end of metapop functions

class Population(object):
    
    def __init__(self, dim, resource_type='OP50', rw_l=5, walk_bias=(1, 1), recog_par=0.95, pred_par=0.1, mig_rate=0.05, egg_var=True, loc_save=False):
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
        ######
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
        return set(self.index).difference(self.top_layer.keys())
    
    @property
    def empty_loc_middle(self):
        return set(self.index).difference(self.middle_layer.keys())
        
    @property
    def empty_loc_bottom(self):
        return set(self.index).difference(self.bottom_layer.keys())


    def random_walk_2d(self, loc, empty_spots, pot_prey_spots=False):
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
        curr_pos_top = [*self.top_layer]
        curr_pos_mid = [(k[0], k[1], 'mid') for i, (k,v) in enumerate(self.middle_layer.items()) if v[0] == 'J']
        mixed_pos = curr_pos_top + curr_pos_mid
        J2A = [k for i, (k,v) in enumerate(self.bottom_layer.items()) if v[0] == 'J2A']
        rnd.shuffle(mixed_pos)
        for worm in mixed_pos:
            #print(worm)
            if len(worm) == 3 and (worm[0], worm[1]) in self.middle_layer:
                    trajectory = self.random_walk_2d((worm[0], worm[1]), self.empty_loc_middle)
                    #print(trajectory)
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
                #print(trajectory)
                if new_pos != worm:
                    self.top_layer[new_pos] = self.top_layer[worm]
                    del self.top_layer[worm]
                if self.pred and self.top_layer[new_pos][3]=='EU':
                    pot_preys = {**{(k, 'bottom'):v[2] for i, (k,v) in enumerate(self.bottom_layer.items()) if k in set(trajectory) and v[0] == 'J2A'}, **{(k, 'middle'):v[2] for i, (k,v) in enumerate(self.middle_layer.items()) if k in set(trajectory)}}
                    #print(pot_preys)
                    recognized_prob = np.random.binomial(1, self.recog_par, size=len(pot_preys))
                    recognized = {k:(v if recognized_prob[i] else 'MC') for i, (k,v) in enumerate(pot_preys.items())}
                    #print(recognized)
                    preys = [k for i, (k,v) in enumerate(recognized.items()) if v!= self.top_layer[new_pos][2] or v=='MC']
                    #print(preys)
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
        self.top_layer = top
        self.middle_layer = mid
        self.bottom_layer = bottom
    
    def add_dauer(self, worm, always_eu=False):
        spots_top = list(self.empty_loc_top)
        if len(spots_top) > 0:
            worm_mf = 'ST'
            if always_eu or worm[2] == 'NP':
                worm_mf = 'EU'
            else:
                eu_mf_prob = np.random.choice(MF_data[worm[2] + '_' + self.resource_type])
                eu_mf = np.random.binomial(1, eu_mf_prob)
                if eu_mf:
                    worm_mf = 'EU'
            worm = ('A', 0, worm[2], worm_mf, worm[4])
            rand_pos =  np.random.choice(range(len(spots_top)))
            self.top_layer[spots_top[rand_pos]] = worm
        
    def add_worm(self, worm):
        if not worm[4]:
            worm = (worm[0], worm[1], worm[2], worm[3], str(uuid.uuid4()))
        spots_top = list(self.empty_loc_top)
        spots_middle = list(self.empty_loc_middle)
        spots_bottom = list(self.empty_loc_bottom)
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
        for i, (k, v) in enumerate(self.top_layer.items()):
            if v[0] != 'dead':
                self.top_layer[k] = (v[0], v[1]+1, v[2], v[3], v[4])
        for i, (k, v) in enumerate(self.middle_layer.items()):
            if v[0] != 'dead':
                self.middle_layer[k] = (v[0], v[1]+1, v[2], v[3], v[4])
        for i, (k, v) in enumerate(self.bottom_layer.items()):
            self.bottom_layer[k] = (v[0], v[1]+1, v[2], v[3], v[4])
    
    def take_a_step(self):
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
        self.init_history()
        self.update_history()
        for i in tqdm(range(t), disable=not verbose):
            self.take_a_step()
            self.update_history()