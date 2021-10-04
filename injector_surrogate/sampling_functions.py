import sys
import numpy as np

#NN Surrogate model class
from injector_surrogate_quads import Model

#input params: solenoid and quads to vary 
opt_var_names = ['SOL1:solenoid_field_scale','SQ01:b1_gradient','CQ01:b1_gradient','QE04:b1_gradient']

#output params: emittance in transverse plane (x & y)
opt_out_names = ['norm_emit_x','norm_emit_y']


def get_ground_truth(Model,ref_point,varx,vary,varz): 
    '''Returns normalized emittance prediction from the surrogate model
       for given settings of SOL1, SQ01, and CQ01 '''
    
    #convert to machine units
    ref_point = Model.sim_to_machine(np.asarray(ref_point))

    #make input array of length model_in_list (inputs model takes)
    x_in = np.empty((1,len(Model.model_in_list)))

    #fill in reference point around which to optimize
    x_in[:,:] = np.asarray(ref_point[0])

    #set solenoid, SQ, CQ to values from optimization step
    x_in[:, Model.loc_in[opt_var_names[0]]] = varx
    x_in[:, Model.loc_in[opt_var_names[1]]] = vary
    x_in[:, Model.loc_in[opt_var_names[2]]] = varz
   
    #output predictions
    y_out = Model.pred_machine_units(x_in) 
    
    #output is geometric emittance in transverse plane
    emitx = y_out[:,Model.loc_out['norm_emit_x']] #grab norm_emit_x out of the model
    emity = y_out[:,Model.loc_out['norm_emit_y']] #grab norm_emit_y out of the model

    return np.sqrt(emitx*emity)

def get_beamsize(Model,ref_point,varx,vary,varz,varscan): 
    '''Returns the beamsize (xrms, yrms) prediction [m] from the surrogate model
    for given settings of SOL1, SQ01, CQ01 and scanning quad QE04 '''
        
    #convert to machine units
    ref_point = Model.sim_to_machine(np.asarray(ref_point))
    
    #make input array of length model_in_list (inputs model takes)
    x_in = np.empty((1,len(Model.model_in_list)))

    #fill in reference point around which to optimize
    x_in[:,:] = np.asarray(ref_point[0])

    #set solenoid, SQ, CQ to values from optimization step
    x_in[:, Model.loc_in[opt_var_names[0]]] = varx
    x_in[:, Model.loc_in[opt_var_names[1]]] = vary
    x_in[:, Model.loc_in[opt_var_names[2]]] = varz
    
    #set quad 525 to values for scan
    x_in[:, Model.loc_in[opt_var_names[3]]] = varscan

    #output predictions
    y_out = Model.pred_machine_units(x_in) 
    
    x_rms = y_out[:,0][0]
    y_rms = y_out[:,1][0]

    return np.array([x_rms, y_rms])