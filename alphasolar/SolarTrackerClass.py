'''
SolarTrackerClass.py: Contains the SolarTracker class. 
'''

# Python imports.
import numpy as np
import random

# Local imports.
import alphasolar.solar_helpers as sh

class SolarTracker(object):
    ''' Class for a Solar Tracker '''

    def __init__(self, tracker, panel_step, actions, dual_axis=False):
        self.tracker = tracker
        self.panel_step = panel_step
        self.dual_axis = dual_axis
        self.actions = actions

    def get_policy(self):
        return self._policy

    def _policy(self, state):
        '''
        Args:
            state (SolarOOMDP state): contains the year, month, hour etc.
            panel_step = int

        Returns:
            (str): Action in the set SolarOOMDPClass.ACTIONS
        '''

        # Compute sun vec and initial panel vec.
        sun_alt, sun_az = self.tracker(state)
        sun_vec = sh._compute_sun_vector(sun_alt, sun_az)
        panel_ew, panel_ns = state.get_panel_angle_ew(), state.get_panel_angle_ns()
        best_action = sh._compute_panel_normal_vector(panel_ns, panel_ew)
        max_cos_sim = float("-inf")

        # Find action that minimizes cos difference to estimate of sun vector.
        for action in self.actions:

            # Grab new normal vector.
            new_panel_ns, new_panel_ew = [float(x) for x in action.split(",")]
            new_panel_normal_vec = sh._compute_panel_normal_vector(new_panel_ns, new_panel_ew)
            
            cos_sim = np.dot(sun_vec, new_panel_normal_vec)
            if cos_sim > max_cos_sim:
                best_action = action
                max_cos_sim = cos_sim

        return best_action

