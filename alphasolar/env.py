import gym
import itertools
import numpy as np
import math as m
import random
import datetime
import alphasolar.solar_helpers as sh
import matplotlib.pyplot as plt
from alphasolar.structures import *

class AlphaSolarEnv(gym.Env):
    ACTIONS = ["panel_forward_ns", "panel_back_ns", "do_nothing", "panel_forward_ew", "panel_back_ew"]
    ATTRIBUTES = ["angle_AZ", "angle_ALT", "angle_ns", "angle_ew"]
    CLASSES = ["agent", "sun", "time", "worldPosition"]

    def __init__(self,
                panel,
                date_time,
                name_ext,
                timestep=3000,
                panel_step=.1,
                reflective_index=0.65,
                latitude_deg=40.7,
                longitude_deg=142.17,
                img_dims=32,
                mode_dict = {'dual_axis':True, 'image_mode':True, 'cloud_mode':True},
                n_steps=1000,
                ):

        if name_ext == "usa_avg":
            self.loc_index = 0
            self.lat_list, self.lon_list = latitude_deg, longitude_deg
            latitude_deg, longitude_deg = latitude_deg[0], longitude_deg[1]

            # Error check the lat/long.
        elif abs(latitude_deg) > 90 or abs(longitude_deg) > 180:
            print ("Error: latitude must be between [-90, 90], longitude between [-180,180]. Lat:", latitude_deg, "Long:", longitude_deg)
            quit()

        if mode_dict['cloud_mode'] and not mode_dict['image_mode']:
            print ("Warning (SolarOOMDP): Clouds were set to active but image mode is off. No cloud simulation supported for non-image-mode.")
            mode_dict['cloud_mode'] = False

        # Mode information
        if not(mode_dict['dual_axis']):
            # If we are in 1-axis tracking mode, change actions accordingly.
            self.ACTIONS = self.get_single_axis_actions
            self.dual_axis = False
        else:
            self.dual_axis = True

        # Image stuff.
        self.img_dims = img_dims
        self.image_mode = mode_dict['image_mode']
        self.cloud_mode = mode_dict['cloud_mode']
        self.clouds = self._generate_clouds() if mode_dict['cloud_mode'] else []

        #get panel information.
        self.panel = panel
        self.sqrt_num_panels = 1 # Assume only 1 panel for now.

        # Global information
        self.latitude_deg = latitude_deg # positive in the northern hemisphere
        self.longitude_deg = longitude_deg # negative reckoning west from prime meridian in Greenwich, England
        self.panel_step = panel_step
        self.timestep = timestep #timestep in minutes
        self.reflective_index = reflective_index
        self.name_ext = name_ext

        # Time stuff.
        self.init_time = date_time
        self.time = date_time

        # Make state and call super.
        self.panels = self._get_default_panel_obj_list()
        self.init_state = self._create_state(self.panels, self.init_time)

        self.observation_space = gym.spaces.Box(shape=(img_dims ** 2,), low=-np.inf, high=np.inf)
        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))

        self.n_steps = n_steps


    def get_bandit_actions(self):
        ns = [str(x) for x in range(-90, 91, self.panel_step)]
        if self.dual_axis:
            ew = [str(x) for x in range(-90, 91, self.panel_step)]
        else:
            ew = [str(0)]

        actions = []

        for element in itertools.product(ns, ew):
            new_action = str(element[0] + "," + element[1])
            actions.append(new_action)

        return actions

        
    def end_of_instance(self):
        if self.name_ext == "usa_avg":
            self.loc_index = (self.loc_index + 1) % len(self.lat_list)
            self.latitude_deg, self.longitude_deg = self.lat_list[self.loc_index], self.lon_list[self.loc_index]

    def _get_default_panel_obj_list(self):
        panels = []
        for i in range(self.sqrt_num_panels**2):
            # Make panel object.
            panel_attributes = {}
            panel_attributes["angle_ew"] = 0.0
            panel_attributes["angle_ns"] = 0.0
            # panel = OOMDPObject(attributes=panel_attributes, name="panel_" + str(i))
            panel = ("panel_" + str(i), panel_attributes)
            panels.append(panel)
        return panels

    def _get_day(self):
        return self.get_local_time().timetuple().tm_yday

    def get_panel_step(self):
        return self.panel_step

    def get_reflective_index(self):
        return self.reflective_index

    def get_single_axis_actions(self):
        return ["do_nothing", "panel_forward_ew", "panel_back_ew"]

    def get_optimal_actions(self):
        return ["optimal"]

    # -------------------
    # --- CLOUD STUFF ---
    # -------------------

    def _generate_clouds(self):
        '''
        Returns:
            (list of Cloud)
        '''
        num_clouds = random.randint(0,4)
        clouds = []

        # Generate info for each cloud.
        dx, dy = 1, 0
        for _ in range(num_clouds):
            x = random.randint(0, self.img_dims)
            y = random.randint(0, self.img_dims)
            rx = random.randint(3,6)
            ry = random.randint(2,rx)
            clouds.append(Cloud(x, y, dx, dy, rx, ry))

        return clouds

    def _move_clouds(self):
        '''
        Summary:
            Moves each cloud in the cloud list (self.clouds).
        '''
        for cloud in self.clouds:
            cloud.move(self.timestep)

    # ----------------------------------
    # --- REWARD AND TRANSITION FUNC ---
    # ----------------------------------

    def _reward_func(self, state, action):
        '''
        Args:
            state (OOMDP State)
            action (str)
        Returns
            (float)
        '''

        # Both altitude_deg and azimuth_deg are in degrees.
        sun_altitude_deg = sh._compute_sun_altitude(self.latitude_deg, self.longitude_deg, self.get_local_time())
        sun_azimuth_deg = sh._compute_sun_azimuth(self.latitude_deg, self.longitude_deg, self.get_local_time())

        # Panel stuff
        panel_ew_deg = state.get_panel_angle_ew()
        panel_ns_deg = state.get_panel_angle_ns()

        if action is "optimal":
            # Compute optimal reward.
            flux = self._compute_optimal_reward(sun_altitude_deg, sun_azimuth_deg)
            # Compute electrical power output of panel for given flux.
            power = self.panel.get_power(flux)

            # Convert timestep to seconds.
            reward = power * self.timestep * 60 # Joules

        else:
            if "energy" in self.name_ext:
                flux, r_d, r_f, r_r = self._compute_flux(sun_altitude_deg, sun_azimuth_deg, panel_ns_deg, panel_ew_deg, breakdown=True)
            
                p_d, p_f, p_r = self.panel.get_power(r_d), self.panel.get_power(r_f), self.panel.get_power(r_r)
                e_d, e_f, e_r = p_d * self.timestep * 60, p_f * self.timestep * 60, p_r * self.timestep * 60
            else:
                flux = self._compute_flux(sun_altitude_deg, sun_azimuth_deg, panel_ns_deg, panel_ew_deg)

            # Compute electrical power output of panel for given flux.
            power = self.panel.get_power(flux)

            # Convert timestep to seconds.
            energy = power * self.timestep * 60 # Joules
            cost = 0 # in Joules

            # Get cost of motion.
            if "ew" in action:
                cost = self.panel.get_rotation_energy_for_axis('ew', np.radians(panel_ew_deg), np.radians(self.panel_step))
            elif "ns" in action:
                cost = self.panel.get_rotation_energy_for_axis('ns', np.radians(panel_ns_deg), np.radians(self.panel_step))

            reward = energy - cost

            # sh._write_datum_to_file(str(self), agent, r_d, "direct")
            # sh._write_datum_to_file(str(self), agent, r_f, "diffuse")
            # sh._write_datum_to_file(str(self), agent, r_r, "reflective")

        reward = (reward) / 1000000.0 # Convert Watts to Megawatts

        # if "energy" in self.name_ext:
        #     e_d, e_f, e_r = e_d / 1000000.0, e_f / 1000000.0, e_r / 1000000.0
        #     return reward, e_d, e_f, e_r

        return reward


    def _compute_flux(self, sun_altitude_deg, sun_azimuth_deg, panel_ns_deg, panel_ew_deg, breakdown=False):        
        '''
        Args:
            sun_altitude_deg (float)
            sun_azimuth_deg (float)
            panel_ns_deg (float)
            panel_ew_deg (float)
            breakdown (bool): If true returns breakdown of energy
        '''
        # Compute direct radiation.
        direct_rads = sh._compute_radiation_direct(self.get_local_time(), sun_altitude_deg)
        diffuse_rads = sh._compute_radiation_diffuse(self.get_local_time(), self._get_day(), sun_altitude_deg)
        reflective_rads = sh._compute_radiation_reflective(self.get_local_time(), self._get_day(), self.reflective_index, sun_altitude_deg)

        # Compute tilted component.
        direct_tilt_factor = sh._compute_direct_radiation_tilt_factor(panel_ns_deg, panel_ew_deg, sun_altitude_deg, sun_azimuth_deg)
        fixed_direct_tilt_factor = sh._compute_direct_radiation_tilt_factor(0, 0, sun_altitude_deg, sun_azimuth_deg)

        diffuse_tilt_factor = sh._compute_diffuse_radiation_tilt_factor(panel_ns_deg, panel_ew_deg)
        reflective_tilt_factor = sh._compute_reflective_radiation_tilt_factor(panel_ns_deg, panel_ew_deg)

        # Compute total.
        flux = direct_rads * direct_tilt_factor + \
                    diffuse_rads * diffuse_tilt_factor + \
                    reflective_rads * reflective_tilt_factor

        r_d = direct_rads * direct_tilt_factor
        r_f = diffuse_rads * diffuse_tilt_factor
        r_r = reflective_rads * reflective_tilt_factor

        if breakdown:
            return flux, r_d, r_f, r_r

        return flux

    def get_local_time(self):
        return (self.time + self.time.utcoffset())

    '''
    Computes the optimal reward possible for a given sun position.
    Ignores current position.
    '''
    def _compute_optimal_reward(self, sun_altitude_deg, sun_azimuth_deg):
        optimal_panel_ew = 0
        optimal_panel_ns = 0
        optimal_reward = -.001

        # Iterate over all possible panel angles.
        for panel_ew_deg in range(-90, 90, 5):
            for panel_ns_deg in range(-90, 90, 5):

                # Check reward.
                reward = self._compute_flux(sun_altitude_deg, sun_azimuth_deg, panel_ns_deg, panel_ew_deg)
                if reward > optimal_reward:
                    optimal_reward = reward

        power = self.panel.get_power(optimal_reward)
        energy = power * self.timestep * 60 # Joules
        optimal_reward = energy / 1000000.0 # Convert Watts to Megawatts

        return optimal_reward


    def _create_moved_panel(self, state, action, panel_index):
        '''
        Args;
            state (State)
            action (str)
            panel_index (int)
        Returns:
            (OOMDPObject): The panel object, moved according to @action.
        '''
        panel_angle_ew = state.get_panel_angle_ew(panel_index=panel_index)
        panel_angle_ns = state.get_panel_angle_ns(panel_index=panel_index)

        if "," in action:
            # Bandit action.
            ns_act, ew_act = action.split(",")
            bounded_panel_angle_ew = max(min(int(ew_act), 90), -90)
            bounded_panel_angle_ns = max(min(int(ns_act), 90), -90)
        else:
            # Compute new angles
            ew_step, ns_step = {"panel_forward_ew": (self.panel_step, 0),
                                "panel_forward_ns": (0, self.panel_step),
                                "panel_back_ew": (-self.panel_step, 0),
                                "panel_back_ns": (0, -self.panel_step),
                                "do_nothing": (0, 0)}[action]
            new_panel_angle_ew, new_panel_angle_ns = panel_angle_ew + ew_step, panel_angle_ns + ns_step
            bounded_panel_angle_ew = max(min(new_panel_angle_ew, 90), -90)
            bounded_panel_angle_ns = max(min(new_panel_angle_ns, 90), -90)

        # Make panel object.
        panel_attributes = {}
        panel_attributes["angle_ew"] = bounded_panel_angle_ew
        panel_attributes["angle_ns"] = bounded_panel_angle_ns
        #panel = OOMDPObject(attributes=panel_attributes, name="panel_" + str(panel_index))
        panel = ("panel_" + str(panel_index), panel_attributes)
        return panel

    def _transition_func(self, state, action):
        '''
        Args:
            (OOMDP State)
            action (str)
        Returns
            (OOMDP State)
        '''
        # self._error_check(state, action)

        if self.get_local_time().timetuple().tm_hour >= 16: # or self.get_local_time().timetuple().tm_hour <= 6:
            self.time += datetime.timedelta(hours=13)
            new_panels = state.get_panels()
        else:
            self.time += datetime.timedelta(minutes=self.timestep)

            # Remake or move clouds.
            if self.get_local_time().hour == 1 and self.get_local_time().minute == 0:
                self.clouds = self._generate_clouds() if self.cloud_mode else []
            elif self.clouds != []:
                self._move_clouds()

            # If we're computing optimal-greedy behavior, the new state is irrelevent (we search over all anyway).
            if action == "optimal":
                new_panels = state.get_panels()
            else:
                # Move all panels.
                new_panels = []
                for i in range(self.sqrt_num_panels**2):

                    next_panel = self._create_moved_panel(state, action, panel_index=i)

                    new_panels.append(next_panel)

        next_state = self._create_state(new_panels, self.time)

        return next_state

    def _create_state(self, panels, time):
        '''
        Args:
            panels (list): Contains attribute dictionaries for panel objects.
            time (datetime)
        Returns:
            (SolarOOMDPState)
        '''
        self.objects = {attr : [] for attr in self.CLASSES}

        self.objects["sun"] = []
        self.objects["panel"] = panels

        # Sun.
        sun_attributes = {}
        sun_angle_AZ = sh._compute_sun_azimuth(self.latitude_deg, self.longitude_deg, time)
        sun_angle_ALT = sh._compute_sun_altitude(self.latitude_deg, self.longitude_deg, time)
        
        # Image stuff.
        if self.image_mode:
            # Grab image relative to first image for now.
            bounded_panel_angle_ew = max(min(panels[0][1]["angle_ew"], 90), -90)
            bounded_panel_angle_ns = max(min(panels[0][1]["angle_ns"], 90), -90)
            # Set attributes as pixels.
            image = self._create_sun_image(sun_angle_AZ, sun_angle_ALT, bounded_panel_angle_ns, bounded_panel_angle_ew)
            for i in range (self.img_dims):
                for j in range (self.img_dims):
                    idx = i*self.img_dims + j
                    sun_attributes['pix' + str(idx)] = image[i][j]    
            sun_attributes["image"] = image
        else:
            sun_attributes["angle_AZ"] = sun_angle_AZ
            sun_attributes["angle_ALT"] = sun_angle_ALT  

        self.objects["sun"].append(sun_attributes)

        return State(self.objects, date_time=time, longitude=self.longitude_deg, latitude=self.latitude_deg, sun_angle_AZ = sun_angle_AZ, sun_angle_ALT = sun_angle_ALT)

    # -------------------
    # --- IMAGE STUFF ---
    # -------------------

    def _get_sun_x_y(self, sun_angle_AZ, sun_angle_ALT):
        x = self.img_dims * (1 + m.sin(m.radians(sun_angle_AZ)))/2
        y = self.img_dims * m.sin(m.radians(sun_angle_ALT))/2
        return x, y

    def _create_sun_image(self, sun_angle_AZ, sun_angle_ALT, panel_angle_ns, panel_angle_ew):
        # Create image of the sun, given alt and az
        sun_dim = self.img_dims/8.0

        # For viewing purposes, we normalize between 0 and 1 on the x axis and 0 to .5 on the y axis
        panel_tilt_offset_y = m.sin(m.radians(panel_angle_ns))
        panel_tilt_offset_x = m.sin(m.radians(panel_angle_ew))

        percent_in_sky_x = m.sin(m.radians(sun_angle_AZ))
        percent_in_sky_y = m.sin(m.radians(sun_angle_ALT))
        x, y = self._get_sun_x_y(sun_angle_AZ, sun_angle_ALT)
        image = [np.ones(self.img_dims)*0.6 for l in [[0] * self.img_dims] * self.img_dims]

        # Make gaussian sun
        for i in range (self.img_dims):
            for j in range (self.img_dims):
                image[i][j] = min(image[i][j] + sh._gaussian(j, x, sun_dim) * sh._gaussian(i, y, sun_dim), 1.0)

                # Add cloud cover.
                for cloud in self.clouds:
                    image[i][j] -= (sh._gaussian(j, cloud.get_mu()[0], cloud.get_sigma()[0][0]) * \
                                    sh._gaussian(i, cloud.get_mu()[1], cloud.get_sigma()[1][1]) * cloud.get_intensity())

                # Backcompute the altitude of the pixel; if it is below the horizon, render black.
                alt_pix = 2*float(i)/self.img_dims + panel_tilt_offset_y
                if alt_pix < 0:
                    image[i][j] = 1

        # Show image (for testing purposes)
        # self._show_image(image)

        return image

    def _show_image(self, image):
        plt.imshow(image, cmap='gray', vmin=00.0, vmax=1.0, interpolation='nearest')
        plt.gca().invert_yaxis()
        plt.title( 'Images used to train model')
        plt.figtext(0.01, 0.95, 'Date and Time: ' + str(self.time.ctime()), fontsize = 11)
        plt.figtext(0.01, 0.9, 'Latitude: ' + str(self.latitude_deg), fontsize = 11)
        plt.figtext(0.01, 0.85, 'Longitude: ' + str(self.longitude_deg), fontsize = 11)
        plt.show()

    def __str__(self):
        percept = "true"
        ext = ""
        if self.image_mode and self.cloud_mode:
            percept = "image"
        if self.dual_axis:
            ext = "d_"
        return "solar_" + ext + self.name_ext + "_p-" + str(self.panel_step) + "_" + percept

    def _error_check(self, state, action):
        '''
        Args:
            state (State)
            action (str)
        Summary:
            Checks to make sure the received state and action are of the right type.
        '''

        if action not in self.ACTIONS + self.get_optimal_actions() + self.get_bandit_actions():
            raise ValueError("Error: the action provided (" + str(action) + ") was invalid.")

        if not isinstance(state, State):
            raise ValueError("Error: the given state (" + str(state) + ") was not of the correct class.")

    def _get_terminal(self):
        self.current_step += 1
        if self.current_step == self.n_steps:
            return True 
        return False

    def reset(self):
        '''
        Summary:
            Resets the OOMDP back to the initial configuration.
        '''
        self.current_step = 0
        self.time = self.init_time
        self.state = self.init_state
        return np.array(self.objects["sun"][0]["image"]).flatten()
    
    def step(self, action):
        action = self.ACTIONS[action]
        reward = self._reward_func(self.state, action)
        self.state = self._transition_func(self.state, action)
        return np.array(self.objects["sun"][0]["image"]).flatten(), reward, self._get_terminal(), {}


def _multivariate_gaussian(x, y, mu_vec, cov_matrix):
    '''
    Args;
        x (float)
        y (float)
        mu_vec (np.array)
        cov_matrix (np.matrix)
    Returns:
        (float): evaluates the PDF of the multivariate at the point x,y.
    '''
    numerator = np.exp(-.5 *np.transpose(np.array([x,y]) - mu_vec) * np.linalg.inv(cov_matrix) * (np.array([x,y]) - mu_vec))
    denominator = np.sqrt(np.linalg.det(2*m.pi*cov_matrix))

    res = numerator / denominator
    return res[0][0] + res[1][1]



