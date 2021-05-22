import numpy as np

g = 9.8 #meters per second

class Panel():
    def __init__(self,
                    x_dim,
                    y_dim,
                    assembly_mass, #total mass of panel assembly, panel + mount
                    COM_offset, #offset between center of mass and radial axis
                    bearing_friction, #friction on bearing, assuming constant?
                    efficiency,
                    offset_angle,
                    actuator_force,
                    actuator_offset_ew,
                    actuator_mount_ew,
                    actuator_offset_ns,
                    actuator_mount_ns):
        '''
        :param x_dim: length of panel (meters)
        :param y_dim: width of panel (meters)
        :param mass: panel mass (kg)
        :param efficiency: energy conversion efficiency of solar panel (pct)
        :param actuator_force: force exerted by actuator (assumed to be constant) (Newtons)
        :param actuator_offset_1: distance between panel center and actuator mount across axis 1 (m)
        :param actuator_mount_1: length of actuator mount arm for axis 1. (m)
        :param actuator_offset_2: distance between panel center and actuator mount across axis 2 (m)
        :param actuator_mount_2: length of actuator mount arm for axis 2. (m)
        '''
        self.x_dim, self.y_dim = x_dim, y_dim
        self.mass = assembly_mass
        self.efficiency = efficiency
        self.COM_offset = COM_offset
        self.offset_angle = offset_angle
        self.bearing_friction = bearing_friction #TODO: implement this

        #actuator specs
        #self.actuator_force = actuator_force

        #dictionary of actuator mount attributes
        self.actuator_attrib = {}

        self.actuator_attrib['ew'] = {'offset': actuator_offset_ew, 'mount':actuator_mount_ew}
        self.actuator_attrib['ns'] = {'offset': actuator_offset_ns, 'mount': actuator_mount_ns}


    def get_power(self, flux):
        '''
        Returns the electrical power output in Watts of panel with the given flux
        :param flux: input flux of solar panel (W)
        :return: electrical power of solar panel (W)
        '''
        return self.x_dim*self.y_dim*self.efficiency*flux

    def __get_load__(self, axis, current_angle):
        '''
        Returns the load on the actuator for the current angle.
        Very simple, ignoring friction for now.
        :return:
        '''

        #TODO: model friction

        #calculate torque due to gravity:

        #torque = COM_offset*mass*g_sin \theta
        torque_g = self.COM_offset*self.mass*g*np.sin(current_angle)

        # print "current angle: {} torque: {}".format(current_angle, torque_g)

        #assume gravitational torque is equal to load, i.e. moving at constant velocity

        #calculate torque offset angle, a function of the current panel config and a constant param of the panel design

        l = self.__get_actuator_length__(axis, current_angle)

        mount = self.actuator_attrib[axis]['mount']
        offset = self.actuator_attrib[axis]['offset']

        #law of cosines: mount^2 = l^2 + offset^2 - 2*l*offset*cos(\phi)

        #angle between mount and actuator
        offset_angle_1 = np.arccos((np.square(mount) - np.square(l) - np.square(offset))/(-2*l*offset))

        total_offset = offset_angle_1 + self.offset_angle

        f_actuator = torque_g/np.sin(total_offset)

        return f_actuator

    def __get_actuator_length__(self, axis, current_angle):
        '''
        Computes the current length of the actuator.
        :param axis: axis indicator
        :param current_angle: current angle of panel for this axis
        :return:
        '''

        #x^2 = a^2 + b^2 - 2abcos(\theta)
        a = self.actuator_attrib[axis]['mount']
        b = self.actuator_attrib[axis]['offset']

        return np.sqrt(np.square(a) + np.square(b) - 2 * a * b * np.cos(current_angle))

    def get_rotation_energy_for_axis(self, axis, current_angle, delta_theta, actuator_efficiency=1.):
        '''
        :param current_angle: the current angle of the panel for this axis (radians)
        :param delta_theta: angle change (radians)
        :param axis: axis indicator (ew or ns)
        :return: energy consumed during rotation operation (Joules)
        '''

        #TODO: model linear actuator efficiency - assuming right now 100% electrical/physical work conversion

        # compute dx
        # (distance extended by linear actuator as a function of current_angle, delta_theta and the panel configuration)

        a = self.actuator_attrib[axis]['mount']
        b = self.actuator_attrib[axis]['offset']

        #computing f(\theta) = x with law of cosines: x^2 = a^2 + b^2 - 2abcos(\theta)

        dx = (a*b*np.sin(current_angle))/np.sqrt(np.square(a) + np.square(b) - 2*a*b*np.cos(current_angle))*delta_theta

        #W = F*dx (Joules)

        #absolute value - delta theta is always positive even when moving backwards

        force = self.__get_load__(axis, current_angle)

        #assumption = friction ~= load*coeff

        force_total = force + self.bearing_friction*force

        work = np.abs(force_total*dx/actuator_efficiency)
        #print "force: {}, work: {}".format(force, work)

        return work


class Cloud(object):

    PIX_INTENSITY = .15

    def __init__(self, x, y, dx, dy, rx, ry, intensity=PIX_INTENSITY):
        self.x, self.y = x, y
        self.dx, self.dy = dx, dy
        self.rx, self.ry = rx, ry
        self.intensity = intensity

    def move(self, timestep):
        # Moves dx,dy every 20 minutes.
        self.x += self.dx * timestep/60.0
        self.y += self.dy * timestep/60.0

    def get_mu(self):
        return np.array([self.x, self.y])

    def get_sigma(self):
        return np.array([[self.rx, 0.2], [0.2, self.ry]])

    def get_intensity(self):
        return self.intensity

    def __str__(self):
        return "cloud: (x=" + str(self.x) + " y=" + str(self.y) + " rx=" + str(self.rx) + " ry=" + str(self.ry)


class State():
    ''' Class for Solar Panel States '''

    def __init__(self, objects, date_time, longitude, latitude, sun_angle_AZ, sun_angle_ALT, clouds=[]):
        self.date_time = date_time
        self.longitude = longitude
        self.latitude = latitude
        self.sun_angle_AZ = sun_angle_AZ
        self.sun_angle_ALT = sun_angle_ALT
        self.objects = objects
        
        # Hm.
        self.clouds = clouds

    # --- Time and Loc (for trackers) ---

    def get_day_of_year(self):
        return self.date_time.timetuple().tm_yday

    def get_year(self):
        return self.date_time.year

    def get_month(self):
        return self.date_time.month

    def get_day(self):
        return self.date_time.day

    def get_hour(self):
        return self.date_time.hour

    def get_longitude(self):
        return self.longitude

    def get_latitude(self):
        return self.latitude

    def get_date_time(self):
        return self.date_time

    # --- State Attributes ---

    def get_sun_angle_AZ(self):
        return self.sun_angle_AZ

    def get_sun_angle_ALT(self):
        return self.sun_angle_ALT

    def get_panels(self):
        return self.objects["panel"]

    def get_panel_angle_ew(self, panel_index=0):
        return self.objects["panel"][panel_index][1]["angle_ew"] * 1

    def get_panel_angle_ns(self, panel_index=0):
        return self.objects["panel"][panel_index][1]["angle_ns"] * 1