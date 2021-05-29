'''
tracking_baselines.py

Contains tracking functions for computing the location of the sun, primarily from:

    "Five new algorithms for the computation of sun position from 2010 to 2110"
    by Roberto Grena, 2012, Solar Energy, Volume 86.
'''

# Python libs.
import math as m
import numpy
from pysolar import solar


import alphasolar.solar_helpers as sh

def _compute_new_times(year, month, day, hour):
    '''
    Args:
        Same as algorithms below

    Returns:
        (tuple):
            year (int)
            month (int)
            time (int)
            rotation_independent_time (int)
    '''
    # From Grena
    if month <= 2:
        month += 12
        year -= 1

    time = int(365.25 * (year - 2000)) + int(30.6001 * (month + 1)) \
        - int(0.01 * year) + day + 0.0416667* hour - 21958

    delta_t = 96.4 + 0.00158 * time # Diff between UT and TT (in seconds). Seems right.

    te = time + 1.1574 * 10**(-5) * delta_t

    return year, month, time, te

def static_policy(state, action="do_nothing"):
    return action

def optimal_policy(state):
    return "optimal"

# ==========================
# ======== TRACKERS ========
# ==========================

def tracker_from_state_info(state):
    '''
    Args:
        state (SolarOOMDP state): contains the panel and sun az/alt.
        panel_shift (int): how much to move the panel by each timestep.

    Returns:
        (tuple): <sun_az, sun_alt>
    '''

    # When state has this stuff.
    sun_az = state.get_sun_angle_AZ()
    sun_alt = state.get_sun_angle_ALT()

    return sun_az, sun_alt

def tracker_from_day_time_loc(state, tracker):
    '''
    Args:
        state (SolarOOMDP state): contains the year, month, hour etc.

    Returns:
        (tuple): <sun_az, sun_alt>
    '''

    # Get relevant data.
    year, month, hour, day, = state.get_year(), state.get_month(), state.get_hour(), state.get_day()
    longitude, latitude = state.get_longitude(), state.get_latitude()

    # Use tracker to compute sun vector.
    sun_az, sun_alt = tracker(year, month, hour, day, delta_t, longitude, latitude)

    return sun_az, sun_alt

def grena_tracker(state):
    '''
    Args:
        state (OOMDPstate)

    Returns:
        (tuple): represents sun location.
            altitude (float): degrees up from the horizon.
            azimuth (float): degrees along the equator from north.
    '''
    date_time_obj = state.get_date_time() + state.get_date_time().utcoffset()
    year, month, day, hour = date_time_obj.year, date_time_obj.month, date_time_obj.day, date_time_obj.hour
    latitude_deg, longitude_deg = state.get_latitude(), state.get_longitude()
    latitude_rad, longitude_rad = m.radians(latitude_deg), m.radians(longitude_deg)

    year, month, time, rot_ind_time = _compute_new_times(year, month, day, hour)

    pressure = 1.0
    temperature = 15.0

    t = int(365.25*float(year-2000) + int(30.6001*float(month+1)) - int(0.01*float(year)) + day) + 0.0416667*hour - 21958.0
    te = t + 1.1574e-5*70;

    wte = 0.017202786*te;
    s1, c1 = m.sin(wte), m.cos(wte)
    s2 = 2 *s1 *c1
    c2 = (c1 + s1) * (c1 - s1)
    s3 = s2*c1 + c2*s1
    c3 = c2*c1 - s2*s1
    s4 = 2.0*s2*c2
    c4 = (c2+s2)*(c2-s2)

    pi_2 = 2 * m.pi
    pi_ov_2 = m.pi / 2.0

    right_asc = -1.38880 + 1.72027920e-2*te + 3.199e-2*s1 - 2.65e-3*c1 + 4.050e-2*s2 + 1.525e-2*c2 + 1.33e-3*s3 + 3.8e-4*c3 + 7.3e-4*s4 + 6.2e-4*c4;
    right_asc = right_asc % pi_2
    if (right_asc < 0.0):
        right_asc += pi_2
    decl = 6.57e-3 + 7.347e-2*s1 - 3.9919e-1*c1 + 7.3e-4*s2 - 6.60e-3*c2 + 1.50e-3*s3 - 2.58e-3*c3 + 6e-5*s4 - 1.3e-4*c4 + 0.2967
    hour_angle = 1.75283 + 6.3003881*t + longitude_rad - right_asc
    hour_angle = ((hour_angle + m.pi) % pi_2) - m.pi

    if (hour_angle < -m.pi):
        hour_angle += pi_2

    sp = m.sin(latitude_rad);
    cp = m.sqrt((1-sp*sp));
    sd = m.sin(decl);
    cd = m.sqrt(1-sd*sd);
    sH = m.sin(hour_angle);
    cH = m.cos(hour_angle);
    se0 = sp*sd + cp*cd*cH;
    ep = m.asin(se0) - 4.26e-5*m.sqrt(1.0-se0*se0);
    azimuth_estimate = m.atan2(sH, cH*sp - sd*cp/cd);

    if (ep > 0.0):
        Pressure = 1.0
        De = (0.08422*Pressure) / ((273.0+temperature)*m.tan(ep + 0.003138/(ep + 0.08919)))
    else:
        De = 0.0

    zenith = pi_2 - ep - De;

    # Flip axis direction.
    azimuth_estimate =  -((360 + m.degrees(azimuth_estimate)) % 360)

    # Estimate altitude.
    altitude_estimate = m.degrees(m.asin(m.cos(latitude_rad) * m.cos(decl) * \
                        m.cos(hour_angle) + m.sin(latitude_rad) * m.sin(decl)))

    return altitude_estimate, azimuth_estimate

def _final_step(right_asc, declination, hour_angle, latitude, longitude):
    '''
    Args:
        right_asc (float): alpha, radians
        declination (float): delta, radians
        hour_angle (float): H, radians
        latitude (float): phi, radians
        longitude (float): theta, radians

    Summary:
        Implements Section 3.7 of [Grena 2012], which determines the azimuth and zenith.
    '''
    lat_radians, long_radians = m.radians(latitude), m.radians(longitude)


    sp = m.sin(lat_radians)
    cp = m.sqrt((1-sp*sp))
    sd = m.sin(declination)
    cd = m.sqrt(1-sd*sd)
    sH = m.sin(hour_angle)
    cH = m.cos(hour_angle)
    se0 = sp*sd + cp*cd*cH
    ep = numpy.arcsin(se0) - 4.26e-5*m.sqrt(1.0-se0*se0)
    azimuth = numpy.arctan2(sH, cH*sp - sd*cp/cd)

    Pressure = 1.0
    temperature = 20
    if (ep > 0.0):
        De = (0.08422*Pressure) / ((273.0+temperature)*m.tan(ep + 0.003138/(ep + 0.08919)));
    else:
        De = 0.0;

    m.piM = 1.57079632679490
    zenith = m.piM - ep - De;

    return m.degrees(azimuth), m.degrees(zenith)

    # e_zero = m.asin(m.sin(lat_radians) * m.sin(declination) + m.cos(lat_radians) * m.cos(declination) * m.cos(hour_angle))

    # dpe = -4.26 * 10**(-5) * m.cos(e_zero)

    # ep = e_zero + dpe

    # # Gamma
    # azimuth = numpy.arctan2(m.sin(hour_angle), m.cos(hour_angle) * m.sin(lat_radians) - m.tan(declination) * m.cos(lat_radians))

    # # dre = Compute zenith offset due to tempreature/pressure

    # zenith = m.pi / 2.0 - ep

def _asc_decl_ha_to_alt_az(right_asc, declination, hour_angle, latitude):
    latitude_radians = m.radians(latitude)
    alt_temp = m.sin(declination)*m.sin(latitude_radians)+m.cos(declination)*m.cos(latitude_radians)*m.cos(hour_angle)
    altitude = numpy.arcsin(alt_temp)

    az_temp = (m.sin(declination) - m.sin(altitude) * m.sin(latitude_radians)) / (m.cos(altitude)*m.cos(latitude_radians))
    azimuth = numpy.arccos(az_temp)
    if m.sin(hour_angle) >= 0:
        azimuth = 360 - azimuth

    return m.degrees(altitude), m.degrees(azimuth)

def main():
    simple_tracker(year=2060, month=1, hour=13, day=26, delta_t=0, longitude=.1, latitude=-.2)

if __name__ == "__main__":
    main()
