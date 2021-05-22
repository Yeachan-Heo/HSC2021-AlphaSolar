
# Python imports.
import os
import math as m
import numpy as np

# Misc. imports.
from pysolar import solar, radiation #capitalize to get it to work?

CLOUD_DIFFUS_FACTOR = 1.0 #0.85 # 10% of light is blocked

def _write_datum_to_file(mdp_name, agent, datum, datum_name):
    out_file = open(os.path.join("..", "results", mdp_name, str(agent)) + "-" + datum_name + ".csv", "a+")
    out_file.write(str(datum) + ",")
    out_file.close()

def _compute_sun_altitude(latitude_deg, longitude_deg, time):
    return solar.get_altitude(latitude_deg, longitude_deg, time)

def _compute_sun_azimuth(latitude_deg, longitude_deg, time):
    return solar.get_azimuth(latitude_deg, longitude_deg, time)

# --- Radiation hitting the surface of the Earth ---

def _compute_radiation_direct(time, sun_altitude_deg):
    return max(_get_radiation_direct(time, sun_altitude_deg), 0.0)

def _compute_radiation_diffuse(time, day, sun_altitude_deg):
    sky_diffus = _compute_sky_diffusion(day)
    return max(sky_diffus * _compute_radiation_direct(time, sun_altitude_deg), 0.0)

def _compute_radiation_reflective(time, day, reflective_index, sun_altitude_deg):
    sky_diffus = _compute_sky_diffusion(day)
    rad_direct = _compute_radiation_direct(time, sun_altitude_deg)
    return max(reflective_index * rad_direct * (m.sin(m.radians(sun_altitude_deg)) + sky_diffus), 0.0)

def _compute_sky_diffusion(day):
    return 0.095 + 0.04 * m.sin(0.99*day - 99)

# --- CLOUDS ---

def _compute_direct_cloud_cover(clouds, sun_x, sun_y, img_dims):
    sun_dim = img_dims / 8.0
    total_sun_light = 0.0
    total_covered_light = 0.0
    sun_x_range = range( int(max(sun_x - sun_dim, 0)), int(min(sun_x + sun_dim, img_dims)))
    sun_y_range = range( int(max(sun_y - sun_dim, 0)), int(min(sun_y + sun_dim, img_dims)))

    for i in sun_x_range:
        for j in sun_y_range:
            sun_light = _gaussian(j, sun_x, sun_dim) * _gaussian(i, sun_y, sun_dim)
            # Loop the central location of the sun and compute cloud cover:
            cloud_cover = 0.0
            for cloud in clouds:
                cloud_cover += (_gaussian(j, cloud.get_mu()[0], cloud.get_sigma()[0][0]) * \
                                    _gaussian(i, cloud.get_mu()[1], cloud.get_sigma()[1][1]) * cloud.get_intensity())

            total_sun_light += sun_light
            total_covered_light += (sun_light - cloud_cover*CLOUD_DIFFUS_FACTOR)

    if total_sun_light > 0 and total_covered_light > 0:
        return float(total_covered_light) / total_sun_light
    return 1.0


# --- Tilt Factors ---

def _compute_direct_radiation_tilt_factor(panel_ns_deg, panel_ew_deg, sun_altitude_deg, sun_azimuth_deg):
    '''
    Args:
        panel_ns_deg (float): in the range [-90, 90], 0 is facing up.
        panel_ew_deg (float): in the range [-90, 90], 0 is facing up.
    Summary:
        Per the model of:
    '''
    sun_vector = _compute_sun_vector(sun_altitude_deg, sun_azimuth_deg)
    panel_normal = _compute_panel_normal_vector(panel_ns_deg, panel_ew_deg)

    cos_diff = np.dot(sun_vector, panel_normal)

    return max(cos_diff, 0)

def _compute_sun_vector(sun_altitude_deg, sun_azimuth_deg):
    '''
    Args:
        sun_altitude_deg (float)
        sun_azimuth_deg (float)
    Notes:
        We assume x+ is North, x- is South, y+ is E, y- is W, z+ is up, z- is down.
    '''

    sun_alt_radians, sun_az_radians = m.radians(sun_altitude_deg), m.radians(sun_azimuth_deg)
    x = m.sin(m.pi - sun_az_radians) * m.cos(sun_alt_radians)
    y = m.cos(m.pi - sun_az_radians) * m.cos(sun_alt_radians)
    z = m.sin(sun_alt_radians)
    
    #x = m.cos(m.pi - sun_az_radians) * m.cos(sun_alt_radians)
    #y = m.cos(m.pi - sun_az_radians) * m.sin(sun_alt_radians)
    #z = m.sin(sun_alt_radians)

    return _normalize(x, y, z)

def _compute_panel_normal_vector(panel_ns_deg, panel_ew_deg):
    panel_ns_radians, panel_ew_radians = m.radians(panel_ns_deg), m.radians(panel_ew_deg)

    # Compute panel normal.
    x = m.sin(panel_ns_radians)*m.cos(panel_ew_radians)
    y = m.sin(panel_ew_radians)*m.cos(panel_ns_radians)
    z = m.cos(panel_ns_radians)*m.cos(panel_ew_radians)

    return _normalize(x, y, z)

def _normalize(x, y, z):
    tot = m.sqrt(x**2 + y**2 + z**2)
    return np.array([x / tot, y / tot, z / tot])

def _compute_diffuse_radiation_tilt_factor(panel_ns_deg, panel_ew_deg):
    '''
    Args:
        panel_ns_deg (float)
        panel_ew_deg (float)
    Returns:
        (float): The diffuse radiation tilt factor.
    '''
    ns_radians = m.radians(abs(panel_ns_deg))
    ew_radians = m.radians(abs(panel_ew_deg))
    diffuse_radiation_angle_factor = (m.cos(ns_radians) + m.cos(ew_radians)) / 2.0

    return diffuse_radiation_angle_factor

def _compute_reflective_radiation_tilt_factor(panel_ns_deg, panel_ew_deg):
    return (2 - m.cos(m.radians(panel_ns_deg)) - m.cos(m.radians(panel_ew_deg))) / 2.0

# --- Misc. ---

# DIRECTLY FROM PYSOLAR (with different conditional)
def _get_radiation_direct(utc_datetime, sun_altitude_deg):
    # from Masters, p. 412
    if 0 < sun_altitude_deg < 180:
        
        day = utc_datetime.day # solar.GetDayOfYear(utc_datetime)
        flux = radiation.get_apparent_extraterrestrial_flux(day)
        optical_depth = radiation.get_optical_depth(day)
        air_mass_ratio = radiation.get_air_mass_ratio(sun_altitude_deg)
        return flux * m.exp(-1 * optical_depth * air_mass_ratio)
    else:
        return 0.0

# Credit to http://stackoverflow.com/questions/14873203/plotting-of-1-dimensional-gaussian-distribution-function
# TODO: fix ^
def _gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))