import os
import argparse

from ray import tune
from ray.tune import stopper
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from alphasolar.env import *
from pytz import timezone
from tzwhere import tzwhere as tzw
from pprint import pprint

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--dual_axis", default=True, type=bool)
    parser.add_argument("--cloud", default=True, type=bool)
    parser.add_argument("--name", type=str)
    parser.add_argument("--panel_step", default=20)
    parser.add_argument("--timestep", default=20)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    
    panel = Panel(x_dim=1,
                  y_dim=1,
                  assembly_mass=15, #kg
                  COM_offset=0.1, #meters
                  bearing_friction=0.1, #coeff of friction, totally made up
                  efficiency=0.9,
                  offset_angle=0.25, #radians
                  actuator_force=1500,#TODO: remove, not used
                  actuator_offset_ew=0.1,
                  actuator_mount_ew=0.5,
                  actuator_offset_ns=0.1,
                  actuator_mount_ns=0.5)
    
    tzwhere = tzw.tzwhere()

    timezone_str = tzwhere.tzNameAt(args.lat, args.lon) # Seville coordinates

    date_time = datetime.datetime(2020, 6, 5, 3)
    localtz = timezone(timezone_str)
    
    date_time = localtz.localize(date_time)
    
    timezone_str = timezone_str.replace("/", "")
    n_steps = (60 // args.timestep) * 24

    env_config = {
        "panel" : panel,
        "date_time" : date_time,
        "name_ext" : timezone_str,
        "timestep" : args.timestep,
        "panel_step" : args.panel_step,
        "latitude_deg" : args.lat,
        "longitude_deg" : args.lon,
        "n_steps" : n_steps,
        "mode_dict" : {"dual_axis" : args.dual_axis, "image_mode" : True, "cloud_mode" : args.cloud}
    }

    config = DEFAULT_CONFIG.copy()

    config["env"] = AlphaSolarEnvRllib
    config["env_config"] = env_config
    config["num_workers"] = 7

    pprint(config)

    local_dir = (f"{timezone_str}_{args.name}_{date_time}_{('dual' if args.dual_axis else 'single')}_{'no' if args.cloud else 'yes'}cloud_p{args.panel_step}")
    local_dir = os.path.join("./", local_dir)

    env = config["env"](env_config)

    env.reset()
    s = 0
    while True:
        _, r, d, _ = env.step(2)
        s += r
        if d:
            break
    print("baseline score: ", s)

main()
#