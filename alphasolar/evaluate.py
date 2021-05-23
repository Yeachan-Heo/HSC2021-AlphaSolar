from alphasolar.run_experiment import *
import ray


def run_with_policy(env_cls, env_config, policy):
    env = env_cls(env_config)
    s = env.reset()
    d = False 
    r_lst = [0]
    s_lst = [s]
    t_lst = [env.time]
    while not d:
        s, r, d, _ = env.step(policy(s))
        r_lst.append(r)
        s_lst.append(s)
        t_lst.append(env.time)
    
    return np.array(r_lst), np.array(s_lst), np.array(t_lst)


def save_arrays(r_lst, s_lst, t_lst, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f"score of {save_dir.split('/')[-1]}: ", sum(r_lst))
    np.save(os.path.join(save_dir, "r_lst.npy"), r_lst)
    np.save(os.path.join(save_dir, "s_lst.npy"), s_lst)
    np.save(os.path.join(save_dir, "t_lst.npy"), t_lst)


def run_with_policy_and_save(env_cls, env_config, policy, save_dir):
    s_lst, r_lst, t_lst = run_with_policy(env_cls, env_config, policy)
    save_arrays(s_lst, r_lst, t_lst, save_dir)


def evaluate():
    args = parse_args()
    panel = Panel(x_dim=1,
                  y_dim=1,
                  assembly_mass=15, #kg
                  COM_offset=0.1, #meters
                  bearing_friction=0.1, #coeff of friction, totally made up
                  efficiency=0.9,
                  offset_angle=0.52, #radians
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

    pprint(config)

    local_dir = (f"{timezone_str}_{args.name}_{date_time}_{('dual' if args.dual_axis else 'single')}_{'no' if args.cloud else 'yes'}cloud_p{args.panel_step}")
    evaluation_dir = os.path.join("./evaluation_results", local_dir)
    local_dir = os.path.join("./ray_results", local_dir)

    restore_dir = os.path.join(local_dir, sorted(os.listdir(local_dir))[-1])
    restore_dir = os.path.join(restore_dir, sorted(list(filter(lambda x: "PPO" in x, os.listdir(restore_dir))))[-1])
    restore_dir = os.path.join(restore_dir, sorted(list(filter(lambda x: "checkpoint" in x, os.listdir(restore_dir))))[-1])
    restore_dir = os.path.join(restore_dir, sorted(list(filter(lambda x: not ("." in x), os.listdir(restore_dir))))[-1])

    print(restore_dir)
    
    ray.init()

    agent = PPOTrainer(config)
    agent.restore(restore_dir)

    agent_policy = agent.compute_action
    optimal_policy = lambda s: "optimal"
    static_policy = lambda s: 2 if args.dual_axis else 1

    run_with_policy_and_save(AlphaSolarEnvRllib, env_config, agent_policy, os.path.join(evaluation_dir, "agent"))
    run_with_policy_and_save(AlphaSolarEnvRllib, env_config, optimal_policy, os.path.join(evaluation_dir, "optimal"))
    run_with_policy_and_save(AlphaSolarEnvRllib, env_config, static_policy, os.path.join(evaluation_dir, "static"))

evaluate()