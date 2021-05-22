from alphasolar.env import *
from pytz import timezone
import imageio

if __name__ == '__main__':
    plt.ion()

    dt = datetime.datetime.now()
    localtz = timezone('America/New_York')
    dt = localtz.localize(dt)

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

    env = AlphaSolarEnv(panel,
    date_time=dt, 
    name_ext="test")

    state = env.reset()
    done = False
    
    images = []

    while not done:
        state, rew, done, _ = env.step(env.action_space.sample())
        images.append(state.reshape(32, 32))
        print(rew, done)
    
    imageio.mimsave("./sample.gif", images)
    