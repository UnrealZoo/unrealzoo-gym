from gym_unrealcv._gym_compat import Wrapper


class ConfigUEWrapper(Wrapper):
    def __init__(self, env, docker=False, resolution=(160, 160), display=None, offscreen=False,
                            use_opengl=False, nullrhi=False, gpu_id=None, sleep_time=5, comm_mode='tcp'):
        super().__init__(env)
        env.unwrapped.docker = docker
        env.unwrapped.display = display
        env.unwrapped.offscreen_rendering = offscreen
        env.unwrapped.use_opengl = use_opengl
        env.unwrapped.nullrhi = nullrhi
        env.unwrapped.gpu_id = gpu_id
        env.unwrapped.sleep_time = sleep_time
        env.unwrapped.resolution = resolution
        env.unwrapped.comm_mode = comm_mode