class Reward:
    '''
    define different type reward function
    '''
    def __init__(self):
        # self.reward_factor = setting['reward_factor']
        # self.reward_th = setting['reward_th']
        self.reward_factor = 10
        self.reward_th = 0.1
        self.dis2target_last = 0

    def reset(self, initial_distance=0):
        self.dis2target_last = initial_distance

    def reward_bbox(self, boxes):
        reward = 0

        # summarize the reward of each detected box
        for box in boxes:
            reward += self.get_bbox_reward(box)

        if reward > self.reward_th:
            # get ideal target
            reward = min(reward * self.reward_factor, 10)
        elif reward == 0:
            # false trigger
            reward = -1
        else:
            # small target
            reward = 0

        return reward, boxes

    def get_bbox_reward(self, box):
        # get reward of single box considering the size and position of box
        (xmin, ymin), (xmax, ymax) = box
        boxsize = (ymax - ymin) * (xmax - xmin)
        x_c = (xmax + xmin) / 2.0
        x_bias = x_c - 0.5
        discount = max(0, 1 - x_bias ** 2)
        reward = discount * boxsize
        return reward

    def reward_distance(self, dis2target_now):
        if self.dis2target_last <= 0:
            self.dis2target_last = dis2target_now
            return 0
        reward = (self.dis2target_last - dis2target_now) / max(self.dis2target_last, 100)
        self.dis2target_last = dis2target_now

        return reward
