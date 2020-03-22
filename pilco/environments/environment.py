import gym


class Environment:

    def __init__(self,
                 name,
                 sub_sampling_factor=1):

        self.env = gym.make(name)
        self.env.reset()
        self.sub_sampling_factor = sub_sampling_factor

    def reset(self):

        self.env.reset()

        return self.env.state

    def step(self, action):

        state = self.env.state.copy()
        
        for i in range(self.sub_sampling_factor):
            
            self.env.step(action)

        next_state = self.env.state.copy()

        return state, action, next_state


