import gym
from gym.utils import seeding
from gym.spaces.discrete import Discrete
from gym.spaces import Box
from .room_utils import generate_room
from .render_utils import room_to_rgb, room_to_tiny_world_rgb
import numpy as np
import torch

class SokobanEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array', 'raw'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array', 'raw']
    }

    def __init__(self,
                 dim_room=(9, 9),
                 max_steps=120,
                 num_boxes=4,
                 num_gen_steps=None,
                 reset=True):

        # General Configuration
        self.dim_room = dim_room
        if num_gen_steps == None:
            self.num_gen_steps = int(1.7 * (dim_room[0] + dim_room[1]))
        else:
            self.num_gen_steps = num_gen_steps

        self.num_boxes = num_boxes
        self.boxes_on_target = 0

        # Penalties and Rewards
        self.penalty_for_step = -0.1
        # self.penalty_box_off_target = -1
        self.penalty_box_off_target = -0.2
        self.reward_box_on_target = 10 # 20, 600 loop
        # self.reward_finished = 10
        self.reward_finished = self.reward_box_on_target * num_boxes * 100 * 3
        self.reward_last = 0

        # Other Settings
        self.viewer = None
        self.max_steps = max_steps
        self.action_space = Discrete(len(ACTION_LOOKUP))
        screen_height, screen_width = (dim_room[0] * 16, dim_room[1] * 16)
        # self.observation_space = Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        self.observation_space = Box(low = 0, high = 1, shape = (4 * dim_room[0] * dim_room[1], ), dtype = np.uint8)
        self.hasBox = False
        
        if reset:
            # Initialize Room
            _ = self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, observation_mode='raw'):
        # assert action in ACTION_LOOKUP # PROではActionの確率密度が入力される
        # action = np.argmax(action)
        assert observation_mode in ['rgb_array', 'tiny_rgb_array', 'raw']

        self.num_env_steps += 1

        self.new_box_position = None
        self.old_box_position = None

        moved_box = False

        moved_player = self._move(action)

        self._calc_reward()
        
        done = self._check_if_done()

        # Convert the observation to RGB frame
        observation = self.render(mode=observation_mode)

        info = {
            "action.name": ACTION_LOOKUP[action],
            "action.moved_player": moved_player,
            "action.moved_box": moved_box,
        }
        if done:
            info["maxsteps_used"] = self._check_if_maxsteps()
            info["all_boxes_on_target"] = self._check_if_all_boxes_on_target()
            print('{} boxes on target, reward {}'.format(self.boxes_on_target, self.reward_last))

        return torch.flatten(torch.from_numpy(observation).clone()).to(device='cuda'), self.reward_last, done, info

    def _push(self, action):
        """
        Perform a push, if a box is adjacent in the right direction.
        If no box, can be pushed, try to move.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # No push, if the push would get the box out of the room's grid
        new_box_position = new_position + change
        if new_box_position[0] >= self.room_state.shape[0] \
                or new_box_position[1] >= self.room_state.shape[1]:
            return False, False


        can_push_box = self.room_state[new_position[0], new_position[1]] in [3, 4]
        can_push_box &= self.room_state[new_box_position[0], new_box_position[1]] in [1, 2]
        if can_push_box:

            self.new_box_position = tuple(new_box_position)
            self.old_box_position = tuple(new_position)

            # Move Player
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            # Move Box
            box_type = 4
            if self.room_fixed[new_box_position[0], new_box_position[1]] == 2:
                box_type = 3
            self.room_state[new_box_position[0], new_box_position[1]] = box_type
            return True, True

        # Try to move if no box to push, available
        else:
            return self._move(action), False

# TYPE_LOOKUP = {
#     0: 'wall',
#     1: 'empty space',
#     2: 'box target',
#     3: 'box on target',
#     4: 'box not on target',
#     5: 'player'
#     6: 'player on target',
#     7: 'player has box'
# }

    def _move(self, action):
        """
        Moves the player to the next field, if it is not occupied.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # new_positionにBoxがあったときhasbox = True
        # Trueになった瞬間に、fix_roomをBoxから1に変更
        if self.room_state[new_position[0], new_position[1]] == 4:
            self.player_position = new_position
            self.room_state[new_position[0], new_position[1]] = 7
            if self.hasBox == False:
                self.hasBox = True
                self.room_fixed[new_position[0], new_position[1]] = 1

            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            return True

        # Move player if the field in the moving direction is either
        # an empty field or an empty box target.
        if self.room_state[new_position[0], new_position[1]] in [1,3]:
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 7 if self.hasBox else 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]
            return True
        
        if self.room_state[new_position[0], new_position[1]] == 2:
            if self.hasBox == False:
                return False

            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 7 
            
            self.room_fixed[(new_position[0], new_position[1])] = 3
            self.hasBox = False
                
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            return True


        return False

    def _calc_reward(self):
        """
        Calculate Reward Based on
        :return:
        """
        # Every step a small penalty is given, This ensures
        # that short solutions have a higher reward.
        self.reward_last += self.penalty_for_step
        
        # When player carries a box, he gets a small penalty.
        # current_position = self.player_position.copy()
        # if self.room_state[current_position[0], current_position[1]] == 7:
        #     self.reward_last += self.penalty_for_step
        
        # # Every moving box step is a small penalty.
        # current_position = self.player_position.copy()
        # if self.room_state[current_position[0], current_position[1]] == 7:
        #     self.reward_last += self.penalty_for_step

        # count boxes on the target
        current_boxes_on_target = np.count_nonzero(self.room_fixed == 3)

        # Add the reward if a box is pushed on the target and give a
        # penalty if a box is pushed off the target.        env.render(mode='human')
        if current_boxes_on_target > self.boxes_on_target:
            self.reward_last += self.reward_box_on_target * current_boxes_on_target
            # print('Box on Target: {}, reward: {}'.format(current_boxes_on_target, self.reward_last))
        elif current_boxes_on_target < self.num_boxes:
            # self.reward_last += self.penalty_box_off_target * 5 * (self.boxes_on_target - current_boxes_on_target)
            self.reward_last += self.penalty_box_off_target

        game_won = self._check_if_all_boxes_on_target()        
        if game_won:
            print('Game Complete!')
            self.reward_last += self.reward_finished
        
        self.boxes_on_target = current_boxes_on_target

    def _check_if_done(self):
        # Check if the game is over either through reaching the maximum number
        # of available steps or by pushing all boxes on the targets.        
        return self._check_if_all_boxes_on_target() or self._check_if_maxsteps()

    def _check_if_all_boxes_on_target(self):
        are_all_boxes_on_target = np.count_nonzero(self.room_fixed == 3)
        return are_all_boxes_on_target == self.num_boxes

    def _check_if_maxsteps(self):
        if self.max_steps == self.num_env_steps:
            # print('Max step: {}, Max Steps Reached: {}'.format(self.max_steps, self.num_env_steps))
            return True
        else:
            return False
        # return (self.max_steps == self.num_env_steps)

    def reset(self, second_player=False, render_mode='raw'):
        try:
            ## create random map
            # self.room_fixed, self.room_state, self.box_mapping = generate_room(
            #     dim=self.dim_room,
            #     num_steps=self.num_gen_steps,
            #     num_boxes=self.num_boxes,
            #     second_player=second_player
            # )
            
            # create fixed map for testing
            # 9 x 9 map and 4 boxes
            self.room_fixed =  np.array([[0,0,0,0,0,0,0,0,0],
                                [0,1,1,1,1,1,1,1,0],
                                [0,1,1,2,0,1,1,1,0],
                                [0,1,0,2,2,1,1,1,0],
                                [0,1,0,0,2,0,0,1,0],
                                [0,1,4,4,4,1,4,1,0],
                                [0,1,1,1,1,0,0,1,0],
                                [0,1,1,1,1,1,1,1,0],
                                [0,0,0,0,0,0,0,0,0],
                                ])
            
            self.room_state =  np.array([[0,0,0,0,0,0,0,0,0],
                                [0,1,1,1,1,1,1,1,0],
                                [0,1,1,2,0,1,1,1,0],
                                [0,1,0,2,2,1,1,1,0],
                                [0,1,0,0,2,0,0,1,0],
                                [0,1,4,4,4,5,4,1,0],
                                [0,1,1,1,1,0,0,1,0],
                                [0,1,1,1,1,1,1,1,0],
                                [0,0,0,0,0,0,0,0,0],
                                ])


            
        except (RuntimeError, RuntimeWarning) as e:
            print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
            print("[SOKOBAN] Retry . . .")
            return self.reset(second_player=second_player, render_mode=render_mode)
        self.player_position = np.argwhere(self.room_state == 5)[0]
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0
        self.hasBox = False

        starting_observation = self.render(render_mode)
        return torch.flatten(torch.from_numpy(starting_observation).clone()).to(device='cuda')

    def render(self, mode='human', close=None, scale=1):
        assert mode in RENDERING_MODES

        img = self.get_image(mode, scale)

        if 'rgb_array' in mode:
            return img

        elif 'human' in mode:
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

        elif 'raw' in mode:
            arr_walls = (self.room_fixed == 0).view(np.int8)
            arr_goals = (self.room_fixed == 2).view(np.int8)
            arr_boxes = ((self.room_state == 4) + (self.room_state == 3) + (self.room_fixed == 3) + (self.room_state == 7)).view(np.int8)
            arr_player = ((self.room_state == 5) + (self.room_state == 7) + (self.room_state == 6)).view(np.int8)

            return np.array([arr_walls, arr_goals, arr_boxes, arr_player])
            # return arr_walls, arr_goals, arr_boxes, arr_player

        else:
            super(SokobanEnv, self).render(mode=mode)  # just raise an exception

    def get_image(self, mode, scale=1):
        
        if mode.startswith('tiny_'):
            img = room_to_tiny_world_rgb(self.room_state, self.room_fixed, scale=scale)
        else:
            img = room_to_rgb(self.room_state, self.room_fixed)

        return img

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def set_maxsteps(self, num_steps):
        self.max_steps = num_steps

    def get_action_lookup(self):
        return ACTION_LOOKUP

    def get_action_meanings(self):
        return ACTION_LOOKUP


ACTION_LOOKUP = {
    0: 'move up',
    1: 'move down',
    2: 'move left',
    3: 'move right',
}
# ACTION_LOOKUP = {
#     0: 'no operation',
#     1: 'push up',
#     2: 'push down',
#     3: 'push left',
#     4: 'push right',
#     5: 'move up',
#     6: 'move down',
#     7: 'move left',
#     8: 'move right',
# }

# Moves are mapped to coordinate changes as follows
# 0: Move up
# 1: Move down
# 2: Move left
# 3: Move right
CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

RENDERING_MODES = ['rgb_array', 'human', 'tiny_rgb_array', 'tiny_human', 'raw']
