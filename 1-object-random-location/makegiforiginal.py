import pygame
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody)
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
import numpy as np
import random
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

SCREEN_WIDTH = 320
SCREEN_HEIGHT = 320
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.DOUBLEBUF)
pygame.display.set_caption('test')
clock = pygame.time.Clock()
PPM = 1
SCALE = PPM
TARGET_FPS = 1000
TIME_STEP = 1.0 / TARGET_FPS

def get_game_screen(screen):

    screen = np.rot90(pygame.surfarray.array3d(screen))[::-1]
    return screen

class Environment:

    def __init__(self):
        self.body_list = []
        self.world = world(gravity=(0, -100), sleep=False)
        self.colour_list = []

        self.possible_actions = [self.move_up, self.move_down, self.move_left, self.move_right, self.rot_left]
        num_actions = len(self.possible_actions)
        self.action_space = np.linspace(0, num_actions - 1, num_actions)
        print(self.action_space)
        base_state = self.reset()
        self.observation_space = np.asarray(base_state)

    def sample(self):
        return random.randint(0, len(self.action_space)-1)

    def add_box(self, position, density, friction, angle, size):
        self.body_list.append(self.world.CreateDynamicBody(position=position, angle=angle))
        box = self.body_list[-1].CreatePolygonFixture(box=size, density=density, friction=friction)
        self.colour_list.append((150, 150, 150, 150))

    def reset(self):
        self.obj_type = 5
        self.timer = 0
        self.trigger = 'stop'
        self.curr_grab = 1000
        self.curr_right = 0
        self.consecutive_wait = 0
        self.dist_from_obj = 0
        self.dist_from_pos = 0
        self.colour_list = []
        self.obj_type = 1
        self.lift_reward = [0.1]*10
        self.max_height = 200
        for i_body in range(len(self.body_list)):
            if i_body != 0:
                self.world.DestroyBody(self.body_list[i_body])
        self.body_list = []
        self.ground_body = self.world.CreateStaticBody(
            position=(160, 0),
            shapes=polygonShape(box=(160, 7.5)),
            angle=0
        )
        self.wall_body1 = self.world.CreateStaticBody(
            position=(320, 80),
            shapes=polygonShape(box=(7.5, 320)),
            angle=0
        )
        self.wall_body2 = self.world.CreateStaticBody(
            position=(0, 80),
            shapes=polygonShape(box=(7.5, 320)),
            angle=0
        )
        self.wall_body3 = self.world.CreateStaticBody(
            position=(100000, 0),
            shapes=polygonShape(box=(5, 70)),
            angle=0
        )
        self.roof = self.world.CreateStaticBody(
            position=(160, 320),
            shapes=polygonShape(box=(160, 7.5)),
            angle=0
        )
        pos1 = np.random.uniform(15, 130)
        pos2 = np.random.uniform(280, 310)
        self.arm = self.world.CreateDynamicBody(
            position=(pos1, 315),
            shapes=polygonShape(box=(6, 12)),
            angle=0
        )
        self.grab = self.world.CreateDynamicBody(
            position=(pos1, pos2),
            shapes=polygonShape(box=(6, 6)),
            angle=0,
            fixedRotation=True,
        )
        self.grab.CreatePolygonFixture(box=(6,6), density=0.1, friction=0)
        self.stick = self.world.CreateDynamicBody(
            position=(pos1, pos2),
            shapes=polygonShape(box=(40, 6)),
            angle=0,
        )
        self.stick.CreatePolygonFixture(box=(40,6), density=0.1, friction=0)
        offset = 2.5
        DENSITY_2 = 0.1
        FRICTION = 10
        if self.obj_type in [2,3,5]:
            OBJ_2 = [(-40,30),(-10,30),(-10,20),(-40,20)]
            OBJ_3 = [(-40,-20),(-10,-20),(-10,-30),(-40,-30)]
        else:
            OBJ_2 = [(-40,30),(-10,30),(-10,10),(-40,10)]
            OBJ_3 = [(-40,-10),(-10,-10),(-10,-30),(-40,-30)]
        OBJ_1 = [(-10,30),(10,30),(10,-30),(-10,-30)]
        OBJ_4 = [(10,10),(40,10),(40,-10),(10,-10)]
        OBJ_5 = [(-40,-5),(-10,-5),(-10,5),(-40,5)]
        definedFixturesObj_1 = fixtureDef(shape=polygonShape(vertices=[ (x / SCALE,y / SCALE) for x,y in OBJ_1 ]),
            density=DENSITY_2,
            friction=FRICTION,
            restitution=0.0)

        definedFixturesObj_2 = fixtureDef(shape=polygonShape(vertices=[ (x / SCALE,y / SCALE) for x,y in OBJ_2 ]),
            density=DENSITY_2,
            friction=FRICTION,
            restitution=0.0)

        definedFixturesObj_3 = fixtureDef(shape=polygonShape(vertices=[ (x / SCALE,y / SCALE) for x,y in OBJ_3 ]),
            density=DENSITY_2,
            friction=FRICTION,
            restitution=0.0)

        definedFixturesObj_4 = fixtureDef(shape=polygonShape(vertices=[ (x / SCALE,y / SCALE) for x,y in OBJ_4 ]),
            density=DENSITY_2,
            friction=FRICTION,
            restitution=0.0)
        definedFixturesObj_5 = fixtureDef(shape=polygonShape(vertices=[ (x / SCALE,y / SCALE) for x,y in OBJ_5 ]),
            density=DENSITY_2,
            friction=FRICTION,
            restitution=0.0)
        if self.obj_type == 1:
            fixturesListObj = [definedFixturesObj_1, definedFixturesObj_2, definedFixturesObj_3, definedFixturesObj_4]
        elif self.obj_type == 2:
            fixturesListObj = [definedFixturesObj_1, definedFixturesObj_3, definedFixturesObj_5]
        elif self.obj_type == 3:
            fixturesListObj = [definedFixturesObj_1, definedFixturesObj_2, definedFixturesObj_5]
        elif self.obj_type == 4:
            fixturesListObj = [definedFixturesObj_1, definedFixturesObj_2, definedFixturesObj_3]
        elif self.obj_type == 5:
            fixturesListObj = [definedFixturesObj_1, definedFixturesObj_2, definedFixturesObj_3, definedFixturesObj_5]
        elif self.obj_type == 6:
            fixturesListObj = [definedFixturesObj_1, definedFixturesObj_2, definedFixturesObj_4]

        self.object = self.world.CreateDynamicBody(position = ( random.randint(140, 280),40 ),
            angle=0,
            angularDamping = 10,
            linearDamping = 1,
            fixtures = fixturesListObj)

        self.object.mass=1000

        self.prev_grab_pos = self.grab.position.copy()

        self.h_pj = self.world.CreatePrismaticJoint(
            bodyA=self.wall_body2,
            bodyB=self.arm,
            anchor=(self.wall_body2.worldCenter[0], self.wall_body2.worldCenter[1]),
            axis=(1, 0),
            lowerTranslation=-100000,
            upperTranslation=1000000,
            enableLimit=True,
            motorSpeed=0,
            maxMotorForce=5000000000000000000000,
            enableMotor=True,
        )
        self.v_pj = self.world.CreatePrismaticJoint(
            bodyA=self.arm,
            bodyB=self.grab,
            anchor=(self.grab.worldCenter[0], self.grab.worldCenter[1]),
            axis=(0, 1),
            lowerTranslation=-100000,
            upperTranslation=0,
            enableLimit=True,
            motorSpeed=0,
            maxMotorForce=5000000000000000000000,
            enableMotor=True,
        )
        self.a_rj = self.world.CreateRevoluteJoint(
                bodyA=self.grab,
                bodyB=self.stick,
                anchor=self.stick.worldCenter,
                maxMotorTorque = 100000000,
                motorSpeed = 0,
                enableMotor = True,
                collideConnected = False,
                )
        self.colour_list.append((255, 255, 255, 255))
        self.colour_list.append((255, 255, 255, 255))
        self.colour_list.append((255, 255, 255, 255))
        self.colour_list.append((255, 255, 255, 255))
        self.colour_list.append((255, 255, 255, 255))
        self.colour_list.append((150, 200, 150, 200))
        self.colour_list.append((200, 150, 200, 150))
        self.colour_list.append((200, 150, 200, 150))
        self.body_list = [self.ground_body, self.wall_body1, self.wall_body2, self.roof, self.arm,
                          self.grab, self.object, self.stick]





        self.reward_list = [1] * len(self.body_list)

        self.update_screen()
        self.state1 = get_game_screen(screen)
        return self.state1

    def move_left(self):
        self.trigger = 'left'
        return 0

    def move_right(self):
        self.trigger = 'right'
        return 0

    def move_up(self):
        self.trigger = 'up'
        return 0


    def move_down(self):
        self.trigger = 'down'
        return 0

    def rot_left(self):
        self.trigger = 'rot_left'
        return 0

    def rot_right(self):
        self.trigger = 'rot_left'
        return 0

    def activate_trigger(self):
        if self.trigger == 'left':
            self.v_pj.motorSpeed = 0
            self.h_pj.motorSpeed = -700
            self.a_rj.motorSpeed = 0
        elif self.trigger == 'right':
            self.v_pj.motorSpeed = 0
            self.h_pj.motorSpeed = 700
            self.a_rj.motorSpeed = 0

        elif self.trigger == 'up':
            self.v_pj.motorSpeed = 700
            self.h_pj.motorSpeed = 0
            self.a_rj.motorSpeed = 0

        elif self.trigger == 'down':
            self.v_pj.motorSpeed = -700
            self.h_pj.motorSpeed = 0
            self.a_rj.motorSpeed = 0
        elif self.trigger == 'rot_left':
            self.v_pj.motorSpeed = 0
            self.h_pj.motorSpeed = 0
            self.a_rj.motorSpeed = 10
        elif self.trigger == 'rot_right':
            self.v_pj.motorSpeed = 0
            self.h_pj.motorSpeed = 0
            self.a_rj.motorSpeed = -10
        return -0.01

    def destroy_joint(self):
        try:
            self.world.DestroyJoint(self.grabber_joint)
            del self.grabber_joint
            self.colour_list[self.curr_grab] = (150, 150, 150, 150)
            self.curr_grab = 1000
        except:
            pass

    def step(self, action):
        self.timer += 1
        reward = 0
        self.possible_actions[int(action)]()
        dist_1 = np.sqrt(2*320**2)
        dist_2a = (self.stick.position[0] - self.object.position[0])**2
        dist_2b = (self.stick.position[1] - self.object.position[1])**2
        dist_2 = np.sqrt(dist_2a + dist_2b)

        multiplier = 1-(dist_1 - dist_2)/dist_1

        reward += self.activate_trigger() * multiplier
        for i in range(10):
            self.world.Step(TIME_STEP, 10, 10)
        self.update_screen()
        total_state = get_game_screen(screen)

        if self.timer >= 1000:
            self.destroy_joint()
            return total_state, 0, True, 'grab'
        compare_height = 60
        heights = np.linspace(60,self.max_height-10,len(self.lift_reward))
        for i, combo in enumerate(zip(self.lift_reward, heights)):
            l_reward, height = combo
            if self.object.position[1] >= height:
                reward += l_reward
                self.lift_reward[i] = 0
        if self.object.position[1] < self.max_height:
            return total_state, reward, False, 'grab'
        else:

            self.destroy_joint()
            return total_state, reward, True, 'grab'

    def update_screen(self):
        screen.fill((70, 70, 70, 70))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    sys.exit()
        for iterator, body in enumerate(self.body_list):
            for fixture in body.fixtures:
                # The fixture holds information like density and friction,g
                # and also the shape.
                shape = fixture.shape

                # Naively assume that this is a polygon shape. (not good normally!)
                # We take the body's transform and multiply it with each
                # vertex, and then convert from meters to pixels with the scale
                # factor.
                vertices = [(body.transform * v) * PPM for v in shape.vertices]

                # But wait! It's upside-down! Pygame and Box2D orient their
                # axes in different ways. Box2D is just like how you learned
                # in high school, with positive x and y directions going
                # right and up. Pygame, on the other hand, increases in the
                # right and downward directions. This means we must flip
                # the y components.
                vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]

                pygame.draw.polygon(screen, self.colour_list[iterator], vertices)
                pygame.draw.polygon(screen, (0, 0, 0, 0), vertices, 1)
        self.world.Step(TIME_STEP, 10, 10)
        plt.pause(0.01)
        # Flip the screen and try to keep at the target FPS
        pygame.display.flip()


import random
import torchvision.transforms as T
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import pickle
from collections import deque

import math
from torch.nn import _reduction as _Reduction
import itertools
import matplotlib.pyplot as plt
#from tensorboardcolab import TensorBoardColab

from torch.nn.init import normal_


class ExplorationExploitationScheduler(object):
    """Determines an action according to an epsilon greedy strategy with annealing epsilon"""
    def __init__(self, eps_initial=1, eps_final=0.1, eps_final_frame=0.01,
                 eps_evaluation=0.0, eps_annealing_frames=1500000,
                 replay_memory_start_size=50000, max_frames=10000000):
        """
        Args:
            DQN: A DQN object
            n_actions: Integer, number of possible actions
            eps_initial: Float, Exploration probability for the first
                replay_memory_start_size frames
            eps_final: Float, Exploration probability after
                replay_memory_start_size + eps_annealing_frames frames
            eps_final_frame: Float, Exploration probability after max_frames frames
            eps_evaluation: Float, Exploration probability during evaluation
            eps_annealing_frames: Int, Number of frames over which the
                exploration probabilty is annealed from eps_initial to eps_final
            replay_memory_start_size: Integer, Number of frames during
                which the agent only explores
            max_frames: Integer, Total number of frames shown to the agent
        """

        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_frame = eps_final_frame
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_frames = eps_annealing_frames
        self.replay_memory_start_size = replay_memory_start_size
        self.max_frames = max_frames

        # Slopes and intercepts for exploration decrease
        self.slope = -(self.eps_initial - self.eps_final)/self.eps_annealing_frames
        self.intercept = self.eps_initial - self.slope*self.replay_memory_start_size
        self.slope_2 = -(self.eps_final - self.eps_final_frame)/(self.max_frames - self.eps_annealing_frames - self.replay_memory_start_size)
        self.intercept_2 = self.eps_final_frame - self.slope_2*self.max_frames

    def get_eps(self,  frame_number, evaluation=False):
        """
        Args:
            session: A tensorflow session object
            frame_number: Integer, number of the current frame
            state: A (84, 84, 4) sequence of frames of an Atari game in grayscale
            evaluation: A boolean saying whether the agent is being evaluated
        Returns:
            An integer between 0 and n_actions - 1 determining the action the agent perfoms next
        """
        if evaluation:
            eps = self.eps_evaluation
        elif frame_number < self.replay_memory_start_size:
            eps = self.eps_initial
        elif frame_number >= self.replay_memory_start_size and frame_number < self.replay_memory_start_size + self.eps_annealing_frames:
            eps = self.slope*frame_number + self.intercept
        elif frame_number >= self.replay_memory_start_size + self.eps_annealing_frames:
            eps = self.slope_2*frame_number + self.intercept_2

        return eps

class BasicBuffer(object):
    """Replay Memory that stores the last size=1,000,000 transitions"""
    def __init__(self, max_size=1000000, frame_height=84, frame_width=84,
                 agent_history_length=4, batch_size=32):
        """
        Args:
            size: Integer, Number of stored transitions

            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
            agent_history_length: Integer, Number of frames stacked together to create a state
            batch_size: Integer, Number if transitions returned in a minibatch
        """
        self.size = max_size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0
        self.current = 0
        self.length = 0

        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)

        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = np.empty((self.batch_size, self.agent_history_length,
                                self.frame_height, self.frame_width), dtype=np.uint8)
        self.new_states = np.empty((self.batch_size, self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.uint8)
        self.indices = np.empty(self.batch_size, dtype=np.int32)

    def push(self, frame, action, reward,  terminal):
        """
        Args:
            action: An integer between 0 and env.action_space.n - 1
                determining the action the agent perfomed
            frame: A (84, 84, 1) frame of an Atari game in grayscale
            reward: A float determining the reward the agent received for performing an action
            terminal: A bool stating whether the episode terminated
        """

        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError('Dimension of frame is wrong!')
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size
        if self.length < self.size:
          self.length += 1

    def _get_state(self, index):
        if self.count is 0:
            raise ValueError("The replay memory is empty!")
        if index < self.agent_history_length - 1:
            raise ValueError("Index must be min 3")
        return self.frames[index-self.agent_history_length+1:index+1, ...]

    def _get_valid_indices(self):
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue
                if self.terminal_flags[index - self.agent_history_length:index].any():
                    continue
                break
            self.indices[i] = index

    def sample(self, batch_size):
        """
        Returns a minibatch of self.batch_size = 32 transitions
        """
        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')

        self._get_valid_indices()

        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)
        return self.states, self.actions[self.indices], self.rewards[self.indices], self.new_states, self.terminal_flags[self.indices]

    def __len__(self):
        return self.length




class StateHolder:

    def __init__(self, number_screens=4):
        self.first_action = True
        self.number_screens = number_screens

    def push(self, screen):
        if self.first_action:
            screen = screen[np.newaxis, ...]
            self.state = screen
            for number in range(self.number_screens-1):
                self.state = np.concatenate((self.state, screen), axis=0)
            self.first_action = False
        else:
            screen = screen[np.newaxis, ...]
            self.state = np.concatenate((self.state, screen), axis=0)[1:]

    def get(self):
        return self.state

    def reset(self):
        self.first_action = True
        self.state = torch.ByteTensor(1, 84, 84).to(device)

class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.constant_(self.conv1.bias, 0)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.constant_(self.conv2.bias, 0)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        nn.init.constant_(self.conv3.bias, 0)

        self.conv4 = nn.Conv2d(64, 1024, kernel_size=7, stride=1)
        nn.init.kaiming_normal_(self.conv4.weight, nonlinearity='relu')
        nn.init.constant_(self.conv4.bias, 0)
        # add comment
        self.fc_value = nn.Linear(512, 1)
        nn.init.kaiming_normal_(self.fc_value.weight, nonlinearity='relu')
        self.fc_advantage = nn.Linear(512, n_actions)
        nn.init.kaiming_normal_(self.fc_advantage.weight, nonlinearity='relu')

    def forward(self, x):
        x = x / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # add comment
        x_value = x[:,:512,:,:].view(-1,512)
        x_advantage = x[:,512:,:,:].view(-1,512)
        x_value = self.fc_value(x_value)
        x_advantage = self.fc_advantage(x_advantage)
        # add comment
        q_value = x_value + x_advantage.sub(torch.mean(x_advantage, 1)[:, None])
        return q_value



class DoubleDQNAgent:

    def __init__(self, learning_rate=0.00001, gamma=0.99, buffer_size=200000):
        self.frame = 0
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.replay_buffer = BasicBuffer(max_size=buffer_size)
        self.epsilon = 0.9999

        self.device = torch.device("cuda")
        init_state = env.reset()
        #init_state = np.moveaxis(init_state, 2, 0)
        init_state = correct_state(init_state)

        init_state = np.asarray([init_state, init_state, init_state, init_state])


        self.model1 = DuelingDQN(init_state.shape, len(env.action_space)).to(self.device)
        self.model2 = DuelingDQN(init_state.shape, len(env.action_space)).to(self.device)
        self.model2.eval()

        self.optimizer1 = torch.optim.Adam(self.model1.parameters(), lr = learning_rate)

        self.loss = nn.SmoothL1Loss()

    def get_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:

            return env.sample()
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals = self.model1(state)
        action = np.argmax(qvals.cpu().detach().numpy())

        return action

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        actions = actions.view(actions.size(0), 1)
        dones = dones.view(dones.size(0), 1)
        rewards = rewards.view(rewards.size(0), 1)
        state_action_values = self.model1(states).gather(1, actions)


        next_state_values = torch.zeros(batch_size, device=self.device)
        next_state_action = self.model1(next_states).detach().max(1)[1].view(-1,1)
        next_state_values = self.model2(next_states).detach().gather(1, next_state_action).view(-1).unsqueeze(1)


        expected_state_action_values = (next_state_values * self.gamma * (1-dones)) + rewards


        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        '''
        curr_Q1 = self.model1.forward(states).gather(1, actions)
        curr_Q2 = self.model2.forward(states).gather(1, actions)

        next_Q1 = self.model1.forward(next_states)
        next_Q2 = self.model2.forward(next_states)
        max_next_Q, index = torch.max(next_Q1, 1)
        next_Q = next_Q2.gather(1, index.view(-1, 1))



        expected_Q = rewards + self.gamma * next_Q * (1-dones)

        loss1 = F.smooth_l1_loss(curr_Q1, expected_Q.detach())

        loss2 = mse_loss(curr_Q2, expected_Q.detach())
        '''
        return loss

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss1= self.compute_loss(batch)
        del batch
        self.optimizer1.zero_grad()
        loss1.backward()
        for param in self.model1.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer1.step()
        '''
        self.optimizer2.zero_grad()
        loss2.backward()
        self.optimizer2.step()
        '''
        return loss1


def correct_state(state):
    normalize = T.Compose(
        [
            T.ToPILImage(),
            T.Grayscale(1),
            T.Resize((84, 84), Image.CUBIC)

        ]
    )

    state = normalize(state)
    return np.array(state, dtype=np.uint8)

directories = ["models/base_model/deepQ_DDQN_p.pickle","models/retrain_model/deepQ_DDQN_p.pickle"]
save_spots = ["gifs/pre_original/","gifs/post_original/"]
for directory, save_spot in zip(directories,save_spots):
    curr_agent = 'grab'
    max_frames = 30000000
    batch_size = 32
    episode_rewards = []
    asp_dict = {}
    env = Environment()
    agent_grab = DoubleDQNAgent()
    episode_reward = 0
    eval_rewards = []
    losses = []
    exp_eps_sched = ExplorationExploitationScheduler()
    start = 0
    frame_subtract = 0
    #tb = TensorBoardColab()
    a = False

    with open(directory, 'rb') as learner:
        agent_grab = pickle.load(learner)
        print('o')
    start = len(episode_rewards) * 1000
    state = env.reset()
    state1 = correct_state(state)
    state2 = state1.copy()
    state3 = state2.copy()
    state4 = state3.copy()
    state = np.asarray([state4, state3, state2, state1])
    print('Successfully Loaded Previous Model')
    episode_rewards = []
    im_num = 0
    init_state = state.copy()
    num_act = 0
    eval_bool = False
    im_list = []
    agent_grab.epsilon = 0
    env.reset()
    print('=' * 10)
    print('Start')
    print('=' * 10)
    while len(episode_rewards) <= 20:
        action = agent_grab.get_action(state)

        state_out, reward, done, info = env.step(action)
        state4 = state3.copy()
        state3 = state2.copy()
        state2 = state1.copy()
        state1 = correct_state(state_out)
        next_state = np.asarray([state4, state3, state2, state1])
        im_list.append(Image.fromarray(np.uint8((state1)*255)))

        episode_reward += reward

        if done:

            im_list[0].save(f'{save_spot}out{im_num}.gif', save_all=True, append_images=im_list[1:], duration = 10)
            im_num += 1
            im_list = []
            curr_agent = 'grab'
            episode_rewards.append(episode_reward)
            if len(episode_rewards) % 1 == 0:
                print('Train Episode Reward: '+ str(np.mean(episode_rewards)))
                print('Current Episode Reward: ' + str(episode_reward))
                print('Episode: ' + str(len(episode_rewards)) + '    Epsilon: ' + str(agent_grab.epsilon))
                if eval_bool:
                    tb.save_value('Train Rewards', 'train_rewards', frame, np.mean(episode_rewards[-100:]))

            episode_reward = 0

            state = env.reset()
            state1 = correct_state(state)
            state2 = state1.copy()
            state3 = state2.copy()
            state4 = state3.copy()
            state = np.asarray([state4, state3, state2, state1])

        else:

            state = next_state.copy()
