import gymnasium.spaces as spaces
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
def adjency_matrix_to_mask(adj):
    mask = np.zeros(adj.shape)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j] > 0:
                mask[i, j] = 1
            else:
                mask[i, j] = 0
    return mask
def plot_graph(graph):
    # plot thief in red, cop in blue
    node_colors = []
    for node in graph.nodes():
        if graph.nodes[node]["agent"] == 1:
            node_colors.append("red")
        elif graph.nodes[node]["agent"] == 2:
            node_colors.append("blue")
        else:
            node_colors.append("green")
    pos = {(i, j): (i, -j) for i, j in graph.nodes()}  # Specify positions for nodes
    plt.figure(figsize=(5, 5))  # Set the size of the plot
    nx.draw(graph, pos, with_labels=True, node_size=500, node_color=node_colors, font_size=10, font_weight='bold')
    plt.title('Grid Graph', fontsize=12)
    plt.show()
def get_node_features(graph ):
    node_features = np.zeros((len(graph.nodes()), 1))
    index = 0
    for node in graph.nodes():
        node_features[index] = graph.nodes[node]["agent"]
        index += 1
    return node_features
def initialize_state(thief_position : tuple, cop_position : tuple, graph: nx.Graph):
    for node in graph.nodes():
        graph.nodes[node]["agent"] = 0

    graph.nodes[thief_position]["agent"] = 1
    graph.nodes[cop_position]["agent"] = 2

    return graph
import networkx as nx

class GridWorld_copsVSthief(gym.Env):
    def __init__(self,WIDTH,HEIGHT):
        grid_graph = nx.grid_2d_graph(WIDTH,HEIGHT)
        self.G = grid_graph
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(WIDTH,HEIGHT,1), dtype=np.float32)

    def reset(self):

        self.thief_position = (np.random.randint(0,self.WIDTH),np.random.randint(0,self.HEIGHT))
        # put the cops at a random position, but not the same as the thief
        self.cops_position = (np.random.randint(0,self.WIDTH),np.random.randint(0,self.HEIGHT))
        while self.cops_position == self.thief_position:
            self.cops_position = (np.random.randint(0,self.WIDTH),np.random.randint(0,self.HEIGHT))


        self.G = initialize_state(self.thief_position,self.cops_position,self.G)

        adj = nx.adjacency_matrix(self.G).todense()
        adj = adj + np.eye(adj.shape[0])
        mask = np.ones(5)
        if self.thief_position[1] == self.HEIGHT - 1:
            mask[0] = 0
        if self.thief_position[0] == self.WIDTH - 1:
            mask[1] = 0
        if self.thief_position[1] == 0:
            mask[2] = 0
        if self.thief_position[0] == 0:
            mask[3] = 0

        self.state = get_node_features(self.G),adjency_matrix_to_mask(adj)
        return self.state,mask

    def step(self,action):
        # 0 : up
        # 1 : right
        # 2 : down
        # 3 : left
        # 4 : stay

        reward = 0

        if action == 0:
            # check if the thief is at the border
            if self.thief_position[1] == self.HEIGHT-1:
                self.thief_position = (self.thief_position[0],self.thief_position[1])

            else:
                self.thief_position = (self.thief_position[0],(self.thief_position[1]+1)%self.HEIGHT)
        elif action == 1:
            if self.thief_position[0] == self.WIDTH-1:
                self.thief_position = (self.thief_position[0],self.thief_position[1])

            else:
                self.thief_position = ((self.thief_position[0]+1)%self.WIDTH,self.thief_position[1])
        elif action == 2:
            # check if the thief is at the border
            if self.thief_position[1] == 0:
                self.thief_position = (self.thief_position[0],self.thief_position[1])


            else:
                self.thief_position = (self.thief_position[0],(self.thief_position[1]-1)%self.HEIGHT)
        elif action == 3:
            # check if the thief is at the border
            if self.thief_position[0] == 0:
                self.thief_position = (self.thief_position[0],self.thief_position[1])
            else:
                self.thief_position = ((self.thief_position[0]-1)%self.WIDTH,self.thief_position[1])
        elif action == 4:
            pass

        # reward is the distance between cops and thief, the closer the better
        reward += -np.sqrt((self.thief_position[0]-self.cops_position[0])**2+(self.thief_position[1]-self.cops_position[1])**2)
        reward -= 1
        #reward += -np.sqrt((self.thief_position[0]-self.cops_position[0])**2+(self.thief_position[1]-self.cops_position[1])**2) / 100
        self.G = initialize_state(self.thief_position,self.cops_position,self.G)

        adj = nx.adjacency_matrix(self.G).todense()
        adj = adj + np.eye(adj.shape[0])

        self.state = get_node_features(self.G),adjency_matrix_to_mask(adj)
        # crate mask for action space.
        # if the thief is at the border, he can't go further

        mask = np.ones(5)
        if self.thief_position[1] == self.HEIGHT - 1:
            mask[0] = 0
        if self.thief_position[0] == self.WIDTH - 1:
            mask[1] = 0
        if self.thief_position[1] == 0:
            mask[2] = 0
        if self.thief_position[0] == 0:
            mask[3] = 0
        info = {'mask':mask}
        done = self.thief_position == self.cops_position

        if done:
            reward += 10
        return self.state,reward, done,None,info

    def render(self):
        plot_graph(self.G)
    def close(self):
        pass
