import sys
sys.path.extend("../")
sys.path.append("../")

import networkx as nx
from io import BytesIO
import matplotlib.image as mpimg
from gym import logger
import numpy as np

from finite_mdp.mdp import DeterministicMDP


class MDPViewer(object):
    def __init__(self, mdp):
        self.mdp = mdp
        self.graph = None
        self.pydot_graph = None

    def build_graph(self):
        if not isinstance(self.mdp, DeterministicMDP):
            raise NotImplementedError()
        self.graph = nx.MultiDiGraph()
        action_labels = 'abcdefghijklmnopqrstuvwxyz'
        for state in range(self.mdp.transition.shape[0]):
            if state == self.mdp.state:
                self.graph.add_node(state, color="green")
            else:
                self.graph.add_node(state, color="black")
        for from_state in range(self.mdp.transition.shape[0]):
            for action in range(self.mdp.transition.shape[1]):
                if not self.mdp.terminal[from_state]:
                    self.graph.add_edge(from_state, self.mdp.transition[from_state, action],
                                        label="{}, {}".format(action_labels[action],
                                                             self.mdp.reward[from_state, action]))
        self.pydot_graph = nx.drawing.nx_pydot.to_pydot(self.graph)
        self.pydot_graph.set("size", "50")
        self.pydot_graph.set("resolution", "100")
        self.pydot_graph.set("autosize", "false")

    def get_image(self):
        try:
            self.build_graph()
            # Render image as bytes
            sio = BytesIO()
            sio.write(self.pydot_graph.create_png(prog='dot'))
            sio.seek(0)
            # Convert to 255-valued RGB
            img = np.round(255*mpimg.imread(sio)).astype(np.uint8)
            # Size must be divisible by two
            shape = [(s // 2) * 2 for s in list(img.shape)]
            img = img[:shape[0], :shape[1], :shape[2]]
            return img
        except Exception as e:
            logger.error("Couldn't render finite-mdp-env: " + str(e))
            return None

