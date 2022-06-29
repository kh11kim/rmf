from .elements import *
from .kin_graph import *
from .rrt import *
from pybullet_suite import *

class Mode1:
    def __init__(self, index: int, kingraph: KinGraph, config: Config):
        self.index = index
        self.kingraph = kingraph
        self.config = config
        self.parent: Optional[Mode1] = None
        self.traj_parent = None
        self.traj = None

class Mode:
    def __init__(self, stage: int, tree: Tree, kingraph: KinGraph, parent_last_node=None, rev=False):
        self.stage = stage
        self.kingraph = kingraph
        self.tree = tree
        self.parent_last_node = parent_last_node
        # self.traj_switch: Optional[List[Config]] = traj_switch
        self.rev = rev
        self.parent_mode: Optional[Mode] = None
        self.traj: Optional[Mode] = None

class ModeTree1:
    def __init__(self, root: KinRelation, total_stages: int):
        self.modes = {step:[] for step in range(total_stages)}
        self.modes[0].append(root)
    
    def sample_mode(self, index: int, all=False)->Optional[KinRelation]:
        if all:
            modes = []
            for key in self.modes:
                modes += self.modes[key]
            return np.random.choice(modes)
        modes = self.modes[index]
        if modes:
            return np.random.choice(modes)
        return None
    
    def add_mode(self, mode: KinRelation, parent: KinRelation, traj:List[Config]):
        mode.traj_parent = traj
        mode.parent = parent
        self.modes[parent.index + 1].append(mode)
    
    def backtrack(self, last_mode: KinRelation)->List[KinRelation]:
        modes = []
        mode = last_mode
        while True:
            modes.append(mode)
            if not mode.parent:
                break
            mode.parent.traj = mode.traj_parent
            mode = mode.parent
        return modes[::-1]

class ModeTree:
    def __init__(self, root: Mode, total_stages: int):
        self.modes: Dict[int, List[Mode]] = {step:[] for step in range(total_stages)}
        self.modes[root.stage].append(root)
    
    def sample_mode(self, stage: int)->Optional[Mode]:
        modes = self.modes[stage]
        if modes:
            return np.random.choice(modes)
        return None
    
    # def get_mode_by_edge(self, stage: int, edge: KinEdge):
    #     modes = self.modes[stage]
    #     for mode in modes:
    #         if mode.kingraph.kin_edge[]

    def get_grow_target(self, stage: int, action: Action):
        modes = self.modes[stage]
        edges = []
        if action.name == "pick":
            edges = [mode.kingraph.kin_edge[action.obj_name] for mode in modes]
        elif action.name == "place":
            edges = [mode.kingraph.kin_edge[action.obj_name] for mode in modes]
        elif action.name == "move":
            edges = []
        return modes, edges

    def add_mode(self, mode: Mode, parent: Mode, parent_last_node:Optional[Config]):
        mode.parent_mode = parent
        mode.parent_last_node = parent_last_node
        self.modes[mode.stage].append(mode)
    
    def backtrack(self, last_mode: Mode)->List[Mode]:
        modes = []
        mode = last_mode
        while True:
            modes.append(mode)
            if not mode.parent_mode:
                break
            mode = mode.parent_mode
        return modes[::-1]