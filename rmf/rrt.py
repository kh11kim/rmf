import numpy as np
from .elements import *
from .kin_graph import *
from pybullet_suite import *

class Tree:
    def __init__(self, root: Config):
        root.index = 0
        self.root = root
        self.data = [root]
        self.parent = {0:-1}
        self.num = 1

    def add_node(self, node: Config, parent: Config):
        assert node.index != parent.index
        node.index = self.num
        self.parent[node.index] = parent.index
        self.data.append(node)
        self.num += 1
    
    def nearest(self, node: Config):
        distances = []
        for node_tree in self.data:
            d = np.linalg.norm(node_tree.q - node.q)
            distances.append(d)
        idx = np.argmin(distances)
        return self.data[idx]
    
    def backtrack(self, node: Config):
        path = []
        node_curr = node
        while True:
            path.append(node_curr)
            parent_index = self.parent[node_curr.index]
            if parent_index == -1:
                break
            node_curr = self.data[parent_index]
        return path[::-1]

class BiRRT:
    def __init__(
        self,
        eps: float = 0.2,
        p_goal: float = 0.2,
        max_iter: int = 100,
        q_delta_max: float = 0.1,
        DLS_damping: float = 0.1,
    ):
        self.eps = eps
        self.p_goal = p_goal
        self.max_iter = max_iter
        self.q_delta_max = q_delta_max
        self.DLS_damping = DLS_damping

        self.is_goal = lambda node: self.distance(node, self.goal) < self.eps
    
    def get_random_node(self):
        lower = self.kingraph.robot.arm_lower_limit
        upper = self.kingraph.robot.arm_lower_limit
        q = np.random.uniform(low=lower, high=upper)
        return Config(q)

    def plan(self, start: Config, goal: Config, kingraph: KinGraph):
        self.init(start, goal, kingraph)
        
        tree_a = self.tree_start
        tree_b = self.tree_goal
        for i in range(self.max_iter):
            node_rand = self.get_random_node()
            
            if not self.extend(tree_a, node_rand) == "trapped":
                if self.connect(tree_b, self._node_new) == "reached":
                    return self.get_path()
            (tree_a, tree_b) = (tree_b, tree_a)
        return []

    def init(self, start: Config, goal: Config, kingraph: KinGraph):
        self.start = start
        self.goal = goal
        self.kingraph = kingraph
        self.tree_start = Tree(start)
        self.tree_goal = Tree(goal)

    def connect(self, tree, node):
        result = "advanced"
        while result == "advanced":
            result = self.extend(tree, node)
        return result

    def distance(self, node1:Config, node2:Config):
        return np.linalg.norm(node1.q - node2.q)

    def extend(self, tree: Tree, node_rand: Config):
        node_near = tree.nearest(node_rand)
        node_new = self.control(node_near, node_rand)
        if node_new is not None:
            tree.add_node(node_new, node_near)
            if not self.distance(node_rand, node_new) > self.eps:
                self.last_node = node_new
                return "reached"
            else:
                self._node_new = node_new #save this to "connect"
            return "advanced"
        return "trapped"
    
    def limit_step_size(self, q_delta: np.ndarray, q_delta_max: Optional[np.ndarray]=None):
        if q_delta_max is None:
            q_delta_max = self.q_delta_max
        mag = np.linalg.norm(q_delta, np.inf)
        if mag > q_delta_max:
            q_delta = q_delta / mag * q_delta_max
        return q_delta

    def control(self, node_near:Config, node_rand:Config):
        mag = self.distance(node_near, node_rand)
        if mag <= self.eps:
            node_new = node_rand.copy()
            node_new.index = -1
        else:
            q_err = node_rand.q - node_near.q
            q_delta = self.limit_step_size(q_err, self.q_delta_max)
            q_new = node_near.q + q_delta
            node_new = Config(q_new)

        if not self.kingraph.is_collision(node_new):
            return node_new
        else:
            return None
    
    def get_path(self):
        node_tree_start = self.tree_start.nearest(self.last_node)
        node_tree_goal = self.tree_goal.nearest(self.last_node)
        path_from_start = self.tree_start.backtrack(node_tree_start)
        path_from_goal = self.tree_goal.backtrack(node_tree_goal)
        
        return [*path_from_start, *path_from_goal[::-1]]


def distance_ts(
    T1: Pose, 
    T2: Pose, 
    rot_weight=0.5
) -> float:
    linear = np.linalg.norm(T1.trans - T2.trans)
    qtn1, qtn2 = T1.rot.as_quat(), T2.rot.as_quat()
    if qtn1 @ qtn2 < 0:
        qtn2 = -qtn2
    angular = np.arccos(np.clip(qtn1 @ qtn2, -1, 1))
    return linear + rot_weight * angular


class TSRRT:
    def __init__(
        self,
        eps: float = 0.2,
        p_goal: float = 0.2,
        max_iter: int = 100,
        q_delta_max: float = 0.1,
        DLS_damping: float = 0.1,
    ):
        self.eps = eps
        self.p_goal = p_goal
        self.max_iter = max_iter
        self.q_delta_max = q_delta_max
        self.DLS_damping = DLS_damping
        
    def check_mode_switch(
        self, 
        node_final: Config, 
        T_target: Pose,
        kingraph: KinGraph
    ):
        EPS = 0.01
        result = False
        traj = []
        with kingraph.robot.no_set_joint():
            config = node_final.copy()
            for _ in range(10):
                q = self.steer(
                    q=config.q,
                    jac=kingraph.robot.get_jacobian(config.q),
                    curr_pose=kingraph.robot.forward_kinematics(config.q),
                    target_pose=T_target,
                    q_delta_max=0.05
                )
                config_new = Config(q)
                kingraph.robot.set_joint_angles(config_new.q)
                config.T = kingraph.robot.get_ee_pose()
                kingraph.assign()
                if not kingraph.is_collision(config):
                    traj.append(config_new)
                    if distance_ts(config.T, T_target) < EPS:
                        result = traj
                        break
                    config = config_new
                else:
                    break
                
        kingraph.assign()
        return result
    
    def steer(
        self,
        q: np.ndarray,
        jac: np.ndarray,
        curr_pose: Pose,
        target_pose: Pose,
        q_delta_max: Optional[float] = None
    ) -> np.ndarray:
        if q_delta_max is None:
            q_delta_max = self.q_delta_max
        pos_err = target_pose.trans - curr_pose.trans
        orn_err = orn_error(target_pose.rot.as_quat(), curr_pose.rot.as_quat())
        err = np.hstack([pos_err, orn_err*2])
        lmbda = np.eye(6) * self.DLS_damping ** 2
        jac_pinv = jac.T @ np.linalg.inv(jac @ jac.T + lmbda)
        q_delta = self.limit_step_size(jac_pinv @ err, q_delta_max)
        return q_delta + q

    def limit_step_size(self, q_delta: np.ndarray, q_delta_max: Optional[np.ndarray]=None):
        if q_delta_max is None:
            q_delta_max = self.q_delta_max
        mag = np.linalg.norm(q_delta, np.inf)
        if mag > q_delta_max:
            q_delta = q_delta / mag * q_delta_max
        return q_delta

        

class ModeForest:
    def __init__(
        self,
        eps: float = 0.1,
        p_goal: float = 0.5,
        max_iter: int = 100,
        q_delta_max: float = 0.1,
        DLS_damping: float = 0.1,
    ):
        self.eps = eps
        self.p_goal = p_goal
        self.max_iter = max_iter
        self.q_delta_max = q_delta_max
        self.DLS_damping = DLS_damping
    
    def distance(self, node1:Config, node2:Config):
        return np.linalg.norm(node1.q - node2.q)
    
    def grow_tree(
        self,
        goal_config: Config,
        tree: Tree,
        kingraph: KinGraph,
    ):
        self.kingraph = kingraph
        
        for _ in range(self.max_iter):
            p = np.random.random()
            if p < self.p_goal:
                node_rand = goal_config.copy()
            else:
                node_rand = Config(kingraph.robot.get_random_arm_angles())
            if self.extend(tree, node_rand) == "reached":
                return self.last_node
        return None
    
    def grow_bi_tree(self, tree_start: Config, tree_goal: Tree, kingraph: KinGraph):
        self.kingraph = kingraph
        tree_a = tree_start
        tree_b = tree_goal
        for i in range(self.max_iter):
            node_rand = Config(kingraph.robot.get_random_arm_angles())
            
            if not self.extend(tree_a, node_rand) == "trapped":
                if self.connect(tree_b, self._node_new) == "reached":
                    return self.get_path(tree_start, tree_goal)
            (tree_a, tree_b) = (tree_b, tree_a)
        return []
    
    def extend(self, tree: Tree, node_rand: Config):
        node_near = tree.nearest(node_rand)
        node_new = self.control(node_near, node_rand)
        if node_new is not None:
            tree.add_node(node_new, node_near)
            if self.distance(node_rand, node_new) < self.eps:
                self.last_node = node_new
                return "reached"
            else:
                self._node_new = node_new #save this to "connect"
                return "advanced"
        return "trapped"
    
    def connect(self, tree, node):
        result = "advanced"
        while result == "advanced":
            result = self.extend(tree, node)
        return result

    def control(self, node_near:Config, node_rand:Config):
        mag = self.distance(node_near, node_rand)
        if mag <= self.eps:
            node_new = node_rand.copy()
            node_new.index = -1
        else:
            q_err = node_rand.q - node_near.q
            q_delta = self.limit_step_size(q_err, self.q_delta_max)
            q_new = node_near.q + q_delta
            node_new = Config(q_new)

        if not self.kingraph.is_collision(node_new):
            return node_new
        else:
            return None
    
    def get_path(self, tree_start: Tree, tree_goal: Tree):
        node_tree_start = tree_start.nearest(self.last_node)
        node_tree_goal = tree_goal.nearest(self.last_node)
        path_from_start = tree_start.backtrack(node_tree_start)
        path_from_goal = tree_goal.backtrack(node_tree_goal)
        
        return [*path_from_start, *path_from_goal[::-1]]

    def limit_step_size(self, q_delta: np.ndarray, q_delta_max: Optional[np.ndarray]=None):
        if q_delta_max is None:
            q_delta_max = self.q_delta_max
        mag = np.linalg.norm(q_delta, np.inf)
        if mag > q_delta_max:
            q_delta = q_delta / mag * q_delta_max
        return q_delta
    
    # def check_mode_switch(
    #     self, 
    #     node_final: Config, 
    #     T_target: Transform,
    #     kingraph: KinGraphReal
    # ):
    #     EPS = 0.01
    #     result = False
    #     traj = []
    #     with kingraph.robot.no_set_joint():
    #         config = node_final.copy()
    #         for _ in range(10):
    #             q = self.steer(
    #                 q=config.q,
    #                 jac=kingraph.robot.get_jacobian(config.q),
    #                 curr_pose=kingraph.robot.forward_kinematics(config.q),
    #                 target_pose=T_target,
    #                 q_delta_max=0.05
    #             )
    #             config_new = Config(q)
    #             kingraph.robot.set_arm_angles(config_new.q)
    #             config.T = kingraph.robot.get_ee_pose()
    #             kingraph.assign()
    #             if not kingraph.is_collision(config):
    #                 traj.append(config_new)
    #                 if distance_ts(config.T, T_target) < EPS:
    #                     result = traj
    #                     break
    #                 config = config_new
    #             else:
    #                 break
                
    #     kingraph.assign()
    #     return result
    
    # def steer(
    #     self,
    #     q: np.ndarray,
    #     jac: np.ndarray,
    #     curr_pose: Transform,
    #     target_pose: Transform,
    #     q_delta_max: Optional[float] = None
    # ) -> np.ndarray:
    #     if q_delta_max is None:
    #         q_delta_max = self.q_delta_max
    #     pos_err = target_pose.translation - curr_pose.translation
    #     orn_err = orn_error(target_pose.rotation.as_quat(), curr_pose.rotation.as_quat())
    #     err = np.hstack([pos_err, orn_err*2])
    #     lmbda = np.eye(6) * self.DLS_damping ** 2
    #     jac_pinv = jac.T @ np.linalg.inv(jac @ jac.T + lmbda)
    #     q_delta = self.limit_step_size(jac_pinv @ err, q_delta_max)
    #     return q_delta + q