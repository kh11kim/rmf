import time
import numpy as np
from pybullet_suite.pybullet_suite import *
from rmf import *
from rmf.kin_graph import KinGraph, KinRelation
from rmf.data_structure import *
from rmf.plan import *
from rmf.mode_tree import *

class Logger:
    def __init__(self):
        self.string = ""
        self.filepath = "log.txt"
        self.f = open(self.filepath, "w")
    
    def __del__(self):
        self.f.close()

    def add(self, log: str):
        self.string += log
    
    def reset(self):
        self.string = ""
    
    def print(self):
        self.f.writelines(self.string+ "\n")
        #print(self.string)

MAX_ITER = 100
P_GOAL = 0.5
logger = Logger()

def filter_top_grasp_only(grasp_set):
    fake_world = BulletWorld(gui=False)
    dice = fake_world.load_urdf(
        "dice", DICE_URDF
    )
    hand = Gripper(fake_world)
    grasp_set = np.load(DICE_GRASP_SET, allow_pickle=True)["grasp_set"]
    top_grasps = []
    for grasp_tf in grasp_set:
        hand.reset(grasp_tf)
        axis = np.array([0,0,-1])
        mat = grasp_tf.rot.as_matrix()
        z_axis = mat[:,-1]
        angle = np.arccos(axis@z_axis)
        if np.abs(angle) < np.pi/4:
            top_grasps.append(grasp_tf)

    for grasp_tf in top_grasps:
        hand.reset(grasp_tf)
    return top_grasps

def set_world(gui:bool):
    distance_between_table = 0.4
    table_height = 0.2
    world = BulletWorld(gui=gui)
    sm = BulletSceneMaker(world)
    sm.create_plane(z_offset=-0.4)
    sm.create_table("ground", 2, 2, 0.4)

    # robot
    robot: Panda = world.load_robot("robot", robot_class=Panda)

    # fixed
    fixed = {}
    for table_name, sig in zip(["table1", "table2"], [+1, -1]):
        table = sm.create_table(
            table_name, 0.3, 0.3, 0.2, 
            x_offset=0.4, y_offset=sig*distance_between_table/2, z_offset=table_height)
        fixed[table_name] = Fixed.from_body(table, table_name)
    
    #movable
    dice_pose = Pose(trans=[0.4, -0.2, 0.2 + 0.03/2])
    dice = world.load_urdf(
        "dice", DICE_URDF, dice_pose
    )
    dice = Movable.from_body(dice, "dice")
    dice_grasp_set = np.load(DICE_GRASP_SET, allow_pickle=True)["grasp_set"]
    if False:
        dice_placement_axis_set = [np.array([0,0,1])]
        dice_grasp_set = filter_top_grasp_only(dice_grasp_set)
    else:
        dice_placement_axis_set = [
            np.array([1,0,0]), 
            np.array([-1,0,0]),
            np.array([0,1,0]), np.array([0,-1,0]),
            np.array([0,0,1]), np.array([0,0,-1]),
        ]
    dice.set_grasp_set(dice_grasp_set)
    dice.set_placement_axis_set(dice_placement_axis_set)

    world.set_gravity([0,0,-9.81])
    world.wait_for_rest()
    # get placement of dice
    dice_placement = Placement.from_bodies(dice, fixed['table2'])

    kingraph = KinGraph(world, sm)
    kingraph.add_robot(robot)
    kingraph.add_object("table1", fixed["table1"])
    kingraph.add_object("table2", fixed["table2"])
    kingraph.add_object("dice", dice, "table2", dice_placement)
    
    config = Config.from_robot(kingraph.robot)
    return config, kingraph

def get_goal_targets(config_init: Config, kingraph: KinGraph):
    dice = kingraph.objects["dice"]
    table: Fixed = kingraph.objects["table2"]

    xyz = table.sample_point()
    target_placement_axis = np.array([0,0,-1])
    placement = Placement.from_body_point_and_placement_axes(
        dice, table,
        xyz, target_placement_axis
    )
    dice_pose = table.get_base_pose() * placement.tf.inverse()
    dice.set_base_pose(dice_pose)

    target1 = Target("dice", placement=placement)
    target2 = Target("robot", config=Config(kingraph.robot.arm_central))
    targets = [target2]

    plan_skeleton = [
        Action("pick", "dice"),
        Action("place", "dice", "table1"),
        Action("pick", "dice"),
        Action("place", "dice", "table2"),
        Action("move_free", "robot")
    ]
    plan = Plan()
    plan.parse(plan_skeleton, targets, config_init, kingraph)
    return plan

def sample_grasp(obj: Movable):
    grasp = obj.sample_grasp()
    if grasp:
        return grasp
    return None

def sample_placement(movable, placeable, kingraph: KinGraph):
    return kingraph.sample_placement(movable.name, placeable.name)

def sample_kin(grasp: Grasp, placement: Placement, kingraph: KinGraph, pre=None):
    if (grasp is None) | (placement is None):
        return None
    movable: Movable = kingraph.objects[placement.movable_name]
    placeable: Fixed = kingraph.objects[placement.placeable_name]
    obj_pose = placeable.get_base_pose() * placement.tf.inverse()
    T_target = obj_pose * grasp.tf

    if pre == "grasp":
        T_target = grasp.get_pre_pose(T_target)
    elif pre == "placement":
        T_target = placement.get_pre_pose(T_target)
    kingraph.robot.set_joint_angles(kingraph.robot.arm_central)
    q = kingraph.robot.inverse_kinematics(pose=T_target)
    if q is not None:
        config = Config(q)
        if not kingraph.is_collision(config):
            if not pre:
                logger.add(f"sample_kin : success ")
            return config
    if not pre:
        logger.add(f"sample_kin : failed ")
    return None

def sample_traj(tree: Tree, config_goal: Config, kingraph: KinGraph):
    if config_goal is not None:
        mf = ModeForest(max_iter=MAX_ITER)
        last_node = mf.grow_tree(config_goal, tree, kingraph)
        if last_node:
            logger.add("sample_traj: success")
            return last_node
    logger.add("sample_traj: failed")
    return None

def pick_move(mode: Mode, action: Action, grow_target: List[Grasp]):
    mode.kingraph.assign(mode.tree.root)
    movable = mode.kingraph.objects[action.obj_name]
    placement = mode.kingraph.kin_edge[action.obj_name]
    if (np.random.rand() < P_GOAL) & (len(grow_target) != 0):
        grasp = np.random.choice(grow_target)
    else:
        grasp = sample_grasp(movable)
    config2 = sample_kin(grasp, placement, mode.kingraph)
    if config2:
        config2 = sample_kin(grasp, placement, mode.kingraph, pre="grasp")
    last_node = sample_traj(mode.tree, config2, mode.kingraph)
    if last_node:
        mode.kingraph.robot.set_joint_angles(config2.q)
        T_target = movable.get_base_pose() * grasp.tf
        mp = TSRRT()
        traj_switch = mp.check_mode_switch(last_node, T_target, mode.kingraph)
        if traj_switch:
            
            return grasp, last_node
    
    return None, None

def place_move(
    mode: Mode,
    action: Action,
    grow_target: List[Placement]
):
    mode.kingraph.assign(mode.tree.root)
    movable = mode.kingraph.objects[action.obj_name]
    placeable = mode.kingraph.objects[action.placeable_name]
    grasp = mode.kingraph.kin_edge[movable.name]
    
    if action.target is not None:
        placement = action.target.placement
    else:
        if (np.random.rand() < P_GOAL) & (len(grow_target) != 0):
            placement = np.random.choice(grow_target)
        else:
            xyz = placeable.sample_point()
            axis = movable.sample_placement_axis()
            placement = Placement.from_body_point_and_placement_axes(
                movable, placeable, xyz, axis
            )
    
    config2 = sample_kin(grasp, placement, mode.kingraph)
    #kingraph.sm.view_frame(pose)
    if config2:
        config2 = sample_kin(grasp, placement, mode.kingraph, pre="placement")
    last_node = sample_traj(mode.tree, config2, mode.kingraph)
    if last_node:
        mode.kingraph.robot.set_joint_angles(config2.q)
        T_target = placeable.get_base_pose() * placement.tf.inverse() * grasp.tf
        mp = TSRRT()
        traj_switch = mp.check_mode_switch(last_node, T_target, mode.kingraph)
        if traj_switch:
            return placement, last_node
    return None, None

def free_move(mode: Mode, action: Action):
    mode.kingraph.assign(mode.tree.root)
    target = action.target
    if target is not None:
        config2 = target.config.copy()
    else:
        config2 = Config(mode.kingraph.robot.get_random_arm_angles())
    last_node = sample_traj(mode.tree, config2, mode.kingraph)
    if last_node:
        return last_node
    return None

def grow_tree(mode: Mode1, action: Action, grow_target: List[KinEdge]):
    edge = last_node = None
    if action.name == "pick":
        logger.add("action: pick ")
        edge, last_node = pick_move(mode, action, grow_target)
    elif action.name == "place":
        logger.add("action: place ")
        edge, last_node = place_move(mode, action, grow_target)
    elif action.name == "move_free":
        logger.add("action: move_free ")
        last_node = free_move(mode, action)
    return edge, last_node



def main():
    config_init, kingraph_init = set_world(gui=True)
    plan = get_goal_targets(config_init, kingraph_init)
    config_init = Config.from_robot(kingraph_init.robot)
    config_goal, kingraph_goal = plan.get_random_goal_config()
    kingraph_goal.assign(config_goal)
    kingraph_init.assign(config_init)

    # elapsed_times = []
    # for i in range(10):
    #     tic = time.time()

    m_init = Mode(0, Tree(config_init), kingraph_init)
    m_goal = Mode(0, Tree(config_goal), kingraph_goal, rev=True)
    mt_fwd = ModeTree(m_init, plan.num_milestones)
    mt_rev = ModeTree(m_goal, plan.num_milestones)
    last_mode_traj = None

    while True: #time.time()- tic < 20:
        logger.reset()

        is_rev = np.random.choice([True, False])
        
        p = np.random.random()
        d = 1.2
        a_diffs = [1]
        for i in range(1, plan.num_actions):
            r = i
            r_next = r+1
            a_diff = r_next**d - r**d
            a_diffs.append(a_diff)
        a_diffs = np.array(a_diffs)
        a_diffs = a_diffs/np.linalg.norm(a_diffs)

        for stage, a in enumerate(a_diffs):
            if p < a:
                break
        mt, mt_guide = (mt_fwd, mt_rev) if not is_rev else (mt_rev, mt_fwd)
        
        next_stage = stage + 1
        mode_curr = mt.sample_mode(stage)
        action = plan.actions[(stage, is_rev)]
        if (not mode_curr)|(not action): continue

        dir = "fwd" if not is_rev else "rev"
        mode_curr_idx = stage if not is_rev else plan.num_actions - stage - 1
        mode_next_idx = mode_curr_idx + 1 if not is_rev else mode_curr_idx - 1
        logger.add(f"dir:{dir}, stage:{stage} - (mode{mode_curr_idx}->mode{mode_next_idx}): ")

        guide_stage_index = plan.num_actions - stage - 2
        if not guide_stage_index < 0:
            modes, grow_target = mt_guide.get_grow_target(guide_stage_index, action)
        else:
            grow_target = []
        logger.add(f"guided: {len(grow_target) != 0} ")

        edge, last_node = grow_tree(mode_curr, action, grow_target)
        

        if last_node is None:
            logger.print()
        else:
            logger.print()
            kingraph_new = mode_curr.kingraph.mode_switch(action, edge)
            mode_new = Mode(next_stage, Tree(last_node.copy()), kingraph_new, last_node, rev=is_rev)
            mt.add_mode(mode_new, mode_curr, last_node)
            if edge in grow_target:
                bool_idx = [(e == edge) for e in grow_target]
                idx = np.where(bool_idx)[0][0]
                mode_guide = modes[idx]
                mp = ModeForest()
                last_mode_traj = mp.grow_bi_tree(mode_new.tree, mode_guide.tree, kingraph_new)
                if last_mode_traj:
                    mode_center = mode_new
                    if is_rev == False: # right direction
                        mode_center.traj = last_mode_traj
                        modes_to_front = mt_fwd.backtrack(mode_new)
                        modes_to_back = mt_rev.backtrack(mode_guide)
                    else:
                        mode_center.traj = last_mode_traj[::-1]
                        modes_to_front = mt_fwd.backtrack(mode_guide)
                        modes_to_back = mt_rev.backtrack(mode_new)
                break
            print(f"mode_switch: rev{is_rev},  {stage}->{next_stage}: success")


        # toc = time.time()
        # elapsed_times.append(toc-tic)


    ## debug        
    #front
    for i, mode in enumerate(modes_to_front[:-1]):
        next_mode = modes_to_front[i+1]
        mode.traj = mode.tree.backtrack(next_mode.parent_last_node)
    #backward
    for i, mode in enumerate(modes_to_back[:-1]):
        next_mode = modes_to_back[i+1]
        mode.traj = mode.tree.backtrack(next_mode.parent_last_node)[::-1]
    modes = [*modes_to_front[:-1], mode_center, *modes_to_back[:-1][::-1]]
    
    for mode in modes:
        traj = mode.traj
        for config in traj:
            mode.kingraph.robot.set_joint_angles(config.q)
            mode.kingraph.assign()
            time.sleep(0.1)

    # print(f"elapsed time mean: {np.mean(elapsed_times)}")
    # print(f"elapsed time std: {np.std(elapsed_times)}")    

if __name__ == "__main__":
    main()