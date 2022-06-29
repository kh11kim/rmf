import time
import numpy as np
from pybullet_suite.pybullet_suite import *
from rmf import *
from rmf.kin_graph import KinGraph, KinRelation
from rmf.mode_tree import *



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
    if True:
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

    return kingraph

def get_goal_targets(kingraph: KinGraph):
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
    target2 = Target("dice", config=Config(kingraph.robot.arm_central))

    plan_skeleton = [
        Action("pick", "dice"),
        Action("place", "dice", "table1"),
        Action("pick", "dice"),
        Action("place", "dice", "table2", target=target1),
        Action("move", target=target2)
    ]
    return plan_skeleton

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
    # with movable.no_set_pose():
    #     movable.set_base_pose(obj_pose)
    #     print("debug")
    if pre == "grasp":
        T_target = grasp.get_pre_pose(T_target)
    elif pre == "placement":
        T_target = placement.get_pre_pose(T_target)
    kingraph.robot.set_joint_angles(kingraph.robot.arm_central)
    q = kingraph.robot.inverse_kinematics(pose=T_target)
    if q is not None:
        config = Config(q)
        if not kingraph.is_collision(config):
            print(f"sample_kin{pre}: success")
            return config
    print(f"sample_kin{pre}: failed")
    return None

def sample_traj(config_init: Config, config_end: Config, kingraph: KinGraph):
    if config_end is None:
        return None
    mp = BiRRT(max_iter=100)
    traj = mp.plan(config_init, config_end, kingraph)
    if traj:
        print("sample_traj: success")
        return traj
    print("sample_traj: failed")
    return None

def pick_move(mode: Mode1, action: Action):
    mode.kingraph.assign(mode.config)
    movable = mode.kingraph.objects[action.obj_name]
    placement = mode.kingraph.kin_edge[action.obj_name]
    grasp = sample_grasp(movable)
    config2 = sample_kin(grasp, placement, mode.kingraph)
    if config2:
        config2 = sample_kin(grasp, placement, mode.kingraph, pre="grasp")
    traj = sample_traj(mode.config, config2, mode.kingraph)
    if traj:
        mode.kingraph.robot.set_joint_angles(config2.q)
        T_target = movable.get_base_pose() * grasp.tf
        mp = TSRRT()
        traj_switch = mp.check_mode_switch(traj[-1], T_target, mode.kingraph)
        if traj_switch:
            return grasp, traj+traj_switch
    return None, None

def place_move(
    mode: Mode1,
    action: Action,
):
    mode.kingraph.assign(mode.config)
    movable = mode.kingraph.objects[action.obj_name]
    placeable = mode.kingraph.objects[action.placeable_name]
    grasp = mode.kingraph.kin_edge[movable.name]
    
    if action.target is not None:
        placement = action.target.placement
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
    traj = sample_traj(mode.config, config2, mode.kingraph)
    if traj:
        mode.kingraph.robot.set_joint_angles(config2.q)
        T_target = placeable.get_base_pose() * placement.tf.inverse() * grasp.tf
        mp = TSRRT()
        traj_switch = mp.check_mode_switch(traj[-1], T_target, mode.kingraph)
        if traj_switch:
            return placement, traj+traj_switch
    return None, None

def free_move(rel: Mode1, action: Action):
    target = action.target
    if target is not None:
        config2 = target.config
    else:
        config2 = rel.kingraph.robot.get_random_arm_angles()
    traj = sample_traj(rel.config, config2, rel.kingraph)
    if traj:
        return traj
    return None


def main():
    kingraph = set_world(gui=True)
    plan_skeleton = get_goal_targets(kingraph)
    config_init = Config.from_robot(kingraph.robot)

    elapsed_times = []
    for i in range(10):
        tic = time.time()
        m = Mode1(0, kingraph, config_init)
        mt = ModeTree1(m, len(plan_skeleton)+1)
        
        while time.time()- tic < 20:
            n = len(plan_skeleton)
            p = np.random.random()
            area = np.arange(1,2*n,2)
            area = (area/np.sum(area)).cumsum()
            for stage, a in enumerate(area):
                if p < a:
                    break

            #stage = np.random.choice(len(actions))
            mode_curr = mt.sample_mode(stage)
            if not mode_curr: continue
            action = plan_skeleton[stage]
            
            if action.name == "pick":
                edge, traj = pick_move(mode_curr, action)
            elif action.name == "place":
                edge, traj = place_move(mode_curr, action)
            elif action.name == "move":
                traj = free_move(mode_curr, action)
            
            if traj is None:
                print(f"mode_switch {stage}->{stage+1}: Failed")
            else:
                kingraph_new = mode_curr.kingraph.mode_switch(action, edge)
                stage_new = Mode1(mode_curr.index+1, kingraph_new, traj[-1])
                mt.add_mode(stage_new, mode_curr, traj)
                if stage == len(plan_skeleton)-1:
                    print("success")
                    break
                else:
                    print(f"mode_switch {stage}->{stage+1}: success")
        toc = time.time()
        elapsed_times.append(toc-tic)


        #debug        
        # modes = mt.backtrack(stage_new)
        # for mode in modes[:-1]:
        #     traj = mode.traj
        #     for config in traj:
        #         mode.kingraph.robot.set_joint_angles(config.q)
        #         mode.kingraph.assign()
        #         time.sleep(0.1)

    print(f"elapsed time mean: {np.mean(elapsed_times)}")
    print(f"elapsed time std: {np.std(elapsed_times)}")    

if __name__ == "__main__":
    main()