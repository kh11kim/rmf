import time
import numpy as np
from pybullet_suite.pybullet_suite import *
from rmf import *
from rmf.kin_graph import KinGraph, KinRelation

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
    
    input()

def main():
    set_world(gui=True)
    input()

if __name__ == "__main__":
    main()