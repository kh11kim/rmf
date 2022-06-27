import numpy as np
from pybullet_suite.pybullet_suite import *
from rmf import *

def get_graspset_of_box(gui):
    """Usage:
    grasp_set = get_graspset_of_box(gui=False)
    np.savez("data/urdfs/block/grasp_set.npz", grasp_set=grasp_set)
    """
    fake_world = BulletWorld(gui=gui)
    scene_maker = BulletSceneMaker(fake_world)
    box_edge = 0.03
    box_size = np.array([box_edge, box_edge, box_edge])
    box1 = fake_world.load_urdf("box1", DICE_URDF)
    hand = Gripper(fake_world)

    sample_per_plane = 3
    yaw_per_point = 10
    yaws = np.linspace(0, 2*np.pi, yaw_per_point, endpoint=False)

    grasp_set = []
    # xz
    sample_points_plane = (np.random.rand(sample_per_plane, 2) - 0.5) * 2 * 0.015
    for pt in sample_points_plane:
        for yaw in yaws:
            for is_rev in [0, 1]:
                tcp = Pose(Rotation.from_euler("zyx",[0,0,0]), [pt[0], 0, pt[1]])
                rot_rev = Pose(Rotation.from_euler("xyz", [0,0,np.pi*is_rev]))
                rot_yaw = Pose(Rotation.from_euler("zxy", [0,0,yaw]))
                obj_to_grasp = tcp*rot_yaw*rot_rev
                hand.reset(obj_to_grasp)
                if not hand.detect_contact():
                    grasp_set.append(obj_to_grasp)
                hand.remove()
    # yz
    sample_points_plane = (np.random.rand(sample_per_plane, 2) - 0.5) * 2 * 0.015
    for pt in sample_points_plane:
        for yaw in yaws:
            for is_rev in [0, 1]:
                tcp = Pose(Rotation.from_euler("zyx",[np.pi/2,0,np.pi/2]), [0, pt[0], pt[1]])
                rot_rev = Pose(Rotation.from_euler("xyz", [0,0,np.pi*is_rev]))
                rot_yaw = Pose(Rotation.from_euler("zxy", [0,0,yaw]))
                obj_to_grasp = tcp*rot_yaw*rot_rev
                hand.reset(obj_to_grasp)
                if not hand.detect_contact():
                    obj_to_grasp = tcp*rot_yaw*rot_rev
                    grasp_set.append(obj_to_grasp)
                hand.remove()
    # xy
    sample_points_plane = (np.random.rand(sample_per_plane, 2) - 0.5) * 2 * 0.015
    for pt in sample_points_plane:
        for yaw in yaws:
            for is_rev in [0, 1]:
                tcp = Pose(Rotation.from_euler("zyx",[0,0,np.pi/2]), [pt[0], pt[1], 0])
                rot_rev = Pose(Rotation.from_euler("xyz", [0,0,np.pi*is_rev]))
                rot_yaw = Pose(Rotation.from_euler("zxy", [0,0,yaw]))
                obj_to_grasp = tcp*rot_yaw*rot_rev
                hand.reset(obj_to_grasp)
                if not hand.detect_contact():
                    obj_to_grasp = tcp*rot_yaw*rot_rev
                    grasp_set.append(obj_to_grasp)
                hand.remove()
    return grasp_set

if __name__ == "__main__":
    grasp_set = get_graspset_of_box(gui=False)
    np.savez(DICE_GRASP_SET, grasp_set=grasp_set)