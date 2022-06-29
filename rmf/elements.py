import numpy as np
from copy import copy, deepcopy
from pybullet_suite import *

# Node : Body (Movable, Fixed)
PRE_POSE_DISTANCE = 0.05

class KinNode(Body):
    """Wrapper function of Body object
    """
    def __init__(
        self,
        physics_client: BulletClient,
        body_uid: int,
        name: str,
    ):
        super().__init__(
            physics_client=physics_client,
            body_uid=body_uid
        )
        self.name = name
    
    @classmethod
    def from_body(cls, body: Body, name: str):
        return cls(body.physics_client, body.uid, name)

class Movable(KinNode):
    """ Movable (Body wrapper class)
    contains grasp set and placement axis set
    and provides sampling functionality
    """
    def __init__(
        self,
        physics_client: BulletClient,
        body_uid: int,
        name: str,
    ):
        super().__init__(
            physics_client=physics_client,
            body_uid=body_uid,
            name=name
        )
    
    def set_grasp_set(self, grasp_pose_set: List["Grasp"]):
        self.grasp_set = [Grasp(self.name, i, T) for i, T in enumerate(grasp_pose_set)]
    
    def set_placement_axis_set(self, placement_axis_set: List[np.ndarray]):
        self.placement_axis_set = placement_axis_set
    
    def sample_grasp(self) -> "Grasp":
        return copy(np.random.choice(self.grasp_set))
    
    def sample_placement_axis(self) -> np.ndarray:
        i = np.random.choice(range(len(self.placement_axis_set)))
        return copy(self.placement_axis_set[i])
    
    def solve_penetration(self):
        """just add half_extent to the z
        """
        pose = self.get_base_pose()
        _, extent = self.get_AABB(output_center_extent=True)
        xy = pose.trans[:2]
        z = pose.trans[-1] + extent[-1]/2
        pose_new = Pose(pose.rot, np.array([*xy, z]))
        self.set_base_pose(pose_new)


class Fixed(KinNode):
    """ Fixed (Body wrapper class)
    provides sampling functionality in placement plane
    """
    def __init__(
        self,
        physics_client: BulletClient,
        body_uid: int,
        name: str
    ):
        super().__init__(
            physics_client=physics_client,
            body_uid=body_uid,
            name=name
        )
    
    def sample_point(self, inner=0.05):
        lower, upper = self.get_AABB()
        xy = np.random.uniform(
            low=lower[:2]+inner, high=upper[:2]-inner
        )
        z = upper[-1]
        return np.array([*xy, z])



# Edge :
class KinEdge:
    def __init__(self, movable_name: str, tf: Pose):
        self.movable_name = movable_name
        self.tf = tf # transform to parent wrt obj

class Grasp(KinEdge):
    def __init__(
        self,
        movable_name: str,
        index: int,
        tf: Pose
    ):
        assert type(movable_name) is str
        super().__init__(movable_name=movable_name, tf=tf)
        self.index = index

    def get_pre_pose(self, pose: Pose):
        pre_pose = Pose(trans=[0,0,-PRE_POSE_DISTANCE])
        return pose * pre_pose
    
    def __eq__(self, other: "Grasp"):
        same_index = self.index == other.index
        same_movable_name = self.movable_name == other.movable_name
        return same_index & same_movable_name
    
class Placement(KinEdge):
    def __init__(
        self,
        movable_name: str,
        placeable_name: str,
        tf: Pose,
    ):
        assert type(movable_name) is str
        assert type(placeable_name) is str
        super().__init__(movable_name=movable_name, tf=tf)
        self.placeable_name = placeable_name

    def get_pre_pose(self, pose: Pose):
        pre_pose = Pose(trans=[0,0,+PRE_POSE_DISTANCE])
        return pre_pose * pose
    
    def __eq__(self, other: "Placement"):
        same_movable_name = self.movable_name == other.movable_name
        same_placeable_name = self.placeable_name == other.placeable_name
        same_tf = self.tf == other.tf
        return same_movable_name & same_placeable_name & same_tf

    @classmethod
    def from_bodies(
        cls, 
        movable: Movable, 
        placeable: Fixed,
    ):
        """ get placement from current object transformations.
        """
        parent_pose = placeable.get_base_pose()
        tf = movable.get_base_pose().inverse() * parent_pose
        return cls(movable.name, placeable.name, tf)
    
    @classmethod
    def from_body_point_and_placement_axes(
        cls,
        movable: Movable,
        placeable: Fixed,
        xyz: np.ndarray,
        z_axis: np.ndarray,
        solve_penetration: bool = True
    ):
        with movable.no_set_pose():
            pose = cls.get_pose_by_z_axis_and_point(z_axis, xyz)
            movable.set_base_pose(pose)
            if solve_penetration:
                movable.solve_penetration()
            placement = Placement.from_bodies(movable, placeable)
        return placement

    @staticmethod
    def get_pose_by_z_axis_and_point(z_axis, xyz):
        if np.allclose(z_axis, np.array([1, 0, 0])) | \
            np.allclose(z_axis, np.array([-1, 0, 0])):
            x_axis = np.array([0, 1, 0])
        else:
            x_axis = np.array([1, 0, 0])
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis/np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        rot = np.vstack([x_axis, y_axis, z_axis]).T
        rot = Rotation.from_matrix(rot)
        yaw = np.random.uniform(0, np.pi*2)
        yaw_rot = Rotation.from_rotvec(yaw * z_axis)
        pose = Pose(rot.inv()*yaw_rot, xyz)
        return pose

class Config:
    """Configuration of robots
    """
    def __init__(
        self, 
        q: np.ndarray, 
        T: Optional[Pose]=None, 
    ):
        self.q = q
        self.T = T
        self.index = -1 # not assigned

    def copy(self):
        return Config(
            q=deepcopy(self.q),
            T=deepcopy(self.T),
        )

    @classmethod
    def from_robot(cls, robot: Panda):
        q = robot.get_joint_angles()
        T = robot.get_ee_pose()
        return cls(q, T)

if __name__ == "__main__":
    world = BulletWorld(gui=True)
    sm = BulletSceneMaker(world)
    box = sm.create_box("box", [0.05]*3, 1, [0,0,0])
    box = Movable.from_body(box, "box")
    input()