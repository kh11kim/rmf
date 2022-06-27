from .elements import *
from .action_plan import *

## Simplified version of Kinematic graph / Plan graph
class KinRelation:
    """ Simplified version of kinematic graph.
    """
    def __init__(self):
        self.obj_type = {} #assume one robot in the world
        self.parent: Dict[str, str] = {}
        self.child = {}
        self.add_robot()

    def copy(self):
        return deepcopy(self)

    @property
    def movables(self):
        return [
            name for name in self.obj_type.keys() \
                if self.obj_type[name] == "Movable"
        ]
    
    @property
    def fixed(self):
        return [
            name for name in self.obj_type.keys() \
                if self.obj_type[name] == "Fixed"
        ]
        
    @property
    def placeables(self):
        return self.fixed + self.movables

    def add_robot(self):
        self.obj_type["robot"] = "robot"
        self.parent["robot"] = None

    # robot, fixed
    def add_object(self, name:str, obj_type:str, parent_name: str=None):
        self.obj_type[name] = obj_type
        if obj_type == "fixed":
            self.parent[name] = None
            self.child[name] = None
        elif obj_type == "movable":
            assert parent_name in self.obj_type.keys(), f"parent_name of {name} should be given"
            self.parent[name] = parent_name
            self.child[parent_name] = name
    
    def apply_action(self, action: Action):
        """Mode switch by action
        """
        relation_new = self.copy()

        if action.name == "move":
            return relation_new

        old_parent = relation_new.parent[action.obj_name]
        if action.name == "pick":
            assert action.placeable_name is None, f"action {action.name} shouldn't have a placement"
            new_parent = "robot"
            
        elif action.name == "place":
            assert action.placeable_name is not None, f"action {action.name} should have a placement"
            new_parent = action.placeable_name

        relation_new.parent[action.obj_name] = new_parent
        relation_new.child[old_parent] = None
        relation_new.child[new_parent] = action.obj_name
        return relation_new

class KinGraph:
    def __init__(
        self, 
        world: BulletWorld,
        scene_maker:Optional[BulletSceneMaker]=None
    ):
        self.world = world #for collision detection
        self.objects: Dict[str, KinNode] = {} # movable, fixed
        self.kin_edge: Dict[str, KinEdge] = {}
        self.relation = KinRelation()
        self.sm = scene_maker #for debug
    
    def copy(self) -> "KinGraph":
        new_graph = KinGraph(world=self.world, scene_maker=self.sm)
        new_graph.objects = copy(self.objects)
        new_graph.relation = self.relation.copy()
        new_graph.kin_edge = deepcopy(self.kin_edge)
        return new_graph
        
    @property
    def robot(self) -> Panda:
        return self.objects["robot"]
    
    @property
    def movables(self) -> List[str]:
        return self.relation.movables
    
    @property
    def placeables(self) -> List[str]:
        return self.relation.placeables

    @property
    def fixed(self) -> List[str]:
        return self.relation.fixed

    def add_robot(self, robot: Panda):
        self.objects["robot"] = robot
        self.relation.add_robot()
    
    def add_object(
        self, 
        name: str, 
        obj: Body, 
        parent_name: str = None, 
        edge: KinEdge = None
    ):
        obj_type = type(obj)
        if (obj_type == Panda) | (obj_type == Fixed):
            pass

        elif obj_type is Movable:
            assert parent_name in self.objects.keys(), f"parent_name of {name} should be given first"
            assert edge is not None, f"edge of {name} should be given"
        
        self.objects[name] = obj
        self.relation.add_object(name, obj_type.__name__, parent_name)

        if obj_type is Movable:
            self.kin_edge[name] = edge
    
    def assign(self, config: Optional[Config] = None):
        def assign_obj(obj_name):
            parent_name = self.relation.parent[obj_name]
            parent_type = type(self.objects[parent_name])
            if parent_type is Panda:
                parent_pose = self.robot.get_ee_pose()
            elif parent_type is Fixed:
                parent_pose = self.objects[parent_name].get_base_pose()
            elif parent_type is Movable:
                parent_pose = assign_obj(parent_name)
            
            obj_pose = parent_pose * self.kin_edge[obj_name].tf.inverse()
            self.objects[obj_name].set_base_pose(obj_pose)
            return obj_pose
        if config:
            self.robot.set_arm_angles(config.q)
        for movable_names in self.movables:
            assign_obj(movable_names)
    
    def mode_switch(
        self, 
        action: Action,
        edge: KinEdge
    ) -> "KinGraph":
        graph = self.copy()
        if action.name == "move":
            return graph
        graph.relation = graph.relation.apply_action(action)
        # obj_name = action.obj_name
        # parent_old_name = graph.relation.parent[obj_name]
        # if action.name == "pick":
        #     parent_name = "robot"
        # elif action.name == "place":
        #     parent_name = action.placeable_name
        #     assert parent_name is not None, "There is no placeable(parent) name in action"
        # graph.parent[obj_name] = parent_name
        # graph.child[parent_old_name] = \
        #     [child for child in graph.child[parent_old_name] if child != obj_name]
        # graph.child[parent_name].append(obj_name)
        graph.kin_edge[action.obj_name] = edge

        return graph

    def is_collision(self, config: Config):
        robot: Panda = self.objects["robot"]
        robot.set_arm_angles(config.q)
        self.assign()
        
        if self.world.is_self_collision(robot):
            return True
        
        for movable_name in self.movables:
            object_names = list(self.objects.keys())
            others = object_names.remove(movable_name) #check except myself
            others = object_names.remove(self.relation.parent[movable_name]) #check except parent
            if self.world.is_body_pairwise_collision(
                body=movable_name, obstacles=others):
                return True
        return False
        
        # for movable_name in self.movables:
        #     parent = self.parent[movable_name]
        #     child = self.child[movable_name]
        #     if self.world.get_contacts(
        #         name=movable_name,
        #         exception=[parent, *child]
        #     ):
        #         return True
        # return False
    
    
    def sample_placement(
        self,
        movable_name: str,
        placeable_name: str,
    ):
        movable: Movable = self.objects[movable_name]
        placeable = self.objects[placeable_name]
        if placeable_name in self.movables:
            xyz = placeable.get_base_pose().trans
        elif placeable_name in self.fixed:
            xyz = placeable.sample_from_plane()
        z_axis = movable.sample_placement_axis()

        return Placement.from_body_point_and_placement_axes(
            movable, placeable, xyz, z_axis)
        
    def get_robot_target_by_grasp(
        self,
        movable_name: str,
        grasp: Grasp
    ) -> Pose:
        movable: Movable = self.objects[movable_name]
        pose = movable.get_base_pose() * grasp.tf
        return pose

    def get_robot_target_by_placement(
        self,
        movable_name: str,
        placeable_name: str,
        placement: Placement
    ) -> Pose:
        placeable: Fixed = self.objects[placeable_name]
        curr_grasp = self.kin_edge[movable_name]
        assert type(curr_grasp) is Grasp, f"Movable is not currently grasped!"
        pose = placeable.get_base_pose() * placement.tf.inverse() * curr_grasp.tf
        #debug
        # with placeable.no_set_pose():
        #     self.objects[movable_name].set_base_pose(pose)
        #     print("pause")
        return pose
