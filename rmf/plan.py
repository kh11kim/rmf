from itertools import product
from typing import Tuple, Optional, List, Dict

from .data_structure import Action, Target
from .kin_graph import KinGraph, KinRelation, Config, Movable

   
class Plan:
    def __init__(self):
        pass

    def parse(
        self, 
        plan_skeleton: List[Action], 
        targets: List[Target],
        config_init: Config,
        kingraph: KinGraph,
    ):
        self.num_actions = len(plan_skeleton)
        self.num_milestones = self.num_actions + 1
        self.actions: Dict[Tuple(int, bool), Action] = {key:[] for key in product(range(self.num_actions), [True, False])} # key is 2d tuple (stage:int, rev:bool)
        self.kingraph = kingraph
        self.targets = targets.copy()

        #go forward
        self.relations = [kingraph.relation]
        for stage in range(self.num_actions - 1): # we need {num_actions} numbers of modes
            action = plan_skeleton[stage]
            action.stage = stage
            next_relation = self.relations[stage].get_next_relation(action)
            self.relations.append(next_relation)
            self.actions[(stage, False)] = action
        final_stage = stage + 1
        final_action = plan_skeleton[final_stage]
        final_action.stage = final_stage
        self.actions[(final_stage, False)] = action

        # go backward to set goal
        for action in plan_skeleton[::-1]:
            for target in targets:
                if action.obj_name == target.obj_name:
                    if (target.obj_name == "robot") | \
                       (action.name == "place") & (target.placement is not None) | \
                       (action.name == "pick") & (target.grasp is not None):
                        self.actions[(action.stage, False)].target= target
                        targets.remove(target)

        for stage in range(self.num_actions - 1): # we need {num_actions} numbers of modes
            rel_index = self.num_actions - stage - 1
            rel_curr = self.relations[rel_index]
            rel_next = self.relations[rel_index-1]
            rev_action = self.get_action(rel_curr, rel_next)
            rev_action.rev = True
            rev_action.stage = stage
            self.actions[(stage, True)] = rev_action
        #final_reverse_action
        final_stage = stage + 1
        rev_action = self.get_action(self.relations[0], self.relations[0])
        rev_action.stage = final_stage
        rev_action.target = Target("robot", config=config_init)
        self.actions[(final_stage, True)] = rev_action


    def get_action(self, rel1: KinRelation, rel2: KinRelation):
        check_movable = []
        for movable in rel1.movables:
            if rel1.parent[movable] != rel2.parent[movable]:
                check_movable.append(movable)

        if len(check_movable) > 1:
            raise ValueError("One action can change only one edge")
        elif len(check_movable) == 0:
            if rel1.holding == True:
                action_name = "move_hold"
            else:
                action_name = "move_free"
            return Action(action_name, obj_name="robot")
        else:
            movable = check_movable[0]
            if (rel1.holding == True) & (rel2.holding == False):
                action_name = "place"
                placeable_name = rel2.placement[movable]
            elif (rel1.holding == False) & (rel2.holding == True):
                action_name = "pick"
                placeable_name = None
            else:
                raise ValueError()
            return Action(action_name, obj_name=movable, placeable_name=placeable_name)
            
        
    
    def get_rev_action(self, fwd_action: Action, prev_relation: KinRelation):
        if fwd_action.name == "pick":
            rev_action_name = "pick"
            rev_placeable_name = prev_relation.parent[fwd_action.obj_name]
        elif fwd_action.name == "place":
            rev_action_name = "pick"
            rev_placeable_name = None
        elif fwd_action.name == "move":
            rev_action_name = "move"
            rev_placeable_name = None
        else:
            raise ValueError(f"wrong value on action.name {fwd_action.name}")

        return Action(
            name=rev_action_name,
            obj_name=fwd_action.obj_name,
            placeable_name=rev_placeable_name,
            rev=True,
            stage=fwd_action.stage+1
        )
    
    def get_random_goal_config(self):
        final_relation = self.relations[-1]
        final_kingraph = self.kingraph.copy()
        final_kingraph.relation = final_relation

        #target assign
        obj_to_assign = ["robot", *final_relation.movables]
        assigned = []
        config = None
        for obj_name in obj_to_assign:
            for target in self.targets:
                if obj_name == target.obj_name:
                    if target.target_type == "Config":
                        config = target.config
                    elif target.target_type == "Grasp":
                        final_kingraph.kin_edge[obj_name] = target.grasp
                    elif target.target_type == "Placement":
                        final_kingraph.kin_edge[obj_name] = target.placement
                    assigned.append(obj_name)
        
        #random assign
        obj_to_assign = [obj for obj in obj_to_assign if obj not in assigned]
        for obj_name in obj_to_assign:
            if obj_name == "robot":
                config = Config(final_kingraph.robot.get_random_arm_angles())
            else:
                obj: Movable = final_kingraph.objects[obj_name]
                parent_name = final_relation.parent[obj_name]
                if parent_name == "robot":
                    #grasp
                    grasp = obj.sample_grasp()
                    final_kingraph.kin_edge[obj_name] = grasp
                else:
                    placement = final_kingraph.sample_placement(
                        obj_name, parent_name
                    )
                    final_kingraph.kin_edge[obj_name] = placement
        
        return config, final_kingraph
            
    