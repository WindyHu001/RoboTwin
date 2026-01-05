from ._base_task import Base_Task  
from .utils import *  
import sapien  
from ._GLOBAL_CONFIGS import *  
import imgaug.augmenters as iaa    
  
  
class pick_hammer_LG_bottom(Base_Task):  
  
    def setup_demo(self, **kwags):  
        super()._init_task_env_(**kwags)  
  
    def load_actors(self):  
        self.hammer = rand_create_actor(
            self,
            xlim=[-0.1, 0.1],
            ylim=[-0.1, 0.1],
            zlim=[0.783,0.783],
            modelname="020_hammer_train_bottom",
            rotate_rand=True,
            rotate_lim=[0, 1, 0],
            qpos=[0, 0, 0.995, 0.105],
            convex=True,
        )

        self.hammer.set_mass(0.01)  
        self.add_prohibit_area(self.hammer, padding=0.10)  

        hammer_fp0 = self.hammer.get_functional_point(0, "pose").p
        self.hammer_init_height = float(hammer_fp0[2])
        self.scene.step()    
        self.scene.update_render()            
  
    def play_once(self):  
        # Determine which arm to use based on hammer position  
        hammer_pose = self.hammer.get_pose().p  
        arm_tag = ArmTag("left" if hammer_pose[0] < 0 else "right")  
  
        # Grasp the hammer  
        self.move(self.grasp_actor(self.hammer, arm_tag=arm_tag, pre_grasp_dis=0.12, grasp_dis=0.01,contact_point_id=0))
          
        # Lift the hammer up  
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))  
          
        self.info["info"] = {
            "{A}": "020_hammer/base0",
            "{a}": str(arm_tag),
        }
        return self.info
  
    def check_success(self):
        # Get current hammer functional point pose (world coordinates)
        hammer_pose = self.hammer.get_functional_point(0, "pose").p

        # --- Early failure condition (similar style to your bottle code) ---
        # If the hammer is *lower* than its initial height minus a small tolerance,
        # we treat it as failure (fell off the table / not picked at all).
        if hammer_pose[2] < self.hammer_init_height - 0.02:
            # Optional: you can track a flag like in your bottle code
            # self.actor_pose = False
            return False

        # --- Main success condition: lifted by enough height ---
        # How much higher is the hammer compared to where it started?
        dz = hammer_pose[2] - self.hammer_init_height

        # Require that the hammer is lifted at least 7cm (you can tune this)
        required_lift = 0.08  # 8cm

        # Optionally, you could also require that it's not moved too far horizontally,
        # but for a pure pick-up task, height is usually enough.
        return dz > required_lift