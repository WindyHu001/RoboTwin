from ._base_task import Base_Task  
from .utils import *  
import sapien  
from ._GLOBAL_CONFIGS import *  
import imgaug.augmenters as iaa    
import math

  
class pick_scanner_LG_top(Base_Task):  

    def quat_from_axis_angle(self, axis, angle_deg):
        """Return [w, x, y, z] given axis=(ax,ay,az) and angle(deg)."""
        ax, ay, az = axis
        theta = math.radians(angle_deg)
        half = theta * 0.5
        s = math.sin(half)
        c = math.cos(half)
        return [c, ax * s, ay * s, az * s]
  
    def setup_demo(self, **kwags):  
        super()._init_task_env_(**kwags)  
  
    def load_actors(self):  
        self.scanner_q = self.quat_from_axis_angle((0, 1/math.sqrt(2), 1/math.sqrt(2)), 180)

        self.scanner = rand_create_actor(
            self,
            xlim=[-0.1, 0.1],
            ylim=[-0.1, 0.1],
            zlim=[0.760],
            modelname="024_scanner_train_top",
            rotate_rand=True,
            rotate_lim=[0, 1, 0],
            convex=True,
            qpos=self.scanner_q,
        )

        self.scanner.set_mass(0.01)  
        self.add_prohibit_area(self.scanner, padding=0.10)  

        scanner_fp0 = self.scanner.get_functional_point(0, "pose").p
        self.scanner_init_height = float(scanner_fp0[2])
        self.scene.step()    
        self.scene.update_render()            
  
    def play_once(self):  
        # Determine which arm to use based on scanner position  
        scanner_pose = self.scanner.get_pose().p  
        arm_tag = ArmTag("left" if scanner_pose[0] < 0 else "right")  
  
        # Grasp the scanner  
        self.move(self.grasp_actor(self.scanner, arm_tag=arm_tag, pre_grasp_dis=0.12, grasp_dis=0.01,contact_point_id=0))
          
        # Lift the scanner up  
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))  
          
        self.info["info"] = {
            "{A}": "024_scanner/base0",
            "{a}": str(arm_tag),
        }
        return self.info
  
    def check_success(self):
        # Get current scanner functional point pose (world coordinates)
        scanner_pose = self.scanner.get_functional_point(0, "pose").p

        # --- Early failure condition (similar style to your bottle code) ---
        # If the scanner is *lower* than its initial height minus a small tolerance,
        # we treat it as failure (fell off the table / not picked at all).
        if scanner_pose[2] < self.scanner_init_height - 0.02:
            # Optional: you can track a flag like in your bottle code
            # self.actor_pose = False
            return False

        # --- Main success condition: lifted by enough height ---
        # How much higher is the scanner compared to where it started?
        dz = scanner_pose[2] - self.scanner_init_height

        # Require that the scanner is lifted at least 7cm (you can tune this)
        required_lift = 0.08  # 8cm

        # Optionally, you could also require that it's not moved too far horizontally,
        # but for a pure pick-up task, height is usually enough.
        return dz > required_lift