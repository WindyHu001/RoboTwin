from ._base_task import Base_Task  
from .utils import *  
import sapien  
from ._GLOBAL_CONFIGS import *  
import imgaug.augmenters as iaa  
import math
  
  
class move_scanner_pad(Base_Task):  
  
    def setup_demo(self, **kwags):  
        super()._init_task_env_(**kwags)  

    def quat_from_axis_angle(self, axis, angle_deg):
        """Return [w, x, y, z] given axis=(ax,ay,az) and angle(deg)."""
        ax, ay, az = axis
        theta = math.radians(angle_deg)
        half = theta * 0.5
        s = math.sin(half)
        c = math.cos(half)
        return [c, ax * s, ay * s, az * s]

    def load_actors(self):  
        # Create scanner at random position  
        self.scanner_q = self.quat_from_axis_angle((0, 1/math.sqrt(2), 1/math.sqrt(2)), 180)
        self.scanner = rand_create_actor(
            self,
            xlim=[-0.05, 0.05],
            ylim=[-0.1, 0.1],
            zlim=[0.760],
            modelname="024_scanner_train_mid",
            rotate_rand=True,
            rotate_lim=[0, 1, 0],
            convex=True,
            qpos=self.scanner_q,
        )
  
        # self.scanner.set_mass(0.01)  
        self.add_prohibit_area(self.scanner, padding=0.10)  
  
        # Store initial height for success checking  
        scanner_fp0 = self.scanner.get_functional_point(0, "pose").p  
        self.scanner_init_height = float(scanner_fp0[2])  
  
        # Add red contact marker to scanner  
        self.add_contact_marker_to_scanner()  
  
        # Create target pad (similar to move_pillbottle_pad)  
        scanner_pos = self.scanner.get_pose().p  
        if scanner_pos[0] > 0:  
            xlim = [0.08, 0.25]  
        else:  
            xlim = [-0.25, -0.08]   
              
        target_rand_pose = rand_pose(  
            xlim=xlim,  
            ylim=[-0.2, 0.1],  
            qpos=[1, 0, 0, 0],  
            rotate_rand=False,  
        )  
          
        # Ensure minimum distance from scanner  
        while (np.sqrt((target_rand_pose.p[0] - scanner_pos[0])**2 +   
                      (target_rand_pose.p[1] - scanner_pos[1])**2) < 0.1):  
            target_rand_pose = rand_pose(  
                xlim=xlim,  
                ylim=[-0.2, 0.1],  
                qpos=[1, 0, 0, 0],  
                rotate_rand=False,  
            )  
  
        # Create pad as thin box  
        half_size = [0.06, 0.16, 0.01]   
        self.pad = create_box(  
            scene=self,  
            pose=target_rand_pose,  
            half_size=half_size,  
            color=(0.3, 0.3, 0.3),  
            name="target_pad",  
            is_static=True,  
        )  
  
        # Add blue target marker on pad  
        self.add_target_marker_to_pad()  
  
        self.add_prohibit_area(self.pad, padding=0.15)  
        self.scene.step()  
        self.scene.update_render()  
  
    def add_contact_marker_to_scanner(self):  
        """Attach a bright red sphere to the scanner's contact point"""  
        if not hasattr(self, 'scanner'):  
            return  
  
        contact_point_local = self.scanner.get_contact_point(0, "pose").p  
  
        builder = self.scene.create_actor_builder()  
        builder.set_physx_body_type("kinematic")  
  
        builder.add_sphere_visual(  
            radius=0.02,  
            material=sapien.render.RenderMaterial(  
                base_color=[1.0, 0.0, 0.0, 1.0],   # Red  
                emission=[1.0, 0.0, 0.0, 1.0],  
                specular=0.5,  
                roughness=0.0,  
                metallic=0.0,  
                transmission=0.0  
            )  
        )  
  
        marker = builder.build(name="contact_marker")  
        marker_pose = sapien.Pose(p=contact_point_local)  
        marker.set_pose(marker_pose)  
        self.contact_marker = marker  
  
    def add_target_marker_to_pad(self):  
        """Add a blue sphere marker at the pad's target location"""  
        if not hasattr(self, 'pad'):  
            return  
  
        # Get pad's functional point as target location  
        target_pose = self.pad.get_functional_point(1, "pose")  
  
        builder = self.scene.create_actor_builder()  
        builder.set_physx_body_type("kinematic")  
  
        builder.add_sphere_visual(  
            radius=0.025,  # Slightly larger than contact marker  
            material=sapien.render.RenderMaterial(  
                base_color=[0.0, 0.0, 1.0, 1.0],   # Blue  
                emission=[0.0, 0.0, 1.0, 1.0],  
                specular=0.5,  
                roughness=0.0,  
                metallic=0.0,  
                transmission=0.0  
            )  
        )  
  
        marker = builder.build(name="target_marker")  
        marker.set_pose(target_pose)  
        self.target_marker = marker  
  
    def _update_render(self):  
        """Override to update marker positions"""  
        # Update scanner contact marker  
        if hasattr(self, 'contact_marker') and hasattr(self, 'scanner'):  
            contact_point_local = self.scanner.get_contact_point(0, "pose").p  
            marker_pose = sapien.Pose(p=contact_point_local)  
            self.contact_marker.set_pose(marker_pose)  
  
        super()._update_render()  
  
    def play_once(self):  
        # Determine which arm to use based on scanner position  
        scanner_pose = self.scanner.get_pose().p  
        arm_tag = ArmTag("left" if scanner_pose[0] < 0 else "right")  
  
        # Grasp the scanner  
        self.move(self.grasp_actor(  
            self.scanner,   
            arm_tag=arm_tag,   
            pre_grasp_dis=0.12,   
            grasp_dis=0.01,  
            contact_point_id=0  
        ))  
  
        # Lift the scanner up  
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))  
  
        # Get target pose from pad's functional point  
        target_pose = self.pad.get_functional_point(1)  
  
        # Place scanner on pad  
        self.move(self.place_actor(  
            self.scanner,  
            arm_tag=arm_tag,  
            target_pose=target_pose,  
            pre_dis=0.05,  
            dis=0,  
            functional_point_id=0,  
            pre_dis_axis='fp'  
        ))  
  
        self.info["info"] = {  
            "{A}": "024_scanner/base0",  
            "{a}": str(arm_tag),  
        }  
        return self.info  
  
    def check_success(self):  
        scanner_pose = self.scanner.get_functional_point(0, "pose").p  
        target_pos = self.pad.get_pose().p  
          
        eps = 0.05
        # return (  
        #     np.all(abs(scanner_pose[:2] - target_pos[:2]) < np.array([eps, eps])) and  
        #     abs(scanner_pose[2] - (0.741 + self.table_z_bias)) < 0.005 and  
        #     self.robot.is_left_gripper_open() and   
        #     self.robot.is_right_gripper_open()  
        # )
        return np.all(abs(scanner_pose[:2] - target_pos[:2]) < np.array([eps, eps]))