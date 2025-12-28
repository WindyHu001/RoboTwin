from ._base_task import Base_Task  
from .utils import *  
import sapien  
from ._GLOBAL_CONFIGS import *  
import imgaug.augmenters as iaa  
import math
import random

  
class move_mug_pad(Base_Task):  
  
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
        # Create mug at random position  
        # self.mug_q = self.quat_from_axis_angle((0, 1/math.sqrt(2), 1/math.sqrt(2)), 180)

        self.mug_id = np.random.choice([i for i in range(10)])

        self.contact_point_id = random.choice([0,1,2,3,4,5])
        self.mug = rand_create_actor(
            self,
            xlim=[-0.2, 0.2],
            ylim=[-0.05, 0.05],
            ylim_prop=True,
            modelname="039_mug",
            rotate_rand=True,
            rotate_lim=[0, 1.57, 0],
            qpos=[0.707, 0.707, 0, 0],
            convex=True,
            model_id=self.mug_id,
        )
  
        # self.mug.set_mass(0.01)  
        self.add_prohibit_area(self.mug, padding=0.10)  
  
        # Store initial height for success checking  
        mug_fp0 = self.mug.get_functional_point(0, "pose").p  
        self.mug_init_height = float(mug_fp0[2])  
  
        # Add red contact marker to mug  
        self.add_contact_marker_to_mug()  
  
        # Create target pad (similar to move_pillbottle_pad)  
        mug_pos = self.mug.get_pose().p  
        if mug_pos[0] > 0:  
            xlim = [0.08, 0.25]  
        else:  
            xlim = [-0.25, -0.08]   
              
        target_rand_pose = rand_pose(  
            xlim=xlim,  
            ylim=[-0.2, 0.1],  
            qpos=[1, 0, 0, 0],  
            rotate_rand=False,  
        )  
          
        # Ensure minimum distance from mug  
        while (np.sqrt((target_rand_pose.p[0] - mug_pos[0])**2 +   
                      (target_rand_pose.p[1] - mug_pos[1])**2) < 0.1):  
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
  
    def add_contact_marker_to_mug(self):  
        """Attach a bright red sphere to the mug's contact point"""  
        if not hasattr(self, 'mug'):  
            return  
  
        contact_point_local = self.mug.get_contact_point(self.contact_point_id, "pose").p  
  
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
        # Update mug contact marker  
        if hasattr(self, 'contact_marker') and hasattr(self, 'mug'):  
            contact_point_local = self.mug.get_contact_point(self.contact_point_id, "pose").p  
            marker_pose = sapien.Pose(p=contact_point_local)  
            self.contact_marker.set_pose(marker_pose)  
  
        super()._update_render()  
  
    def play_once(self):  
        # Determine which arm to use based on mug position  
        mug_pose = self.mug.get_pose().p  
        arm_tag = ArmTag("left" if mug_pose[0] < 0 else "right")  
  
        # Grasp the mug  
        self.move(self.grasp_actor(  
            self.mug,   
            arm_tag=arm_tag,   
            pre_grasp_dis=0.12,   
            grasp_dis=0.01,  
            contact_point_id=self.contact_point_id  
        ))  
  
        # Lift the mug up  
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))  
  
        # Get target pose from pad's functional point  
        target_pose = self.pad.get_functional_point(1)  
  
        # Place mug on pad  
        self.move(self.place_actor(  
            self.mug,  
            arm_tag=arm_tag,  
            target_pose=target_pose,  
            pre_dis=0.05,  
            dis=0,  
            functional_point_id=1,  
            pre_dis_axis='fp'  
        ))  
  
        self.info["info"] = {  
            "{A}": f"039_mug/base{self.mug_id}",  
            "{a}": str(arm_tag),  
        }  
        return self.info  
  
    def check_success(self):  
        mug_pose = self.mug.get_functional_point(0, "pose").p  
        target_pos = self.pad.get_pose().p  
          
        eps = 0.05

        return np.all(abs(mug_pose[:2] - target_pos[:2]) < np.array([eps, eps]))