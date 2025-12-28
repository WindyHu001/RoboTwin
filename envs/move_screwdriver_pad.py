from ._base_task import Base_Task  
from .utils import *  
import sapien  
from ._GLOBAL_CONFIGS import *  
import imgaug.augmenters as iaa  
import math
  
  
class move_screwdriver_pad(Base_Task):  
  
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
        # Create screwdriver at random position  
        x = np.random.uniform(-0.05, 0.05)
        y = np.random.uniform(-0.1, 0.1)
        z = 0.741  
        self.screwdriver_q = self.quat_from_axis_angle((0, 1, 0), 90)

        # def quat_from_axis_angle(axis, angle_deg):
        #     ax, ay, az = axis
        #     theta = math.radians(angle_deg)
        #     half = 0.5 * theta
        #     s = math.sin(half)
        #     c = math.cos(half)
        #     return np.array([c, ax*s, ay*s, az*s], dtype=np.float64)  # [w,x,y,z]

        # def quat_mul(q1, q2):
        #     # Hamilton product, both in [w,x,y,z]
        #     w1, x1, y1, z1 = q1
        #     w2, x2, y2, z2 = q2
        #     return np.array([
        #         w1*w2 - x1*x2 - y1*y2 - z1*z2,
        #         w1*x2 + x1*w2 + y1*z2 - z1*y2,
        #         w1*y2 - x1*z2 + y1*w2 + z1*x2,
        #         w1*z2 + x1*y2 - y1*x2 + z1*w2
        #     ], dtype=np.float64)

        # def quat_normalize(q):
        #     return q / np.linalg.norm(q)

        # # 先 world-Y 90°，再 world-X 30°
        # qy = quat_from_axis_angle((0, 1, 0), 90)
        # qx = quat_from_axis_angle((1, 0, 0), 45)

        # q_world = quat_normalize(quat_mul(qy,qx))



        # self.screwdriver_q = q_world.tolist()

        # self.screwdriver = create_actor(
        #     scene=self,
        #     pose=sapien.Pose(
        #         p=[x, y, z],
        #         q=self.screwdriver_q,
        #     ),
        #     modelname="032_screwdriver_train_mid_test",
        #     convex=True,
        #     model_id=0,
        # )
        self.screwdriver = rand_create_actor(  
            self,  
            xlim=[-0.1, 0.1],  
            ylim=[-0.1, 0.1],  
            modelname="032_screwdriver_train_mid_test",  
            rotate_rand=True,  
            rotate_lim=[0, 1, 0],  
            qpos=self.screwdriver_q,  
            convex=True,  
        )  
        fp1 = self.screwdriver.get_functional_point(1, "pose")
        c1 = self.screwdriver.get_contact_point(1, "pose")


        print("FP1 pose p:", fp1)
        print("C1 pose p:", c1)

        # self.screwdriver.set_mass(0.01)  
        self.add_prohibit_area(self.screwdriver, padding=0.10)  
  
        # Store initial height for success checking  
        screwdriver_fp0 = self.screwdriver.get_functional_point(0, "pose").p  
        self.screwdriver_init_height = float(screwdriver_fp0[2])  
  
        # Add red contact marker to screwdriver  
        self.add_contact_marker_to_screwdriver()  
  
        # Create target pad (similar to move_pillbottle_pad)  
        screwdriver_pos = self.screwdriver.get_pose().p  
        if screwdriver_pos[0] > 0:  
            xlim = [0.12, 0.25]  
        else:  
            xlim = [-0.25, -0.12]   
              
        target_rand_pose = rand_pose(  
            xlim=xlim,  
            ylim=[-0.2, 0.1],  
            qpos=[1, 0, 0, 0],  
            rotate_rand=False,  
        )  
          
        # Ensure minimum distance from screwdriver  
        while (np.sqrt((target_rand_pose.p[0] - screwdriver_pos[0])**2 +   
                      (target_rand_pose.p[1] - screwdriver_pos[1])**2) < 0.1):  
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
  
    def add_contact_marker_to_screwdriver(self):  
        """Attach a bright red sphere to the screwdriver's contact point"""  
        if not hasattr(self, 'screwdriver'):  
            return  
  
        contact_point_local = self.screwdriver.get_contact_point(0, "pose").p  
  
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
        # Update screwdriver contact marker  
        if hasattr(self, 'contact_marker') and hasattr(self, 'screwdriver'):  
            contact_point_local = self.screwdriver.get_contact_point(0, "pose").p  
            marker_pose = sapien.Pose(p=contact_point_local)  
            self.contact_marker.set_pose(marker_pose)  
  
        super()._update_render()  
  
    def play_once(self):  
        # Determine which arm to use based on screwdriver position  
        screwdriver_pose = self.screwdriver.get_pose().p  
        arm_tag = ArmTag("left" if screwdriver_pose[0] < 0 else "right")  
  
        # Grasp the screwdriver  
        self.move(self.grasp_actor(  
            self.screwdriver,   
            arm_tag=arm_tag,   
            pre_grasp_dis=0.12,   
            grasp_dis=0.01,  
            contact_point_id=0
        ))  
  
        # Lift the screwdriver up  
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))  
  
        # Get target pose from pad's functional point  
        target_pose = self.pad.get_functional_point(1)  
        screw_fp = self.screwdriver.get_functional_point(0, "pose").q
        pad_fp1_pose = self.pad.get_functional_point(1, "pose")  # Pose(p,q)
        target_pose = sapien.Pose(p=pad_fp1_pose.p, q=screw_fp)

        self.move(self.place_actor(  
            self.screwdriver,  
            arm_tag=arm_tag,  
            target_pose=target_pose,  
            pre_dis=0.05,  
            dis=0,  
            functional_point_id=0,  
            pre_dis_axis='fp',
            constrain="free"
        ))  
  
        self.info["info"] = {  
            "{A}": "032_screwdriver/base0",  
            "{a}": str(arm_tag),  
        }  
        return self.info  
  
    def check_success(self):  
        screwdriver_pose = self.screwdriver.get_functional_point(0, "pose").p  
        target_pos = self.pad.get_pose().p  
          
        eps = 0.05
        return np.all(abs(screwdriver_pose[:2] - target_pos[:2]) < np.array([eps, eps]))