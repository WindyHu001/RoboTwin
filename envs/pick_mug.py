from ._base_task import Base_Task  
from .utils import *  
import sapien  
from ._GLOBAL_CONFIGS import *  
import imgaug.augmenters as iaa    
import math
import random


  
  
class pick_mug(Base_Task):  
  
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

        q_x_90  = self.quat_from_axis_angle((1, 0, 0),  0)
        self.mug_q = q_x_90

        # self.mug = create_actor(  
        #     scene=self,  
        #     pose=sapien.Pose([0, -0.06, 0.760], self.mug_q),  
        #     modelname="030_mug",  
        #     convex=True,  
        #     model_id=0,  
        # )  
        # self.mug = rand_create_actor(  
        #     scene=self,  
        #     modelname="039_mug", 
        #     model_id=0,  
        #     xlim=[-0.05, 0.05],  
        #     ylim=[-0.05, 0.05],  
        #     # zlim=[0.75,0.71],
        #     # rotate_rand=False,  
        # )  
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
        
        # [Debug]: Add the axis 
        # pose = self.mug.get_pose()
        # T = pose.to_transformation_matrix()
        # R = T[:3, :3]  # 3x3 rotation part
        # self._create_mug_axes()
        # print("local x axis in world:", R[:, 0])
        # print("local y axis in world:", R[:, 1])
        # print("local z axis in world:", R[:, 2])
        # self.mug.set_mass(0.1)  
        # self.add_prohibit_area(self.mug, padding=0.10)  
        
        mug_fp0 = self.mug.get_functional_point(0, "pose").p
        self.mug_init_height = float(mug_fp0[2])
        mug_pos = self.mug.get_pose().p
        print(mug_pos)     
        self.add_contact_marker_to_mug()  

        

        self.scene.step()    
        self.scene.update_render()         

    def _create_mug_axes(self):
        axis_length = 0.1
        axis_radius = 0.01

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("kinematic")

        mat_x = sapien.render.RenderMaterial(
            base_color=[1, 0, 0, 1],
            emission=[1, 0, 0, 1],
        )
        builder.add_capsule_visual(
            radius=axis_radius,
            half_length=axis_length / 2,
            material=mat_x,
            pose=sapien.Pose(p=[axis_length / 2, 0, 0], q=[1, 0, 0, 0]),
        )

        mat_y = sapien.render.RenderMaterial(
            base_color=[0, 1, 0, 1],
            emission=[0, 1, 0, 1],
        )
        q_rot_z_90 = self.quat_from_axis_angle((0, 0, 1), 90)
        builder.add_capsule_visual(
            radius=axis_radius,
            half_length=axis_length / 2,
            material=mat_y,
            pose=sapien.Pose(p=[0, axis_length / 2, 0], q=q_rot_z_90),
        )

        mat_z = sapien.render.RenderMaterial(
            base_color=[0, 0, 1, 1],
            emission=[0, 0, 1, 1],
        )
        q_rot_y_neg90 = self.quat_from_axis_angle((0, -1, 0), 90)
        builder.add_capsule_visual(
            radius=axis_radius,
            half_length=axis_length / 2,
            material=mat_z,
            pose=sapien.Pose(p=[0, 0, axis_length / 2], q=q_rot_y_neg90),
        )

        self.mug_axes = builder.build(name="mug_axes")
        self._update_mug_axes_pose()



    def _update_mug_axes_pose(self):
        if not hasattr(self, "mug") or not hasattr(self, "mug_axes"):
            return
        mug_pose = self.mug.get_pose()
        self.mug_axes.set_pose(mug_pose) 

    def add_contact_marker_to_mug(self):    
        """Attach a bright colored sphere to the mug's contact point"""    
        if not hasattr(self, 'mug'):    
            return    
          
        # Get contact point position relative to mug    
        contact_point_local = self.mug.get_contact_point(self.contact_point_id, "pose").p  
          
        # Create a small sphere as visual marker    
        builder = self.scene.create_actor_builder()    
        builder.set_physx_body_type("kinematic")  # No physics, just visual    
            
        # Add visual sphere with bright red color    
        builder.add_sphere_visual(    
            radius=0.02,  # 5mm radius (small marker)    
            material = sapien.render.RenderMaterial(  
                base_color=[1.0, 0.0, 0.0, 1.0],   # bright opaque red  
                emission=[1.0, 0.0, 0.0, 1.0],     # pure red glow  
                specular=0.5,                      # some highlight  
                roughness=0.0,                     # shiny → brighter  
                metallic=0.0,  
                transmission=0.0  
            )  
        )    
  
        # Build and attach to mug    
        marker = builder.build(name="contact_marker")    
        
        # Set marker pose relative to mug    
        marker_pose = sapien.Pose(p=contact_point_local)    
        marker.set_pose(marker_pose)    
            
        # Store reference to update position each frame    
        self.contact_marker = marker  


    def _update_render(self):    
        """Override to update marker position"""    
        # Update marker to follow mug    
        
        if hasattr(self, 'contact_marker') and hasattr(self, 'mug'):    
            contact_point_local = self.mug.get_contact_point(self.contact_point_id, "pose").p  
            marker_pose = sapien.Pose(p=contact_point_local)    
            self.contact_marker.set_pose(marker_pose)  
        self._update_mug_axes_pose()
        super()._update_render()    
  
    def play_once(self):  
        # Determine which arm to use based on mug position  
        mug_pose = self.mug.get_pose().p  
        arm_tag = ArmTag("left" if mug_pose[0] < 0 else "right")  
  
        # Grasp the mug  
        self.move(self.grasp_actor(self.mug, arm_tag=arm_tag, pre_grasp_dis=0.05,contact_point_id=self.contact_point_id))
          
        # Lift the mug up  
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))  
          
        self.info["info"] = {
            "{A}": f"039_mug/base{self.mug_id}",
            "{a}": str(arm_tag),
        }
        return self.info
  
    def check_success(self):
        # Get current mug functional point pose (world coordinates)
        mug_pose = self.mug.get_functional_point(0, "pose").p

        # --- Early failure condition (similar style to your bottle code) ---
        # if mug_pose[2] < self.mug_init_height - 0.02:
        #     # Optional: you can track a flag like in your bottle code
        #     # self.actor_pose = False
        #     return False

        # --- Main success condition: lifted by enough height ---
        # How much higher is the mug compared to where it started?
        dz = mug_pose[2] - self.mug_init_height

        # Require that the mug is lifted at least 7cm (you can tune this)
        required_lift = 0.05  # 7 cm

        # Optionally, you could also require that it's not moved too far horizontally,
        # but for a pure pick-up task, height is usually enough.
        return dz > required_lift