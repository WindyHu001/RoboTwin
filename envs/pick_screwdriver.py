from ._base_task import Base_Task  
from .utils import *  
import sapien  
from ._GLOBAL_CONFIGS import *  
import numpy as np
import math

class pick_screwdriver(Base_Task):  
    def setup_demo(self, **kwargs):  
        super()._init_task_env_(**kwargs)  

    def quat_from_axis_angle(self, axis, angle_deg):
        """Return [w, x, y, z] given axis=(ax,ay,az) and angle(deg)."""
        ax, ay, az = axis
        theta = math.radians(angle_deg)
        half = theta * 0.5
        s = math.sin(half)
        c = math.cos(half)
        return [c, ax * s, ay * s, az * s]
  
    def load_actors(self):  
        x = np.random.uniform(-0.25, 0.25)
        y = np.random.uniform(-0.1, 0.1)
        z = 0.741  

        q = [0, 0, 0.995, 0.105]
        self.screwdriver_q = self.quat_from_axis_angle((0, 1, 0), 90)

        # self.screwdriver = create_actor(
        #     scene=self,
        #     pose=sapien.Pose(
        #         p=[x, y, z],
        #         q=self.screwdriver_q,
        #     ),
        #     modelname="032_screwdriver_train_bottom",
        #     convex=True,
        #     model_id=0,
        # )
        self.screwdriver = rand_create_actor(  
            self,  
            xlim=[-0.25, 0.25],  
            ylim=[-0.1, 0.1],  
            modelname="032_screwdriver_train_mid",  
            rotate_rand=True,  
            rotate_lim=[0, 1, 0],  
            qpos=self.screwdriver_q,  
            convex=True,  
        )  
        self.add_prohibit_area(self.screwdriver, padding=0.10) 
        screwdriver_fp0 = self.screwdriver.get_functional_point(0, "pose").p
        self.screwdriver_init_height = float(screwdriver_fp0[2])
        # Set screwdriver mass and add visual marker 
        screwdriver_pos = self.screwdriver.get_pose().p
        print(screwdriver_pos) 
        self.add_contact_marker_to_screwdriver()  

        self.scene.step()    
        self.scene.update_render()  
  
    def add_contact_marker_to_screwdriver(self):    
        """Attach a bright colored sphere to the screwdriver's contact point"""    
        if not hasattr(self, 'screwdriver'):    
            
            return    
        # Get contact point position relative to screwdriver    
        contact_point_local = self.screwdriver.get_contact_point(0, "pose").p  
          
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
        
        # Build and attach to screwdriver    
        marker = builder.build(name="contact_marker")    
            
        # Set marker pose relative to screwdriver    
        marker_pose = sapien.Pose(p=contact_point_local)    
        marker.set_pose(marker_pose)    
            
        # Store reference to update position each frame    
        self.contact_marker = marker  
        
    def _update_render(self):    
        """Override to update marker position"""    
        # Update marker to follow screwdriver    
        if hasattr(self, 'contact_marker') and hasattr(self, 'screwdriver'):    
            contact_point_local = self.screwdriver.get_contact_point(0, "pose").p  
            marker_pose = sapien.Pose(p=contact_point_local)    
            self.contact_marker.set_pose(marker_pose)  
  
        super()._update_render()    
  
    def play_once(self):  
        # Determine which arm to use based on screwdriver position  
        screwdriver_pose = self.screwdriver.get_pose().p  
        arm_tag = ArmTag("left" if screwdriver_pose[0] < 0 else "right")  
        # import pdb;pdb.set_trace()
        # Pick up the screwdriver  
        self.move(self.grasp_actor(self.screwdriver, arm_tag=arm_tag, pre_grasp_dis=0.1,grasp_dis=0.01,contact_point_id=1))  
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.2, move_axis="arm"))  
        # Store info for instruction generation  
        self.info["info"] = {  
            "{A}": f"032_screwdriver/base0",  
            "{a}": str(arm_tag),  
        }  
        return self.info  

    def check_success(self):
        # Get current screwdriver functional point pose (world coordinates)
        screwdriver_pose = self.screwdriver.get_functional_point(0, "pose").p

        # --- Early failure condition (similar style to your bottle code) ---
        # If the screwdriver is *lower* than its initial height minus a small tolerance,
        # we treat it as failure (fell off the table / not picked at all).
        if screwdriver_pose[2] < self.screwdriver_init_height - 0.02:
            # Optional: you can track a flag like in your bottle code
            # self.actor_pose = False
            return False

        # --- Main success condition: lifted by enough height ---
        # How much higher is the screwdriver compared to where it started?
        dz = screwdriver_pose[2] - self.screwdriver_init_height

        # Require that the screwdriver is lifted at least 7cm (you can tune this)
        required_lift = 0.05  # 7 cm

        # Optionally, you could also require that it's not moved too far horizontally,
        # but for a pure pick-up task, height is usually enough.
        return dz > required_lift