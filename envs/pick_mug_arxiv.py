from ._base_task import Base_Task  
from .utils import *  
import sapien  
from ._GLOBAL_CONFIGS import *  
import numpy as np

class pick_mug(Base_Task):  
    def setup_demo(self, **kwargs):  
        super()._init_task_env_(**kwargs)  
  
    def load_actors(self):  
        self.mug_id = np.random.choice([i for i in range(10)])          
        self.mug = rand_create_actor(
            self,
            xlim=[-0.1, 0.1],
            ylim=[-0.05, 0.05],
            # ylim_prop=True,
            modelname="039_mug_train_handle",
            # rotate_rand=True,
            # rotate_lim=[0, 1.57, 0],
            # qpos=[0.707, 0.707, 0, 0],
            convex=True,
            model_id=self.mug_id,
        )
        mug_fp0 = self.mug.get_functional_point(0, "pose").p
        self.mug_init_height = float(mug_fp0[2])
        # Set mug mass and add visual marker 
        mug_pos = self.mug.get_pose().p
        print(mug_pos) 
        self.add_contact_marker_to_mug()  

        self.scene.step()    
        self.scene.update_render()  
  
    def add_contact_marker_to_mug(self):    
        """Attach a bright colored sphere to the mug's contact point"""    
        if not hasattr(self, 'mug'):    
            
            return    
        # Get contact point position relative to mug    
        contact_point_local = self.mug.get_contact_point(0, "pose").p  
          
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
            contact_point_local = self.mug.get_contact_point(0, "pose").p  
            marker_pose = sapien.Pose(p=contact_point_local)    
            self.contact_marker.set_pose(marker_pose)  
  
        super()._update_render()    
  
    def play_once(self):  
        # Determine which arm to use based on mug position  
        mug_pose = self.mug.get_pose().p  
        # arm_tag = ArmTag("left" if mug_pose[0] < 0 else "right")  
        arm_tag = ArmTag("right")  
        # import pdb;pdb.set_trace()
        # Pick up the mug  
        self.move(self.grasp_actor(self.mug, arm_tag=arm_tag, pre_grasp_dis=0.1,contact_point_id=6))  
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.2, move_axis="arm"))  
        # Store info for instruction generation  
        self.info["info"] = {  
            "{A}": f"032_mug/base0",  
            "{a}": str(arm_tag),  
        }  
        return self.info  

    def check_success(self):
        # Get current mug functional point pose (world coordinates)
        mug_pose = self.mug.get_functional_point(0, "pose").p

        # --- Early failure condition (similar style to your bottle code) ---
        # If the mug is *lower* than its initial height minus a small tolerance,
        # we treat it as failure (fell off the table / not picked at all).
        if mug_pose[2] < self.mug_init_height - 0.02:
            # Optional: you can track a flag like in your bottle code
            # self.actor_pose = False
            return False

        # --- Main success condition: lifted by enough height ---
        # How much higher is the mug compared to where it started?
        dz = mug_pose[2] - self.mug_init_height

        # Require that the mug is lifted at least 7cm (you can tune this)
        required_lift = 0.05  # 7 cm

        # Optionally, you could also require that it's not moved too far horizontally,
        # but for a pure pick-up task, height is usually enough.
        return dz > required_lift