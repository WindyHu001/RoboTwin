from ._base_task import Base_Task  
from .utils import *  
import sapien  
from ._GLOBAL_CONFIGS import *  
import imgaug.augmenters as iaa    
  
  
class pick_hammer(Base_Task):  
  
    def setup_demo(self, **kwags):  
        super()._init_task_env_(**kwags)  
  
    def load_actors(self):  
        # self.hammer = create_actor(  
        #     scene=self,  
        #     pose=sapien.Pose([0, -0.06, 0.783], [0, 0, 0.995, 0.105]),  
        #     modelname="020_hammer_eval_bottom",  
        #     convex=True,  
        #     model_id=0,  
        # )  
        self.hammer = rand_create_actor(
            self,
            xlim=[-0.1, 0.1],
            ylim=[-0.1, 0.1],
            zlim=[0.783,0.783],
            modelname="020_hammer_train_mid",
            rotate_rand=True,
            rotate_lim=[0, 1, 0],
            qpos=[0, 0, 0.995, 0.105],
            convex=True,
        )

        self.hammer.set_mass(0.01)  
        self.add_prohibit_area(self.hammer, padding=0.10)  

        hammer_fp0 = self.hammer.get_functional_point(0, "pose").p
        self.hammer_init_height = float(hammer_fp0[2])
        hammer_pos = self.hammer.get_pose().p
        print(hammer_pos)     
        self.add_contact_marker_to_hammer()  
        self.scene.step()    
        self.scene.update_render()            
  
    def add_contact_marker_to_hammer(self):    
        """Attach a bright colored sphere to the hammer's contact point"""    
        if not hasattr(self, 'hammer'):    
            return    
          
        # Get contact point position relative to hammer    
        contact_point_local = self.hammer.get_contact_point(0, "pose").p  
          
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
  
        # Build and attach to hammer    
        marker = builder.build(name="contact_marker")    
        
        # Set marker pose relative to hammer    
        marker_pose = sapien.Pose(p=contact_point_local)    
        marker.set_pose(marker_pose)    
            
        # Store reference to update position each frame    
        self.contact_marker = marker  


    def _update_render(self):    
        """Override to update marker position"""    
        # Update marker to follow hammer    
        if hasattr(self, 'contact_marker') and hasattr(self, 'hammer'):    
            contact_point_local = self.hammer.get_contact_point(0, "pose").p  
            marker_pose = sapien.Pose(p=contact_point_local)    
            self.contact_marker.set_pose(marker_pose)  
          
        super()._update_render()    
  
    def play_once(self):  
        # Determine which arm to use based on hammer position  
        hammer_pose = self.hammer.get_pose().p  
        arm_tag = ArmTag("left" if hammer_pose[0] < 0 else "right")  
  
        # Grasp the hammer  
        self.move(self.grasp_actor(self.hammer, arm_tag=arm_tag, pre_grasp_dis=0.12, grasp_dis=0.01,contact_point_id=0))
          
        # Check if grasp succeeded  
        # if not self.plan_success:  
        #     self.plan_success = True  
        #     self.info["info"] = {"{A}": "020_hammer/base0", "{a}": str(arm_tag)}  
        #     return self.info  
          
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
        required_lift = 0.05  # 7 cm

        # Optionally, you could also require that it's not moved too far horizontally,
        # but for a pure pick-up task, height is usually enough.
        return dz > required_lift