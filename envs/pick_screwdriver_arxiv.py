from ._base_task import Base_Task  
from .utils import *  
import sapien  
from ._GLOBAL_CONFIGS import *  

class pick_screwdriver(Base_Task):  
    def setup_demo(self, **kwargs):  
        super()._init_task_env_(**kwargs)  
  
    def load_actors(self):  
        # self.screwdriver = create_actor(  
        #     scene=self,  
        #     pose=sapien.Pose([0, -0.06, 0.66], [0, 0, 0.995, 0.105]),  
        #     modelname="032_screwdriver",  
        #     convex=True,  
        #     model_id=0,  
        # )  

        # True
        # self.screwdriver = create_actor(  
        #     scene=self,  
        #     pose=sapien.Pose(  
        #         p=[0.193685, -0.0614598, 0.741],  # Position from your debug output  
        #         q=[0, 0, 0.995, 0.105]  # Quaternion from your debug output  
        #     ),  
        #     modelname="032_screwdriver",  
        #     convex=True,  
        #     model_id=0,  
        # )

        # self.screwdriver = create_actor(  
        #     scene=self,  
        #     pose=sapien.Pose(  
        #         p=[-0.0236761, -0.18805, 0.741],  # Position from your debug output  
        #         q=[0, 0, 0.995, 0.105]  # Quaternion from your debug output  
        #     ),  
        #     modelname="032_screwdriver_test",  
        #     convex=True,  
        #     model_id=0,  
        # )
        # True
        # self.screwdriver = create_actor(  
        #     scene=self,  
        #     pose=sapien.Pose(  
        #         p=[0.0295476, -0.0589536, 0.741],  # Position from your debug output  
        #         q=[0, 0, 0.995, 0.105]  # Quaternion from your debug output  
        #     ),  
        #     modelname="032_screwdriver",  
        #     convex=True,  
        #     model_id=0,  
        # )

        # screwdriver
        # self.screwdriver = create_actor(  
        #     scene=self,  
        #     pose=sapien.Pose(  
        #         p=[0, -0.06, 0.741],  # Position from your debug output  
        #         q=[0, 0, 0.995, 0.105]  # Quaternion from your debug output  
        #     ),  
        #     modelname="032_screwdriver",  
        #     convex=True,  
        #     model_id=0,  
        # )

        # self.screwdriver = rand_create_actor(    
        #     scene=self,    
        #     modelname="032_screwdriver",    
        #     model_id=0,    
        #     xlim=[-0.2, 0.2],    
        #     ylim=[-0.2, 0.2],    
        #     zlim = [0.741,0.741],
        #     rotate_rand=True,  # This is False  
        #     rotate_lim=[1, 0, 0],
        #     qpos=[1, 0, 0, 0],  # Fixed quaternion  
        # )
        # Set screwdriver mass and add visual marker 
        screwdriver_pos = self.screwdriver.get_pose().p
        print(screwdriver_pos) 
        self.screwdriver.set_mass(0.1)  
        self.add_contact_marker_to_screwdriver()  
        ee_r = np.array(self.robot.get_right_ee_pose()[:3])
        dist_r0 = np.linalg.norm(ee_r[:2] - screwdriver_pos[:2])
        self._init_dist_xy = dist_r0
        print("init ee_r:",ee_r)
        screwdriver_fp0 = self.screwdriver.get_functional_point(0, "pose").p
        self.screwdriver_init_height = float(screwdriver_fp0[2])
        print("init_dist:",self._init_dist_xy)
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
        self.move(self.grasp_actor(self.screwdriver, arm_tag=arm_tag, pre_grasp_dis=0.1))  
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.2, move_axis="arm"))  
        # Store info for instruction generation  
        self.info["info"] = {  
            "{A}": f"032_screwdriver/base0",  
            "{a}": str(arm_tag),  
        }  
        return self.info  
          
    # def check_success(self):  
    #     # screwdriver_pos = self.screwdriver.get_pose().p  
    #     # print(screwdriver_pos[2])
    #     # # # Check if screwdriver is lifted to sufficient height  
    #     # height_ok = screwdriver_pos[2] > 0.73
        
    #     # # Check if right gripper is closed (holding the object)  
    #     # gripper_closed = self.is_right_gripper_close()  
        
    #     # # Check if there's contact between gripper and screwdriver  
    #     # contacts = self.get_gripper_actor_contact_position("032_screwdriver_test")  
    #     # has_contact = len(contacts) > 0  
    #     # screwdriver_pos = self.screwdriver.get_pose().p
    #     # screwdriver_pos_all = self.screwdriver.get_pose()
    #     # ee_pose = self.robot.get_right_ee_pose()
    #     # # import pdb;pdb.set_trace()
    #     # ee_pos = np.array(ee_pose[:3])

    #     # dist_xy = np.linalg.norm(ee_pos[:2] - screwdriver_pos[:2])

    #     # print("screwdriver pos:", screwdriver_pos, "screwdriver_pos_all:",screwdriver_pos_all, "ee pos:", ee_pos, "dist_xy:", dist_xy)
    #     # return height_ok

    #     screwdriver_pos = self.screwdriver.get_pose().p

    #     ee_r = np.array(self.robot.get_right_ee_pose()[:3])
    #     ee_l = np.array(self.robot.get_left_ee_pose()[:3])

    #     dist_r = np.linalg.norm(ee_r[:2] - screwdriver_pos[:2])
    #     dist_l = np.linalg.norm(ee_l[:2] - screwdriver_pos[:2])

    #     dist_xy = dist_r

    #     print(f"[CHECK] screw pos: {screwdriver_pos}, "
    #         f"dist_r: {dist_r:.3f}, dist_l: {dist_l:.3f}, min: {dist_xy:.3f}, "
    #         f"init: {getattr(self, '_init_dist_xy', None)}")

    #     if not hasattr(self, "_init_dist_xy") or self._init_dist_xy is None:
    #         return False

    #     init_dist = self._init_dist_xy
    #     print("ee_r_check_suc:",ee_r)
    #     print("init_dist_check_suc:",init_dist)
    #     print("dist_xy_check_suc:",dist_xy)
    #     improved = dist_xy < init_dist - 0.1
    #     print("improved_suc:",improved)
    #     close_enough = dist_xy < 0.02

    #     return improved and close_enough
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