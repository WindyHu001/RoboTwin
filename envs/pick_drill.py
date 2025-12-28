from ._base_task import Base_Task  
from .utils import *  
import sapien  
from ._GLOBAL_CONFIGS import *  
import imgaug.augmenters as iaa    
import math


  
  
class pick_drill(Base_Task):  
  
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

        q_x_90  = self.quat_from_axis_angle((0, 0, 1),  90)
        self.drill_q = q_x_90

        # self.drill = create_actor(  
        #     scene=self,  
        #     pose=sapien.Pose([0, -0.06, 0.760], self.drill_q),  
        #     modelname="030_drill",  
        #     convex=True,  
        #     model_id=0,  
        # )  
        self.drill = rand_create_actor(  
            scene=self,  
            modelname="030_drill", 
            model_id=1,  
            xlim=[-0.05, 0.05],  
            ylim=[-0.05, 0.05],  
            zlim=[0.75,0.76],
            # rotate_rand=False,  
            qpos=q_x_90,  
        )  
        pose = self.drill.get_pose()
        T = pose.to_transformation_matrix()
        R = T[:3, :3]  # 3x3 rotation part
        self._create_drill_axes()
        print("local x axis in world:", R[:, 0])
        print("local y axis in world:", R[:, 1])
        print("local z axis in world:", R[:, 2])
        # self.drill.set_mass(0.1)  
        self.add_prohibit_area(self.drill, padding=0.10)  
        
        drill_fp0 = self.drill.get_functional_point(0, "pose").p
        self.drill_init_height = float(drill_fp0[2])
        drill_pos = self.drill.get_pose().p
        print(drill_pos)     
        self.add_contact_marker_to_drill()  
        self.scene.step()    
        self.scene.update_render()         

    def _create_drill_axes(self):
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

        self.drill_axes = builder.build(name="drill_axes")
        self._update_drill_axes_pose()



    def _update_drill_axes_pose(self):
        if not hasattr(self, "drill") or not hasattr(self, "drill_axes"):
            return
        drill_pose = self.drill.get_pose()
        self.drill_axes.set_pose(drill_pose) 

    def add_contact_marker_to_drill(self):    
        """Attach a bright colored sphere to the drill's contact point"""    
        if not hasattr(self, 'drill'):    
            return    
          
        # Get contact point position relative to drill    
        contact_point_local = self.drill.get_contact_point(0, "pose").p  
          
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
  
        # Build and attach to drill    
        marker = builder.build(name="contact_marker")    
        
        # Set marker pose relative to drill    
        marker_pose = sapien.Pose(p=contact_point_local)    
        marker.set_pose(marker_pose)    
            
        # Store reference to update position each frame    
        self.contact_marker = marker  


    def _update_render(self):    
        """Override to update marker position"""    
        # Update marker to follow drill    
        if hasattr(self, 'contact_marker') and hasattr(self, 'drill'):    
            contact_point_local = self.drill.get_contact_point(0, "pose").p  
            marker_pose = sapien.Pose(p=contact_point_local)    
            self.contact_marker.set_pose(marker_pose)  
        self._update_drill_axes_pose()
        super()._update_render()    
  
    def play_once(self):  
        # Determine which arm to use based on drill position  
        drill_pose = self.drill.get_pose().p  
        arm_tag = ArmTag("left" if drill_pose[0] < 0 else "right")  
  
        # Grasp the drill  
        self.move(self.grasp_actor(self.drill, arm_tag=arm_tag, pre_grasp_dis=0.1))
          
        # Lift the drill up  
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))  
          
        self.info["info"] = {
            "{A}": "030_drill/base0",
            "{a}": str(arm_tag),
        }
        return self.info
  
    def check_success(self):
        # Get current drill functional point pose (world coordinates)
        drill_pose = self.drill.get_functional_point(0, "pose").p

        # --- Early failure condition (similar style to your bottle code) ---
        # If the drill is *lower* than its initial height minus a small tolerance,
        # we treat it as failure (fell off the table / not picked at all).
        if drill_pose[2] < self.drill_init_height - 0.02:
            # Optional: you can track a flag like in your bottle code
            # self.actor_pose = False
            return False

        # --- Main success condition: lifted by enough height ---
        # How much higher is the drill compared to where it started?
        dz = drill_pose[2] - self.drill_init_height

        # Require that the drill is lifted at least 7cm (you can tune this)
        required_lift = 0.05  # 7 cm

        # Optionally, you could also require that it's not moved too far horizontally,
        # but for a pure pick-up task, height is usually enough.
        return dz > required_lift