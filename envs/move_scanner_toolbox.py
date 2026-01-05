from ._base_task import Base_Task  
from .utils import *  
import sapien  
from ._GLOBAL_CONFIGS import *  
import imgaug.augmenters as iaa  
import math
from sapien import Pose
import random

  
class move_scanner_toolbox(Base_Task):  
  
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
    def build_toolbox_2x2(self, task,
                        center=(0.18, 0.10, 0.76),
                        outer_xy=(0.22, 0.16),
                        base_thk=0.01,
                        wall_thk=0.008,
                        wall_h=0.06,
                        color=(0.2, 0.2, 0.2),
                        cell_floor_color=(0.25, 0.25, 0.25),
                        name_prefix="toolbox"):
        cx, cy, cz = center
        L, W = outer_xy

        parts = []
        cell_floors = []

        # ---- base plate ----
        base_half = (L/2, W/2, base_thk/2)
        base_pose = sapien.Pose([cx, cy, cz + base_thk/2], [1, 0, 0, 0])
        parts.append(create_box(
            scene=task,
            pose=base_pose,
            half_size=base_half,
            color=color,
            name=f"{name_prefix}_base",
            is_static=True,
        ))

        wall_zc = cz + base_thk + wall_h/2

        # ---- outer walls ----
        side_half = (wall_thk/2, W/2, wall_h/2)
        for tag, x in [("L", cx - L/2 + wall_thk/2), ("R", cx + L/2 - wall_thk/2)]:
            parts.append(create_box(
                scene=task,
                pose=sapien.Pose([x, cy, wall_zc], [1, 0, 0, 0]),
                half_size=side_half,
                color=color,
                name=f"{name_prefix}_wall_{tag}",
                is_static=True,
            ))

        fb_half = (L/2, wall_thk/2, wall_h/2)
        for tag, y in [("F", cy - W/2 + wall_thk/2), ("B", cy + W/2 - wall_thk/2)]:
            parts.append(create_box(
                scene=task,
                pose=sapien.Pose([cx, y, wall_zc], [1, 0, 0, 0]),
                half_size=fb_half,
                color=color,
                name=f"{name_prefix}_wall_{tag}",
                is_static=True,
            ))

        # ---- inner dividers ----
        # ✅ REMOVE the horizontal divider (y=cy) to merge top+bottom
        # parts.append(create_box(... name=f"{name_prefix}_div_y"...))

        # ✅ KEEP ONLY the vertical divider (x=cx): split into left/right
        parts.append(create_box(
            scene=task,
            pose=sapien.Pose([cx, cy, wall_zc], [1, 0, 0, 0]),
            half_size=(wall_thk/2, W/2 - wall_thk, wall_h/2),
            color=color,
            name=f"{name_prefix}_div_x",
            is_static=True,
        ))

        # ---- define 2 cell regions + create 2 floor tiles ----
        floor_z = cz + base_thk
        margin = 0.01

        # Each cell is half in x, full in y (minus walls)
        cell_L = L/2 - wall_thk   # interior half-length in x
        cell_W = W - 2 * wall_thk # full interior width in y

        # tile (anchor) thickness
        tile_thk = min(0.004, base_thk * 0.6)
        tile_half = (cell_L/2 - margin, cell_W/2 - margin, tile_thk/2)
        tile_zc = floor_z + tile_thk/2 + 1e-4

        # centers for 2 cells: left / right
        x_offsets = [-L/4, +L/4]

        cell_aabbs = []
        cell_centers = []

        for cell_id, xo in enumerate(x_offsets):
            cxy = np.array([cx + xo, cy])

            mn = np.array([cxy[0] - cell_L/2 + margin,
                        cxy[1] - cell_W/2 + margin,
                        floor_z])
            mx = np.array([cxy[0] + cell_L/2 - margin,
                        cxy[1] + cell_W/2 - margin,
                        floor_z + wall_h])
            cell_aabbs.append((mn, mx))

            cell_centers.append(np.array([cxy[0], cxy[1], floor_z + 0.03]))

            tile_pose = sapien.Pose([cxy[0], cxy[1], tile_zc], [1, 0, 0, 0])
            tile = create_box(
                scene=task,
                pose=tile_pose,
                half_size=tile_half,
                color=cell_floor_color,
                name=f"{name_prefix}_cell_floor_{cell_id}",
                is_static=True,
            )
            cell_floors.append(tile)

        parts.extend(cell_floors)

        return parts, cell_aabbs, cell_centers, cell_floors
    # def build_toolbox_2x2(self, task,
    #                     center=(0.18, 0.10, 0.76),
    #                     outer_xy=(0.22, 0.16),
    #                     base_thk=0.01,
    #                     wall_thk=0.008,
    #                     wall_h=0.06,
    #                     color=(0.2, 0.2, 0.2),
    #                     cell_floor_color=(0.25, 0.25, 0.25),
    #                     name_prefix="toolbox"):
    #     cx, cy, cz = center
    #     L, W = outer_xy

    #     parts = []
    #     cell_floors = []  # ✅ new: 4 tiles

    #     # ---- base plate (keep a single plate for stable collision) ----
    #     base_half = (L/2, W/2, base_thk/2)
    #     base_pose = sapien.Pose([cx, cy, cz + base_thk/2], [1, 0, 0, 0])
    #     parts.append(create_box(
    #         scene=task,
    #         pose=base_pose,
    #         half_size=base_half,
    #         color=color,
    #         name=f"{name_prefix}_base",
    #         is_static=True,
    #     ))

    #     # Common z center for walls
    #     wall_zc = cz + base_thk + wall_h/2

    #     # ---- outer walls ----
    #     side_half = (wall_thk/2, W/2, wall_h/2)
    #     for tag, x in [("L", cx - L/2 + wall_thk/2), ("R", cx + L/2 - wall_thk/2)]:
    #         parts.append(create_box(
    #             scene=task,
    #             pose=sapien.Pose([x, cy, wall_zc], [1, 0, 0, 0]),
    #             half_size=side_half,
    #             color=color,
    #             name=f"{name_prefix}_wall_{tag}",
    #             is_static=True,
    #         ))

    #     fb_half = (L/2, wall_thk/2, wall_h/2)
    #     for tag, y in [("F", cy - W/2 + wall_thk/2), ("B", cy + W/2 - wall_thk/2)]:
    #         parts.append(create_box(
    #             scene=task,
    #             pose=sapien.Pose([cx, y, wall_zc], [1, 0, 0, 0]),
    #             half_size=fb_half,
    #             color=color,
    #             name=f"{name_prefix}_wall_{tag}",
    #             is_static=True,
    #         ))

    #     # ---- inner dividers ----
    #     parts.append(create_box(
    #         scene=task,
    #         pose=sapien.Pose([cx, cy, wall_zc], [1, 0, 0, 0]),
    #         half_size=(L/2 - wall_thk, wall_thk/2, wall_h/2),
    #         color=color,
    #         name=f"{name_prefix}_div_y",
    #         is_static=True,
    #     ))
    #     parts.append(create_box(
    #         scene=task,
    #         pose=sapien.Pose([cx, cy, wall_zc], [1, 0, 0, 0]),
    #         half_size=(wall_thk/2, W/2 - wall_thk, wall_h/2),
    #         color=color,
    #         name=f"{name_prefix}_div_x",
    #         is_static=True,
    #     ))

    #     # ---- define 4 cell regions + create 4 floor tiles ----
    #     cell_L = L/2 - wall_thk
    #     cell_W = W/2 - wall_thk
    #     floor_z = cz + base_thk

    #     x_offsets = [-L/4, +L/4]
    #     y_offsets = [-W/4, +W/4]

    #     cell_aabbs = []
    #     cell_centers = []
    #     margin = 0.01

    #     # ✅ tile should not intersect walls; slightly smaller than interior
    #     tile_thk = min(0.004, base_thk * 0.6)     # thin visual/anchor layer
    #     tile_half = (cell_L/2 - margin, cell_W/2 - margin, tile_thk/2)
    #     tile_zc = floor_z + tile_thk/2 + 1e-4     # tiny offset to avoid z-fighting / penetration

    #     cell_id = 0
    #     for xo in x_offsets:
    #         for yo in y_offsets:
    #             cxy = np.array([cx + xo, cy + yo])

    #             mn = np.array([cxy[0] - cell_L/2 + margin,
    #                         cxy[1] - cell_W/2 + margin,
    #                         floor_z])
    #             mx = np.array([cxy[0] + cell_L/2 - margin,
    #                         cxy[1] + cell_W/2 - margin,
    #                         floor_z + wall_h])
    #             cell_aabbs.append((mn, mx))

    #             cell_centers.append(np.array([cxy[0], cxy[1], floor_z + 0.03]))

    #             # ✅ create a per-cell floor tile actor (static)
    #             tile_pose = sapien.Pose([cxy[0], cxy[1], tile_zc], [1, 0, 0, 0])
    #             tile = create_box(
    #                 scene=task,
    #                 pose=tile_pose,
    #                 half_size=tile_half,
    #                 color=cell_floor_color,
    #                 name=f"{name_prefix}_cell_floor_{cell_id}",
    #                 is_static=True,
    #             )
    #             cell_floors.append(tile)

    #             cell_id += 1

    #     # parts doesn't include tiles by default; you can if you want:
    #     parts.extend(cell_floors)

    #     # ✅ return tiles too
    #     return parts, cell_aabbs, cell_centers, cell_floors

    def sample_toolbox_pose_opposite(self,
                                    obj_pose_p,
                                    z=0.76,
                                    y_lim=(-0.1, 0.1),
                                    x_band=(0.18, 0.22),
                                    min_dist=0.12,
                                    max_tries=200):
        """
        Sample a toolbox pose on the opposite side of the object (by x sign),
        using rejection sampling with a minimum XY distance.

        obj_pose_p: object world position (np.array-like length 3)
        Returns: sapien.Pose
        """
        obj_p = np.array(obj_pose_p, dtype=float)

        # opposite side in x
        if obj_p[0] > 0:
            xlim = [-x_band[1], -x_band[0]]  # [-0.22, -0.18]
            print("[debug] -x")
        else:
            xlim = [ x_band[0],  x_band[1]]  # [ 0.18,  0.22]

        for _ in range(max_tries):
            # sample in the band
            x = np.random.uniform(xlim[0], xlim[1])
            y = np.random.uniform(y_lim[0], y_lim[1])

            # min XY distance constraint
            if np.hypot(x - obj_p[0], y - obj_p[1]) >= min_dist:
                return sapien.Pose([x, y, z], [1, 0, 0, 0])

        # fallback: if constraints too strict, just return center of band
        x = 0.5 * (xlim[0] + xlim[1])
        y = 0.5 * (y_lim[0] + y_lim[1])
        return sapien.Pose([x, y, z], [1, 0, 0, 0])


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

        obj_pos = self.scanner.get_pose().p
        toolbox_pose = self.sample_toolbox_pose_opposite(obj_pos, z=0.76)
        self.parts, self.cell_aabbs, self.cell_centers, self.cell_floors = self.build_toolbox_2x2(
            task=self,
            center=(toolbox_pose.p[0], toolbox_pose.p[1], toolbox_pose.p[2]),     # choose a spot on table
            outer_xy = (0.25,0.4),
            wall_h=0.06,
        )
        # idx = random.randint(0, 3)
        # print("[debug] Slot:",idx)
        
        # idx = np.random.choice([i for i in range(4)])
        idx = 0
        self.pad = self.cell_floors[idx]
  
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
        arm_tag = ArmTag("right" if scanner_pose[0] < 0 else "left")  
  
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
        print(target_pose)
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
        
        return np.all(abs(scanner_pose[:3] - target_pos[:3]) < np.array([eps, eps, eps]))