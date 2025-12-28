from ._base_task import Base_Task
from .utils import *
import sapien
from ._GLOBAL_CONFIGS import *
import imgaug.augmenters as iaa
import math
import numpy as np


class move_screwdriver_pad(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    # -------------------------
    # Math helpers
    # -------------------------
    def quat_from_axis_angle(self, axis, angle_deg):
        """Return [w, x, y, z] given axis=(ax,ay,az) and angle(deg)."""
        ax, ay, az = axis
        theta = math.radians(angle_deg)
        half = theta * 0.5
        s = math.sin(half)
        c = math.cos(half)
        return [c, ax * s, ay * s, az * s]

    # -------------------------
    # Visualization helpers
    # -------------------------
    def _make_material(self, rgba, emission=None):
        if emission is None:
            emission = rgba
        return sapien.render.RenderMaterial(
            base_color=list(rgba),
            emission=list(emission),
            specular=0.5,
            roughness=0.0,
            metallic=0.0,
            transmission=0.0,
        )

    def _create_axes_actor(self, name, axis_length=0.10, axis_radius=0.004):
        """
        Create a small RGB axis gizmo actor.
        IMPORTANT: capsule long axis is along local +X, so Y/Z need rotations.
        """
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("kinematic")

        q_I = [1.0, 0.0, 0.0, 0.0]  # identity [w,x,y,z]
        q_rot_z_90 = self.quat_from_axis_angle((0, 0, 1), 90)        # X->Y
        q_rot_y_neg90 = self.quat_from_axis_angle((0, -1, 0), 90)    # X->Z

        # X axis (red)
        builder.add_capsule_visual(
            radius=axis_radius,
            half_length=axis_length / 2,
            material=self._make_material([1, 0, 0, 1]),
            pose=sapien.Pose(p=[axis_length / 2, 0, 0], q=q_I),
        )
        # Y axis (green)
        builder.add_capsule_visual(
            radius=axis_radius,
            half_length=axis_length / 2,
            material=self._make_material([0, 1, 0, 1]),
            pose=sapien.Pose(p=[0, axis_length / 2, 0], q=q_rot_z_90),
        )
        # Z axis (blue)
        builder.add_capsule_visual(
            radius=axis_radius,
            half_length=axis_length / 2,
            material=self._make_material([0, 0, 1, 1]),
            pose=sapien.Pose(p=[0, 0, axis_length / 2], q=q_rot_y_neg90),
        )

        return builder.build(name=name)

    def _create_point_marker(self, name, radius=0.02, rgba=(1, 1, 0, 1)):
        """Create a kinematic sphere marker; you update its pose each frame."""
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("kinematic")
        builder.add_sphere_visual(radius=radius, material=self._make_material(list(rgba)))
        return builder.build(name=name)

    def _update_debug_frames(self):
        """Update FP/Target axes & dots to follow their world poses."""
        # Screwdriver FP1
        if hasattr(self, "screwdriver"):
            if hasattr(self, "screw_fp1_axes") or hasattr(self, "screw_fp1_dot"):
                screw_fp1_pose = self.screwdriver.get_functional_point(1, "pose")  # Pose in world
                if hasattr(self, "screw_fp1_axes"):
                    self.screw_fp1_axes.set_pose(screw_fp1_pose)
                if hasattr(self, "screw_fp1_dot"):
                    self.screw_fp1_dot.set_pose(sapien.Pose(p=screw_fp1_pose.p))

            if hasattr(self, "screw_fp0_axes") or hasattr(self, "screw_fp0_dot"):
                screw_fp0_pose = self.screwdriver.get_functional_point(0, "pose")
                if hasattr(self, "screw_fp0_axes"):
                    self.screw_fp0_axes.set_pose(screw_fp0_pose)
                if hasattr(self, "screw_fp0_dot"):
                    self.screw_fp0_dot.set_pose(sapien.Pose(p=screw_fp0_pose.p))

            if hasattr(self, "contact_marker"):
                c0_pose = self.screwdriver.get_contact_point(0, "pose")  # Pose in world
                self.contact_marker.set_pose(sapien.Pose(p=c0_pose.p))

        # Pad FP1 (target)
        if hasattr(self, "pad"):
            if hasattr(self, "target_axes") or hasattr(self, "target_dot"):
                pad_fp1_pose = self.pad.get_functional_point(1, "pose")  # Pose in world
                if hasattr(self, "target_axes"):
                    self.target_axes.set_pose(pad_fp1_pose)
                if hasattr(self, "target_dot"):
                    self.target_dot.set_pose(sapien.Pose(p=pad_fp1_pose.p))

    # -------------------------
    # Markers
    # -------------------------
    def add_contact_marker_to_screwdriver(self):
        """Attach a bright red sphere to screwdriver contact point 0 (visual debug)."""
        if not hasattr(self, "screwdriver"):
            return
        self.contact_marker = self._create_point_marker("contact_marker", radius=0.02, rgba=(1, 0, 0, 1))
        c0_pose = self.screwdriver.get_contact_point(0, "pose")  # world
        self.contact_marker.set_pose(sapien.Pose(p=c0_pose.p))

    def add_target_marker_to_pad(self):
        """Add a cyan sphere marker at pad FP1 position (visual debug)."""
        if not hasattr(self, "pad"):
            return
        self.target_dot = self._create_point_marker("target_dot", radius=0.02, rgba=(0, 1, 1, 1))
        pad_fp1_pose = self.pad.get_functional_point(1, "pose")
        self.target_dot.set_pose(sapien.Pose(p=pad_fp1_pose.p))

    def quat_to_euler_zyx(self,q):
        """
        Convert quaternion [w, x, y, z] to Euler angles (roll, pitch, yaw)
        using ZYX convention (yaw-pitch-roll).
        
        Returns angles in degrees.
        """
        w, x, y, z = q

        # roll (X axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # pitch (Y axis rotation)
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # gimbal lock
        else:
            pitch = math.asin(sinp)

        # yaw (Z axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return np.degrees([roll, pitch, yaw])
    # -------------------------
    # Scene update
    # -------------------------
    def _update_render(self):
        # Keep all debug visuals following objects each frame
        self._update_debug_frames()
        super()._update_render()

    # -------------------------
    # Main logic
    # -------------------------


    def load_actors(self):
        # Create screwdriver at random position
        x = np.random.uniform(-0.05, 0.05)
        y = np.random.uniform(-0.1, 0.1)
        z = 0.741

        # Lay it down: rotate around Y by +90 deg (wxyz)
        # self.screwdriver_q = self.quat_from_axis_angle((0, 1, 0), 90)

        def quat_from_axis_angle(axis, angle_deg):
            ax, ay, az = axis
            theta = math.radians(angle_deg)
            half = 0.5 * theta
            s = math.sin(half)
            c = math.cos(half)
            return np.array([c, ax*s, ay*s, az*s], dtype=np.float64)  # [w,x,y,z]

        def quat_mul(q1, q2):
            # Hamilton product, both in [w,x,y,z]
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            return np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ], dtype=np.float64)

        def quat_normalize(q):
            return q / np.linalg.norm(q)

        # 先 world-Y 90°，再 world-X 30°
        qy = quat_from_axis_angle((0, 1, 0), 90)
        qx = quat_from_axis_angle((1, 0, 0), 45)

        q_world = quat_normalize(quat_mul(qx,qy))

        self.screwdriver_q = q_world.tolist()

        self.screwdriver = create_actor(
            scene=self,
            pose=sapien.Pose(p=[x, y, z], q=self.screwdriver_q),
            modelname="032_screwdriver_train_mid_test",
            convex=True,
            model_id=0,
        )
        fp1 = self.screwdriver.get_functional_point(1, "pose")

        rpy = self.quat_to_euler_zyx(fp1.q)

        print("FP1 pose p:", fp1.p)
        print("FP1 pose q:", fp1.q)
        print("FP1 Euler (roll, pitch, yaw) [deg]:", rpy)

        c0 = self.screwdriver.get_contact_point(0, "pose")
        rpy_c0 = self.quat_to_euler_zyx(c0.q)

        print("C0 pose p:", c0.p)
        print("C0 pose q:", c0.q)
        print("C0 Euler (roll, pitch, yaw) [deg]:", rpy_c0)
        self.add_prohibit_area(self.screwdriver, padding=0.10)

        # Store initial height for success checking
        screwdriver_fp0 = self.screwdriver.get_functional_point(0, "pose").p
        self.screwdriver_init_height = float(screwdriver_fp0[2])

        def normalize(q):
            q = np.array(q, dtype=np.float64)
            return q / np.linalg.norm(q)

        def quat_inv(q):
            w, x, y, z = q
            return np.array([w, -x, -y, -z], dtype=np.float64) / np.dot(q, q)

        def quat_mul(q1, q2):
            # q = q1 ⊗ q2  (both wxyz)
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            return np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
            ], dtype=np.float64)

        def best_dot(q_pred, q_meas):
            q_pred = normalize(q_pred)
            q_meas = normalize(q_meas)
            return float(abs(np.dot(q_pred, q_meas)))  # abs handles q == -q

        def xyzw_to_wxyz(q):
            x, y, z, w = q
            return np.array([w, x, y, z], dtype=np.float64)

        def test_point(name, q_local_model, q_world_meas, q_actor_world):
            q_actor_world = normalize(q_actor_world)
            q_world_meas = normalize(q_world_meas)

            # assume model is wxyz
            q_local_wxyz = normalize(q_local_model)
            q_pred1 = quat_mul(q_actor_world, q_local_wxyz)
            score1 = best_dot(q_pred1, q_world_meas)

            # assume model is xyzw
            q_local_wxyz2 = normalize(xyzw_to_wxyz(q_local_model))
            q_pred2 = quat_mul(q_actor_world, q_local_wxyz2)
            score2 = best_dot(q_pred2, q_world_meas)

            print(f"\n[{name}]")
            print("  actor world q:", q_actor_world)
            print("  meas  world q:", q_world_meas)
            print("  model local q (raw):", q_local_model)
            print("  pred(world)=actor⊗local (model as wxyz):", normalize(q_pred1), " score:", score1)
            print("  pred(world)=actor⊗local (model as xyzw):", normalize(q_pred2), " score:", score2)

        # ---- use your actual runtime values ----
        q_actor = self.screwdriver.get_pose().q                      # world, wxyz
        q_fp1_world = self.screwdriver.get_functional_point(1, "pose").q
        q_c0_world  = self.screwdriver.get_contact_point(0, "pose").q

        # ---- fill in your model local definitions (raw numbers as stored) ----
        q_c0_model  = np.array([0.707, 0.0, -0.707, 0.0], dtype=np.float64)
        q_fp1_model = np.array([0.498, -0.504, 0.475, -0.522], dtype=np.float64)

        test_point("C0",  q_c0_model,  q_c0_world,  q_actor)
        test_point("FP1", q_fp1_model, q_fp1_world, q_actor)
        # Create pad pose
        screwdriver_pos = self.screwdriver.get_pose().p
        if screwdriver_pos[0] > 0:
            xlim = [0.15, 0.25]
        else:
            xlim = [-0.25, -0.15]

        target_rand_pose = rand_pose(
            xlim=xlim,
            ylim=[-0.2, 0.1],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )

        # Ensure minimum distance from screwdriver
        while np.sqrt((target_rand_pose.p[0] - screwdriver_pos[0]) ** 2 + (target_rand_pose.p[1] - screwdriver_pos[1]) ** 2) < 0.1:
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

        self.add_prohibit_area(self.pad, padding=0.15)

        # --- Debug visuals ---
        # Contact marker (red point at contact0)
        self.add_contact_marker_to_screwdriver()

        # Target marker (cyan sphere at pad FP1 position)
        self.add_target_marker_to_pad()

        # Axes for screw FP0 / FP1 and target FP1
        self.screw_fp0_axes = self._create_axes_actor("screw_fp0_axes", axis_length=0.10, axis_radius=0.004)
        self.screw_fp1_axes = self._create_axes_actor("screw_fp1_axes", axis_length=0.10, axis_radius=0.004)
        self.target_axes = self._create_axes_actor("target_axes", axis_length=0.12, axis_radius=0.004)

        # Dots for screw FP0 / FP1
        self.screw_fp0_dot = self._create_point_marker("screw_fp0_dot", radius=0.015, rgba=(1, 1, 0, 1))  # yellow
        self.screw_fp1_dot = self._create_point_marker("screw_fp1_dot", radius=0.015, rgba=(1, 0, 1, 1))  # magenta

        # Init update
        self._update_debug_frames()

        self.scene.step()
        self.scene.update_render()



    def _debug_print_alignment(self, tag=""):
        """Print positions/orientations of relevant frames."""
        pad_fp1 = self.pad.get_functional_point(1, "pose")
        screw_pose = self.screwdriver.get_pose()
        c0_pose = self.screwdriver.get_contact_point(0, "pose")
        fp0 = self.screwdriver.get_functional_point(0, "pose")
        fp1 = self.screwdriver.get_functional_point(1, "pose")

        dp_fp1 = pad_fp1.p - fp1.p
        dist_fp1 = float(np.linalg.norm(dp_fp1))

        dot_q = float(abs(np.dot(np.array(pad_fp1.q, dtype=np.float32), np.array(fp1.q, dtype=np.float32))))
        target_pose = self.pad.get_functional_point(1)  
        screw_fp = self.screwdriver.get_functional_point(1, "pose").q
        pad_fp1_pose = self.pad.get_functional_point(1, "pose")  # Pose(p,q)
        target_pose = sapien.Pose(p=pad_fp1_pose.p, q=screw_fp)

        print(f"\n[DEBUG {tag}]")
        print("  screw pose p:", screw_pose.p, " q:", screw_pose.q)
        print("  screw C0   p:", c0_pose.p, " q:", c0_pose.q)
        print("  screw FP0  p:", fp0.p, " q:", fp0.q)
        print("  screw FP1  p:", fp1.p, " q:", fp1.q)
        print("  pad   FP1  p:", pad_fp1.p, " q:", pad_fp1.q)
        print("  dp(fp1->target):", dp_fp1, " dist:", dist_fp1)
        print("  |dot(q_fp1, q_target)|:", dot_q, "(~1 means same orientation)")
        print("  actual target:",target_pose)

    def play_once(self):
        # Determine which arm to use based on screwdriver position
        screwdriver_pose = self.screwdriver.get_pose().p
        arm_tag = ArmTag("left" if screwdriver_pose[0] < 0 else "right")

        # Grasp
        self.move(self.grasp_actor(
            self.screwdriver,
            arm_tag=arm_tag,
            pre_grasp_dis=0.12,
            grasp_dis=0.01,
            contact_point_id=0,
            
        ))

        # Lift up
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))

        # Debug before placing
        

        # ---------------------------------------------------------
        # OPTION 1: Keep using place_actor (6DoF alignment) [often causes rotation]
        # ---------------------------------------------------------
        # If you still want to try "orientation = current FP1 orientation", do:
        

        target_pose = self.pad.get_functional_point(1)  
        screw_fp = self.screwdriver.get_functional_point(0, "pose").q
        pad_fp1_pose = self.pad.get_functional_point(1, "pose")  # Pose(p,q)
        target_pose = sapien.Pose(p=pad_fp1_pose.p, q=screw_fp)

        self._debug_print_alignment(tag="AFTER_GRASP_LIFT")
        # screw_fp1_pose = self.screwdriver.get_functional_point(1, "pose")
        # target_pose = sapien.Pose(p=pad_fp1_pose.p, q=screw_fp1_pose.q)

        # Visualize this target_pose as well (optional): move target_axes to this target_pose
        # (target_axes already follows pad_fp1_pose; uncomment if you want to see "forced target q" instead)
        # self.target_axes.set_pose(target_pose)

        self.move(self.place_actor(
            self.screwdriver,
            arm_tag=arm_tag,
            target_pose=target_pose,
            pre_dis=0.05,
            dis=0,
            functional_point_id=0,
            pre_dis_axis="fp",
            constrain="free"
        ))

        # Debug after placing
        self._debug_print_alignment(tag="AFTER_PLACE_ACTOR")

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
