import time
from enum import Enum

import numpy as np
import pygame
from OpenGL import GL
from OpenGL.GLU import gluPerspective, gluLookAt

from ..skeleton import _Skeleton


class Viewer:
    class Events(Enum):
        PREV_FRAME = 0
        NEXT_FRAME = 1

    z = np.array([0, 0, 1])

    def __init__(self, landmarks, width=800, height=600, fps=30):
        """
        """
        skeleton = _Skeleton(landmarks)
        self.bones = skeleton.get_bone_list(
            skeleton_sections=_Skeleton.Sections.MAIN
            | _Skeleton.Sections.HEAD
            | _Skeleton.Sections.HANDS | _Skeleton.Sections.FEET
            | _Skeleton.Sections.FACE)
        self._bone_colours = skeleton.get_colour_dict(
            skeleton_sections=_Skeleton.Sections.MAIN
            | _Skeleton.Sections.HEAD
            | _Skeleton.Sections.HANDS | _Skeleton.Sections.FEET
            | _Skeleton.Sections.FACE)
        self._bone_radius = 0.01

        pygame.init()
        pygame.display.set_caption("Skeleton Viewer")
        pygame.display.set_mode((width, height),
                                pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.key.set_repeat(150)

        GL.glEnable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glMatrixMode(GL.GL_PROJECTION)
        gluPerspective(60, (width / height), 0.1, 100.0)

        self._cam_position = np.array([0, 1, 10], dtype=np.float32)
        self._cam_direction = np.array([0, 0, -1], dtype=np.float32)
        self._cam_up = np.array([0, 1, 0])
        self._cam_pitch = 0.0
        self._cam_yaw = -90.0
        self._cam_step_speed = 0.01
        self._cam_rot_speed = 0.2
        self._cam_zoom_speed = 0.2
        self._scene_yaw = 0.0

        self._keypoints = None

        self._last_update = 0
        self._frame_interval = 1 / fps

        self._is_running = True

    @property
    def skeletons(self):
        return self._keypoints

    @skeletons.setter
    def skeletons(self, skeletons_array):
        if skeletons_array is None:
            self._keypoints = None
            return
        if len(skeletons_array.shape) not in (3, 4):
            raise Exception(
                "Skeletons must be specified in person-wise fashion, i.e. in "
                "shape [person, landmark, coordinates]!")
        elif skeletons_array.shape[-1] != 3:
            raise Exception("Skeletons must be 3-dimensional!")
        self._keypoints = skeletons_array

    @property
    def bones(self):
        return self._bones

    @bones.setter
    def bones(self, bone_list):
        self._bones = bone_list

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._is_running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self._is_running = False
                elif event.key == pygame.K_f:
                    return Viewer.Events.NEXT_FRAME
                elif event.key == pygame.K_b:
                    return Viewer.Events.PREV_FRAME
            elif event.type == pygame.MOUSEMOTION:
                if event.buttons[2]:
                    self._cam_yaw += event.rel[0] * self._cam_rot_speed
                    self._cam_pitch -= event.rel[1] * self._cam_rot_speed

                    self._cam_yaw %= 360
                    self._cam_pitch %= 360

                    self._cam_direction = np.array([
                        np.cos(np.deg2rad(self._cam_yaw)) *
                        np.cos(np.deg2rad(self._cam_pitch)),
                        np.sin(np.deg2rad(self._cam_pitch)),
                        np.sin(np.deg2rad(self._cam_yaw)) *
                        np.cos(np.deg2rad(self._cam_pitch)),
                    ])
                elif event.buttons[0]:
                    self._scene_yaw += event.rel[0] * self._cam_rot_speed
                    self._scene_yaw %= 360
            elif event.type == pygame.MOUSEWHEEL:
                self._cam_position += (event.y * self._cam_direction *
                                       self._cam_zoom_speed)
        return None

    def draw_coordinate_system(self):
        GL.glColor3ub(100, 100, 100)
        GL.glBegin(GL.GL_LINES)
        # x-z-GRID
        for x in range(-5, 6):
            GL.glVertex3f(x, 0, -5)
            GL.glVertex3f(x, 0, 5)
        for z in range(-5, 6):
            GL.glVertex3f(-5, 0, z)
            GL.glVertex3f(5, 0, z)
        # Coordinate arrows
        GL.glColor3ub(255, 0, 0)
        GL.glVertex3f(0, 0, 0)
        GL.glVertex3f(1, 0, 0)
        GL.glColor3ub(0, 0, 255)
        GL.glVertex3f(0, 0, 0)
        GL.glVertex3f(0, 1, 0)
        GL.glColor3ub(0, 255, 0)
        GL.glVertex3f(0, 0, 0)
        GL.glVertex3f(0, 0, 1)
        GL.glEnd()

    def draw_line(self, a, b, colour=(255, 255, 255)):
        GL.glColor3ub(*colour)
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3f(*a)
        GL.glVertex3f(*b)
        GL.glEnd()

    def draw_skeleton(self):
        if self._keypoints is not None:
            for skeleton in self._keypoints:
                for i, (a, b) in enumerate(self._bones):
                    if np.all(skeleton[a] == 0) or np.all(skeleton[b] == 0):
                        # TODO: remove
                        continue
                    GL.glColor3ub(*self._bone_colours[a])
                    self.draw_bone(skeleton[a], skeleton[b])
                for i, colour in self._bone_colours.items():
                    if np.all(skeleton[i] == 0):
                        # TODO: remove
                        continue
                    GL.glColor3ub(*colour)
                    self.draw_joint(skeleton[i])

    def draw_joint(self, x):
        # This is currently a diamond, could/should make that something
        # nicer/smoother/a ball
        GL.glBegin(GL.GL_TRIANGLE_FAN)
        GL.glVertex3f(x[0], x[1], x[2] - 2 * self._bone_radius)
        GL.glVertex3f(x[0] - 2 * self._bone_radius,
                      x[1] - 2 * self._bone_radius, x[2])
        GL.glVertex3f(x[0] - 2 * self._bone_radius,
                      x[1] + 2 * self._bone_radius, x[2])
        GL.glVertex3f(x[0] + 2 * self._bone_radius,
                      x[1] + 2 * self._bone_radius, x[2])
        GL.glVertex3f(x[0] + 2 * self._bone_radius,
                      x[1] - 2 * self._bone_radius, x[2])
        GL.glVertex3f(x[0] - 2 * self._bone_radius,
                      x[1] - 2 * self._bone_radius, x[2])
        GL.glEnd()

        GL.glBegin(GL.GL_TRIANGLE_FAN)
        GL.glVertex3f(x[0], x[1], x[2] + 2 * self._bone_radius)
        GL.glVertex3f(x[0] - 2 * self._bone_radius,
                      x[1] - 2 * self._bone_radius, x[2])
        GL.glVertex3f(x[0] + 2 * self._bone_radius,
                      x[1] - 2 * self._bone_radius, x[2])
        GL.glVertex3f(x[0] + 2 * self._bone_radius,
                      x[1] + 2 * self._bone_radius, x[2])
        GL.glVertex3f(x[0] - 2 * self._bone_radius,
                      x[1] + 2 * self._bone_radius, x[2])
        GL.glVertex3f(x[0] - 2 * self._bone_radius,
                      x[1] - 2 * self._bone_radius, x[2])
        GL.glEnd()

    def draw_bone(self, a, b):
        GL.glPushMatrix()
        GL.glTranslatef(*a)
        ba = b - a
        len_ba = np.sqrt((ba**2).sum())
        rot_ax = np.cross(Viewer.z, ba)
        angle = np.rad2deg(np.arccos(np.dot(ba, Viewer.z) / len_ba))
        GL.glRotatef(angle, *rot_ax)
        self.cylinder(len_ba)
        GL.glPopMatrix()

    def cylinder(self, height=2):
        steps = 20

        # HULL
        GL.glBegin(GL.GL_QUAD_STRIP)
        for i in range(steps + 1):
            x = self._bone_radius * np.cos(2 * i / steps * np.pi)
            y = self._bone_radius * np.sin(2 * i / steps * np.pi)
            GL.glVertex3f(x, y, height)
            GL.glVertex3f(x, y, 0.0)
        GL.glEnd()
        # TOP
        GL.glBegin(GL.GL_TRIANGLE_FAN)
        GL.glVertex3f(0, 0, height)
        for i in range(steps + 1):
            x = self._bone_radius * np.cos(2 * i / steps * np.pi)
            y = self._bone_radius * np.sin(2 * i / steps * np.pi)
            GL.glVertex3f(x, y, height)
        GL.glEnd()
        # BOTTOM
        GL.glBegin(GL.GL_TRIANGLE_FAN)
        GL.glVertex3f(0, 0, 0)
        for i in range(steps, -1, -1):
            x = self._bone_radius * np.cos(2 * i / steps * np.pi)
            y = self._bone_radius * np.sin(2 * i / steps * np.pi)
            GL.glVertex3f(x, y, 0)
        GL.glEnd()

    def draw_other(self):
        pass

    def update(self):
        now = time.monotonic()
        delta_time = now - self._last_update
        # Handle moving around in the scene by keyboard here as it should run
        # much faster than normal keypress events
        key_press = pygame.key.get_pressed()
        shift = None
        if key_press[pygame.K_LEFT] or key_press[pygame.K_a]:
            shift = np.cross(self._cam_up, self._cam_direction)
            shift /= np.sqrt((shift**2).sum())
        if key_press[pygame.K_RIGHT] or key_press[pygame.K_d]:
            shift = np.cross(self._cam_direction, self._cam_up)
            shift /= np.sqrt((shift**2).sum())
        if key_press[pygame.K_UP] or key_press[pygame.K_w]:
            shift = self._cam_up / 4
        if key_press[pygame.K_DOWN] or key_press[pygame.K_s]:
            shift = -self._cam_up / 4
        if shift is not None:
            self._cam_position += shift * self._cam_step_speed * delta_time

        if delta_time > self._frame_interval:
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadIdentity()
            gluLookAt(*self._cam_position,
                      *(self._cam_position + self._cam_direction),
                      *self._cam_up)
            GL.glRotatef(self._scene_yaw, 0, 1, 0)
            self.draw_coordinate_system()
            self.draw_skeleton()
            self.draw_other()

            pygame.display.flip()
            self._last_update = now

        if not self._is_running:
            pygame.quit()

    @property
    def is_running(self):
        return self._is_running
