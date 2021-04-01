from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from .skeleton import _Skeleton


class Skeleton3D(_Skeleton):
    """
    Class to visualise 3D skeletons using matplotlibs 3D plotting.
    """
    # Controls to allow shuffling of axes for 3D plotting
    _X = 0
    _Y = 2
    _Z = 1

    ###########################################################################
    # 3D drawing functions
    ###########################################################################
    def plot3d(self,
               keypoints,
               fig=None,
               ax=None,
               skeleton_sections=_Skeleton.Sections.MAIN
               | _Skeleton.Sections.HEAD,
               radius=None):
        """
        3D plot of a single skeleton.
        Parameters
        ----------
        keypoints : numpy array
            The keypoints to be drawn
        fig : matplotlib.figure object, optional
            Figure object to use, if not given a new one is created
        ax : matplotlib.Axes object, optional
            Axis object to be used for drawing. must be a 3d axis. If not given
            a new one is created
        skeleton_sections : _Skeleton.Sections flags
            Selection of shich sections of the skeleton to draw
        radius :  int, optional (default is class property radius)
            Radius for points to be drawn

        Returns
        -------
        ax : matplotlib.Axes object
        """
        if radius is None:
            radius = self._radius
        if ax is None:
            fig, ax = Skeleton3D.get_plot3d_view(fig)
        colours = self.get_colour_dict(skeleton_sections)
        for b in self.get_bone_list(skeleton_sections):
            colour = tuple(c / 255.0 for c in colours[b[0]])
            ax.plot(
                keypoints[b, Skeleton3D._X],
                keypoints[b, Skeleton3D._Y],
                keypoints[b, Skeleton3D._Z],
                c=colour,
                marker="o",
                markersize=radius,
            )

        Skeleton3D._axis_settings(ax, keypoints)
        return ax

    def plot3d_animated(self,
                        keypoints,
                        fig=None,
                        ax=None,
                        skeleton_sections=_Skeleton.Sections.MAIN
                        | _Skeleton.Sections.HEAD,
                        radius=None):
        """
        Create an animated 3D plot of a skeleton video.
        Parameters
        ----------
        keypoints : numpy array
            The keypoints to be drawn
        fig : matplotlib.figure object, optional
            Figure object to use, if not given a new one is created
        ax : matplotlib.axes object, optional
            Axis object to be used for drawing. must be a 3d axis. If not given
            a new one is created
        skeleton_sections : _Skeleton.Sections flags
            Selection of shich sections of the skeleton to draw
        radius :  int, optional (default is class property radius)
            Radius for points to be drawn

        Returns
        -------
        animation : matplotlib.animation.FuncAnimation object
        """
        if radius is None:
            radius = self._radius
        if ax is None:
            fig, ax = Skeleton3D.get_plot3d_view(fig)
        bone_list = self.get_bone_list(skeleton_sections)
        colours = self.get_colour_dict(skeleton_sections)
        bones = [
            ax.plot([], [], [],
                    c=tuple(c / 255.0 for c in colours[b[0]]),
                    marker="o",
                    markersize=self._radius)[0] for b in bone_list
        ]

        Skeleton3D._axis_settings(ax, keypoints)

        update_animated_plot = partial(self._update_animated_plot,
                                       skeleton_sections=skeleton_sections)

        print(keypoints.shape)
        animation = FuncAnimation(fig,
                                  update_animated_plot,
                                  len(keypoints),
                                  fargs=(bones, keypoints),
                                  interval=20)
        return animation

    @staticmethod
    def get_plot3d_view(fig=None, rows=1, cols=1, index=1):
        """
        Convenience function to create 3d matplotlib axis object.
        Wraps figure creation if need be and add_subplot.
        Parameters
        ----------
        fig : matplotlib.figure object, optional
            For re-use of an existing figure object. A new one is created if
            not given.
        rows : int
            Number of subplot rows. Like fig.add_subplot
        cols : int
            Number of subplot cols. Like fig.add_subplot
        index : int
            Index of subplot to use. Like fig.add_subplot

        Returns
        -------
        (fig, ax)
            fig : matplotlib.figure object
            ax : matplotlib.Axes object
        """
        if fig is None:
            fig = plt.figure()
        ax = fig.add_subplot(rows, cols, index, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        return (fig, ax)

    ###########################################################################
    # Plotting helper functions
    ###########################################################################

    def _update_animated_plot(self,
                              frame,
                              bone_plots,
                              keypoints,
                              skeleton_sections=_Skeleton.Sections.MAIN
                              | _Skeleton.Sections.HEAD):
        """
        Updates the data of the animated plot to the next frame.

        Parameters
        ----------
        frame : int
            The frame number to advance to. This is automatically generated.
        bone_plots : matplotlib plots
            The plot updates for which to update the data
        keypoints : np.ndarray
            The full keypoint array as passed in at creation of the animated
            plot.
        """
        bone_list = self.get_bone_list(skeleton_sections)
        for i, b in enumerate(bone_list):
            bone_plots[i].set_data(keypoints[frame, b, Skeleton3D._X],
                                   keypoints[frame, b, Skeleton3D._Y])
            bone_plots[i].set_3d_properties(keypoints[frame, b, Skeleton3D._Z])

    @staticmethod
    def _axis_settings(ax, keypoints):
        """
        Sets the axes limits for the 3d plot so that they have an equal length.
        This is not natively done by matplotlib as it is projection dependend.
        Parameters
        ----------
        ax : matplotlib.axes object
            Axis object for which to set the axis limits
        keypoints : np.ndarray
            Keypoint array to determine required axis limits. If a video is
            being plotted the axis limits will include the skeleton throughout
            the entire video.
        """
        axes = tuple(i for i in range(len(keypoints.shape) - 1))
        val_range = np.ceil(np.amax(np.ptp(keypoints, axis=axes)) * 10) / 20
        val_centres = (np.amax(keypoints, axis=axes) +
                       np.amin(keypoints, axis=axes)) / 2

        ax.set_xlim3d(val_centres[Skeleton3D._X] - val_range,
                      val_centres[Skeleton3D._X] + val_range)
        ax.set_ylim3d(val_centres[Skeleton3D._Y] - val_range,
                      val_centres[Skeleton3D._Y] + val_range)
        ax.set_zlim3d(val_centres[Skeleton3D._Z] - val_range,
                      val_centres[Skeleton3D._Z] + val_range)
