import numpy as np
import cv2

from .skeleton import _Skeleton


class Skeleton2D(_Skeleton):
    """
    Class to visualise 2D skeletons on neutral background or original RGB.
    """

    ###########################################################################
    # 2D drawing functions
    ###########################################################################
    def draw(self,
             keypoints,
             img=None,
             img_filename=None,
             skeleton_sections=_Skeleton.Sections.MAIN
             | _Skeleton.Sections.HEAD,
             radius=None):
        """
        Draw skeleton onto black background or given image.

        Parameters
        ----------
        keypoints : numpy array
            The keypoints to be drawn
        img : numpy array, optional (default is None)
            Image to draw on, if None then the image specified by img_filename
            is loaded or  a black image is generated
        img_filename : string, optional (default is None)
            If given and img is None then loads the specified file to draw on
        skeleton_sections : _Skeleton.Sections flags
            Selection of shich sections of the skeleton to draw
        radius :  int, optional (default is class property radius)
            Radius for points to be drawn

        Returns
        -------
        img (h,w,c) with skeleton drawn on.
        """
        rounded_keypoints = np.around(keypoints).astype(np.int32)
        if img is None:
            if img_filename is not None:
                img = cv2.imread(img_filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                rounded_keypoints -= np.amin(rounded_keypoints, axis=0) - 10
                dims = (np.amax(rounded_keypoints, axis=0) // 10) * 10 + 20
                img = np.zeros(tuple(dims[::-1]) + (3, ), dtype=np.uint8)
        if radius is None:
            radius = self._radius

        colours = self.get_colour_dict(skeleton_sections)
        for c1, c2 in self.get_bone_list(skeleton_sections):
            cv2.line(img, tuple(rounded_keypoints[c1, 0:2]),
                     tuple(rounded_keypoints[c2, 0:2]), colours[c1],
                     max(radius // 2, 1))
        for i, colour in colours.items():
            cv2.circle(img, tuple(rounded_keypoints[i, 0:2]), radius, colour,
                       -1)
        return img

    def draw_frame(self,
                   keypoints,
                   video_filename,
                   frame_id,
                   skeleton_sections=_Skeleton.Sections.MAIN
                   | _Skeleton.Sections.HEAD,
                   radius=None):
        """
        Shortcut to extract a single frame and draw skeleton onto it.
        Extracts a frame from anywhere in the video and draws the skeleton onto
        it. Returns image array. This is not efficient for actually showing the
        full video but good for just randomly smapling the odd frame from
        within the video.

        Parameters
        ----------
        keypoints : numpy array
            The keypoints to be drawn
        video_filename : string
            Filename of the video file to extract the frame image from
        frame_id : int
            ID of the frame to draw
        skeleton_sections : _Skeleton.Sections flags
            Selection of shich sections of the skeleton to draw
        radius :  int, optional (default is class property radius)
            Radius for points to be drawn

        Returns
        -------
        Frame from the video (h,w,c) with skeleton drawn on.
        """
        video_file = cv2.VideoCapture(video_filename)
        video_file.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        __, frame = video_file.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.draw(keypoints=keypoints,
                  img=frame,
                  skeleton_sections=skeleton_sections,
                  radius=radius)
        return frame

    def animate(self,
                keypoints,
                video_filename=None,
                skeleton_sections=_Skeleton.Sections.MAIN
                | _Skeleton.Sections.HEAD,
                radius=None):
        """
        Return a numpy array of all frames with skeletons drawn on.

        Parameters
        ----------
        keypoints : numpy array
            The keypoints to be drawn
        video_filename : string
            Filename of the video file
        skeleton_sections : _Skeleton.Sections flags
            Selection of shich sections of the skeleton to draw
        radius :  int, optional (default is class property radius)
            Radius for points to be drawn

        Returns
        -------
        Sequence of frames (h,w,c) with skeleton drawn on.
        """
        if video_filename is not None:
            video_file = cv2.VideoCapture(video_filename)
            video = []
            while (video_file.isOpened()):
                ret, frame = video_file.read()
                if not ret:
                    break
                video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            video_file.release()
            video = np.array(video)
        else:
            rounded_keypoints = (np.around(keypoints) -
                                 np.amin(keypoints, axis=0) + 5)
            dims = (np.amax(rounded_keypoints, axis=0) // 10) * 10 + 20
            video = np.zeros((len(rounded_keypoints), ) + tuple(dims))

        for frame_id in range(len(keypoints)):
            self.draw(keypoints=keypoints[frame_id],
                      img=video[frame_id],
                      skeleton_sections=skeleton_sections,
                      radius=radius)
        return video
