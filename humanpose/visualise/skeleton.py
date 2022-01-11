from enum import IntFlag


class _Skeleton:
    """
    Define the structural elements to work with. Mostly individual parts of the
    skeleton are linear and defined as lists which are assembled sequentially,
    skipping joints which don't exist in a given configuration. The finer
    details, such as hands and feet can have more of a tree structure and are
    defined as dicts in a child:parent fashion. Connecting the limbs to the
    spine is complex and varies largely based on which spine elements exist in
    a given configuration. Thus working out these connections is done in code.
    """
    class Sections(IntFlag):
        MAIN = 1
        HEAD = 2
        FACE = 4
        HANDS = 8
        FEET = 16
        FULL = 31
        OBJECTS = 32

    # This is for easy looping over symmetric connections.
    _sides = ("left ", "right ")
    _body_parts = {
        # Connections in the extremeties can be expected to be symmetric. Most
        # of these exist in all datasets, differences are only within hands and
        # feet or the very ends if the sequence.
        "leg": ["foot", "ankle", "knee", "hip"],
        "arm": ["hand", "wrist", "elbow", "shoulder", "clavicle"],
        # The spine runs down the middle of the body from the base of the head
        # to the pelvis. Keypoints exist only once and different datasets/pose
        # systems will have a different set of points here.
        # Heuristically we want to go simply from one to the next, skipping
        # anything that doesn't exist. If too much of the spine doesn't exist
        # draw a box using shoulders and hips.
        "spine": [
            "head centre", "head", "neck", "shoulder centre", "thorax",
            "mid torso", "belly", "pelvis"
        ],
        # For a simple skeleton the head usually doesn't need much detail, only
        # the centre line, which is given by the nose or head top and should be
        # connected to the top-most spine keypoint.
        # The remaining keypoints occuring in datasets are symmetric keypoints
        # with facial details (eyes, ears, mouth). These I will only rarely
        # want to draw.
        "head": ["head top", "nose"],
        # TODO: there are sometimes more face keypoints
        "face": {
            "ear": "eye",
            "eye": "nose"
        },
        # Hands and feet rarely have more detail than given above, but e.g.
        # OpenPose and the Kinect has more detail so include support here.
        # TODO : Hands might have a hand centre keypoint or just the
        # finger/knuckle points connecting to the wrist. Need to support that.
        "hand": {
            "handtip": "hand",
            "thumb": "hand"
        },
        "foot": {
            "big toe": "ankle",
            "small toe": "foot",
            "heel": "ankle"
        },
        # Sometimes I might have other, non-body bits such as Hockey sticks to
        # draw. For now I am going to connect object keypoints simply
        # sequentially like the spinal keypoints as for hockey sticks that
        # makes sense. This may need to become more sophisticated if this
        # category grows.
        "objects": ["stick top", "stick end"]
    }
    _sided_bodyparts = ["leg", "arm", "hand", "foot", "face"]

    _part_colours = {
        "spine": {
            "start": (255, 255, 0),  # head
            "end": (255, 0, 0)  # pelvis
        },
        "left leg": {
            "start": (255, 0, 255),  # foot
            "end": (75, 0, 150)  # hip
        },
        "right leg": {
            "start": (100, 100, 255),  # foot
            "end": (0, 0, 150)  # hip
        },
        "left arm": {
            "start": (0, 255, 0),  # hand
            "end": (0, 120, 0)  # clavicle
        },
        "right arm": {
            "start": (0, 255, 255),  # hand
            "end": (0, 120, 120)  # clavicle
        },
        "head": {
            "start": (170, 170, 0),  # head top
            "end": (210, 210, 0)  # nose
        },
        "objects": {
            "start": (175, 0, 0),
            "end": (175, 100, 0)
        },
        "left hand": (120, 255, 120),
        "right hand": (180, 255, 255),
        "left foot": (255, 145, 255),
        "right foot": (130, 180, 255),
        "left face": (120, 120, 0),
        "right face": (120, 120, 0),
    }

    def __init__(self, landmarks, radius=5):
        """
        """
        self._compute_bones(landmarks)
        self._compute_colours(landmarks)
        self._radius = radius

    ###########################################################################
    # Helper functions to get/print bone and colour information
    ###########################################################################

    def get_bone_list(self, skeleton_sections=Sections.MAIN | Sections.HEAD):
        """
        Get the list of all bones forming the selected skeleton groups.
        Parameters
        ----------
        skeleton_sections : Sections flags
            Selection of shich sections of the skeleton to draw
        """
        bones = []
        for sec in _Skeleton.Sections:
            if sec & skeleton_sections:
                bones.extend(self._bones[sec])
        return bones

    def print_bones(self,
                    landmarks,
                    skeleton_sections=Sections.MAIN
                    | Sections.HEAD):
        """
        Returns tuples of the landmark names of selected bones.
        Convenience function for testing, pass in the original list of
        landmarks to get a list of tuples of landmark names rather than a list
        of array indicies for easy checking of the list of bones.
        Parameters
        ----------
        landmarks : list of strings
            List of the landmark names
        skeleton_sections : Sections flags
            Selection of shich sections of the skeleton to draw
        """
        bones = self.get_bone_list(skeleton_sections)
        return [(landmarks[bone[0]], landmarks[bone[1]]) for bone in bones]

    def get_colour_dict(self, skeleton_sections=Sections.MAIN | Sections.HEAD):
        """
        Get the dictionary of colours for drawing the selected keypoints.
        Parameters
        ----------
        skeleton_sections : Sections flags
            Selection of shich sections of the skeleton to draw
        """
        colours = {}
        for sec in _Skeleton.Sections:
            if sec & skeleton_sections:
                colours.update(self._colours[sec])
        return colours

    def print_colours(self,
                      landmarks,
                      skeleton_sections=Sections.MAIN
                      | Sections.HEAD):
        """
        Returns a dictionary with landmark names as keys and colours as values.
        Convenience function for testing, pass in the original list of
        landmarks to get a dictionary with landmark names as keys rather than
        landmark ids, for easy checking of the colouring of landmarks.
        Parameters
        ----------
        landmarks : list of strings
            List of the landmark names
        skeleton_sections : Sections flags
            Selection of shich sections of the skeleton to draw
        """
        colours = self.get_colour_dict(skeleton_sections)
        return {landmarks[key]: val for key, val in colours.items()}

    ###########################################################################
    # Compute colours and bone structure
    ###########################################################################

    def _compute_colours(self, landmarks):
        """
        Computes colours for each landmark based on the landmark list.
        Each section of the skeleton has a defined colour range. This function
        makes the colour gradually shift along that range, taking into account
        how many joints there are in the given configuration.
        Parameters
        ----------
        landmarks : list of strings
            Names of tha landmarks in the given configuration
        """
        self._colours = {sec: {}
                         for sec in _Skeleton.Sections
                         }  # MAIN HEAD FACE HANDS FEET OBJECTS
        for body_part, joints in _Skeleton._body_parts.items():
            if (body_part == "spine" or body_part == "arm"
                    or body_part == "leg"):
                skeleton_sec = _Skeleton.Sections.MAIN
            elif body_part == "head":
                skeleton_sec = _Skeleton.Sections.HEAD
            elif body_part == "face":
                skeleton_sec = _Skeleton.Sections.FACE
            elif body_part == "hand":
                skeleton_sec = _Skeleton.Sections.HANDS
            elif body_part == "foot":
                skeleton_sec = _Skeleton.Sections.FEET
            elif body_part == "objects":
                skeleton_sec = _Skeleton.Sections.OBJECTS
            # body parts for asembly are side agnostic, colours are not
            # => add side key to sided body parts for colouring
            if body_part in _Skeleton._sided_bodyparts:
                sides = _Skeleton._sides
            else:
                sides = ("", )
            for side in sides:
                if isinstance(_Skeleton._part_colours[side + body_part],
                              tuple):
                    for joint in joints.keys():
                        if side + joint in landmarks:
                            self._colours[skeleton_sec][landmarks.index(
                                side +
                                joint)] = _Skeleton._part_colours[side +
                                                                  body_part]
                else:
                    # Find all landmarks that exist in the given config
                    part_joints = []
                    for joint in joints:
                        if side + joint in landmarks:
                            part_joints.append(landmarks.index(side + joint))
                    # Colour each existing landmarks of the current set
                    # gradually shifting the colour going along the part
                    if len(part_joints) > 0:
                        for i, joint in enumerate(part_joints):
                            if len(part_joints) > 1:
                                t = i / (len(part_joints) - 1)
                            else:
                                t = 1
                            self._colours[skeleton_sec][joint] = tuple(
                                (int(_Skeleton._part_colours[
                                    side + body_part]["start"][j] * (1 - t) +
                                     _Skeleton._part_colours[
                                         side + body_part]["end"][j] * t)
                                 for j in range(3)))

    def _compute_bones(self, landmarks):
        """
        Work out all the possible skeletal connections.
        I can later choose which subsets of them to use to build skeletons of
        varying detail.
        Parameters
        ----------
        landmarks : list of strings
            Names of tha landmarks in the given configuration
        """
        self._bones = {sec: []
                       for sec in _Skeleton.Sections
                       }  # MAIN HEAD FACE HANDS FEET OBJECTS
        # ---> MAIN
        self._bones[_Skeleton.Sections.
                    MAIN] = _Skeleton._connect_sequence_symmetrically(
                        landmarks, _Skeleton._body_parts["arm"])
        # Work out connections of shoulder to spine and each other
        for shoulder_joint in ("clavicle", "shoulder"):
            if _Skeleton._sides[0] + shoulder_joint in landmarks:
                self._bones[_Skeleton.Sections.MAIN].extend(
                    _Skeleton._limb2spine(landmarks, shoulder_joint,
                                          ("shoulder centre", "neck"),
                                          _Skeleton._body_parts["spine"][3:]))
                break

        self._bones[_Skeleton.Sections.MAIN].extend(
            _Skeleton._connect_sequence_symmetrically(
                landmarks, _Skeleton._body_parts["leg"]))
        # Work out connections of hips to spine and each other
        self._bones[_Skeleton.Sections.MAIN].extend(
            _Skeleton._limb2spine(landmarks, "hip", ("pelvis", ),
                                  _Skeleton._body_parts["spine"][-2:1:-1]))

        self._bones[_Skeleton.Sections.MAIN].extend(
            _Skeleton._connect_sequence(landmarks,
                                        _Skeleton._body_parts["spine"]))
        # ---> HEAD
        self._bones[_Skeleton.Sections.HEAD] = _Skeleton._connect_sequence(
            landmarks, _Skeleton._body_parts["head"])
        # connect the head to the top-most point of the spine
        for head_pt in _Skeleton._body_parts["head"][::-1]:
            for top_spine in _Skeleton._body_parts["spine"]:
                if head_pt in landmarks and top_spine in landmarks:
                    self._bones[_Skeleton.Sections.HEAD].append(
                        (landmarks.index(head_pt), landmarks.index(top_spine)))
                    break
            else:
                # if there was no break we need to keep searching
                continue
            # if there was a break we are done
            break
        # ---> FACE
        self._bones[
            _Skeleton.Sections.FACE] = _Skeleton._connect_tree_symmetrically(
                landmarks, _Skeleton._body_parts["face"])
        # ---> HANDS and FEET
        self._bones[
            _Skeleton.Sections.HANDS] = _Skeleton._connect_tree_symmetrically(
                landmarks, _Skeleton._body_parts["hand"])
        self._bones[
            _Skeleton.Sections.FEET] = _Skeleton._connect_tree_symmetrically(
                landmarks, _Skeleton._body_parts["foot"])
        # ---> OBJECTS
        self._bones[_Skeleton.Sections.OBJECTS] = _Skeleton._connect_sequence(
            landmarks, _Skeleton._body_parts["objects"])

    ##################################################
    # Some of the bone structures have good regularity and can be build up in a
    # fairly general way. Below functions build up sections of the skeleton
    ##################################################

    @staticmethod
    def _connect_sequence_symmetrically(landmarks, connection_list):
        """
        Wrapper for sequential connections to be made on both body sides.
        Parameters
        ----------
        landmarks : list of strings
            Names of tha landmarks in the given configuration
        connection_list : list of strings
            One of the lists describing the structure of a bodypart defined at
            the top of this class.
        """
        connections = []
        for side in _Skeleton._sides:
            connections.extend(
                _Skeleton._connect_sequence(
                    landmarks, [side + lm for lm in connection_list]))
        return connections

    @staticmethod
    def _connect_tree_symmetrically(landmarks, connection_dict):
        """
        Wrapper for tree connections to be made on both body sides.
        Parameters
        ----------
        landmarks : list of strings
            Names of tha landmarks in the given configuration
        connection_dict : dictionary
            One of the dictionaries describing the structure of a bodypart
            defined at the top of this class.
        """
        connections = []
        for side in _Skeleton._sides:
            connections.extend(
                _Skeleton._connect_tree(landmarks, {
                    side + key: side + val
                    for key, val in connection_dict.items()
                }))
        return connections

    @staticmethod
    def _connect_tree(landmarks, connection_dict):
        """
        Work out tree structure of a body part.
        Parameters
        ----------
        landmarks : list of strings
            Names of tha landmarks in the given configuration
        connection_dict : dictionary
            One of the dictionaries describing the structure of a bodypart
            defined at the top of this class.
        """
        connections = []
        for src, dest in connection_dict.items():
            if src in landmarks and dest in landmarks:
                connections.append(
                    (landmarks.index(src), landmarks.index(dest)))
        return connections

    @staticmethod
    def _connect_sequence(landmarks, connection_list):
        """
        Work out the sequential structure of a body part.
        Parameters
        ----------
        landmarks : list of strings
            Names of tha landmarks in the given configuration
        connection_list : list of strings
            One of the lists describing the structure of a bodypart defined at
            the top of this class.
        """
        connections = []
        pt_it = iter(connection_list)
        # Find initial point of sequence that exists in landmarks list
        for pt in pt_it:
            if pt in landmarks:
                break
        # Find the next one to match and move along
        for next_pt in pt_it:
            if next_pt in landmarks:
                connections.append(
                    (landmarks.index(pt), landmarks.index(next_pt)))
                pt = next_pt
        return connections

    @staticmethod
    def _limb2spine(landmarks, limb_end, spine_attachment, spine_list):
        """
        Work out how to connect either the arms or the legs to the spine.
        Shoulders have a set of preferred attachment points along the spine, if
        those don't exist they should connect to each other and whichever spine
        point is closest to them, forming a triangle. If there are no spine
        points between the shoulders and the hips, connect them to each other
        to form a square for the torso.
        Parameters
        ----------
        landmarks :
            The list of landmarks in the given config
        limb_end :
            'shoulder' or 'hip'
        spine_attachment :
            joints along the spine to try and connect to
        spine_list :
            remaining spine list to search for first possible connection along
            the spine
        """
        connections = []
        limb_ends = [
            landmarks.index(side + limb_end) for side in _Skeleton._sides
        ]
        for attachment in spine_attachment:
            if attachment in landmarks:
                lm = landmarks.index(attachment)
                for end in limb_ends:
                    connections.append((end, lm))
                break
        else:
            # The preferred connection along the spine was not found, so
            # connect limb ends to each other
            connections.append(tuple(limb_ends))
            # Also connect to the first suitable point along the spine
            for next_spine in spine_list:
                if next_spine in landmarks:
                    lm = landmarks.index(next_spine)
                    for end in limb_ends:
                        connections.append((end, lm))
                    break
            else:
                # Did not find a point along the spine, if these are the
                # shoulders also join up with hips
                if limb_end != "hip":
                    for i, side in enumerate(_Skeleton._sides):
                        connections.append(
                            (limb_ends[i], landmarks.index(side + "hip")))
        return connections
