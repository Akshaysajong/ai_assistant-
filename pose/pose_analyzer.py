import numpy as np


class PoseAnalyzer:
    """Analyze pose data to extract useful information."""
    
    def __init__(self):
        pass
    
    def analyze(self, pose_result):
        """
        Analyze pose result and extract information.
        
        Args:
            pose_result: YOLO pose result object
            
        Returns:
            dict: Analysis results with keypoints, actions, etc.
        """
        if not pose_result or len(pose_result) == 0:
            return None
        
        analysis = {
            "num_persons": len(pose_result),
            "poses": []
        }
        
        for result in pose_result:
            if result.keypoints is not None:
                keypoints_array = result.keypoints.xy.cpu().numpy()
                if len(keypoints_array) > 0:
                    keypoints = keypoints_array[0]  # Get keypoints for first person
                    
                    pose_info = {
                        "keypoints": keypoints,
                        "bbox": result.boxes.xyxy.cpu().numpy()[0] if result.boxes is not None and len(result.boxes.xyxy) > 0 else None,
                        "confidence": result.boxes.conf.cpu().numpy()[0] if result.boxes is not None and len(result.boxes.conf) > 0 else None,
                        "actions": self.detect_actions(keypoints)
                    }
                    analysis["poses"].append(pose_info)
        
        return analysis
    
    def detect_actions(self, keypoints):
        """
        Detect basic actions from pose keypoints.
        
        Args:
            keypoints: Array of shape (17, 2) with (x, y) coordinates
            
        Returns:
            list: Detected actions
        """
        actions = []
        
        if len(keypoints) < 17:
            return actions
        
        # Key point indices (COCO format)
        # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
        # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
        # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
        # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
        
        # Check if arms are raised
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        
        if left_wrist[1] < left_shoulder[1] and right_wrist[1] < right_shoulder[1]:
            actions.append("arms_raised")
        elif left_wrist[1] < left_shoulder[1]:
            actions.append("left_arm_raised")
        elif right_wrist[1] < right_shoulder[1]:
            actions.append("right_arm_raised")
        
        # Check if person is standing or sitting
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_knee = keypoints[13]
        right_knee = keypoints[14]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        
        # Calculate leg angles and ratios for better sitting/standing detection
        hip_y = (left_hip[1] + right_hip[1]) / 2
        knee_y = (left_knee[1] + right_knee[1]) / 2
        ankle_y = (left_ankle[1] + right_ankle[1]) / 2
        
        # Calculate leg segment ratios
        hip_knee_dist = abs(knee_y - hip_y)
        knee_ankle_dist = abs(ankle_y - knee_y)
        
        # Standing: legs are more vertical, knee-ankle distance is larger
        # Sitting: legs are bent, hip-knee distance is larger relative to knee-ankle
        if hip_knee_dist > 0 and knee_ankle_dist > 0:
            leg_ratio = hip_knee_dist / knee_ankle_dist
            # If hip-knee is significantly larger than knee-ankle, likely sitting
            if leg_ratio > 0.8:
                actions.append("sitting")
            else:
                actions.append("standing")
        else:
            # Fallback to simple distance check
            if knee_y - hip_y > 80:  # Increased threshold
                actions.append("sitting")
            else:
                actions.append("standing")
        
        return actions
    
    def get_keypoint_positions(self, keypoints):
        """
        Get specific keypoint positions.
        
        Args:
            keypoints: Array of shape (17, 2) with (x, y) coordinates
            
        Returns:
            dict: Dictionary of keypoint names and positions
        """
        if len(keypoints) < 17:
            return {}
        
        return {
            "nose": keypoints[0],
            "left_shoulder": keypoints[5],
            "right_shoulder": keypoints[6],
            "left_elbow": keypoints[7],
            "right_elbow": keypoints[8],
            "left_wrist": keypoints[9],
            "right_wrist": keypoints[10],
            "left_hip": keypoints[11],
            "right_hip": keypoints[12],
            "left_knee": keypoints[13],
            "right_knee": keypoints[14],
            "left_ankle": keypoints[15],
            "right_ankle": keypoints[16]
        }
