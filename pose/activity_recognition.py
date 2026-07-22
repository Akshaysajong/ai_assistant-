import numpy as np
from collections import deque


class ActivityRecognizer:
    """Recognize activities from pose data over time."""
    
    def __init__(self, window_size=30):
        """
        Initialize activity recognizer.
        
        Args:
            window_size: Number of frames to analyze for activity recognition
        """
        self.window_size = window_size
        self.pose_history = deque(maxlen=window_size)
        self.current_activity = "unknown"
        self.activity_confidence = 0.0
    
    def update(self, pose_analysis):
        """
        Update with new pose analysis and recognize activity.
        
        Args:
            pose_analysis: Pose analysis result from PoseAnalyzer
            
        Returns:
            dict: Current activity with confidence
        """
        if pose_analysis is None or not pose_analysis["poses"]:
            return {"activity": "unknown", "confidence": 0.0}
        
        # Store pose data
        self.pose_history.append(pose_analysis)
        
        # Recognize activity if we have enough data
        if len(self.pose_history) >= 5:
            activity, confidence = self.recognize_activity()
            self.current_activity = activity
            self.activity_confidence = confidence
        else:
            self.current_activity = "analyzing"
            self.activity_confidence = 0.0
        
        return {
            "activity": self.current_activity,
            "confidence": self.activity_confidence
        }
    
    def recognize_activity(self):
        """
        Recognize activity from pose history.
        
        Returns:
            tuple: (activity_name, confidence)
        """
        if len(self.pose_history) < 5:
            return "unknown", 0.0
        
        # Get recent poses
        recent_poses = list(self.pose_history)[-10:]
        
        # Analyze movement patterns
        activities = {
            "walking": 0.0,
            "standing": 0.0,
            "sitting": 0.0,
            "waving": 0.0,
            "raising_hands": 0.0,
            "unknown": 0.0
        }
        
        for pose_data in recent_poses:
            if not pose_data["poses"]:
                continue
            
            for pose_info in pose_data["poses"]:
                actions = pose_info.get("actions", [])
                keypoints = pose_info.get("keypoints")
                
                # Count action occurrences
                for action in actions:
                    if "standing" in action:
                        activities["standing"] += 1
                    elif "sitting" in action:
                        activities["sitting"] += 1
                    elif "arms_raised" in action:
                        activities["raising_hands"] += 1
                    elif "arm_raised" in action:
                        activities["waving"] += 0.5
                
                # Analyze movement for walking detection
                if keypoints is not None and len(keypoints) >= 17:
                    movement = self.calculate_movement(keypoints)
                    if movement > 20:  # Threshold for significant movement
                        activities["walking"] += 1
        
        # Normalize scores
        total = sum(activities.values())
        if total > 0:
            for key in activities:
                activities[key] /= total
        
        # Get best activity
        best_activity = max(activities, key=activities.get)
        confidence = activities[best_activity]
        
        # Apply minimum confidence threshold
        if confidence < 0.3:
            best_activity = "unknown"
            confidence = 0.0
        
        return best_activity, confidence
    
    def calculate_movement(self, keypoints):
        """
        Calculate overall movement from keypoints.
        
        Args:
            keypoints: Current frame keypoints
            
        Returns:
            float: Movement magnitude
        """
        if len(self.pose_history) < 2:
            return 0.0
        
        # Get previous keypoints
        prev_pose = self.pose_history[-2]
        if not prev_pose["poses"]:
            return 0.0
        
        prev_keypoints = prev_pose["poses"][0].get("keypoints")
        if prev_keypoints is None or len(prev_keypoints) < 17:
            return 0.0
        
        # Calculate movement
        movement = 0.0
        valid_points = 0
        
        for i in range(min(len(keypoints), len(prev_keypoints))):
            if keypoints[i][0] > 0 and keypoints[i][1] > 0:
                dx = keypoints[i][0] - prev_keypoints[i][0]
                dy = keypoints[i][1] - prev_keypoints[i][1]
                movement += np.sqrt(dx**2 + dy**2)
                valid_points += 1
        
        if valid_points > 0:
            movement /= valid_points
        
        return movement
    
    def detect_waving(self, keypoints):
        """
        Detect waving gesture from keypoints.
        
        Args:
            keypoints: Array of keypoints
            
        Returns:
            bool: True if waving detected
        """
        if len(keypoints) < 11:
            return False
        
        # Check wrist positions relative to shoulders
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        
        # Simple wave detection: one arm raised and moving
        left_raised = left_wrist[1] < left_shoulder[1]
        right_raised = right_wrist[1] < right_shoulder[1]
        
        return left_raised or right_raised
    
    def get_activity_history(self):
        """
        Get recent activity history.
        
        Returns:
            list: Recent activities
        """
        return [self.current_activity]
