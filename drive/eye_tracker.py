import cv2
import mediapipe as mp
from scipy.spatial import distance as dist

mp_face_mesh=mp.solutions.face_mesh
face_mesh=mp_face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1)

LEFT_EYE_INDEXS=[33,160,158,133,153,144]
RIGHT_EYE_INDEXS=[362,385,387,263,373,380]

def caculate_EAR(eye_points):
    A=dist.euclidean(eye_points[1],eye_points[5])
    B=dist.euclidean(eye_points[2],eye_points[4])
    C=dist.euclidean(eye_points[0],eye_points[3])
    EAR=(A+B)/(2.0*C)
    return EAR

def extract_EAR(image,draw=True):
    h,w=image.shape[:2]
    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    results=face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    landmarks=results.multi_face_landmarks[0].landmark
    left_eye=[(int(landmarks[i].x*w),int(landmarks[i].y*h)) for i in LEFT_EYE_INDEXS]
    right_eye=[(int(landmarks[i].x*w),int(landmarks[i].y*h)) for i in RIGHT_EYE_INDEXS]

    left_EAR=caculate_EAR(left_eye)
    right_EAR=caculate_EAR(right_eye)
    avg_EAR=(left_EAR+right_EAR)/2.0

    if draw:
        for pt in left_eye+right_eye:
            cv2.circle(image,pt,2,(0,255,0),1)
        def draw_line(eye_pts):
            for i in range(len(eye_pts)):
                pt1=eye_pts[i]
                pt2=eye_pts[(i+1)%len(eye_pts)]
                cv2.line(image,pt1,pt2,(255,0,0),1)
        draw_line(left_eye)
        draw_line(right_eye)

        cv2.putText(image,f"EAR:{avg_EAR}",(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    return avg_EAR