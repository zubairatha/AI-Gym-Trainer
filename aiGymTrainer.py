import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
from playsound import  playsound
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def checkHandInBox(handX, handY, tlX, tlY, brX, brY):
    if handX>tlX and handX<brX and handY>tlY and handY<brY:
        return True
    else:
        return False

def createMenu():
    cv2.destroyAllWindows()

    cap = cv2.VideoCapture(0)

    curlTlx=220;curlBrx=400;curlTly=100;curlBry=160
    deadliftTlx=220;deadliftBrx=400;deadliftTly=200;deadliftBry=260
    squatTlx=220;squatBrx=400;squatTly=300;squatBry=360

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates

                leftHand = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].y]
                rightHand = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].y]

                if(checkHandInBox(leftHand[0]*640,leftHand[1]*480 ,curlTlx, curlTly, curlBrx, curlBry) or checkHandInBox(rightHand[0]*640,rightHand[1]*480 ,curlTlx, curlTly, curlBrx, curlBry)):
                    cap.release()
                    cv2.destroyAllWindows()
                    curl()
                    break

                if(checkHandInBox(leftHand[0]*640,leftHand[1]*480 ,deadliftTlx, deadliftTly, deadliftBrx, deadliftBry) or checkHandInBox(rightHand[0]*640,rightHand[1]*480 ,deadliftTlx, deadliftTly, deadliftBrx, deadliftBry)):
                    cap.release()
                    cv2.destroyAllWindows()
                    deadlift()
                    break

                if(checkHandInBox(leftHand[0]*640,leftHand[1]*480 ,squatTlx, squatTly, squatBrx, squatBry) or checkHandInBox(rightHand[0]*640,rightHand[1]*480 ,squatTlx, squatTly, squatBrx, squatBry)):
                    cap.release()
                    cv2.destroyAllWindows()
                    squat()
                    break
                        
            except:
                pass

            # curlTlx=220;curlBrx=400;curlTly=200;curlBry=260
            cv2.rectangle(image, (curlTlx,curlTly), (curlBrx,curlBry), (245,117,16), -1)
            cv2.putText(image, 'BICEP CURL', (220,160), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
            # deadliftTlx=220;deadliftBrx=400;deadliftTly=300;deadliftBry=360
            cv2.rectangle(image, (deadliftTlx,deadliftTly), (deadliftBrx,deadliftBry), (245,117,16), -1)
            cv2.putText(image, 'DEADLIFT', (220,260), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
            # squatTlx=220;squatBrx=400;squatTly=400;squatBry=460
            cv2.rectangle(image, (squatTlx,squatTly), (squatBrx,squatBry), (245,117,16), -1)
            cv2.putText(image, 'SQUAT', (220,360), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )  
            cv2.imshow('Mediapipe Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 


def curl():
    cv2.destroyAllWindows()

    cap = cv2.VideoCapture(0)

    # Curl counter variables
    counter = 0 
    stage = None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                leftHand = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].y]
                rightHand = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].y]

                if(checkHandInBox(leftHand[0]*640,leftHand[1]*480 ,590,20,670,80) or checkHandInBox(rightHand[0]*640,rightHand[1]*480,600,0,680,60)):
                    cap.release()
                    cv2.destroyAllWindows()
                    createMenu()
                    break
                
                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)
                
                # Visualize angle
                cv2.putText(image, str(angle), 
                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                # Curl counter logic
                if angle > 160:
                    stage = "down"
                if angle < 30 and stage =='down':
                    stage="up"
                    counter +=1
                    mytext=str(counter)
                    language='en'
                    myobj=gTTS(text=mytext,lang=language,slow=True)
                    myobj.save("welcome1.mp3")
                    playsound("welcome1.mp3")
                    print(counter)
                        
            except:
                pass
            
            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (240,73), (245,117,16), -1)
            cv2.rectangle(image, (590,20), (670,80), (245,117,16), -1)
            cv2.putText(image, 'S', (590,79), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            # cv2.putText(image, 'CURL', 
            #             (480,60), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (85,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (80,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def checkDeadlift(a,b,c,d,checkUp=False):
    if c.visibility>0.5 and d.visibility>0.5:
        print("visible")
        if checkUp==True:
            if a.y<c.y and b.y<d.y:
                return True
            else:
                return False
        else:
            if a.y>c.y and b.y>d.y:
                return True
            else:
                return False
    else:
        print("not visible")
        return False


def deadlift():
    cv2.destroyAllWindows()

    cap = cv2.VideoCapture(0)

    # Deadlift counter variables
    counter = 0 
    stage = None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # left_hand = landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value]
                # right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value]
                # left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                # right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

                leftHand = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].y]
                rightHand = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].y]

                if(checkHandInBox(leftHand[0]*640,leftHand[1]*480 ,590,20,670,80) or checkHandInBox(rightHand[0]*640,rightHand[1]*480,600,0,680,60)):
                    cap.release()
                    cv2.destroyAllWindows()
                    createMenu()
                    break

                left_hand = landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value]
                right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value]
                left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

                if checkDeadlift(left_hand,right_hand,left_knee,right_knee):
                    stage = "down"
                if stage=="down" and checkDeadlift(left_hand,right_hand,left_knee,right_knee,checkUp=True):
                    stage="up"
                    counter+=1
                    mytext=str(counter)
                    language='en'
                    myobj=gTTS(text=mytext,lang=language,slow=True)
                    myobj.save("welcome1.mp3")
                    playsound("welcome1.mp3")
                    print(counter)
                        
            except:
                pass
            
            # Setup status box
            cv2.rectangle(image, (0,0), (240,73), (245,117,16), -1)
            cv2.rectangle(image, (590,20), (670,80), (245,117,16), -1)
            cv2.putText(image, 'S', (590,79), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            # cv2.putText(image, 'DEADLIFT', 
            #             (460,60), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (85,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (80,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def checkSquat(a,b,c,d,checkUp=False):
    if a.visibilty>0.4 and b.visibility>0.4 and c.visibility>0.4 and d.visibility>0.4:
        print("visible")
        # Cup20=c.y+c.y*0.2
        Cdown30=c.y-c.y*0.6
        # Dup20=d.y+d.y*0.2
        Ddown30=d.y-d.y*0.6
        if checkUp==True:
            if a.y<Cdown30 and b.y<Ddown30:
                return True
            else:
                return False
        else:
            if a.y>Cdown30 and b.y>Ddown30:
                return True
            else:
                return False
    else:
        print("not visible")
        return False


def squat():
    cv2.destroyAllWindows()
    cap = cv2.VideoCapture(0)

    # Deadlift counter variables
    counter = 0 
    stage = "None"

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

                leftHand = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].y]
                rightHand = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].y]

                if(checkHandInBox(leftHand[0]*640,leftHand[1]*480 ,590,20,670,80) or checkHandInBox(rightHand[0]*640,rightHand[1]*480,600,0,680,60)):
                    cap.release()
                    cv2.destroyAllWindows()
                    createMenu()
                    break

                if checkSquat(left_hip,right_hip,left_knee,right_knee):
                    stage = "down"
                if stage=="down" and checkSquat(left_hip,right_hip,left_knee,right_knee,checkUp=True):
                    stage="up"
                    counter+=1
                    mytext=str(counter)
                    language='en'
                    myobj=gTTS(text=mytext,lang=language,slow=True)
                    myobj.save("welcome1.mp3")
                    playsound("welcome1.mp3")
                    print(counter)
                        
            except:
                pass
            
            # Setup status box
            cv2.rectangle(image, (0,0), (240,73), (245,117,16), -1)
            cv2.rectangle(image, (590,20), (670,80), (245,117,16), -1)
            cv2.putText(image, 'S', (590,79), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (85,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (80,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            # cv2.putText(image, 'SQUAT', 
            #             (470,60), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

createMenu()
