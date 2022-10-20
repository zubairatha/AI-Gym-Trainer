# AI-Gym-Trainer
An interactive gym trainer which lets you navigate between various exercise counters using hand gestures. The exercise counters count each valid rep of the user and let you navigate between different exercises.


## Abstract
Today we are knee deep in the technological era where even fitness has become solely connected to highly curated tech. People carried out their fitness workouts in the gym. The pandemic restricted us a lot, but the routines of life must go on. Tons of people went on about their regimes at home but with no knowledge of whether they are performing the exercises right. Interactive Gym Buddy is an AI-powered bot which allows you to choose exercises using hand gestures through your camera and counts the reps you do only when you perform an exercise right. 


## Tech stack
Python - OpenCV and Mediapipe

## Mediapipe
Mediapipe segments the subject to 33 3D landmarks. These landmarks are tracked and it also uses z-index analysis to measure depth as in to indicate whether user's hands are behind their hips or not.
![mediapipe-png](/mediapipe.png)

## Excercises
### Bicep Curl
The angle between shoulder-elbow and elbow-wrist is used to count one valid rep.
### Deadlifts
One valid rep will be considered when hands go below knees and then come back up.
### Squats
One valid rep will be counted when hip-knee range is 30deg.
