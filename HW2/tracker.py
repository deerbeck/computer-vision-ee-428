import cv2
from magicwand import *
import argparse
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('video', help='path to input video file')
parser.add_argument('--output', help='path to output video file (optional)')
parser.add_argument('--calibration', default='iphone_calib.txt',
                    help='path to calibration file')
parser.add_argument('--ball_radius', type=float,
                    default=3, help='radius of ball in cm')
args = parser.parse_args()

wand = MagicWand(calibration_path=args.calibration, R=args.ball_radius)

cap = cv2.VideoCapture(args.video)

single_balls = []
distances = []

to_im = 0
double = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # draw circles, cube and get detected circles in the frame
    all_balls = wand.process_frame(frame)

    # capture single ball for 3D Plot
    if len(all_balls) == 1:
        single_balls.append(all_balls[0])

    # calculate distance for wand
    elif len(all_balls) == 2:
        # get delta coordinates for 3D Distance Calculation
        dX, dY, dZ = np.array(all_balls[0])-np.array(all_balls[1])
        distance = np.sqrt(dX**2 + dY**2 + dZ**2)

        # append to distances list to mean out later
        distances.append(distance)
        double += 1

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
    to_im += 1

print(f"Single Ball Ration : {len(single_balls)/to_im}")
print(f"Double Ball Ration : {double/to_im}")

# plotting 3D trajectory of Ball:
single_circles = np.array(single_balls)

fig = plt.figure()
ax = plt.axes(projection='3d')

# extract 3D points from list
xdata = single_circles[:, 0]
ydata = single_circles[:, 1]
zdata = single_circles[:, 2]

# plot Ball trajectory
ax.plot3D(xdata, ydata, zdata, 'red')

# Set labels for the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# set initial view to the same as POV of camera
ax.view_init(azim=-90, elev=-90)

# show 3D Plot
plt.show()

# print out mean of distances
print(f"Wand Distance: {np.mean(distances):.3f} cm")

# mean distance for wand.mov: 36.902 cm, so pretty close to ground truth of 36
# cm
