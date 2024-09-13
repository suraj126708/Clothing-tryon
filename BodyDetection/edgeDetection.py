import cv2


Frame_Width = 1020
Frame_Height = int(Frame_Width/16*9)


Border_Size = 3


White = (255, 255, 255)
Light_Blue = (173, 216, 230)
Red = (0, 0, 255)
Black = (0, 0, 0)


cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, Frame_Width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, Frame_Height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))


Upper_Body_Cascade = cv2.CascadeClassifier('Haar/haarcascade_upperbody.xml')
Lower_Body_Cascade = cv2.CascadeClassifier('Haar/haarcascade_lowerbody.xml')


while True:
    ignore, Frame = cam.read()
    cv2.rectangle(Frame, (0, 0), (Frame_Width, Frame_Height), (White), Border_Size)
    Frame = cv2.flip(Frame, 1)
    Frame_Gray = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
    Upper_Bodies = Upper_Body_Cascade.detectMultiScale(Frame_Gray, 1.3, 5)
    Lower_Bodies = Lower_Body_Cascade.detectMultiScale(Frame_Gray, 1.3, 5)


    for Upper_Body in Upper_Bodies:
        Upper_Body_X, Upper_Body_Y, Upper_Body_Width, Upper_Body_Height = Upper_Body
        cv2.rectangle(Frame, (Upper_Body_X, Upper_Body_Y), (Upper_Body_X + Upper_Body_Width, Upper_Body_Y + Upper_Body_Height), (Light_Blue), Border_Size)


    for Lower_Body in Lower_Bodies:
        Lower_Body_X, Lower_Body_Y, Lower_Body_Width, Lower_Body_Height = Lower_Body
        cv2.rectangle(Frame, (Lower_Body_X, Lower_Body_Y), (Lower_Body_X + Lower_Body_Width, Lower_Body_Y + Lower_Body_Height), (Red), Border_Size)


    cv2.imshow('Body Detection', Frame)
    cv2.moveWindow('Body Detection', 0, 0)


    if cv2.waitKey(1) & 0xff == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()