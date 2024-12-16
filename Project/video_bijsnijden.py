import cv2

def crop_video(input_path, output_path, top, bottom, left, right):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) - left - right
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) - top - bottom
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cropped_frame = frame[top:frame.shape[0]-bottom, left:frame.shape[1]-right]
        out.write(cropped_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

input_path = 'output-temp/output/output_obama-15-12-2024_180548.mp4'
output_path = 'output-temp/output/output_obama-nieuw.mp4'
top = 50    # Aantal pixels om van de bovenkant af te trekken
bottom = 10 # Aantal pixels om van de onderkant af te trekken
left = 20   # Aantal pixels om van de linkerkant af te trekken
right = 55  # Aantal pixels om van de rechterkant af te trekken
crop_video(input_path, output_path, top, bottom, left, right)
