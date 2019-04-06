import cognitive_face as CF
import time
import os


KEY = 'fae146c993af496f94abb9b2fecdb426'  # Replace with a valid subscription key (keeping the quotes in place). Lucia
CF.Key.set(KEY)
BASE_URL = 'https://northeurope.api.cognitive.microsoft.com/face/v1.0'  # Replace with your regional Base URL
CF.BaseUrl.set(BASE_URL)

origin = 'data'
person_folders = os.listdir(origin)
destination = 'microsoft'

for person_folder in person_folders:
    print("PERSON = " + person_folder)
    result = "image,anger,contempt,disgust,fear,happiness,neutral,sadness,surprise"
    image_names = os.listdir(os.path.join(origin, person_folder, "test"))
    for image_name in image_names:
        print(image_name)
        try:
            faces = CF.face.detect(os.path.join(origin, person_folder, "test", image_name), attributes="emotion")
        except CF.util.CognitiveFaceException as e:
            if e.status_code == 429:
                print("WAITING")
                time.sleep(59)
                faces = CF.face.detect(os.path.join(origin, person_folder, "test", image_name), attributes="emotion")
            else:
                raise
        result += "\n{}".format(image_name)
        if faces:
            for key, value in faces[0].get('faceAttributes').get('emotion').items():
                result += ",{}".format(value)
        else:
            result += ",,,,,,,,"
    print("WRITING")
    with open(os.path.join(destination, person_folder + ".csv"), "w") as f:
        f.write(result)
