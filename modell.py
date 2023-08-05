def predict_image(image):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import MinMaxScaler
    import sklearn.preprocessing
    from sklearn.preprocessing import normalize
    from pandas.core.common import random_state
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import numpy as np

    import math
    import cv2
    import numpy as np
    from time import time
    import mediapipe as mp
    import matplotlib.pyplot as plt
    from IPython.display import HTML


    file_1 = pd.read_csv('3d_distances.csv')
    file_2 = pd.read_csv('angles.csv')
    file_3 = pd.read_csv('labels.csv')
    file_4 = pd.read_csv('landmarks.csv')
    file_5 = pd.read_csv('xyz_distances.csv')

    merged_data = pd.merge(file_4,file_5,on='pose_id')
    merged_data = pd.merge(merged_data,file_1,on='pose_id')
    merged_data = pd.merge(merged_data,file_2,on='pose_id')

    merged_data = pd.merge(merged_data,file_3,on='pose_id')

    merged_data = merged_data.drop('pose_id',axis=1)
    merged_data = merged_data.drop('right_knee_mid_hip_left_knee',axis=1)

    x= merged_data.drop(['pose'], axis='columns')
    y = merged_data['pose']

    encoder = LabelEncoder()
    y = merged_data['pose']
    y = encoder.fit_transform(y)
    
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weights_dict = {}
    for idx, k in enumerate(class_weights):
        class_weights_dict[idx] = k 

    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    classifier = RandomForestClassifier(random_state=42)

    classifier.fit(X_train,Y_train)
  
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
    mp_drawing = mp.solutions.drawing_utils

    points = mp_pose.PoseLandmark   
    data = []

    for p in points:
            x = str(p)[13:]
            data.append(x + "_x")
            data.append(x + "_y")
            data.append(x + "_z")
            data.append(x + "_vis")
    data = pd.DataFrame(columns = data) 

    count = 0
    temp = []
    img_array = np.array(image)

    img_arrayRGB = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    blackie = np.zeros(img_array.shape) 

    results = pose.process(img_arrayRGB)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(blackie, results.pose_landmarks, mp_pose.POSE_CONNECTIONS) 

        landmarks = results.pose_landmarks.landmark


        dict = {}
        for point, j in zip(points, landmarks):
            temp = temp + [j.x, j.y, j.z]
            key = point.name
            value = [temp[count], temp[count+1], temp[count+2]]
            dict[key] = value
            count = count + 1

    #xyz_distances
    c1 = dict['LEFT_WRIST'][0] - dict['LEFT_SHOULDER'][0]
    c2 = dict['LEFT_WRIST'][1] - dict['LEFT_SHOULDER'][1]
    c3 =  dict['LEFT_WRIST'][2] - dict['LEFT_SHOULDER'][2]

    c4 = dict['RIGHT_WRIST'][0] - dict['RIGHT_SHOULDER'][0]
    c5 = dict['RIGHT_WRIST'][1] - dict['RIGHT_SHOULDER'][1]
    c6 = dict['RIGHT_WRIST'][2] - dict['RIGHT_SHOULDER'][2]

    c7	= dict['LEFT_ANKLE'][0] - dict['LEFT_HIP'][0]
    c8	= dict['LEFT_ANKLE'][1] - dict['LEFT_HIP'][1]
    c9	= dict['LEFT_ANKLE'][2] - dict['LEFT_HIP'][2]

    c10	=  dict['RIGHT_ANKLE'][0] - dict['RIGHT_HIP'][0]
    c11	= dict['RIGHT_ANKLE'][1] - dict['RIGHT_HIP'][1]
    c12 = dict['RIGHT_ANKLE'][2] - dict['RIGHT_HIP'][2]

    c13	= dict['LEFT_WRIST'][0] - dict['LEFT_HIP'][0]
    c14	=  dict['LEFT_WRIST'][1] - dict['LEFT_HIP'][1]
    c15=  dict['LEFT_WRIST'][2] - dict['LEFT_HIP'][2]

    c16	= dict['RIGHT_WRIST'][0] - dict['RIGHT_HIP'][0]
    c17	= dict['RIGHT_WRIST'][1] - dict['RIGHT_HIP'][1]
    c18= dict['RIGHT_WRIST'][2] - dict['RIGHT_HIP'][2]

    c19	= dict['LEFT_ANKLE'][0] - dict['LEFT_SHOULDER'][0]
    c20 = dict['LEFT_ANKLE'][1] - dict['LEFT_SHOULDER'][1]
    c21 = dict['LEFT_ANKLE'][2] - dict['LEFT_SHOULDER'][2]

    c22 = dict['RIGHT_ANKLE'][0] - dict['RIGHT_SHOULDER'][0]
    c23 = dict['RIGHT_ANKLE'][1] - dict['RIGHT_SHOULDER'][1]
    c24 = dict['RIGHT_ANKLE'][2] - dict['RIGHT_SHOULDER'][2]

    c25 = dict['RIGHT_WRIST'][0] - dict['LEFT_HIP'][0]
    c26 = dict['RIGHT_WRIST'][1] - dict['LEFT_HIP'][1]
    c27 = dict['RIGHT_WRIST'][2] - dict['LEFT_HIP'][2]

    c28 = dict['LEFT_WRIST'][0] - dict['RIGHT_HIP'][0]
    c29	= dict['LEFT_WRIST'][1] - dict['RIGHT_HIP'][1]
    c30	= dict['LEFT_WRIST'][2] - dict['RIGHT_HIP'][2]

    c31 = dict['RIGHT_ELBOW'][0] - dict['LEFT_ELBOW'][0]
    c32 = dict['RIGHT_ELBOW'][1] - dict['LEFT_ELBOW'][1]
    c33 = dict['RIGHT_ELBOW'][2] - dict['LEFT_ELBOW'][2]

    c34 = dict['RIGHT_KNEE'][0] - dict['LEFT_KNEE'][0]
    c35	= dict['RIGHT_KNEE'][1] - dict['LEFT_KNEE'][1]
    c36	= dict['RIGHT_KNEE'][2] - dict['LEFT_KNEE'][2]

    c37 = dict['RIGHT_WRIST'][0] - dict['LEFT_WRIST'][0]
    c38 = dict['RIGHT_WRIST'][1] - dict['LEFT_WRIST'][1]
    c39 = dict['RIGHT_WRIST'][2] - dict['LEFT_WRIST'][2]

    c40 = dict['RIGHT_ANKLE'][0] - dict['LEFT_ANKLE'][0]
    c41	= dict['RIGHT_ANKLE'][1] - dict['LEFT_ANKLE'][1]
    c42= dict['RIGHT_ANKLE'][2] - dict['LEFT_ANKLE'][2]

    c43 = dict['LEFT_HIP'][0]-(dict['LEFT_WRIST'][0] + dict['LEFT_ANKLE'][0])/2
    c44 = dict['LEFT_HIP'][1]-(dict['LEFT_WRIST'][1] + dict['LEFT_ANKLE'][1])/2
    c45 = dict['LEFT_HIP'][2]-(dict['LEFT_WRIST'][2] + dict['LEFT_ANKLE'][2])/2

    c46 = dict['RIGHT_HIP'][0]-(dict['RIGHT_WRIST'][0] + dict['RIGHT_ANKLE'][0])/2
    c47	= dict['RIGHT_HIP'][1]-(dict['RIGHT_WRIST'][1] + dict['RIGHT_ANKLE'][1])/2
    c48 = dict['RIGHT_HIP'][2]-(dict['RIGHT_WRIST'][2] + dict['RIGHT_ANKLE'][2])/2



    #3d_distances

    c49 = np.sqrt((dict['LEFT_WRIST'][0] - dict['LEFT_SHOULDER'][0])**2 + (dict['LEFT_WRIST'][1] - dict['LEFT_SHOULDER'][1])**2 + (dict['LEFT_WRIST'][2] - dict['LEFT_SHOULDER'][2])**2)
    c50 = np.sqrt((dict['RIGHT_WRIST'][0] - dict['RIGHT_SHOULDER'][0])**2 + (dict['RIGHT_WRIST'][1] - dict['RIGHT_SHOULDER'][1])**2 + (dict['RIGHT_WRIST'][2] - dict['RIGHT_SHOULDER'][2])**2)
    c51= np.sqrt((dict['LEFT_ANKLE'][0] - dict['LEFT_HIP'][0])**2 + (dict['LEFT_ANKLE'][1] - dict['LEFT_HIP'][1])**2 + (dict['LEFT_ANKLE'][2] - dict['LEFT_HIP'][2])**2)
    c52= np.sqrt((dict['RIGHT_ANKLE'][0] - dict['RIGHT_HIP'][0])**2 + (dict['RIGHT_ANKLE'][1] - dict['RIGHT_HIP'][1])**2 + (dict['RIGHT_ANKLE'][2] - dict['RIGHT_HIP'][2])**2)
    c53	= np.sqrt((dict['LEFT_WRIST'][0] - dict['LEFT_HIP'][0])**2 + (dict['LEFT_WRIST'][1] - dict['LEFT_HIP'][1])**2 + (dict['LEFT_WRIST'][2] - dict['LEFT_HIP'][2])**2)
    c54	= np.sqrt((dict['RIGHT_WRIST'][0] - dict['RIGHT_HIP'][0])**2 + (dict['RIGHT_WRIST'][1] - dict['RIGHT_HIP'][1])**2 + (dict['RIGHT_WRIST'][2] - dict['RIGHT_HIP'][2])**2)
    c55 = np.sqrt((dict['LEFT_ANKLE'][0] - dict['LEFT_SHOULDER'][0])**2 + (dict['LEFT_ANKLE'][1] - dict['LEFT_SHOULDER'][1])**2 + (dict['LEFT_ANKLE'][2] - dict['LEFT_SHOULDER'][2])**2)
    c56 = np.sqrt((dict['RIGHT_ANKLE'][0] - dict['RIGHT_SHOULDER'][0])**2 + (dict['RIGHT_ANKLE'][1] - dict['RIGHT_SHOULDER'][1])**2 + (dict['RIGHT_ANKLE'][2] - dict['RIGHT_SHOULDER'][2])**2)
    c57 = np.sqrt((dict['RIGHT_WRIST'][0] - dict['LEFT_HIP'][0])**2 + (dict['RIGHT_WRIST'][1] - dict['LEFT_HIP'][1])**2 + (dict['RIGHT_WRIST'][2] - dict['LEFT_HIP'][2])**2)
    c58 = np.sqrt((dict['LEFT_WRIST'][0] - dict['RIGHT_HIP'][0])**2 + (dict['LEFT_WRIST'][1] - dict['RIGHT_HIP'][1])**2 + (dict['LEFT_WRIST'][2] - dict['RIGHT_HIP'][2])**2)
    c59= np.sqrt((dict['RIGHT_ELBOW'][0] - dict['LEFT_ELBOW'][0])**2 + (dict['RIGHT_ELBOW'][1] - dict['LEFT_ELBOW'][1])**2 + (dict['RIGHT_ELBOW'][2] - dict['LEFT_ELBOW'][2])**2)
    c60 = np.sqrt((dict['RIGHT_KNEE'][0] - dict['LEFT_KNEE'][0])**2 + (dict['RIGHT_KNEE'][1] - dict['LEFT_KNEE'][1])**2 + (dict['RIGHT_KNEE'][2] - dict['LEFT_KNEE'][2])**2)
    c61 = np.sqrt((dict['RIGHT_WRIST'][0] - dict['LEFT_WRIST'][0])**2 + (dict['RIGHT_WRIST'][1] - dict['LEFT_WRIST'][1])**2 + (dict['RIGHT_WRIST'][2] - dict['LEFT_WRIST'][2])**2)
    c62 = np.sqrt((dict['RIGHT_ANKLE'][0] - dict['LEFT_ANKLE'][0])**2 + (dict['RIGHT_ANKLE'][1] - dict['LEFT_ANKLE'][1])**2 + (dict['RIGHT_ANKLE'][2] - dict['LEFT_ANKLE'][2])**2)

    x_avg = (dict['LEFT_WRIST'][0] + dict['LEFT_ANKLE'][0])/2
    y_avg = (dict['LEFT_WRIST'][1] + dict['LEFT_ANKLE'][1])/2
    z_avg = (dict['LEFT_WRIST'][2] + dict['LEFT_ANKLE'][2])/2

    c63 = np.sqrt((x_avg - dict['LEFT_HIP'][0])**2 + (y_avg - dict['LEFT_HIP'][1])**2 + (z_avg - dict['LEFT_HIP'][2])**2)



    x_avg = (dict['RIGHT_WRIST'][0] + dict['LEFT_ANKLE'][0])/2
    y_avg = (dict['RIGHT_WRIST'][1] + dict['LEFT_ANKLE'][1])/2
    z_avg = (dict['RIGHT_WRIST'][2] + dict['LEFT_ANKLE'][2])/2
    c64 = np.sqrt((x_avg - dict['RIGHT_HIP'][0])**2 + (y_avg - dict['RIGHT_HIP'][1])**2 + (z_avg - dict['RIGHT_HIP'][2])**2)

    #angles
    import math
    import numpy as np

    Ax = dict['RIGHT_SHOULDER'][0] - dict['RIGHT_ELBOW'][0]
    Ay = dict['RIGHT_SHOULDER'][1] - dict['RIGHT_ELBOW'][1]
    Az = dict['RIGHT_SHOULDER'][2] - dict['RIGHT_ELBOW'][2]
    Bx = dict['RIGHT_SHOULDER'][0] - dict['RIGHT_HIP'][0]
    By = dict['RIGHT_SHOULDER'][1] - dict['RIGHT_HIP'][1]
    Bz = dict['RIGHT_SHOULDER'][2] - dict['RIGHT_HIP'][2]
    AdotB = (Ax * Bx) + (Ay * By) + (Az * Bz)
    modA = np.sqrt(Ax**2 + Ay**2 + Az**2)
    modB = np.sqrt(Bx**2 + By**2 + Bz**2)
    cos_angle = AdotB / (modA * modB)
    angle = np.arccos(cos_angle)
    c65= np.degrees(angle)

    Ax = dict['LEFT_SHOULDER'][0] - dict['LEFT_ELBOW'][0]
    Ay = dict['LEFT_SHOULDER'][1] - dict['LEFT_ELBOW'][1]
    Az = dict['LEFT_SHOULDER'][2] - dict['LEFT_ELBOW'][2]
    Bx = dict['LEFT_SHOULDER'][0] - dict['LEFT_HIP'][0]
    By = dict['LEFT_SHOULDER'][1] - dict['LEFT_HIP'][1]
    Bz = dict['LEFT_SHOULDER'][2] - dict['LEFT_HIP'][2]
    AdotB = (Ax * Bx) + (Ay * By) + (Az * Bz)
    modA = np.sqrt(Ax**2 + Ay**2 + Az**2)
    modB = np.sqrt(Bx**2 + By**2 + Bz**2)
    cos_angle = AdotB / (modA * modB)
    angle = np.arccos(cos_angle)
    c66	= np.degrees(angle)

    Ax = (dict['LEFT_HIP'][0] + dict['RIGHT_HIP'][0])/2 - dict['RIGHT_KNEE'][0]
    Ay =(dict['LEFT_HIP'][1] + dict['RIGHT_HIP'][1])/2 - dict['RIGHT_KNEE'][1]
    Az = (dict['LEFT_HIP'][2] + dict['RIGHT_HIP'][2])/2 - dict['RIGHT_KNEE'][2]
    Bx = (dict['LEFT_HIP'][0] + dict['RIGHT_HIP'][0])/2 - dict['LEFT_KNEE'][0]
    By = (dict['LEFT_HIP'][1] + dict['RIGHT_HIP'][1])/2 - dict['LEFT_KNEE'][1]
    Bz = (dict['LEFT_HIP'][2] + dict['RIGHT_HIP'][2])/2 - dict['LEFT_KNEE'][2]
    AdotB = (Ax * Bx) + (Ay * By) + (Az * Bz)
    modA = np.sqrt(Ax**2 + Ay**2 + Az**2)
    modB = np.sqrt(Bx**2 + By**2 + Bz**2)
    cos_angle = AdotB / (modA * modB)
    angle = np.arccos(cos_angle)
    c67 = np.degrees(angle)


    Ax = dict['RIGHT_KNEE'][0] - dict['RIGHT_HIP'][0]
    Ay = dict['RIGHT_KNEE'][1] - dict['RIGHT_HIP'][1]
    Az = dict['RIGHT_KNEE'][2] - dict['RIGHT_HIP'][2]
    Bx = dict['RIGHT_KNEE'][0] - dict['RIGHT_ANKLE'][0]
    By = dict['RIGHT_KNEE'][1] - dict['RIGHT_ANKLE'][1]
    Bz = dict['RIGHT_KNEE'][2] - dict['RIGHT_ANKLE'][2]
    AdotB = (Ax * Bx) + (Ay * By) + (Az * Bz)
    modA = np.sqrt(Ax**2 + Ay**2 + Az**2)
    modB = np.sqrt(Bx**2 + By**2 + Bz**2)
    cos_angle = AdotB / (modA * modB)
    angle = np.arccos(cos_angle)
    c68 = np.degrees(angle)

    Ax = dict['LEFT_KNEE'][0] - dict['LEFT_HIP'][0]
    Ay = dict['LEFT_KNEE'][1] - dict['LEFT_HIP'][1]
    Az = dict['LEFT_KNEE'][2] - dict['LEFT_HIP'][2]
    Bx = dict['LEFT_KNEE'][0] - dict['LEFT_ANKLE'][0]
    By = dict['LEFT_KNEE'][1] - dict['LEFT_ANKLE'][1]
    Bz = dict['LEFT_KNEE'][2] - dict['LEFT_ANKLE'][2]
    AdotB = (Ax * Bx) + (Ay * By) + (Az * Bz)
    modA = np.sqrt(Ax**2 + Ay**2 + Az**2)
    modB = np.sqrt(Bx**2 + By**2 + Bz**2)
    cos_angle = AdotB / (modA * modB)
    angle = np.arccos(cos_angle)
    c69=  np.degrees(angle)


    Ax = dict['RIGHT_ELBOW'][0] - dict['RIGHT_WRIST'][0]
    Ay = dict['RIGHT_ELBOW'][1] - dict['RIGHT_WRIST'][1]
    Az = dict['RIGHT_ELBOW'][2] - dict['RIGHT_WRIST'][2]
    Bx = dict['RIGHT_ELBOW'][0] - dict['RIGHT_SHOULDER'][0]
    By = dict['RIGHT_ELBOW'][1] - dict['RIGHT_SHOULDER'][1]
    Bz = dict['RIGHT_ELBOW'][2] - dict['RIGHT_SHOULDER'][2]
    AdotB = (Ax * Bx) + (Ay * By) + (Az * Bz)
    modA = np.sqrt(Ax**2 + Ay**2 + Az**2)
    modB = np.sqrt(Bx**2 + By**2 + Bz**2)
    cos_angle = AdotB / (modA * modB)
    angle = np.arccos(cos_angle)
    c70 = np.degrees(angle)

    Ax = dict['LEFT_ELBOW'][0] - dict['LEFT_WRIST'][0]
    Ay = dict['LEFT_ELBOW'][1] - dict['LEFT_WRIST'][1]
    Az = dict['LEFT_ELBOW'][2] - dict['LEFT_WRIST'][2]
    Bx = dict['LEFT_ELBOW'][0] - dict['LEFT_SHOULDER'][0]
    By = dict['LEFT_ELBOW'][1] - dict['LEFT_SHOULDER'][1]
    Bz = dict['LEFT_ELBOW'][2] - dict['LEFT_SHOULDER'][2]
    AdotB = (Ax * Bx) + (Ay * By) + (Az * Bz)
    modA = np.sqrt(Ax**2 + Ay**2 + Az**2)
    modB = np.sqrt(Bx**2 + By**2 + Bz**2)
    cos_angle = AdotB / (modA * modB)
    angle = np.arccos(cos_angle)
    c71 = np.degrees(angle)

    for i in range(1,72):
        if(i!=67):
            temp.append(locals()['c' + str(i)])

    temp = np.array(temp)         
    temp = temp.reshape(-1, 169)   
    
    predictions = classifier.predict(temp)
    y_pred_class_names = encoder.inverse_transform(predictions)
    return y_pred_class_names