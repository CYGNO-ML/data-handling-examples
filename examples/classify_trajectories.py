import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def encode(traj):
    shape = traj.shape
    if len(shape) == 1:
        x = [traj[0]]
        y = [traj[1]]
        z = [traj[2]]
        en = [traj[3]]
        t = [traj[4]]
    else:
        x = traj[:, 0]
        y = traj[:, 1]
        z = traj[:, 2]
        en = traj[:, 3]
        t = traj[:, 4]
    no_samples = len(t)
    tot_energy = np.sum(en)

    mean_vel = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2)/np.diff(t))
    dir_per = 1/no_samples * np.sum(np.arctan2(np.diff(y), np.diff(x)))

    # print(no_samples, tot_energy, mean_vel, dir_per)
    return np.array([no_samples, tot_energy, mean_vel, dir_per])


def encode_trajs(trajs, normalize=True):
    encoded_trajs = []
    for traj in trajs:
        encoded = encode(traj)

        encoded_trajs.append(encoded)
    enc = np.array(encoded_trajs)
    if normalize:
        scaler = MinMaxScaler()
        scaler.fit(enc)
        enc = scaler.transform(enc)
    return enc


def encode_classes(classes):
    d = {ni: indi for indi, ni in enumerate(set(classes))}
    numbers = [d[ni] for ni in classes]
    return np.array(numbers), d


def classify(x, y, test_size=0.3):

    x, y = shuffle(x, y)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=109)

    # Create a Classifier
    clf = LogisticRegression()

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred_train = clf.predict(X_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    print('Accuracy of Logistic regression classifier on training set: {:.2f}'
          .format(clf.score(X_train, y_train)))
    print('Accuracy of Logistic regression classifier on test set: {:.2f}'
          .format(clf.score(X_test, y_test)))


if __name__ == "__main__":
    from cygnods import CygnoDataset
    dataset_path = "../../CYGNO-ML-DATASET"

    # Create an instance of the dataset handler providing the path
    dataset = CygnoDataset(dataset_path)
    trajs, classes = dataset.load_all_trajs()
    encoded_data = encode_trajs(trajs, normalize=True)
    encoded_classes, d = encode_classes(classes)

    classify(encoded_data, encoded_classes, test_size=0.3)
