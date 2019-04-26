from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt


def outlier_detection(data, eps):

    envelope = EllipticEnvelope(contamination=eps)
    X_train = data[1].values.reshape(-1, 1)
    envelope.fit(X_train)
    data['deviation'] = envelope.decision_function(X_train)
    data['anomaly'] = envelope.predict(X_train)

    fig, ax = plt.subplots(figsize=(10, 6))
    a = data.loc[data['anomaly'] == -1, (0, 1)]  # anomaly
    ax.plot(data[0], data[1], color='blue', label='Normal')
    ax.scatter(a[0], a[1], color='red', label='Anomaly')
    plt.legend()
    plt.show()
