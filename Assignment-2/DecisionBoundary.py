
def plotDecisionBoundary(X1, X2, y, model):
    """
    Plots the decision boundary for a binary classification model along with the training data points.

    Parameters:
        X1 (array-like): Feature values for the first feature.
        X2 (array-like): Feature values for the second feature.
        y (array-like): Target labels.
        model (object): Trained binary classification model with a `predict` method.

    Returns:
        None
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    X1 = np.array(X1)
    X2 = np.array(X2)
    y = np.array(y)

    h = 0.01  # Step size in the mesh grid
    x_min, x_max = X1.min() - 0.1, X1.max() + 0.1
    y_min, y_max = X2.min() - 0.1, X2.max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    x1,x2 = xx.ravel(), yy.ravel()

    # Predict the class for each point in the mesh grid
    Z = model.predict(np.column_stack((x1,x2)))
    Z = Z.reshape(xx.shape)

   
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF']) 
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')
    plt.scatter(X1, X2, c=y, cmap=cmap_bold, edgecolor='k', s=20, alpha=0.7)
    