from pennylane import numpy as np


class Sampler:
    @staticmethod
    def circle(n_points=10, radius:float = 0.5, center: list = [0, 0], spread: float = 1):
        if not isinstance(center, np.ndarray):
            center = np.array(center, requires_grad=False)
        points = 2 * spread * (np.random.rand(n_points, 2) - 0.5)
        distances = np.sqrt(np.sum((points-center)**2, axis=1))
        labels = np.where(distances <= radius, 0, 1)

        # Transform normal points to pennylane tensors
        points = np.array(points, requires_grad=False)
        labels = np.array(labels, requires_grad=False)

        return points, labels

    @staticmethod
    def annulus(n_points:int =10,
                inner_radius=0.5*(2/np.pi)**0.5,
                outer_radius=(2/np.pi)**0.5,
                center=[0,0],
                scale:float=1):

        if not isinstance(center, np.ndarray):
            center = np.array(center)
        points = (np.random.rand(n_points, 2) - 0.5) * 2 * scale
        distances = np.sqrt(np.sum((points - center) ** 2, axis=1))

        # Points inside the annulus have distances between inner_radius and outer_radius
        labels = np.where((distances >= inner_radius) & (distances <= outer_radius), 0, 1)

        # Transform normal points to pennylane tensors
        points = np.array(points, requires_grad=False)
        labels = np.array(labels, requires_grad=False)

        return points, labels

    @staticmethod
    def multi_circle(centers=[[0.3, 0.6], [0.7, 0.3]], radii=[0.3, 0.25], n_points=100):
        if not isinstance(centers, np.ndarray):
            centers = np.array(centers, requires_grad=False)
        if not isinstance(radii, np.ndarray):
            radii = np.array(radii, requires_grad=False)

        points = np.random.rand(n_points, 2)
        labels = np.ones(n_points)  # Default to 1 (outside all circles)

        for center, radius in zip(centers, radii):
            distances = np.sqrt(np.sum((points - center) ** 2, axis=1))
            inside = distances <= radius
            labels[inside] = 0  # Set label to 0 if inside any circle

        points = np.array(points, requires_grad=False)
        labels = np.array(labels, requires_grad=False)

        return points, labels


    @staticmethod
    def sinus(n_points):
        """
        Generates a dataset of points with 1/0 labels
        depending on whether they are above or below the sine function
        Args:
            n_points (int): number of samples to generate

        Returns:
            Xvals (array[tuple]): coordinates of points
            yvals (array[int]): classification labels
        """
        Xvals, yvals = [], []

        for i in range(n_points):
            x = 2 * (np.random.rand(2)) - 1
            y = 0
            f= 0.8*np.sin(-2*np.pi*x[0]/2)
            point = x[1]
            if f < point:
                y = 1
            Xvals.append(x)
            yvals.append(y)
        return np.array(Xvals, requires_grad=False), np.array(yvals, requires_grad=False)


    @staticmethod
    def sinus_dif(n_points):
        """
        Generates a dataset of points with 1/0 labels
        depending on whether they are above or below the sine function
        Args:
            n_points (int): number of samples to generate

        Returns:
            Xvals (array[tuple]): coordinates of points
            yvals (array[int]): classification labels
        """
        Xvals, yvals = [], []

        for i in range(n_points):
            x = 2 * (np.random.rand(2)) - 1
            y = 0
            #f= np.sin(2*np.pi*x[0]/2+np.pi/2)
            f= np.sin(-2*np.pi*x[0]/1)
            point = x[1]
            if f < point:
                y = 1
            Xvals.append(x)
            yvals.append(y)
        return np.array(Xvals, requires_grad=False), np.array(yvals, requires_grad=False)


    @staticmethod
    def corners(n_points):
        Xvals, yvals = [], []

        tl=[-1.0, 1.0]
        tr=[1.0, 1.0]
        bl=[-1.0, -1.0]
        br=[1.0, -1.0]

        radius = 0.75

        for i in range(n_points):
            x = 2 * (np.random.rand(2)) - 1
            y = 0

            d_tl = np.linalg.norm(x-tl)
            d_tr = np.linalg.norm(x-tr)
            d_bl = np.linalg.norm(x-bl)
            d_br = np.linalg.norm(x-br)

            if d_tl<radius or d_tr<radius or d_bl<radius or d_br<radius:
                y=1
            Xvals.append(x)
            yvals.append(y)
        return np.array(Xvals, requires_grad=False), np.array(yvals, requires_grad=False)

    @staticmethod
    def spiral(n_points):
        n_points= n_points // 2
        theta = np.sqrt(np.random.rand(n_points)) * 2 * np.pi # np.linspace(0,2*pi,100)

        r_a = 2*theta + np.pi
        data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
        x_a = (data_a + np.random.randn(n_points, 2))/20

        r_b = -2*theta - np.pi
        data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
        x_b = (data_b + np.random.randn(n_points, 2))/20
        xvals = np.append(x_a, x_b, axis=0)
        yvals = np.concatenate([np.ones(n_points), np.zeros(n_points)], axis=0)

        data = list(zip(np.array(xvals, requires_grad=False), np.array(yvals, requires_grad=False)))
        np.random.shuffle(data)

        equis, yes = zip(*data)

        equis, yes = np.array(equis, requires_grad=False), np.array(yes, requires_grad=False)

        return equis, yes


    @staticmethod
    def rectangle(width=1/3, height=1/2, center=[1/2, 1/2], n_points=10):
        if not isinstance(center, np.ndarray):
            center = np.array(center, requires_grad=False)
        points = np.random.rand(n_points, 2)
        # Adjust conditions based on your center definition (center vs top-left corner)
        labels = np.where((points[:, 0] >= center[0] - width/2) & (points[:, 0] <= center[0] + width/2) &
                          (points[:, 1] >= center[1] - height/2) & (points[:, 1] <= center[1] + height/2), 0, 1)

        # Transform normal points to pennylane tensors (same as circle)
        points = np.array(points, requires_grad=False)
        labels = np.array(labels, requires_grad=False)

        return points, labels