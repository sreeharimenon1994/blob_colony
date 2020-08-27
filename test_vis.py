import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time



from matplotlib import colors
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

delay = 0.01


class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, ):
        self.stream = self.data_stream()

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots(figsize=(11, 9))
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5, 
                                          init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        data= next(self.stream).T
        x = data[:, 0]
        y = data[:, 1]
        s = data[:, 2]
        c = data[:, 3]

        self.scat = self.ax.scatter(x, y, c=c, s=s)
        self.ax.axis([0, 200, 0, 200])
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def data_stream(self):
        """Generate a random walk (brownian motion). Data is scaled to produce
        a soft "flickering" effect."""
        root = 'C:\\Users\\sreeh\\Downloads\\projects\\ant_food_test_1\\visualise_itr\\'
        # root = 'visualise_itr/'
        path = 'data.npy'
        df = np.load(root+path, allow_pickle=True)

        path = 'wall.npy'
        wall = np.load(root+path, allow_pickle=True)

        path = 'hill.npy'
        hill = np.load(root+path, allow_pickle=True)

        # colr = ['#0E1414','#1F2FF', '#008141', '#96a4ea', '#ff6396', '#a10000']
        # colr = [[14,20,20], [0,0,0], [0,129,65], [150,164,234], [255,99,150], [161,0,0]]

        # colr = np.array([[14,20,20], [00,00,00], [00,129,65],\
        #                 [150,164,234], [255, 89,150], [161,00,00]])
        # colr = np.random.randint(1, 255, size=6)
        colr = np.array([0, 250, 100, 50, 75, 15])
        # colr = np.array([200, 250, 150, 50, 50, 50])/255
        # wall, hill, food, phero, pherol,   blobs
        # model_path = '23-8-9.pt'
        xy = wall
        s = np.ones(xy.shape[0]) * 100.0
        c = np.zeros([xy.shape[0], 3]) + colr[0]

        tmp = hill
        xy_init = np.append(xy, tmp, axis=0)
        s_init = np.append(s, np.ones(tmp.shape[0]) * 100.0, axis=0)
        # c = c + [colr[1] for x in range(tmp.shape[0])]
        c_init = np.append(c, np.zeros([tmp.shape[0], 3]) + colr[1], axis=0)

        for i in range(df.shape[0]):

            xy =- xy_init.copy()
            s = s_init.copy()
            c = c_init.copy()

            tmp = df[i][0] # food
            xy = np.append(xy, tmp, axis=0)
            s = np.append(s, np.ones(tmp.shape[0]) * 100.0, axis=0)
            # c = c + [colr[2] for x in range(tmp.shape[0])]
            # c = np.append(c, np.ones(tmp.shape[0]) * colr[2], axis=0)
            c = np.append(c, np.zeros([tmp.shape[0], 3]) + colr[2], axis=0)

            tmp = df[i][1] # phero
            xy = np.append(xy, tmp, axis=0)
            s = np.append(s, np.ones(tmp.shape[0]) * 10.0, axis=0)
            # c = c + [colr[3] for x in range(tmp.shape[0])]
            # c = np.append(c, np.ones(tmp.shape[0]) * colr[3], axis=0)        
            c = np.append(c, np.zeros([tmp.shape[0], 3]) + colr[3], axis=0)

            tmp = df[i][2] # phero
            xy = np.append(xy, tmp, axis=0)
            s = np.append(s, np.ones(tmp.shape[0]) * 10.0, axis=0)
            # c = c + [colr[4] for x in range(tmp.shape[0])]
            # c = np.append(c, np.ones(tmp.shape[0]) * colr[4], axis=0)
            c = np.append(c, np.zeros([tmp.shape[0], 3]) + colr[4], axis=0)

            tmp = df[i][3] # blob
            xy = np.append(xy, tmp, axis=0)
            s = np.append(s, np.ones(tmp.shape[0]) * 100.0, axis=0)
            # c = c + [colr[5] for x in range(tmp.shape[0])]
            # c = np.append(c, np.ones(tmp.shape[0]) * colr[5], axis=0)
            c = np.append(c, np.zeros([tmp.shape[0], 3]) + colr[5], axis=0)

            time.sleep(.01)

            yield np.c_[xy[:,0], xy[:,1], s, c]

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)
        # print(data.shape)

        # Set x and y data...
        self.scat.set_offsets(data[:, :2])
        # Set sizes...
        self.scat.set_sizes(abs(data[:, 2]))
        # Set colors..
        self.scat.set_array(data[:, 3])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,


a = AnimatedScatter()
plt.show()