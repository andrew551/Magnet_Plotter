'''
author: Andrew Smith, 9 May 2023

Code allows for finding and plotting of magnetic equilibrium states
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cProfile


def get_filled_hexagon_circular_state(alpha = 1.0):
    '''
    return the circular state equilibrium (if it exists)
    '''
    thetas1 = np.array([0] + [np.pi/2 + np.pi/3 * x for x in range(6)])
    positions = np.array([np.array([0, 0, 0], dtype=np.float64)]+[np.array([np.cos(x*np.pi/3), np.sin(x*np.pi/3), 0], dtype=np.float64) for x in range(6)])
    moments = np.array([np.array([np.cos(x), np.sin(x), 0.0], dtype=np.float64) for x in thetas1])
    moments[0, :] *= alpha
    system = magnet_system()
    system.set_state(positions, moments)
    system.find_stable_equilibrium()
    return system

def get_filled_hexagon_dipolar_state(alpha = 1.0):
    '''
    return the dipolar state equilibrium (if it exists)
    '''
    guess = 1
    thetas1 = np.array([0, 0, guess, -guess, 0, guess, -guess])
    positions = np.array([np.array([0, 0, 0], dtype=np.float64)]+[np.array([np.cos(x*np.pi/3), np.sin(x*np.pi/3), 0], dtype=np.float64) for x in range(6)])
    moments = np.array([np.array([np.cos(x), np.sin(x), 0.0], dtype=np.float64) for x in thetas1])
    moments[0, :] *= alpha
    system = magnet_system()
    system.set_state(positions, moments)
    system.find_stable_equilibrium()
    return system

def get_filled_hexagon_unstable_circular_state(alpha = 1.0):
    '''
    return the unstable circular state by freezing dipole 0, 1, and 4 in the correct orientations
    '''
    thetas1 = np.array([0] + [np.pi/2 + np.pi/3 * x for x in range(6)])
    thetas1[0] = np.pi/2
    thetas1[1] = np.pi/2
    thetas1[3] = -np.pi/2
    is_frozen = np.array([1, 1, 0, 0, 1, 0, 0])
    positions = np.array([np.array([0, 0, 0], dtype=np.float64)]+[np.array([np.cos(x*np.pi/3), np.sin(x*np.pi/3), 0], dtype=np.float64) for x in range(6)])
    moments = np.array([np.array([np.cos(x), np.sin(x), 0.0], dtype=np.float64) for x in thetas1])
    moments[0, :] *= alpha
    system = magnet_system()
    system.set_state(positions, moments, is_frozen = is_frozen)
    system.find_stable_equilibrium()
    return system

def get_polygon(n = 3):
    '''
    get a (non-filled) n-gon with spacing of 1 between nearest neighbours
    '''
    thetas = np.pi/2 + np.linspace(0, 2*np.pi, num = n) + 1
    r = 0.5 / np.sin(np.pi / n)
    positions = np.array([np.array([r * np.cos(x * 2 * np.pi / n), r * np.sin(x * 2 * np.pi / n), 0]) for x in range(n)])
    moments = np.array([np.array([np.cos(x), np.sin(x), 0.0], dtype=np.float64) for x in thetas])
    system = magnet_system()
    system.set_state(positions, moments)
    system.find_stable_equilibrium()
    return system

class magnet_system:        

    def set_state(self, positions, moments, is_frozen=None):
        self.positions = np.array(positions)
        self.moments = np.array(moments)
        self.rotations = moments / np.linalg.norm(moments, axis = -1)[..., np.newaxis]
        assert(self.positions.shape == self.moments.shape)
        self.n = positions.shape[0]
        self.is_frozen = np.array(is_frozen) if not is_frozen is None else np.zeros(self.positions.shape[0])
        assert(self.is_frozen.shape[0] == self.positions.shape[0])


    def find_stable_equilibrium(self):
        '''
        Find a stable equilibrium by relaxing the system (minimising energy)
        The orientations of the dipole moments are changed by computing the torques and applying small changes to
        their rotations based on them. "Angular Momentum" is used for faster convergence.
        ** works for 3D systems
        '''
        omega = np.zeros(self.moments.shape)
        inertia = 100
        gamma = 0.95
        niters = 500
        for i in range(niters):
            B = self.get_B_field(self.positions)
            omega *= gamma
            tau = np.cross(self.moments, B)
            tau *= (1 - self.is_frozen[..., np.newaxis])
            omega += tau / inertia
            theta = np.linalg.norm(omega, axis = -1)[..., np.newaxis]
            _theta = np.copy(theta)
            _theta[_theta == 0] = np.inf # hack to remove division by zero error
            axis = omega / _theta
            # apply Rodrigues' rotation formula
            self.moments = np.cos(theta) * self.moments \
                           + np.cross(axis, self.moments) * np.sin(theta) \
                           + axis * np.einsum('...j,...j', axis, self.moments)[..., np.newaxis] * (1-np.cos(theta))
        u = self.get_energy()
        self.rotations = self.moments / np.linalg.norm(self.moments, axis = -1)[..., np.newaxis]
        print(f'Relaxation method found state with energy {u}')

    def get_energy(self):
        '''
        returns: magnetostatic energy of the system.
        '''
        B = self.get_B_field(self.positions)
        return -np.sum(np.einsum('...j,...j', B, self.moments)) / 2
        
    
    def get_B_field(self, r):
        '''
            r : numpy array of shape (..., 3)
            get B field at every point
            return numpy array of same shape as r
        '''
        B = np.zeros(r.shape)
        for i in range(self.n):
            dr = r - self.positions[i, :]
            m = self.moments[i, :]
            a = np.linalg.norm(dr, axis = -1) # find distance to each point
            a[a == 0] = np.inf # sneaky way to fix division by zero problems
            a3 = a ** -3
            a5 = a ** -5
            B += 3 * dr * np.einsum('...i,i', dr, m)[..., np.newaxis] * a5[..., np.newaxis] \
                 - m * a3[..., np.newaxis]
        return B


class displayer:

    def __init__(self, system, r_max, nx, ny, Z_plane=0.5):
        self.system = system
        self.r_max = r_max
        self.x = np.linspace(-r_max, r_max, nx, dtype=np.float64) 
        self.y = np.linspace(-r_max, r_max, ny, dtype=np.float64)
        self.X, self.Y = np.meshgrid(self.x, self.y) # mesh of (X, Y) coordinates in a plane
        self.Z_plane = Z_plane # the Z-coordinate of the plane we are examing
        self.grid = np.zeros((nx, ny, 3))
        self.grid[..., 0] = self.X
        self.grid[..., 1] = self.Y
        self.grid[..., 2] = Z_plane
        self.B = system.get_B_field(self.grid)
        self.B_mag = np.linalg.norm(self.B, axis = -1)

    def plot_field_lines(self, dest, show_result = False):
        '''
        draw B-field lines, save the results to dest
        '''
        fig, ax = plt.subplots()
        my_color =  'blue' #np.log(self.B_mag) # old option color-coded by field-strength
        ax.streamplot(self.x, self.y, self.B[..., 0], self.B[..., 1], color=my_color, linewidth=0.4
                      , arrowstyle='->', arrowsize=0.5, density=[3.5, 7.5], cmap='inferno', zorder=20)

        # Add filled circles for the dipoles themselves
        charge_colors = {True: '#aa0000', False: '#0000aa'}
        Xm = np.array(self.system.positions[:, 0])
        Ym = np.array(self.system.positions[:, 1])
        Um = np.array(self.system.rotations[:, 0])
        Vm = np.array(self.system.rotations[:, 1])
        for i in range(self.system.n):
            ax.add_artist(Circle((Xm[i], Ym[i]), 0.1, color='#00FF00', zorder=5))
        arrow_sz = 1.6
        Xm -= Um / 2 / arrow_sz
        Ym -= Vm / 2 / arrow_sz
        ax.quiver(Xm, Ym, Um, Vm, units='xy', scale=arrow_sz, color='r', zorder=10)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_xlim(-self.r_max,self.r_max)
        ax.set_ylim(-self.r_max,self.r_max)
        ax.set_aspect('equal')
        plt.axis('off')
        plt.savefig(dest, bbox_inches='tight', pad_inches=0, dpi=600)
        if show_result:
            plt.show() # this line can be commented out if you just want the file saved

    def plot_field_strength_contours(self, dest, *, img_underlay=None, img_extent=None, show_result = False):
        '''
        draw B-field strength contours, save the results to dest
        optionally draw them on top of an image provided by img_underlay
        img_extent allows the scaling of the image to be controlled, with
        img_extent saying how large the image actually is in the system units.
        '''
        if img_underlay:
            try:
                img = plt.imread(img_underlay)
            except FileNotFoundError:
                print(f"Error: No such file '{img_underlay}', no image underlay will be used")
                img_underlay = None
                
        if img_extent is None:
            img_extent = self.r_max

        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        if img_underlay:
            ax.imshow(img, extent=[-img_extent, img_extent, -img_extent, img_extent], cmap = 'gray')

        levels= [2.0, 4.5] # change these to get other contours
        CS = ax.contour(self.X, self.Y, self.B_mag, levels=levels, colors=['magenta','cyan'], linestyles=['solid', 'dashed'])
        legend = ax.legend(*CS.legend_elements(variable_name='B'), \
                    loc="upper left", title=None)
        ax.add_artist(legend)
        plt.axis('off')
        plt.xlim((-self.r_max, self.r_max))
        plt.ylim((-self.r_max, self.r_max))
        plt.savefig(dest, bbox_inches='tight', pad_inches=0, dpi=600)
        if show_result:
            plt.show() # this line can be commented out if you just want the file saved
        
if __name__ == '__main__':

    '''
    plot paramters: nx, ny : number of data points
    r_max : plot fields in square the [-r_max, r_max]^2
    Z_plane: the Z_plane in which to take field data;
             Z = 0.5 corresponds to a sheet of paper resting on top of spherical mangets of radius 0.5.
    '''
    nx, ny = 2048, 2048
    r_max = 2.5
    Z_plane = 0.5
    
    system = get_filled_hexagon_circular_state(alpha = 1)
    #system = get_filled_hexagon_dipolar_state(alpha=2)  # uncomment one of the following lines for other examples
    #system = get_polygon(n = 5)
    #system = get_filled_hexagon_unstable_circular_state()
    disp = displayer(system, r_max, nx, ny, Z_plane = Z_plane)
    disp.plot_field_lines('plots/plot_field_lines_test.png', show_result = True)
    disp.plot_field_strength_contours('plots/plot_field_strength_test.png', img_underlay='source_images/395c.jpg', img_extent=6, show_result = True)

