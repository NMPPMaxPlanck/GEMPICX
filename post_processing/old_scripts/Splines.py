#!/usr/bin/env python2
# -*- coding: utf-8 -*-

### the goal of this script is to compute deposition weights "by hand" and
### compare them to the results from the Code

import numpy as np
import matplotlib.pylab as plt

################################# Parameters #################################

# Problem parameters
k = 1.25
L = 2*np.pi/k
dx = L/4
dxi = 1/dx
charge = -1

# Real positions of particle at beginning of push
x_old_R = 2.512
y_old_R = 2.2
z_old_R = 2.3

# Real positions of particle at end of push
x_new_R = 2.51398888
y_new_R = 2.201991985
z_new_R = 2.301990498

# Velocities to multiply with integration weights
vx = 0.09944398036
vy = 0.09959926604
vz = 0.09952487823

# Relative positions of particle at beginning of push
x_old = (x_old_R-dx)/dx
y_old = (y_old_R-dx)/dx
z_old = (z_old_R-dx)/dx

# Poitions of particle at end of push
x_new = (x_new_R-dx)/dx
y_new = (y_new_R-dx)/dx
z_new = (z_new_R-dx)/dx

# Distance travelled
delta_x = x_new-x_old
delta_y = y_new-y_old
delta_z = z_new-z_old

# Time at which particle crosses boundary of cell in x-direction
t_cross = (1-x_old)/(x_new-x_old)

# functions
def sigma_1(t, delta_y, delty_z, y_old, z_old):
    return t**3/3*delta_y*delta_z + t**2/2*(delta_y*(z_old-1)+delta_z*(y_old-1)) + t*(1-z_old-y_old+z_old*y_old)

def sigma_2(t, delta_y, delty_z, y_old, z_old):
    return t**3/3*(-delta_y*delta_z) + t**2/2*(delta_y*(1-z_old)+delta_y*(-z_old)) + t*(y_old*(1-z_old))

def sigma_3(t, delta_y, delty_z, y_old, z_old):
    return t**3/3*(-delta_z*delta_y) + t**2/2*(delta_z*(1-y_old)+delta_z*(-y_old)) + t*(z_old*(1-y_old))

def sigma_4(t, delta_y, delty_z, y_old, z_old):
    return t**3/3*delta_y*delta_z + t**2/2*(delta_y*z_old+delta_z*y_old) + y_old*z_old

###############################################################################
######################## VALUES COMPUTED ANALYTICALLY #########################
###############################################################################

# Wijk is weight at cell i,j,k

W111 = dxi*sigma_1(t_cross, delta_y, delta_z, y_old, z_old)
W211 = dxi*(sigma_1(1, delta_y, delta_z, y_old, z_old) - sigma_1(t_cross, delta_y, delta_z, y_old, z_old))
W121 = dxi*sigma_2(t_cross, delta_y, delta_z, y_old, z_old)
W221 = dxi*(sigma_2(1, delta_y, delta_z, y_old, z_old) - sigma_2(t_cross, delta_y, delta_z, y_old, z_old))
W112 = dxi*sigma_3(t_cross, delta_y, delta_z, y_old, z_old)
W212 = dxi*(sigma_3(1, delta_y, delta_z, y_old, z_old) - sigma_3(t_cross, delta_y, delta_z, y_old, z_old))
W122 = dxi*sigma_4(t_cross, delta_y, delta_z, y_old, z_old)
W222 = dxi*(sigma_4(1, delta_y, delta_z, y_old, z_old) - sigma_4(t_cross, delta_y, delta_z, y_old, z_old))

W = vx*charge*np.array([W111, W211, W121, W221, W112, W212, W122, W222])

###############################################################################
########################## VALUES RETURNED BY CODE ############################
###############################################################################

V111 = -0.00134082
V121 = 0.02040222744 + 0.02034719105
V112 = 0.03320426812 + 0.03314921008
V122 = 0.1005301037 + 0.100679474

V211 = -0.00074625
V221 = 0.01150296974 + 0.01148527708
V212 = 0.01875164394 + 0.01873394433
V222 = 0.05705463168 + 0.05710256765

V_h = np.array([V111, V211, V121, V221, V112, V212, V122, V222])

plt.plot(W_h, 'o', V_h, 'o')

###############################################################################
####### VALUES COMPUTED ANALYTICALLY (average instead of line integral) #######
###############################################################################

U111 = 0.5*(1-y_old)*(1-z_old)
U211 = 0.5*(1-y_new)*(1-z_new)
U121 = 0.5*y_old*(1-z_old)
U221 = 0.5*y_new*(1-z_new)
U112 = 0.5*(1-y_old)*z_old
U212 = 0.5*(1-y_new)*z_new
U122 = 0.5*y_old*z_old
U222 = 0.5*y_new*z_new

U = np.array([U111, U211, U121, U221, U112, U212, U122, U222])
U_h = dxi*U

plt.plot(W_h, 'o', V_h, 'o', U_h, 'o')
plt.plot(np.array([W122, W222])/dxi, 'o', [V111, V211], 'o', np.array([U122, U222])/dxi, 'o')

###############################################################################
######### VALUES RETURNED BY CODE (average instead of line integral) ##########
###############################################################################

T111 = 0.02098859312
T121 = 0.06347259674
T112 = 0.1032613325
T122 = 0.3122774776
T211 = 0.02066061956
T221 = 0.06301235613
T212 = 0.1027999087
T222 = 0.3135271156

T = np.array([T111, T211, T121, T221, T112, T212, T122, T222])
T_h = dxi*T

###############################################################################
##################### VALUES COMPUTED ANALYTICALLY (J_y) ######################
###############################################################################

iS1at1 = 0.00017127291278542866
iS2at1 = 0.0008426419572145924
iS2at1 = 0.1687511069172146
iS2at1 = 0.8302349782127854

t = t_cross
iS1 = t**3/3*delta_x*delta_z + t**2/2*(delta_z*(x_old-1)+delta_x*(z_old-1)) + t*(1-z_old)*(1-x_old)
iS2 = t**3/3*(-delta_z*delta_x) + t**2/2*(delta_z*(1-x_old)-delta_x*z_old) + t*(z_old*(1-x_old))
iS3 = t**3/3*(-delta_x*delta_z) + t**2/2*(delta_x*(1-z_old)-delta_z*x_old) + t*(x_old*(1-z_old))
iS4 = t**3/3*delta_x*delta_z + t**2/2*(x_old*delta_z+z_old*delta_x) + t*(x_old*z_old)

###############################################################################
####################### VALUES RETURNED BY CODE (J_y) #########################
###############################################################################

Q111 = iS1
Q112 = iS2
Q211 = iS3 + iS1at1 - iS1
Q212 = iS4 + iS2at1 - iS2
Q311 = iS3at1 - iS3
Q312 = iS4at1 - iS4

Q = np.array([Q111, Q112, Q211, Q212, Q311, Q312])

Q_h = dxi*Q
