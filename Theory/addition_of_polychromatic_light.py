import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import trapezoid as trapz
import cmocean.cm as cmo

#From Section 4.7.2 of PLOS

# --- Electric field ---
def Efield(z, t, A, k, omega, phi):
    """
    Compute the electric field Ex/y(z, t).

    Parameters:
        z : float
            Position (in meters)
        t : ndarray
            Time array (in seconds)
        A : ndarray
            Real amplitudes
        k : ndarray
            Wave numbers
        omega : ndarray
            Angular frequencies
        phi : ndarray
            Phases (at t=0)

    Returns:
        E_x/y_real : ndarray
            Real part of E_x/y(z, t) as a function of time
    """
    A = np.array(A)[:, np.newaxis]       # shape (N, 1)
    k = np.array(k)[:, np.newaxis]       # shape (N, 1)
    omega = np.array(omega)[:, np.newaxis] # shape (N, 1)
    phi = np.array(phi)[:, np.newaxis]   # shape (N, 1)
    t = np.array(t)[np.newaxis, :]       # shape (1, M)

    # Compute the complex field from all components
    complex_sum = np.sum(A * np.exp(1j * (k * z - omega * t - phi)), axis=0)

    return np.real(complex_sum)


#The combo of two beams and stokes parameters for adding horizontal and vertical 
#polarized light will be E(r,t) = E_x(r,t)+E_y(r,t)

# --- Constants ---
S=R=8
c = 3e8 #speed of light m/s
epsilon_0 = 8.854e-12
#randomly Q/R # generate frequecnies, amplitudes, and phases
#for white light the wavelengths are between 400-700 nm and the frequencies are
#750 to 430 terahertz (10^12 Hz) (recall: angular freq is 2pi*f). since each value within these ranges are equally 
#probable you can randomly sample a uniform distrubution of this range... 

A_x = np.random.uniform(0.1,1,S) 
A_y = np.random.uniform(0.1,1,R) 

#angular freq range 
high = 2*np.pi*750*10**12
low = 2*np.pi*430*10**12

omega_x = np.random.uniform(low,high,S)
omega_y = np.random.uniform(low,high,R)


k_x = omega_x / c
k_y = omega_y / c
phi_x = np.random.uniform(0, 2*np.pi, S)
phi_y = np.random.uniform(0, 2*np.pi, R)

#A_x=A_y
#omega_y=omega_x
#phi_x=phi_y

z=0 #place on z-axis
t = np.linspace(0,5e-14,3000) #seconds
t_int = np.linspace(0, 60,10000000) #seconds

#Calculate E_x and E_y over a longer time period 
E_x = Efield(z,t_int,A_x,k_x,omega_x,phi_x)
E_y = Efield(z, t_int, A_y, k_y, omega_y, phi_y)

# Instantaneous fluxes
p_x = (epsilon_0 * c / 2) * np.abs(E_x)**2
p_y = (epsilon_0 * c / 2) * np.abs(E_y)**2

E_45  = (E_x + E_y) / np.sqrt(2)
E_135 = (E_x - E_y) / np.sqrt(2)

p_45  = (epsilon_0 * c / 2) * np.abs(E_45)**2
p_135 = (epsilon_0 * c / 2) * np.abs(E_135)**2

# Time averaging
T = t_int[-1] - t_int[0]
p_x = np.trapz(p_x, t_int, axis=-1) / T
p_y = np.trapz(p_y, t_int, axis=-1) / T

p_45  = np.trapz(p_45, t_int, axis=-1) / T
p_135 = np.trapz(p_135, t_int, axis=-1) / T

# Stokes parameters
s0 = p_x + p_y
s1 = p_x - p_y
s2 = p_45 - p_135

dolp = np.sqrt(s1**2 + s2**2)/s0 
print(dolp*100)

E_x = Efield(z,t,A_x,k_x,omega_x,phi_x)
E_y = Efield(z, t, A_y, k_y, omega_y, phi_y)

plt.figure()
plt.plot(t, E_x, label="E_x")
plt.plot(t,E_y,label="E_y")
plt.xlabel("Time (s)")
plt.ylabel("Electric Field")
plt.legend(loc='upper right')
plt.show()

plt.plot(t,E_x+E_y,label="E")
plt.xlabel("Time (s)")
plt.ylabel("Electric Field")
plt.legend(loc='upper right')
plt.show()


fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(t, E_x, E_y, color='red')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('x')
ax.set_zlabel('y')
plt.show()

plt.figure()
plt.plot(E_y,E_x,color='red')
plt.xlabel("x")
plt.ylabel("y")

# plt.figure()
# # plt.plot(t,p_x,label="P_x")
# # plt.plot(t,p_y,label="P_y")
# plt.plot(t, s1,label='s1',color="purple")
# plt.plot(t, s0,label='s0',color="red")
# plt.plot(t, s2,label='s1',color="blue")
# plt.xlabel("Time (s)")
# plt.ylabel("Poynting Vector")
# plt.legend(loc='upper right')
# plt.show()


# xsize = 512
# ysize = 512

# #Polychromatic vertically polarized
# Ex_pvert = np.zeros((xsize, ysize),dtype=complex)
# Ey_pvert = np.zeros((xsize,ysize),dtype=complex)

# A = np.random.uniform(0.1,1,R) #example doesnt specifiy a range...

# #angular freq range 
# high = 2*np.pi*750*10**12
# low = 2*np.pi*430*10**12
# omega = np.random.uniform(low,high,R)
# k = omega / c
# phi = np.random.uniform(0, 2*np.pi, R)
# A = np.array(A)[:, np.newaxis]       # shape (N, 1)
# k = np.array(k)[:, np.newaxis]       # shape (N, 1)
# omega = np.array(omega)[:, np.newaxis] # shape (N, 1)
# phi = np.array(phi)[:, np.newaxis]   # shape (N, 1)
# t=0


# for xx in range(0,xsize,1):
#     for yy in range(0,ysize,1):
#         xidx = (xx-(xsize/2))/xsize
#         yidx = (yy-(ysize/2))/ysize
#         Ex_pvert[xx,yy] = np.sum(A *xidx* np.exp(1j * (k - omega * t - phi)),axis=0)
#         Ey_pvert[xx,yy] = 0

        
# plt.figure()
# plt.imshow(np.real(Ex_pvert), cmap='jet',interpolation=None)
# plt.ylabel('y')
# plt.xlabel('x')
# plt.title("Ex")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(np.real(Ey_pvert), cmap='jet',interpolation=None)
# plt.ylabel('y')
# plt.xlabel('x')
# plt.title("Ey")
# plt.colorbar()
# plt.show()

# I = np.abs(Ex_pvert)**2 + np.abs(Ey_pvert)**2
# Q =  np.abs(Ex_pvert)**2 - np.abs(Ey_pvert)**2
# U = 2*np.abs(Ex_pvert)*np.abs(Ey_pvert)*np.cos(np.angle(Ex_pvert-Ey_pvert))

# dolp = np.sqrt(Q**2+U**2)/I
# aolp = 0.5*np.atan2(U,Q)
# aolp = np.mod(np.degrees(aolp),180)

# plt.figure()
# plt.imshow(I, cmap='Blues',vmin=0,vmax=1,interpolation=None)
# plt.ylabel('y')
# plt.xlabel('x')
# plt.title("I")
# plt.colorbar()
# plt.show()


# plt.figure()
# plt.imshow(Q, cmap='Blues',interpolation=None)
# plt.ylabel('y')
# plt.xlabel('x')
# plt.title("Q")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(U, cmap='Blues',interpolation=None)
# plt.ylabel('y')
# plt.xlabel('x')
# plt.title("U")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(dolp, cmap='gray',vmin=0,vmax=1,interpolation=None)
# plt.ylabel('y')
# plt.xlabel('x')
# plt.title("dolp")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(aolp, cmap=cmo.phase,vmin=0,vmax=180,interpolation=None)
# plt.ylabel('y')
# plt.xlabel('x')
# plt.title("aolp")
# plt.colorbar()
# plt.show()

# #Polychromatic horizontal polarized
# Ex_phor = np.zeros((xsize, ysize),dtype=complex)
# Ey_phor = np.zeros((xsize,ysize),dtype=complex)

# A = np.random.uniform(0.1,1,R) #example doesnt specifiy a range...

# #angular freq range 
# high = 2*np.pi*750*10**12
# low = 2*np.pi*430*10**12
# omega = np.random.uniform(low,high,R)
# k = omega / c
# phi = np.random.uniform(0, 2*np.pi, R)
# A = np.array(A)[:, np.newaxis]       # shape (N, 1)
# k = np.array(k)[:, np.newaxis]       # shape (N, 1)
# omega = np.array(omega)[:, np.newaxis] # shape (N, 1)
# phi = np.array(phi)[:, np.newaxis]   # shape (N, 1)


# for xx in range(0,xsize,1):
#     for yy in range(0,ysize,1):
#         xidx = (xx-(xsize/2))/xsize
#         yidx = (yy-(ysize/2))/ysize
#         Ex_phor[xx,yy] = 0
#         Ey_phor[xx,yy] = np.sum(A * yidx * np.exp(1j * (k  - omega * t - phi)),axis=0)

        
# plt.figure()
# plt.imshow(np.real(Ex_phor), cmap='jet',interpolation=None)
# plt.ylabel('y')
# plt.xlabel('x')
# plt.title("Ex")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(np.real(Ey_phor), cmap='jet',interpolation=None)
# plt.ylabel('y')
# plt.xlabel('x')
# plt.title("Ey")
# plt.colorbar()
# plt.show()

# I = np.abs(Ex_phor)**2 + np.abs(Ey_phor)**2
# Q =  np.abs(Ex_phor)**2 - np.abs(Ey_phor)**2
# U = 2*np.abs(Ex_phor)*np.abs(Ey_phor)*np.cos(np.angle(Ex_phor-Ey_phor))

# dolp = np.sqrt(Q**2+U**2)/I
# aolp = 0.5*np.atan2(U,Q)
# aolp = np.mod(np.degrees(aolp),180)

# plt.figure()
# plt.imshow(I, cmap='Blues',vmin=0,vmax=1,interpolation=None)
# plt.ylabel('y')
# plt.xlabel('x')
# plt.title("I")
# plt.colorbar()
# plt.show()


# plt.figure()
# plt.imshow(Q, cmap='Blues',interpolation=None)
# plt.ylabel('y')
# plt.xlabel('x')
# plt.title("Q")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(U, cmap='Blues',interpolation=None)
# plt.ylabel('y')
# plt.xlabel('x')
# plt.title("U")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(dolp, cmap='gray',vmin=0,vmax=1,interpolation=None)
# plt.ylabel('y')
# plt.xlabel('x')
# plt.title("dolp")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(aolp, cmap=cmo.phase,vmin=0,vmax=180,interpolation=None)
# plt.ylabel('y')
# plt.xlabel('x')
# plt.title("aolp")
# plt.colorbar()
# plt.show()



# E1 = Ex_pvert + Ex_phor
# E2 = Ey_pvert + Ey_phor


# I = np.abs(E1)**2 + np.abs(E2)**2
# Q =  np.abs(E1)**2 - np.abs(E2)**2
# U = 2*np.abs(E1)*np.abs(E2)*np.cos(np.angle(E1-E2))

# dolp = np.sqrt(Q**2+U**2)/I
# aolp = 0.5*np.atan2(U,Q)
# aolp = np.mod(np.degrees(aolp),180)

# plt.figure()
# plt.imshow(I, cmap='Blues',vmin=0,vmax=1,interpolation=None)
# plt.ylabel('y')
# plt.xlabel('x')
# plt.title("I")
# plt.colorbar()
# plt.show()


# plt.figure()
# plt.imshow(Q, cmap='Blues',interpolation=None)
# plt.ylabel('y')
# plt.xlabel('x')
# plt.title("Q")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(U, cmap='Blues',interpolation=None)
# plt.ylabel('y')
# plt.xlabel('x')
# plt.title("U")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(dolp, cmap='gray',vmin=0,vmax=1,interpolation=None)
# plt.ylabel('y')
# plt.xlabel('x')
# plt.title("dolp")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(aolp, cmap=cmo.phase,vmin=0,vmax=180,interpolation=None)
# plt.ylabel('y')
# plt.xlabel('x')
# plt.title("aolp")
# plt.colorbar()
# plt.show()

