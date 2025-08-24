import math
import numpy as np
import pylab
import qec
import matplotlib.pyplot as plt


def qber(px_correct, px_incorrect, pz_correct, pz_incorrect, num):
    value_a_x = px_correct+px_incorrect
    value_b_x = px_correct-px_incorrect
    value_a_z = pz_correct+pz_incorrect
    value_b_z = pz_correct-pz_incorrect
    #print("pz_incorrect")
    #print(pz_incorrect)

    #print("value_b_x")
    #print(value_b_x)
    #print("value_a_x")
    #print(value_a_x)
    error_x = 0.5-0.5*((value_b_x/value_a_x)**num)
    #print("value_b_z")
    #print(value_b_z)
    #print("value_a_z")
    #print(value_a_z)
    error_z = 0.5-0.5*((value_b_z/value_a_z)**num)
    qber = (error_x + error_z)*0.5
    #print("qber")
    #print(qber)
    return qber


def key_rate(qber, p_success, num):
    t0 = 1e-6
    if qber==0:
        key_rate = (p_success**num)/t0
        return key_rate
    elif qber!=0:
        if np.signbit(((p_success**num)/t0)*(1-2*((-qber)*math.log(qber, 2) - (1-qber)*math.log(1-qber, 2)))):
            key_rate=0.0
        else:
            key_rate = ((p_success**num)/t0)*(1-2*((-qber)*math.log(qber, 2) - (1-qber)*math.log(1-qber, 2)))
        return key_rate


def cost_function(n, m, key_rate, l_rep):
    if key_rate < 10**(-5):
        key_rate = 10**(-5)
    cost = 2*m*n/(key_rate*l_rep)
    return cost


def calculate_cost_coefficient(num_points, max_l_rep, L_att, eta_min, L_tot, epg, epm, max_n, max_m):
    niter = 10
    cost = [ 0 for i in range(num_points) ] 
    axes = [ 0 for i in range(num_points) ]
    print('Calculating the cost coefficient for the one-way quantum repeater scheme with QPC')
    for k in range(num_points):
        var_2 = [1e5] * niter
        eta_c = k * ((1-eta_min)/num_points) + eta_min
        l_rep = np.linspace(1, max_l_rep, 10, dtype=int).tolist()
        for indx in range(niter):
            loss = 1 - (eta_c) * math.exp (-l_rep[indx] / L_att)
            num = math.floor (L_tot / l_rep[indx])
            var_1 = [[1e5 for i in range (max_n)] for j in range (max_m)] #to use in (n,m) parity code
            for i in range(2,max_n):
                for j in range(2,max_m):
                    px_correct,px_incorrect,pz_correct,pz_incorrect,p_success = qec.qpc (epg,epm,loss,i,j)
                    qber_val = qber (px_correct,px_incorrect,pz_correct,pz_incorrect,num)
                    key_rate_val = key_rate (qber_val,p_success,num)
                    var_1[i][j] = cost_function (i,j,key_rate_val,l_rep[indx])
            var_2[indx] = np.amin(var_1)
        cost[k] = np.amin(var_2)
        axes[k] = eta_c
    return [cost,axes]
    

L_tot_epg = [[0.90,1000,1e-3,6], [0.92,1000,1e-4,5], [0.94,10000,1e-3,4], [0.92,10000,1e-4,5]]
max_m = 21
max_n = 21
max_l_rep = 10 #km
L_att = 22
cost = [ 0 for i in range(4)]
axes = [ 0 for i in range(4)]
for cnt in range(4):
    eta_min = L_tot_epg[cnt][0]
    L_tot = L_tot_epg[cnt][1]
    epg = L_tot_epg[cnt][2]
    num_points = L_tot_epg[cnt][3]
    epm = 0.25*epg
    [cost[cnt],axes[cnt]] = calculate_cost_coefficient(num_points, max_l_rep, L_att, eta_min, L_tot, epg, epm, max_n, max_m)



L_tot_epg = [[0.90,10000,1e-3,6], [0.92,10000,1e-4,5], [0.94,100000,1e-3,4], [0.92,100000,1e-4,5]]
max_m = 11
max_n = 11
max_l_rep = 100 #km
L_att = 4e5
cost_vac = [ 0 for i in range(4)]
axes_vac = [ 0 for i in range(4)]
for cnt in range(4):
    eta_min = L_tot_epg[cnt][0]
    L_tot = L_tot_epg[cnt][1]
    epg = L_tot_epg[cnt][2]
    num_points = L_tot_epg[cnt][3]
    epm = 0.25*epg
    [cost_vac[cnt],axes_vac[cnt]] = calculate_cost_coefficient(num_points, max_l_rep, L_att, eta_min, L_tot, epg, epm, max_n, max_m)





import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.figure(figsize=(10, 6))  # Set figure size for better clarity
# Plot the lines with color and marker codes
plt.plot(axes[0], cost[0], marker='^', linestyle='-', color='b', markersize=8)
plt.plot(axes[1], cost[1], marker='s', linestyle='-', color='b', markersize=8)
plt.plot(axes_vac[0], cost_vac[0], marker='^', linestyle='-', color='g', markersize=8)
plt.plot(axes_vac[1], cost_vac[1], marker='s', linestyle='-', color='g', markersize=8)
# Add labels and title
plt.xlabel('Coupling efficiency (η)', fontsize=12)
plt.ylabel('Cost coefficient (qubits/sbit/s)', fontsize=12)
plt.text(0.35, 1.05, 'Vaccum Beam Guide', color='green', fontsize=14, ha='center', va='bottom', transform=plt.gca().transAxes)
plt.text(0.5, 1.05, ' Vs ', fontsize=14, ha='center', va='bottom', transform=plt.gca().transAxes, color='black')
plt.text(0.6, 1.05, 'Optical Fiber', color='blue', fontsize=14, ha='center', va='bottom', transform=plt.gca().transAxes)

plt.yscale('log')  # Set y-axis to logarithmic scale
# Create custom legend entries
legend_elements = [
    Line2D([0], [0], color='b', marker='^', linestyle='-', markersize=8, label='Distance = 1000 km, epg = 1e-3'),
    Line2D([0], [0], color='b', marker='s', linestyle='-', markersize=8, label='Distance = 1000 km, epg = 1e-4'),
    Line2D([0], [0], color='g', marker='^', linestyle='-', markersize=8, label='Distance = 10000 km, epg = 1e-3'),
    Line2D([0], [0], color='g', marker='s', linestyle='-', markersize=8, label='Distance = 10000 km, epg = 1e-4')
]
# Add custom legend
plt.legend(handles=legend_elements, loc='best', fontsize=10, title="Parameters")
# Additional plot settings
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add gridlines for both major and minor ticks
plt.minorticks_on()  # Enable minor ticks on both axes for more precise scale reading
# Customize tick parameters
plt.tick_params(axis='both', which='major', labelsize=10)  # Customize major ticks
plt.tick_params(axis='both', which='minor', labelsize=8)   # Customize minor ticks


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.figure(figsize=(10, 6))  # Set figure size for better clarity
# Plot the lines with color and marker codes
plt.plot(axes[2], cost[2], marker='^', linestyle='-', color='b', markersize=8)
plt.plot(axes[3], cost[3], marker='s', linestyle='-', color='b', markersize=8)
plt.plot(axes_vac[2], cost_vac[2], marker='^', linestyle='-', color='g', markersize=8)
plt.plot(axes_vac[3], cost_vac[3], marker='s', linestyle='-', color='g', markersize=8)
# Add labels and title
plt.xlabel('Coupling efficiency (η)', fontsize=12)
plt.ylabel('Cost coefficient (qubits/sbit/s)', fontsize=12)
plt.text(0.35, 1.05, 'Vaccum Beam Guide', color='green', fontsize=14, ha='center', va='bottom', transform=plt.gca().transAxes)
plt.text(0.5, 1.05, ' Vs ', fontsize=14, ha='center', va='bottom', transform=plt.gca().transAxes, color='black')
plt.text(0.6, 1.05, 'Optical Fiber', color='blue', fontsize=14, ha='center', va='bottom', transform=plt.gca().transAxes)

plt.yscale('log')  # Set y-axis to logarithmic scale
# Create custom legend entries
legend_elements = [
    Line2D([0], [0], color='b', marker='^', linestyle='-', markersize=8, label='Distance = 10000 km, epg = 1e-3'),
    Line2D([0], [0], color='b', marker='s', linestyle='-', markersize=8, label='Distance = 10000 km, epg = 1e-4'),
    Line2D([0], [0], color='g', marker='^', linestyle='-', markersize=8, label='Distance = 100000 km, epg = 1e-3'),
    Line2D([0], [0], color='g', marker='s', linestyle='-', markersize=8, label='Distance = 100000 km, epg = 1e-4')
]
# Add custom legend
plt.legend(handles=legend_elements, loc='best', fontsize=10, title="Parameters")
# Additional plot settings
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add gridlines for both major and minor ticks
plt.minorticks_on()  # Enable minor ticks on both axes for more precise scale reading
# Customize tick parameters
plt.tick_params(axis='both', which='major', labelsize=10)  # Customize major ticks
plt.tick_params(axis='both', which='minor', labelsize=8)   # Customize minor ticks



plt.show()

