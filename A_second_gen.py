import math
import numpy as np
import pylab
import qec
import matplotlib.pyplot as plt


def qber(px_correct, px_incorrect, pz_correct, pz_incorrect, num):
    value_a_x = px_correct-px_incorrect
    value_a_z = pz_correct-pz_incorrect
    error_x = 0.5-0.5*((value_a_x)**num)
    error_z = 0.5-0.5*((value_a_z)**num)
    qber = (error_x + error_z)*0.5
    return qber


def key_rate(qber, p_success, num, nEG, l_rep):
    t0 = 1e-6
    c = 2 * 1e5
    capitalt0 = l_rep / c
    if qber==0:
        key_rate = (p_success**num) / (nEG * (t0 + capitalt0))
    elif qber!=0:
        if np.signbit(1-2*((-qber)*math.log(qber, 2) - (1-qber)*math.log(1-qber, 2))):
            key_rate=0.0
        else:
            key_rate = ((p_success**num) / (nEG * (t0 + capitalt0)) ) * (
                1-2*((-qber)*math.log(qber, 2) - (1-qber)*math.log(1-qber, 2)))
    return key_rate
    
def cost_function(M, key_rate, l_rep):
    if key_rate < 10**(-5):
        key_rate = 10**(-5)
    cost = 2*M/(key_rate*l_rep)
    return cost

def generate_congested_points(n, start=0.4, end=0.88, factor=3):
    # Generate `n` linearly spaced points between 0 and 1
    linear_points = np.linspace(0, 1, n)
    
    # Apply the exponential transformation to increase density towards 1
    congested_points = linear_points**(1/factor)
    
    # Rescale to the desired range [start, end]
    congested_points = start + congested_points * (end - start)
    
    return congested_points

def calculate_cost_coefficient(num_points, max_l_rep, L_att, L_tot, epg, epm, max_nEG, max_M):
    cost = [ 0 for i in range(num_points) ] 
    axes = [ 0 for i in range(num_points) ]
    print('Calculating the cost coefficient for the two-way quantum repeater scheme with CSS code')
    eta_c_list = generate_congested_points(num_points)
    for k in range(num_points):
        eta_c = eta_c_list[k]
        l_rep = np.arange(0.01, max_l_rep, 0.02) # This is how I defined l_rep, min of 0.1 and max of 5 
        var_2 = [1e5] * len(l_rep)
        for indx in range(len(l_rep)):
            loss = 1 - 0.5 * ((eta_c)**2) * math.exp (-l_rep[indx] / L_att)
            num = math.floor (L_tot / l_rep[indx])
            var_1 = [[1e5] * max_nEG] * max_M
            for i in range(2,max_M):
                for j in range(2,max_nEG):
                    px_correct,px_incorrect,pz_correct,pz_incorrect,p_success = qec.css(epg, epm, loss, i, j)
                    qber_val = qber(px_correct, px_incorrect, pz_correct, pz_incorrect, num)
                    key_rate_val = key_rate(qber_val, p_success, num, j, l_rep[indx])
                    var_1[i][j] = cost_function (i,key_rate_val,l_rep[indx])
            var_2[indx] = np.amin(var_1)
        cost[k] = np.amin(var_2)
        axes[k] = eta_c
    return [cost,axes]

num_points = 20
max_l_rep = 10 #km 
L_att = 22
eta_min = 0.1
L_tot = 10000
epg = 1e-4 #also 1e-3
epm = 0.25*epg
max_nEG = 20
max_M = 20

[cost,axes] = calculate_cost_coefficient(num_points, max_l_rep, L_att, L_tot, epg, epm, max_nEG, max_M)

plt.plot(axes, cost, label='epg = 1e-3', marker='^')
plt.xlabel('eta_c')
plt.ylabel('Cost coefficient (qubits/sbit/s)')
plt.title('total distance is 10000 (km)')
plt.yscale('log')
plt.legend()

plt.show()