#from dhCheck_Task1 import dhCheckCorrectness
#from dhCheck_Task2 import dhCheckCorrectness
#from dhCheck_Task3 import dhCheckCorrectness


from scipy.stats import triang, lognorm, pareto # triang for t1, others for t2, last for t3
import numpy as np # for all
#import sklearn.linear_model # for t3
from scipy.optimize import linprog 

def Task1(a, b, c, point1, number_set, prob_set, num, point2, mu, sigma, xm, alpha, point3, point4):
    # TODO
    # task 1.1
    # Triangular distrubution for Asset Value
    # find range of distrubtion
    # find the mode relative to the lower bound
    rang = b - a
    c_relative_to_lower = (c-a) / rang
    
    # Use scipy triang module to create  triangular distrubution
    tri_distrubution = triang(c_relative_to_lower, loc=a, scale=rang)
    
    # i) find the probability that AV <= point1 
    prob1 = tri_distrubution.cdf(point1)
    
    # ii) find the mean and median of the triangular distrubtion 
    MEAN_t = tri_distrubution.mean()
    MEDIAN_t = tri_distrubution.median()
    
    # task 1.2
    # convert annual occurance to numpy array
    # convert probabilities to numpy array
    num_set = np.array(number_set) 
    p_set = np.array(prob_set)
    
    # calculate mean 
    # calculate variance = sum(number_set_i^2 * prob_set_i) - (sum(number_set_i * prob_set_i))^2   
    MEAN_d = np.sum(num_set * p_set)
    VARIANCE_d = np.sum((num_set ** 2) * p_set) - MEAN_d ** 2
    
    # task 1.3 
    # monte carlo simulation for total impact
    # use scipy to calculate the following
    # calculate Flaw A impact (log-nomral)
    # claculate Flaw B impact (pareto)
    lognorm_samples = lognorm.rvs(sigma, scale=np.exp(mu), size=num)
    pareto_samples = pareto.rvs(alpha, scale=xm, size=num)
    
    # i) compute sum of total impact samples
    total_impact_samples = lognorm_samples + pareto_samples
    
    # ii) compute probability that total impact > prob2 
    # utilise np.mean 
    prob2 = np.mean(total_impact_samples > point2)
    
    # iii) compute probability that total impact is between point3 and point4
    prob3 = np.mean((total_impact_samples >= point3) & (total_impact_samples >= point4))

    # task 1.4
    # utilise median as AV and prob2 as EF to calculate SLE
    # use MEAN as ARO and SLE to calculate ALE 
    SLE = MEDIAN_t * prob2
    ALE = MEAN_d * SLE
    
    return (prob1, MEAN_t, MEDIAN_t, MEAN_d, VARIANCE_d, prob2, prob3, ALE)


def Task2(num, table, probs):
    # TODO
    # extract x and y values 
    X = np.array([2, 3, 4, 5])
    Y = np.array([6, 7, 8])
    
    # convert table into numpy array for indexing
    np_table = np.array(table)
    
    # 1) compute probabilities 1 and 2
    # p1 = (3 <= X <= 4)
    # p2 = P(X + Y <= 10)
    mask_x = (X >= 3) & (X <= 4)
    prob1 = np.sum(np_table[:, mask_x]) / num  # Correctly selects X = 3, 4
    valid_xy_mask = (X[None, :] + Y[:, None]) <= 10  # Mask for valid (X, Y) pairs
    prob2 = np.sum(np_table[valid_xy_mask]) / num  # Select correct values
    
    # 2) compute P(Y=8 | T)
    # initalise probabilities givin by spec
    # compute P(T) using the law of total probability
        # sum over x values
        # sum over y values
    PX2, PX3, PX4, PX5, PY6, PY7 = probs
    
    total_Y8_cases = np.sum(np_table[2, :])  # Sum over all X values in row Y=8
    P_Y8 = total_Y8_cases / num
    
    if total_Y8_cases == 0:
        return (prob1, prob2, 0) 
        
    P_T_given_Y8 = np.sum((np_table[2, :] / total_Y8_cases) * np.array([PX2, PX3, PX4, PX5]))

    P_T = (
        np.sum(np_table[0, :] * np.array([PX2, PX3, PX4, PX5])) * PY6 +
        np.sum(np_table[1, :] * np.array([PX2, PX3, PX4, PX5])) * PY7 +
        np.sum(np_table[2, :] * np.array([PX2, PX3, PX4, PX5])) * P_Y8
    ) / num
    
    # compute P(Y = 8 | T) using bayes' theorem
    #P_T_given_Y8 = P_T_given_Y8 = np.sum(np_table[2, :] * np.array([PX2, PX3, PX4, PX5])) / np.sum(np_table[2, :])
    #P_Y8 = np.sum(np_table[2, :]) / num
    #prob3 = (P_T_given_Y8 * P_Y8) / P_T
    
    
    # Compute P(Y=8 | T) using Bayes' Theorem
    if P_T == 0:
        prob3 = 0  # Avoid division by zero
    else:
        prob3 = (P_T_given_Y8 * P_Y8) / P_T
    
    return (prob1, prob2, prob3)

# -- END OF YOUR CODERUNNER SUBMISSION CODE


# -- END OF YOUR CODERUNNER SUBMISSION CODE




"""def Task3(x, y, z, x_initial, c, x_bound, se_bound, ml_bound):
    # convert inputs to numpy arrays
    # X shape = (9,4) corresponding to x1, x2, x3, x4
    # Y shape = (9,) corresponding to total safeguard effect
    # Z shape = (9,) corresponding to total maintenance load 
    X = np.array(x).T
    Y = np.array(y)
    Z = np.array(z)
    
    # 1)
    # perform linear regression for safeguard effect y
    # extract coefficients (b0 to b4)
    model_y = sklearn.linear_model.LinearRegression()
    model_y.fit(X, Y)
    weights_b = [model_y.intercept_] + list(model_y.coef_)
    
    # perform linear regression for maintenance load z
    # extract coefficients (d0 to d4)
    model_z = sklearn.linear_model.LinearRegression()
    model_z.fit(X, Z)
    weights_d = [model_z.intercept_] + list(model_z.coef_)
    
    # 2)
    # linear programming to find x_add
    # calculate number of security controls
    # convert following to numpy arrays
    # cost coefficients
    # calculate inital security controls
    # upper bounds for each security control
    num_controls = len(c)
    c = np.array(c)
    x_initial = np.array(x_initial)
    x_bound = np.array(x_bound)
    
    # constraints matricies
    # for A
    # ensure total safegaurd effect meets the requirement
    # ensure maintenance load doesn't exceed limit
    # for B
    # adjust safeguard constraint
    # adjust maintenance load constraint
    A = [
        -np.array(weights_b[1:]),
        np.array(weights_d[1:])
    ]
    B = [-se_bound + weights_b[0] + np.dot(weights_b[1:], x_initial),
         ml_bound - weights_d[0] - np.dot(weights_b[1:], x_initial)]

    # bounds for additional controls
    bounds = [(0, x_bound[i] - x_initial[i]) for i in range(num_controls)]
    
    # solve linear program
    # extract optimal additional security controls if solution exists
    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    x_add = result.x if result.success else None
    
    return (weights_b, weights_d, x_add)"""

def Task3_revamp(x, y, z, x_initial, c, x_bound, se_bound, ml_bound):
    # convert inputs to numpy arrays
    # X shape = (9,4) corresponding to x1, x2, x3, x4 transposed
    # Y shape = (9,) corresponding to total safeguard effect
    # Z shape = (9,) corresponding to total maintenance load 
    X = np.array(x).T
    Y = np.array(y)
    Z = np.array(z)
    

    # 1) Perform linear regression using NumPy's least squares method
    # Compute intercept (b0, d0) using mean differences
    #format weights as arrays
    b_coef = np.linalg.lstsq(X, Y, rcond=None)[0]
    d_coef = np.linalg.lstsq(X, Z, rcond=None)[0]
    
    b0 = np.mean(Y) - np.dot(np.mean(X, axis=0), b_coef)
    d0 = np.mean(Z) - np.dot(np.mean(X, axis=0), d_coef)
    
    weights_b = np.array([b0] + list(b_coef))
    weights_d = np.array([d0] + list(d_coef))
    
    
    # 2)
    # linear programming to find x_add
    # calculate number of security controls
    # convert following to numpy arrays
    # cost coefficients
    # calculate inital security controls
    # upper bounds for each security control
    num_controls = len(c)
    c = np.array(c)
    x_initial = np.array(x_initial)
    x_bound = np.array(x_bound)
    
    # constraints matricies
    # for A
    # ensure total safegaurd effect meets the requirement
    # ensure maintenance load doesn't exceed limit
    # for B
    # adjust safeguard constraint
    # adjust maintenance load constraint
    A = np.array([
        -weights_b[1:],
        weights_d[1:]
    ])
    B = np.array([
        -se_bound + weights_b[0] + np.dot(weights_b[1:], x_initial),
        ml_bound - weights_d[0] - np.dot(weights_d[1:], x_initial)
    ])
    
    # bounds for additional controls
    bounds = [(0, x_bound[i] - x_initial[i]) for i in range(num_controls)]
    
    # solve linear program
    # extract optimal additional security controls if solution exists
    result = linprog(c, A_ub=A, b_ub=B, bounds=bounds, method='highs')
    x_add = result.x if result.success else None
    
    return (weights_b, weights_d, x_add)


def __main__():
    	
    num = 120
    probs = [0.7, 0.6, 0.5, 0.63, 0.44, 0.36]
    table = [[6, 10, 11, 9], [9, 12, 15, 8], [7, 14, 10, 9]]
    (prob1, prob2, prob3) = Task2(num, table, probs)
    print(prob1)
    print(prob2)
    print(prob3)
    
__main__()