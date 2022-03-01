"""
hw4.py
Name(s): William "Cason" Pierce
NetID(s): wcp17
Date: 2-22-22
"""

import numpy as np
import matplotlib.pyplot as plt

"""
This function longTime determine the dominant eigenvalue and eigenvector 
of its input

INPUTS:
A: a numpy.array that is used to calculate the eigenvectors/eigenvalues

OUTPUTS:
domEig: the dominant (greatest) eigenvalue
domVec: the eigenvector that corresponds with domEig
"""
def longTime(A):

    # Assign w to the eigenvalues of A and v to the eigenvectors of A
    eigens = np.linalg.eig(A)
    w = eigens[0]
    v = eigens[1]

    # Iterate through w and v to find the greatest eigenvalue and its
    # corresponding eigenvector
    i = 0
    for k in range(len(w)):
        if abs(w[k]) > abs(i):
            domEig = w[k]
            domVec = v[:, k]
            i = w[k]

    return (domEig, domVec)

"""
This function getAk is used to compute A^k where A is a numpy.array and 
k is a constant

INPUTS:
A: a diagonalizable numpy.array
k: an integer that represents the exponent on A

OUTPUTS:
AK: the computation of A^k as a numpy.array
"""
def getAk(A,k):

    # Find the eigenvalues, eigenvectors, and number of rows in A
    eigens = np.linalg.eig(A)
    n = A.shape[0]

    # Derive the matrix P
    P = eigens[1]

    # Compute P^-1
    invP = np.linalg.inv(P)

    # Compute D by sorting the elements and assigning them to Dk
    # in descending order
    eigenvalues = eigens[0]
    sortedEigVals = np.sort(abs(eigenvalues))[::-1]
    Dk = np.zeros((n, n))
    for i in range(n):
        Dk[i][i] = (sortedEigVals[i] ** k)

    # Compute A^k
    tempProd = np.matmul(P, Dk)
    AK = np.matmul(tempProd, invP)
    return AK


"""
This error function calculates the 2-norm difference between them

INPUTS:
uCurr: a normalized numpy.array column vector
uLong: another normalized numpy.array column vector

OUTPUTS:
err: the normed difference between the two vectors that were inputted into
the function
"""
def error(uCurr, uLong):
    # Calculate the difference between the vectors and find the norm
    diff = uCurr - uLong
    err = np.linalg.norm(diff)
    return err

"""
This normalize function is used to find the corresponding normal vector 
for some numpy.array vector

INPUTS:
v: a numpy.array column or row vector

OUTPUTS:
v: the normalized vector of the input v
"""
def normalize(v):
    # Find the norm of v and divide the elements of v by this norm
    normed = np.linalg.norm(v)
    v = np.array(v / normed)
    return v


"""
This simulate function is used to simulate long term behavior by 
repetitively multiplying matrix A by the current condition

INPUTS:
A: a numpy.array that is used to model how the initial condition will 
change with the passing of time periods
k: an integer used to determine how many times to repeat the simulation
u0: a numpy.array column vector representing the initial condition

OUTPUTS:
sim: a numpy.array with the same number of rows as u0 and k columns. The 
kth column will correspond to the product of A with the current condition
"""
def simulate(A,k,u0):
    # Create a matrix to store the simulation results
    n = A.shape[0]
    sim = np.array([[0.0 for i in range(k+1)] for j in range(n)])
    tempU = u0

    # Populate the columns of sim with each simulation
    sim[:, [0]] = tempU
    for g in range(k):
        tempU = np.matmul(A, tempU)
        sim[:, [g + 1]] = tempU

    return sim


"""
main function
"""
if __name__ == '__main__':


    print("Scenario A: Frogs")
    # Scenario A: Frogs
    frogMat = np.array([[0.0, 0.0, 3.0,  8.0], \
               [0.4, 0.0, 0.0,  0.0], \
               [0.0, 0.5, 0.0,  0.0], \
               [0.0, 0.0, 0.25, 0.0]])
    frogInit = np.array([[0.0], [0.0], [0.0], [250.0]])

    # Expected growth rate and population fraction:
    longTimeBehavFrog = longTime(frogMat)
    print('Long Time Expected Growth Rate:')
    print(longTimeBehavFrog[0])
    print('Long Time Population Fraction:')
    print(longTimeBehavFrog[1])

    # Compute the value of A^k for k = 250, and use it to compute the value
    # of u250 = A^{250}u0.
    print("A^250:")
    A250Frog = getAk(frogMat, 250)
    print(A250Frog)
    print("A^250 * u0 = u250:")
    u250Frog = A250Frog @ frogInit
    print(u250Frog)

    # Simulate the system directly through iteration k = 250 and output the result
    print("A^k * u_k simulation results:")
    simmedFrog = simulate(frogMat, 250, frogInit)
    print(simmedFrog)
    print("u250 using simulation:")
    u_sFrog = simmedFrog[:, [-1]]
    print(u_sFrog)

    # Compute the error between the two methods calculated above
    print("Relative Error:")
    relErrorFrogs = error(normalize(u250Frog), normalize(u_sFrog)) / np.linalg.norm(u250Frog)
    print(relErrorFrogs)

    # The following code is used to create visuals to better understand
    # the data computed above

    # Create a list of integers from 0 to 250
    listIntsFrog = range(251)

    # Store the rows of the simulation for each stage class in a variable
    eggs = simmedFrog[0, :]
    tadpoles = simmedFrog[1, :]
    metamorphs = simmedFrog[2, :]
    frogs = simmedFrog[3, :]

    # Create a numpy.array row vector using the list of iterations
    iterationsFrog = np.array(listIntsFrog)

    # Create simulation.png
    plt.figure()
    fig, frogSim = plt.subplots()
    frogSim.plot(iterationsFrog, eggs, label='Eggs')
    frogSim.plot(iterationsFrog, tadpoles, label='Tadpoles')
    frogSim.plot(iterationsFrog, metamorphs, label='Metamorphs')
    frogSim.plot(iterationsFrog, frogs, label='Frogs')
    legend = frogSim.legend(loc='upper right')
    plt.title('Frog Stage Class Simulation')
    plt.xlabel('Iteration')
    plt.ylabel('Population Size')
    plt.savefig('simulationFrog.png', bbox_inches='tight')
    plt.close('all')


    # Create a numpy.array of the error values at each simulation
    errorsFrog = np.zeros(251)
    expectedLongTermFrog = longTime(frogMat)[1]
    normLongTermFrog = normalize(expectedLongTermFrog)
    for i in listIntsFrog:
        simulated = simmedFrog[:, i]
        normalizedSim = normalize(simulated)
        errFrog = error(normalizedSim, normLongTermFrog)
        errorsFrog[i] = errFrog


    plt.figure()
    fig, frogError = plt.subplots()
    frogError.plot(iterationsFrog, errorsFrog, label='Error')
    legend = frogError.legend(loc='upper right')
    plt.title('Frog Stage Class Error Simulation')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.savefig('errorFrog.png', bbox_inches='tight')
    plt.close('all')



    # Scenario B: Owls
    print("Scenario B: Owls")
    owlMat = np.array([[0.2,  0.1,  0.4,  1/3], \
              [0.4,  0.4,  0.2,  1/3], \
              [0.2,  0.3,  0.2,  1/3], \
              [0.01, 0.01, 0.01, 1.5]])
    owlInit = np.array([[100.0],[100.0],[0.0],[0.0]])

    # Expected growth rate and population fraction:
    longTimeBehavOwl = longTime(owlMat)
    print('Long Time Expected Growth Rate:')
    print(longTimeBehavOwl[0])
    print('Long Time Population Fraction:')
    print(longTimeBehavOwl[1])


    # Compute the value of A^k for k = 250, and use it to compute the value
    # of u250 = A^{250}u0.
    print("A^250:")
    A250Owl = getAk(owlMat, 250)
    print(A250Owl)
    print("A^250 * u0 = u_250:")
    u250Owl = A250Owl @ owlInit
    print(u250Owl)

    # Simulate the system directly through iteration k = 250 and output the result
    print("A^k * u_k simulation results:")
    simmedOwl = simulate(owlMat, 250, owlInit)
    print(simmedOwl)
    print("u250 using simulation:")
    u_sOwl = simmedOwl[:, [-1]]
    print(u_sOwl)


    # Compute the error between the two methods calculated above
    print("Relative Error:")
    relErrorOwl = error(normalize(u250Owl), normalize(u_sOwl)) / np.linalg.norm(u250Owl)
    print(relErrorOwl)
    
    # The following code is used to create visuals to better understand
    # the data computed above

    # Create a list of integers from 0 to 250
    listIntsOwl = range(251)

    # Store the rows of the simulation for each stage class in a variable
    location1 = simmedOwl[0, :]
    location2 = simmedOwl[1, :]
    location3 = simmedOwl[2, :]
    hatchery = simmedOwl[3, :]

    # Create a numpy.array row vector using the list of iterations
    iterationsOwl = np.array(listIntsOwl)

    # Create simulationOwl.png
    plt.figure()
    fig, owlSim = plt.subplots()
    owlSim.plot(iterationsOwl, location1, label='Location 1')
    owlSim.plot(iterationsOwl, location2, label='Location 2')
    owlSim.plot(iterationsOwl, location3, label='Location 3')
    owlSim.plot(iterationsOwl, hatchery, label='Hatchery')
    legend = owlSim.legend(loc='upper right')
    plt.title('Owl Meta-Population Model')
    plt.xlabel('Iteration')
    plt.ylabel('Population Size')
    plt.savefig('simulationOwl.png', bbox_inches='tight')
    plt.close('all')

    # Create a numpy.array of the error values at each simulation
    errorsOwl = np.zeros(251)
    expectedLongTermOwl = longTime(owlMat)[1]
    normLongTermOwl = normalize(expectedLongTermOwl)
    for i in listIntsOwl:
        simulated = simmedOwl[:, i]
        normalizedSim = normalize(simulated)
        errOwl = error(normalizedSim, normLongTermOwl)
        errorsOwl[i] = errOwl

    plt.figure()
    fig, owlError = plt.subplots()
    owlError.plot(iterationsOwl, errorsOwl, label='Error')
    legend = owlError.legend(loc='upper right')
    plt.title('Owl Stage Class Error Simulation')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.savefig('errorOwl.png', bbox_inches='tight')
    plt.close('all')



