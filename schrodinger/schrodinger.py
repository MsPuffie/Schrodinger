# -*- coding: utf-8 -*-

"""Main module."""

import tensorflow as tf
import numpy as np  # only use numpy to read files
tf.enable_eager_execution()
tfe = tf.contrib.eager
c = tf.constant(1.0, dtype=tf.float32)


def readfile(f1):  # this function aims at generate v0_list including vo(potential energy) and x(x_position,domain) from v0.txt file
    array_tmp = np.loadtxt(f1)
    array = array_tmp[1:, :].astype(np.float)
    v0_temp = []
    x_temp = []
    for i in range(len(array)):
        v0_temp.append(array[i, 1])
    v0 = tf.reshape(tf.Variable(v0_temp, dtype=tf.float32), [-1, 1])
    for i in range(len(array)):
        x_temp.append(array[i][0])
    x = x_temp
    return x, v0


def readfile2(f2):  # this function reads N value from N.txt, N is the number of basis set
    n = np.loadtxt(f2)
    n = int(n)
    return n


N = readfile2("N.txt")  # get N at the very beginning


def basis(n):  # generate basis sets, take an input n as the number of basis sets, return a list of basis set in forms of functions
    print('generating fourier basis set')
    basis_set = [lambda x: tf.math.pow(x, 0)]  # I want it be 1, but it need to be a function
    if n < 2:  # N should >=2
        raise ValueError
    for i in range(1, n, 1):
            fcn = my_fcn(i)  # there appeals some problem when I didn't assign the lambda to different function names
            basis_set.append(fcn)  # I guess that's the tensorflow issue.
    return basis_set


def my_fcn(i):  # this function takes i as input. returns the corresponding basis.
    if i % 2.0 != 0:
        n = 0.5 * (i + 1.0)
        return lambda x: tf.sin(n*x)
    else:
        n = 0.5 * i
        return lambda x: tf.cos(n*x)


def differentiate(f):  # differentiate function f and return its 1st order differential function
    return lambda x: tfe.gradients_function(f)(x)[0]


def inner_product(f1, f2, position):  # calculating inner product of f1,f2, given the range mini & maxi & n
    # return the inner product value of f1, f2
    n = len(position)
    sum_value = 0
    f_1 = tf.map_fn(lambda x: f1(x), tf.Variable(position, dtype=tf.float32))
    f_2 = tf.map_fn(lambda y: f2(y), tf.Variable(position, dtype=tf.float32))
    for i in range(0, n-1, 1):  # doing numerically, the grid depends on the domain input by users
        dx = position[i+1] - position[i]
        sum_value += f_1[i] * f_2[i] * dx  # inner product is a kind of sum of multiplication of two functions at every x
    return sum_value


def projection_v0(v_list, position, f_basis):  # take v_list(potential energy, position, and basis as input. returns the coefficient of v0 pojrcted on this basis set
    # b11 b12 ...     e1       a1
    # b21 b22 ...  *  e1   =   a2      e is what we get from this function
    # b31 ... ...     e3       a3
    # ... ... ...     ...      ...
    n_basis = len(f_basis)
    n_v0 = tf.size(v_list)
    a = []  # rhs vector
    print('projecting V0 :)')
    for i in range(0, n_basis, 1):  # calculate the rhs vector
        sum_v0_hat = 0.0  # type: float
        for j in range(0, n_v0, 1):
            sum_v0_hat += tf.constant(v_list[j], dtype=tf.float32)*f_basis[i](tf.Variable(position[j], dtype=tf.float32))
        a.append(tf.Variable(sum_v0_hat, dtype=tf.float32))
    b = tf.Variable(tf.zeros(shape=[n_basis, n_basis]))  # lhs matrix
    b_temp = [[0 for i in range(n_basis)] for j in range(n_basis)]
    for i in range(0, n_basis, 1):  # calculate the lhs matrix
        for j in range(0, n_basis, 1):
            b_temp[i][j] = inner_product(f_basis[i], f_basis[j], position)
    tf.assign(b, tf.Variable(b_temp, dtype=tf.float32))
    print('End projecting V0 :D')
    coefficient_v0 = tf.linalg.solve(b, tf.reshape(a, [n_basis, 1]))  # solve the equation
    return coefficient_v0


def laplace_matrix(f_basis):  # take an basis set and returns the laplacian operator matrix (together with c)
    # 1-step Laplacian with constant c: -c*d^2/(dx)^2
    print('Calculating hamiltonian')
    n = len(f_basis)
    matrix_temp = [[0 for i in range(n)] for j in range(n)]
    value = 1.0  # this value can be anything as long as neither sin(n*value) nor cos(n*value) equals to 0, n > 0
    for i in range(0, n, 1):
        matrix_temp[i][i] = -c * differentiate(differentiate(f_basis[i]))(value) / f_basis[i](value)  # get the coefficient by one over the other, when assigned a value
    return matrix_temp


def hamiltonian(coefficient_v, laplace_mat):  # this function combine the operator and the projection together, get the hamiltonian matrix
    n = tf.size(coefficient_v)
    v = coefficient_v
    # a
    # b
    # c
    vv = tf.tile(tf.transpose(v), (n, 1))
    # a b c
    # a b c
    # a b c
    h_matrix = tf.Variable(tf.zeros(shape=[n, n]))
    print(laplace_mat)
    tf.assign(h_matrix, tf.Variable(laplace_mat + tf.transpose(vv), dtype=tf.float32))
    print('End calculating hamiltonian')
    return h_matrix


def wave_coefficient(hamiltonian_matrix):  # this function is simple, returns the eigenvalues(energy) and eigenvectors(wave function coefficient)
    eigenvalues, e_vectors = tf.self_adjoint_eig(hamiltonian_matrix)
    return eigenvalues, e_vectors


def main():  # main function, run all the functions above and get the lowest energy and corressponding wave function
    finish = False
    print('Start now')
    x, v0 = readfile("v0.txt")
    basis_set = basis(N)
    v0_coefficient = projection_v0(v0, x, basis_set)
    print(v0_coefficient)
    laplace_m = laplace_matrix(basis_set)
    hamiltonian_mat = hamiltonian(v0_coefficient, laplace_m)
    energy, wave_function = wave_coefficient(hamiltonian_mat)
    min_energy = tf.math.reduce_min(energy)
    min_energy_x = tf.math.argmin(energy)
    print(' ')
    print('result:')
    print(' ')
    print('minimum energy is: ', min_energy)
    print(' ')
    print('corresponding wave function coefficient is (a0, a1, a2...): ', wave_function[min_energy_x, :])
    finish = not finish  # used in test, see whether main() runs completely
    return finish


main()

