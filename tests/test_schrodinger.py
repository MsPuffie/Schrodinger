#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `schrodinger` package."""

import unittest
import tensorflow as tf
from click.testing import CliRunner
from schrodinger import schrodinger
from schrodinger import cli


class Test(unittest.TestCase):

    def test_command_line_interface(self):
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'schrodinger.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output

    def test_readfile(self):
        print('readfile')
        x, v0 = schrodinger.readfile("./schrodinger/v0.txt")
        self.assertTrue(tf.equal(tf.to_int32(len(x)), tf.size(v0)))

    def test_readfile2(self):
        print('readfile2')
        n = schrodinger.readfile2("./schrodinger/N.txt")
        self.assertTrue(n > 2)

    def test_basis(self):
        print('basis')
        n = 8
        f_basis = schrodinger.basis(n)
        self.assertTrue(len(f_basis) == n)
        self.assertTrue(tf.equal(f_basis[0](3.0), tf.to_float(1)))
        self.assertTrue(tf.equal(f_basis[2](0.0), tf.to_float(1)))

    def test_my_fcn(self):
        print('my_fcn')
        self.assertTrue(tf.equal(schrodinger.my_fcn(1)(0.0), tf.to_float(0)))
        self.assertTrue(tf.equal(schrodinger.my_fcn(2)(0.0), tf.to_float(1)))

    def test_differentiate(self):
        print('differentiate')
        d_fcn = schrodinger.differentiate(lambda x: x*x)
        self.assertTrue(tf.equal(d_fcn(1.0), tf.to_float(2)))

    def test_inner_product(self):
        print('inner_product')
        position = [1, 2]
        value = schrodinger.inner_product(lambda x: x, lambda y: y, position)
        self.assertTrue(tf.equal(value, tf.to_float(1.0)))

    def test_projection_v0(self):
        print('projection_v0')
        v_list = tf.Variable([1, 2, 3, 4, 5], dtype=tf.float32)
        position = [1, 2, 3, 4, 5]
        f_basis = []
        f_basis.append(lambda x: x**x)
        f_basis.append(lambda x: pow(x, 3))
        f_basis.append(lambda x: pow(x, 4))
        coefficient_v0 = schrodinger.projection_v0(v_list, position, f_basis)
        self.assertTrue(tf.equal(tf.size(coefficient_v0), tf.to_int32(3)))

    def test_laplace_matrix(self):
        print('laplace_matrix')
        f_basis = []
        f_basis.append(lambda x: tf.math.sin(x))
        f_basis.append(lambda x: tf.math.sin(2*x))
        f_basis.append(lambda x: tf.math.sin(3*x))
        mat = schrodinger.laplace_matrix(f_basis)
        self.assertTrue(len(mat) == 3)

    def test_hamiltonian(self):
        print('hamiltonian')
        laplace_mat = tf.Variable(tf.zeros(shape=[3, 3]))
        e_v = tf.Variable([1, 2, 3], dtype=tf.float32)
        e_v = tf.reshape(tf.Variable(e_v, dtype=tf.float32), [-1, 1])
        self.assertTrue(tf.equal(tf.size(schrodinger.hamiltonian(e_v, laplace_mat)), tf.size(laplace_mat)))

    def test_wave_coefficient(self):
        print('wave_function')
        mat = tf.Variable(tf.eye(3), dtype=tf.float32)
        e, v = schrodinger.wave_coefficient(mat)
        self.assertTrue(tf.equal(e[0], tf.to_float(1)))

    def test_main(self):
        print('main')
        bul = schrodinger.main()
        self.assertTrue(bul)


if __name__ == "__main__":
    tf.test.main()

