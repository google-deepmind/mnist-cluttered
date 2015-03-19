Cluttered MNIST Dataset
=======================

A setup script will download MNIST and produce `mnist/*.t7` files:

    luajit download_mnist.lua

Example usage:

    local mnist_cluttered = require 'mnist_cluttered'
    -- The observation will have size 1x100x100 with 8 distractors.
    local dataConfig = {megapatch_w=100, num_dist=8}
    local dataInfo = mnist_cluttered.createData(dataConfig)
    local observation, target = unpack(dataInfo.nextExample())
