--[[
Copyright 2014 Google Inc. All Rights Reserved.

Use of this source code is governed by a BSD-style
license that can be found in the LICENSE file or at
https://developers.google.com/open-source/licenses/bsd
]]

--[[
  Script to download and save mnist data.

  - gets files from Yann LeCun's web site (http://yann.lecun.com/exdb/mnist/)
  - Processes data into a table containing 'data' and 'labels' tensors.
]]

require 'os'
require 'torch'
require 'paths'


local DIR = "mnist"
local FILENAMES = { "train-images-idx3-ubyte", "train-labels-idx1-ubyte",
                    "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte" }
local URLS = { "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
               "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
               "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
               "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz" }
local TRAINSIZE = 60000
local TESTSIZE = 10000
local trainSet = { data = torch.ByteTensor(TRAINSIZE, 28, 28),
                   labels = torch.ByteTensor(TRAINSIZE, 1) }
local testSet = { data = torch.ByteTensor(TESTSIZE, 28, 28),
                  labels = torch.ByteTensor(TESTSIZE, 1) }

local function runCmd(cmd)
  local exitCode = os.execute(cmd)
  if exitCode ~= true and exitCode ~= 0 then
    error("failed cmd: " .. cmd)
  end
end

local function downloadData()
  -- download
  print "Downloading data"
  for i = 1, 4 do
    os.remove(FILENAMES[i] .. ".gz")
    runCmd("wget " .. URLS[i])
  end

  -- Unpack and store
  print "Unpacking data"
  runCmd("mkdir -p " .. DIR)

  for i = 1, 4 do
    runCmd("gunzip " .. FILENAMES[i] .. ".gz")
    assert(os.rename(FILENAMES[i], paths.concat(DIR, FILENAMES[i])))
  end
end


function processTrainData()
  -- see data format as described on http://yann.lecun.com/exdb/mnist/
  print "Reformatting training set"

  -- open training data file and check headers
  local trainData = torch.DiskFile("mnist/" .. FILENAMES[1], "r")
  trainData:binary()
  trainData:bigEndianEncoding()
  local magicNumber = trainData:readInt()
  local numberOfItems = trainData:readInt()
  local nRows = trainData:readInt()
  local nCols = trainData:readInt()
  assert(magicNumber == 2051)
  assert(numberOfItems == TRAINSIZE)
  assert(nRows == 28)
  assert(nCols == 28)

  -- open labels data file and check headers
  local trainLabels = torch.DiskFile("mnist/" .. FILENAMES[2], "r")
  trainLabels:binary()
  trainLabels:bigEndianEncoding()
  magicNumber = trainLabels:readInt()
  numberOfItems = trainLabels:readInt()
  assert(magicNumber == 2049)
  assert(numberOfItems == TRAINSIZE)

  -- read all the data
  for i = 1, TRAINSIZE do
    if i % 1000 == 0 then
      print(i .. "/" .. TRAINSIZE .. " done.")
    end
    -- read training image
    trainSet.data[i]:apply(function()
      return trainData:readByte()
    end)
    -- read label
    local byte = trainLabels:readByte()
    trainSet.labels[i][1] = byte
  end

  -- close input files
  trainData:close()
  trainLabels:close()

  -- output torch files
  local nValidExamples = 10000
  torch.save('mnist/train.t7', {
    data = trainSet.data[{{1, TRAINSIZE - nValidExamples}}]:clone(),
    labels = trainSet.labels[{{1, TRAINSIZE - nValidExamples}}]:clone(),
  })
  torch.save('mnist/valid.t7', {
    data = trainSet.data[{{TRAINSIZE - nValidExamples + 1, -1}}]:clone(),
    labels = trainSet.labels[{{TRAINSIZE - nValidExamples + 1, -1}}]:clone(),
  })
end


local function processTestData()
  -- see data format as described on http://yann.lecun.com/exdb/mnist/
  print "Reformatting test set"

  -- open training data file and check headers
  local testData = torch.DiskFile("mnist/" .. FILENAMES[3], "r")
  testData:binary()
  testData:bigEndianEncoding()
  local magicNumber = testData:readInt()
  local numberOfItems = testData:readInt()
  local nRows = testData:readInt()
  local nCols = testData:readInt()
  assert(magicNumber == 2051)
  assert(numberOfItems == TESTSIZE)
  assert(nRows == 28)
  assert(nCols == 28)

  -- open labels data file and check headers
  local testLabels = torch.DiskFile("mnist/" .. FILENAMES[4], "r")
  testLabels:binary()
  testLabels:bigEndianEncoding()
  magicNumber = testLabels:readInt()
  numberOfItems = testLabels:readInt()
  assert(magicNumber == 2049)
  assert(numberOfItems == TESTSIZE)

  -- read all the data
  for i = 1, TESTSIZE do
    if i % 1000 == 0 then
      print(i .. "/" .. TESTSIZE .. " done.")
    end
    -- read the image
    testSet.data[i]:apply(function()
      return testData:readByte()
    end)
    -- read label
    local byte = testLabels:readByte()
    testSet.labels[i][1] = byte
  end

  -- close input files
  testData:close()
  testLabels:close()

  -- output torch files
  torch.save('mnist/test.t7', testSet)
end

local function processData()
  print "Processing data"
  processTrainData()
  processTestData()
end

-- Execution starts here
downloadData()

-- Process data into Lua table & tensors
processData()
