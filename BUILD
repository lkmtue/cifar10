cc_binary(
  name = "main",
  srcs = ["main.cc"],
  deps = [
    ":data",
    ":net",
    ":input_layer",
    ":conv_layer",
    ":max_pooling_layer",
    ":fully_connected_layer",
    ":util",
    ":softmax_loss_layer",
    ":relu_layer",
    ":average_pooling_layer"
  ],
  copts = [
  "-I/usr/local/opt/openblas/include"
  ],
  linkopts = [
    "-L/usr/local/opt/openblas/lib",
    "-lopenblas"
  ]
)

cc_binary(
  name = "gradient_test",
  srcs = ["gradient_test.cc"],
  deps = [
    ":average_pooling_layer",
    ":data",
    ":net",
    ":input_layer",
    ":conv_layer",
    ":max_pooling_layer",
    ":fully_connected_layer",
    ":util",
    ":relu_layer",
    ":softmax_loss_layer"
  ],
  copts = [
    "-I/usr/local/opt/openblas/include"
  ],
  linkopts = [
    "-L/usr/local/opt/openblas/lib",
    "-lopenblas"
  ]
)

cc_library(
  name = "net",
  srcs = ["net.h"],
  deps = [
    ":data",
    ":input_layer",
    ":softmax_loss_layer",
    ":util"
  ]
)

cc_library(
  name = "input_layer",
  srcs = [
    "layer.h",
    "input_layer.h"
  ],
  deps = [":util"]
)

cc_library(
  name = "conv_layer",
  srcs = [
    "layer.h",
    "conv_layer.h"
  ],
  deps = [
    ":activation",
    ":filler",
    ":im2col",
    ":util"
  ]
)

cc_library(
  name = "max_pooling_layer",
  srcs = [
    "layer.h",
    "max_pooling_layer.h"
  ],
  deps = [":util"]
)

cc_library(
  name = "average_pooling_layer",
  srcs = [
    "layer.h",
    "average_pooling_layer.h"
  ],
  deps = [":util"]
)

cc_library(
  name = "relu_layer",
  srcs =[
    "layer.h",
    "relu_layer.h"
  ],
  deps = [":util"]
)

cc_library(
  name = "fully_connected_layer",
  srcs = [
    "layer.h",
    "fully_connected_layer.h"
  ],
  deps = [
    ":activation",
    ":filler",
    ":util"
  ]
)

cc_library(
  name = "softmax_loss_layer",
  srcs = [
    "layer.h",
    "softmax_loss_layer.h"
  ],
  deps = [":util"]
)

cc_library(
  name = "util",
  srcs = ["util.h"]
)

cc_library(
  name = "data",
  srcs = ["data.h"]
)

cc_library(
  name = "filler",
  srcs = ["filler.h"]
)

cc_library(
  name = "activation",
  srcs = ["activation.h"]
)


cc_library(
  name = "im2col",
  srcs = ["im2col.h"]
)
