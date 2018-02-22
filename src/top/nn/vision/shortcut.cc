/*!
 *  Copyright (c) 2017 by Contributors
 * \file shortcut.cc
 * \brief Property def of pooling operators.
 */
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/top/nn.h>
#include "../nn_common.h"
#include "../../op_common.h"
#include "shortcut.h"

namespace nnvm {
namespace top {

    NNVM_REGISTER_OP(shortcut)
        .describe(R"code(Shortcut layer (sometimes called Shortcut Connections).
                Shortcut connections are the connections which skip one or more layers.
                )code"  NNVM_ADD_FILELINE)
        .set_num_inputs(2)
        .set_num_outputs(1)
        .set_support_level(1)
        .add_argument("lhs", "Tensor", "first input")
        .add_argument("rhs", "Tensor", "second input")
        .set_attr<FInferType>("FInferType", ShortcutType<2, 1>)
        .set_attr<FInferShape>("FInferShape", ShortcutShape<2, 1>)
        .set_attr<FInplaceOption>(
                "FInplaceOption", [](const NodeAttrs& attrs) {
                return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}};
            })
        .set_attr<FGradient>(
                "FGradient", [](const NodePtr& n,
                        const std::vector<NodeEntry>& ograds){
                return std::vector<NodeEntry>{ograds[0], ograds[0]};
            });
}  // namespace top
}  // namespace nnvm
