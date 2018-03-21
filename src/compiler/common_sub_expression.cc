/*!
 *  Copyright (c) 2017 by Contributors
 * \file common_sub_expression.cc
 * \brief take common expressions out and replace with one.
 *
 */
#include <nnvm/graph.h>
#include <nnvm/pass.h>

namespace nnvm {
namespace compiler {

nnvm::Graph CommonSubExpression(nnvm::Graph src) {
  std::vector<NodeEntry> commonNodeList;
  std::unordered_map<std::string, int> unique_expressn;
  std::string expressn = "";
  std::function<std::string(const NodeEntry& e, std::string)>find_cummulative_expression
    = [&find_cummulative_expression, &commonNodeList, &unique_expressn]
    (const NodeEntry& e, std::string expressn)->std::string {
    std::string tempExprssn = "";
    if (e.node->is_variable()) {
      tempExprssn += e.node->attrs.name;
    } else {
      for (auto& e : e.node->inputs) {
        tempExprssn = find_cummulative_expression(e, tempExprssn);
        // Replace already commoned expression with its global index
        if (unique_expressn.count(tempExprssn)) {
          e = commonNodeList[unique_expressn.at(tempExprssn)];
          tempExprssn = std::to_string(unique_expressn.at(tempExprssn));
        }
        expressn += tempExprssn;
        tempExprssn = "";
      }

      tempExprssn = e.node->op()->name + expressn;
    }
    return tempExprssn;
  };

  DFSVisit(src.outputs, [&](const nnvm::NodePtr& n) {
    // If variable then, there is nothing to take common
    if (!n->is_variable()) {
      // If non-variable, then form logical expression
      expressn = "";
      expressn = find_cummulative_expression(nnvm::NodeEntry{n, 0, 0}, expressn);
      commonNodeList.push_back(nnvm::NodeEntry{n, 0, 0});
      unique_expressn.emplace(expressn, commonNodeList.size() - 1);
    }
  });

  return src;
}

NNVM_REGISTER_PASS(CommonSubExpression)
.set_body(CommonSubExpression);
}  // namespace compiler
}  // namespace nnvm
