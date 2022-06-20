import numpy as np
import onnx

#load model
model_name = "my_model/2021_12_17_model2-sim.onnx"
onnx_model = onnx.load(model_name)

def addpassnode(passnodelist,node):
    for nodei in passnodelist:
        for j in range(len(nodei.output)):
            if nodei.output[j] in node.input:

                passnodelist.append(node)
                return passnodelist
    return passnodelist

graph = onnx_model.graph
old_node = graph.node
print(len(old_node))
passnodelist = []
for i in range(len(old_node)):
    if old_node[i].name == 'Conv_245' or old_node[i].name == 'Reshape_315':
        passnodelist.append(old_node[i])

new_concat_332 = onnx.helper.make_node(
    "Concat",
    inputs=['866','874'],
    outputs=['875'],
    name= "Concat_332",
    axis=2
)
for j in range(len(old_node)):
    if old_node[j] not in passnodelist and old_node[j].name != 'Concat_332':
        passlist = addpassnode(passnodelist,old_node[j])
    if old_node[j].name == 'Concat_332':
        print(old_node[j].output)
        print("----------------------------")
        graph.node.remove(old_node[j])
        graph.node.insert(j,new_concat_332)
        print(old_node[j])
print(len(passlist))
for k in range(len(passlist)):
    graph.node.remove(passlist[k])

onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, 'out.onnx')


